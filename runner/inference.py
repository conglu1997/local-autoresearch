"""Recorded, resumable streaming requests shared by research modes."""

import json
import os
import ssl
import subprocess
import time
from datetime import datetime, timezone
from http.client import IncompleteRead
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener

from runner.core import save


CALL_FILES = {
    "request.json",
    "response.json",
    "response.txt",
    "stream.jsonl",
    "http-error.txt",
    "failure.json",
}
RETRYABLE_STATUS = {408, 429, 500, 502, 503, 504}


class NoRedirects(HTTPRedirectHandler):
    """Keep API credentials on the configured endpoint."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


OPENER = build_opener(NoRedirects)


class ServerError(RuntimeError):
    """The model server reported an error inside the response stream."""


def api_key(inference):
    if "api_key_env" not in inference:
        return subprocess.check_output(["rec", "auth", "token"], text=True).strip()
    name = inference["api_key_env"]
    if name is None:
        return None
    key = os.environ[name].strip()
    if not key:
        raise ValueError(f"{name} must contain an API key")
    return key


def model_server(base_url, model, api_key_env):
    if base_url is None:
        if model is not None or api_key_env is not None:
            raise ValueError("--model and --api-key-env require --base-url")
        status = json.loads(subprocess.check_output(["rec", "status"], text=True))
        if not status["ready"] or status["model"] != "flint":
            raise RuntimeError("Start the local Flint server before running research")
    else:
        if not model:
            raise ValueError("--base-url requires --model")
        status = {
            "endpoint": base_url.rstrip("/"),
            "model": model,
            "api_key_env": api_key_env,
        }
    url = urlsplit(status["endpoint"])
    if (
        url.scheme not in {"http", "https"}
        or not url.hostname
        or (url.scheme == "http" and url.hostname not in {"127.0.0.1", "localhost", "::1"})
        or url.username is not None
        or url.password is not None
        or url.query
        or url.fragment
    ):
        raise ValueError(
            "Use an HTTPS base URL or a loopback HTTP URL, "
            "without credentials, query parameters or fragments"
        )
    return status


def stream_text(delta):
    return {
        "content": delta.get("content") or "",
        "reasoning": delta.get("reasoning") or delta.get("reasoning_content") or "",
    }


def stream_tools(calls, delta):
    """Assemble tool calls whose names and JSON arguments arrive in separate chunks."""
    for fragment in delta.get("tool_calls") or []:
        call = calls.setdefault(
            fragment["index"],
            {"id": "", "type": "function", "function": {"name": "", "arguments": ""}},
        )
        call["id"] += fragment.get("id") or ""
        if fragment.get("type"):
            call["type"] = fragment["type"]
        for field in ("name", "arguments"):
            call["function"][field] += (fragment.get("function") or {}).get(field) or ""


def complete_once(body, endpoint, key, directory):
    save(directory / "request.json", body)
    headers = {"Content-Type": "application/json"}
    if key is not None:
        headers["Authorization"] = "Bearer " + key
    request = Request(
        endpoint + "/chat/completions",
        data=json.dumps(body).encode(),
        headers=headers,
    )
    started = time.monotonic()
    text, reasoning, usage, finish = "", "", None, None
    first_token = None
    metadata = {}
    calls, reasoning_details = {}, []
    try:
        with (
            OPENER.open(request, timeout=300) as response,
            (directory / "stream.jsonl").open("w") as log,
        ):
            for raw in response:
                line = raw.decode().rstrip("\r\n")
                elapsed = time.monotonic() - started
                log.write(json.dumps({"seconds": elapsed, "line": line}, ensure_ascii=False) + "\n")
                log.flush()
                if not line.startswith("data: ") or line == "data: [DONE]":
                    continue
                chunk = json.loads(line[6:])
                for name in ("id", "model", "provider", "error"):
                    if chunk.get(name) is not None:
                        metadata[name] = chunk[name]
                if chunk.get("usage") is not None:
                    usage = chunk["usage"]
                for choice in chunk.get("choices", []):
                    delta = choice.get("delta") or {}
                    parts = stream_text(delta)
                    content, thought = parts["content"], parts["reasoning"]
                    text += content
                    reasoning += thought
                    if body.get("tools"):
                        stream_tools(calls, delta)
                        reasoning_details.extend(delta.get("reasoning_details") or [])
                    if first_token is None and (content or thought or calls or reasoning_details):
                        first_token = elapsed
                    if choice.get("finish_reason") is not None:
                        finish = choice["finish_reason"]
    except HTTPError as error:
        evidence = directory / "http-error.txt"
        evidence.write_text(error.read().decode(errors="replace"))
        error.add_note(f"Model request to {endpoint} failed; response body saved to {evidence}")
        raise
    except (URLError, TimeoutError, ssl.SSLError, json.JSONDecodeError) as error:
        error.add_note(
            f"Model request to {endpoint} failed; request and any partial stream saved in {directory}"
        )
        raise
    result = {
        "content": text,
        "reasoning": reasoning,
        "usage": usage,
        "finish_reason": "error" if "error" in metadata else finish,
        "seconds": time.monotonic() - started,
        "first_token_seconds": first_token,
        **metadata,
    }
    if body.get("tools"):
        result["tool_calls"] = [calls[index] for index in sorted(calls)]
        if reasoning_details:
            result["reasoning_details"] = reasoning_details
    save(directory / "response.json", result)
    (directory / "response.txt").write_text(text)
    return result


def retry_reason(response):
    """Only transport failures are retried; completed model decisions are left intact."""
    error = response.get("error")
    if error is not None:
        code = error.get("code") if isinstance(error, dict) else None
        if str(code) in {str(status) for status in RETRYABLE_STATUS}:
            return f"Transient provider error: {json.dumps(error, ensure_ascii=False)}"
        return None
    if response["finish_reason"] is None:
        return "Stream ended before a finish_reason; partial tool calls were not executed"
    return None


def response_problem(response):
    return retry_reason(response) or (
        f"Provider error: {json.dumps(response['error'], ensure_ascii=False)}"
        if response.get("error") is not None
        else None
    )


def archive_request(directory, reason, delay=0, backoff_index=None):
    """Preserve failed or interrupted requests across repeated resumes."""
    retries = directory / "retries"
    number = max((int(p.name) for p in retries.glob("*") if p.name.isdigit()), default=0) + 1
    archive = retries / f"{number:03d}"
    archive.mkdir(parents=True)
    for name in CALL_FILES:
        path = directory / name
        if path.exists():
            path.rename(archive / name)
    save(
        archive / "retry.json",
        {
            "reason": reason,
            "retry": number,
            "delay_seconds": delay,
            "backoff_index": backoff_index,
            "at": datetime.now(timezone.utc).isoformat(),
        },
    )


def complete(body, endpoint, key, directory, *, max_retries=5):
    """Retry the same request before its response reaches the agent or evaluator."""
    if max_retries < 0:
        raise ValueError("max_retries must be nonnegative")
    # Rebuild the same conversation, reusing only complete responses. An interrupted
    # request may already have been moved into retries/ before a backoff or shutdown.
    requests = sorted((directory / "retries").glob("*/request.json"))
    if (directory / "request.json").exists():
        requests.append(directory / "request.json")
    for path in requests:
        if json.loads(path.read_text()) != body:
            raise RuntimeError(f"Saved request differs from the resumed request: {path}")
    if (directory / "response.json").exists():
        recorded = json.loads((directory / "response.json").read_text())
        if response_problem(recorded) is None:
            if not (directory / "request.json").exists():
                raise RuntimeError(f"Saved response has no request: {directory}")
            return recorded
    if (directory / "request.json").exists():
        failure = directory / "failure.json"
        reason = (
            json.loads(failure.read_text())["error_summary"]
            if failure.exists()
            else (
                response_problem(recorded)
                if (directory / "response.json").exists()
                else "Interrupted before a complete response was saved"
            )
        )
        archive_request(directory, reason)
    for retry in range(max_retries + 1):
        caught = None
        result = None
        try:
            result = complete_once(body, endpoint, key, directory)
        except (
            URLError,
            TimeoutError,
            ssl.SSLError,
            ConnectionError,
            IncompleteRead,
            json.JSONDecodeError,
            UnicodeDecodeError,
        ) as error:
            caught = error
            reason = f"{type(error).__name__}: {error}"
            cause = error.reason if isinstance(error, URLError) else error
            retryable = not isinstance(cause, ssl.SSLCertVerificationError) and (
                not isinstance(error, HTTPError) or error.code in RETRYABLE_STATUS
            )
            evidence = directory / "http-error.txt"
            if isinstance(error, HTTPError) and evidence.exists():
                reason += "; " + evidence.read_text()[:2000]
        else:
            reason = response_problem(result)
            if reason is None:
                (directory / "failure.json").unlink(missing_ok=True)
                return result
            retryable = retry_reason(result) is not None
        save(
            directory / "failure.json",
            {
                "failure_kind": "inference_error",
                "error_summary": reason,
                "retryable": retryable,
                "attempts_this_call": retry + 1,
                "at": datetime.now(timezone.utc).isoformat(),
            },
        )
        if not retryable or retry == max_retries:
            print(
                f"  request {directory}: stopped after {retry + 1} attempts: {reason}; "
                "checkpoint preserved for --resume",
                flush=True,
            )
            if caught is not None:
                caught.add_note(f"Failed after {retry + 1} attempts; evidence saved in {directory}")
                raise caught
            provider = result.get("error")
            message = provider.get("message") if isinstance(provider, dict) else None
            error = ServerError(
                message or f"Model request failed: {reason}; checkpoint saved in {directory}"
            )
            error.add_note(f"Model request failed: {reason}; checkpoint saved in {directory}")
            raise error
        delay = min(30 * 2**retry, 300)
        archive_request(directory, reason, delay, retry + 1)
        print(
            f"  request {directory}: {reason}; retry {retry + 1}/{max_retries} in {delay}s",
            flush=True,
        )
        time.sleep(delay)
