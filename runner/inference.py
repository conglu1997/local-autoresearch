"""Recorded streaming chat requests to rec or an OpenAI-compatible endpoint."""

import json
import os
import subprocess
import time
from urllib.error import HTTPError
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener

from runner.core import save


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


def complete(body, endpoint, key, directory):
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
                if "error" in chunk:
                    raise ServerError(chunk["error"]["message"])
                if chunk.get("usage") is not None:
                    usage = chunk["usage"]
                for choice in chunk["choices"]:
                    delta = stream_text(choice["delta"])
                    content, thought = delta["content"], delta["reasoning"]
                    text += content
                    reasoning += thought
                    if first_token is None and (content or thought):
                        first_token = elapsed
                    if choice["finish_reason"] is not None:
                        finish = choice["finish_reason"]
    except HTTPError as error:
        (directory / "http-error.txt").write_text(error.read().decode())
        raise
    result = {
        "content": text,
        "reasoning": reasoning,
        "usage": usage,
        "finish_reason": finish,
        "seconds": time.monotonic() - started,
        "first_token_seconds": first_token,
    }
    save(directory / "response.json", result)
    (directory / "response.txt").write_text(text)
    return result
