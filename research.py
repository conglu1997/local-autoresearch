"""A model proposes; the local evaluator measures; improvements become the next parent."""

import argparse
import ast
import hashlib
import json
import os
import re
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

from runner.evaluate import CPU_SECONDS, ROOT, WALL_SECONDS, evaluate
from tasks import TASKS


def save(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


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


def feedback(result):
    if "trials" not in result:
        return result
    return {
        "valid": result["valid"],
        "mean_score": result["mean_score"],
        "trials": [
            {
                key: value
                for key, value in trial.items()
                if key not in ("circles", "certificate", "solution", "correlations")
            }
            for trial in result["trials"]
        ],
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
            urlopen(request, timeout=300) as response,
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


def candidate_source(response):
    if response["finish_reason"] != "stop":
        raise ValueError(f"Model response did not finish: {response['finish_reason']}")
    blocks = re.findall(r"```python\s*\n(.*?)```", response["content"], re.DOTALL)
    if len(blocks) != 1:
        raise ValueError("Expected exactly one complete Python code block")
    source = blocks[0].strip() + "\n"
    ast.parse(source)
    return source


def make_prompt(brief, best, ledger):
    prompt = (
        brief
        + "\nCurrent best solver:\n```python\n"
        + (best / "solver.py").read_text()
        + "```\nRecent results:\n"
        + json.dumps(ledger[-4:], indent=2)
    )
    previous = best.parent / ledger[-1]["attempt"]
    if previous != best and (previous / "solver.py").exists():
        prompt += (
            f"\nPrevious attempt ({ledger[-1]['decision']}):\n```python\n"
            + (previous / "solver.py").read_text()
            + "```\n"
        )
    return prompt + "\nWhat would you try next?"


def run(args):
    status = model_server(args.base_url, args.model, args.api_key_env)
    endpoint = status["endpoint"]
    key = api_key(status)
    root = args.run.resolve()
    if args.resume:
        manifest = json.loads((root / "manifest.json").read_text())
        for name, expected in manifest["fixed_files"].items():
            if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != expected:
                raise RuntimeError(f"Fixed file changed since this run began: {name}")
        if status != manifest["inference"]:
            raise RuntimeError("Model configuration changed; start a new run")
    else:
        root.mkdir(parents=True, exist_ok=False)
        task_files = [f"tasks/{args.task}/{name}" for name in ("baseline.py", "program.md")]
        fixed_files = [
            "research.py",
            "runner/__init__.py",
            "runner/evaluate.py",
            "runner/worker.py",
            "uv.lock",
            *task_files,
        ]
        fixed_files += ["tasks/__init__.py", *[f"tasks/{name}/__init__.py" for name in TASKS]]
        fixed_files += [
            str(path.relative_to(ROOT))
            for path in sorted((ROOT / "tasks" / args.task).glob("*.csv"))
        ]
        if hasattr(TASKS[args.task], "FINAL_PROTOCOL"):
            fixed_files += ["tools/__init__.py", "tools/final_test.py"]
        manifest = {
            "task": args.task,
            "started_at": datetime.now(timezone.utc).isoformat(),
            "model": status["model"],
            "inference": status,
            "problem": TASKS[args.task].PROBLEM,
            "cpu_seconds_per_seed": CPU_SECONDS,
            "wall_seconds_per_seed": WALL_SECONDS,
            "seeds": [0],
            "sampling": {
                "temperature": 1.0,
                "top_p": 0.95,
                "max_tokens": args.max_tokens,
            },
            "fixed_files": {
                name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in fixed_files
            },
        }
        if args.base_url is None:
            manifest.update(engine=status["engine"], rec_version=status["version"])
            manifest["sampling"]["top_k"] = 20
        if args.task == "circle":
            manifest["tolerance"] = TASKS[args.task].TOLERANCE
        if hasattr(TASKS[args.task], "FINAL_PROTOCOL"):
            manifest["final_test"] = TASKS[args.task].FINAL_PROTOCOL
        save(root / "manifest.json", manifest)
        baseline = root / "000"
        baseline.mkdir()
        source = (ROOT / task_files[0]).read_text()
        (baseline / "solver.py").write_text(source)
        result = evaluate(source, manifest["seeds"], manifest["task"])
        if not result["valid"]:
            save(baseline / "evaluation.json", result)
            raise RuntimeError("Baseline failed; inspect 000/evaluation.json before research")
        save(baseline / "evaluation.json", result)
        save(baseline / "decision.json", {"decision": "baseline", "parent": None})
        print(f"000 baseline {result['mean_score']:.9f}", flush=True)
    task = manifest["task"]
    brief = ROOT / f"tasks/{task}/program.md"
    attempts = sorted(path for path in root.iterdir() if path.is_dir() and path.name.isdigit())
    best = root / "000"
    ledger = []
    for path in attempts:
        decision = json.loads((path / "decision.json").read_text())["decision"]
        result = json.loads((path / "evaluation.json").read_text())
        ledger.append({"attempt": path.name, "decision": decision, **feedback(result)})
        if decision in ("keep", "baseline"):
            best = path
    best_score = json.loads((best / "evaluation.json").read_text())["mean_score"]
    for number in range(len(attempts), args.experiments + 1):
        directory = root / f"{number:03d}"
        directory.mkdir()
        prompt = make_prompt(brief.read_text(), best, ledger)
        body = {
            "model": manifest["model"],
            "messages": [{"role": "user", "content": prompt}],
            "stream": True,
            "stream_options": {"include_usage": True},
            **manifest["sampling"],
        }
        print(f"{number:03d} asking {manifest['model']}; parent {best.name}", flush=True)
        response = complete(body, endpoint, key, directory)
        try:
            source = candidate_source(response)
        except (ValueError, SyntaxError) as error:
            result = {"valid": False, "mean_score": None, "error": str(error)}
        else:
            (directory / "solver.py").write_text(source)
            result = evaluate(source, manifest["seeds"], task)
        save(directory / "evaluation.json", result)
        decision = "invalid"
        if result["valid"]:
            decision = "keep" if result["mean_score"] > best_score + 1e-9 else "discard"
        save(directory / "decision.json", {"decision": decision, "parent": best.name})
        if decision == "keep":
            best, best_score = directory, result["mean_score"]
        ledger.append({"attempt": directory.name, "decision": decision, **feedback(result)})
        (root / "best.py").write_text((best / "solver.py").read_text())
        save(
            root / "summary.json",
            {"best_attempt": best.name, "best_score": best_score, "completed_experiments": number},
        )
        print(f"{number:03d} {decision}: {result['mean_score']}; best {best_score:.9f}", flush=True)
    print(f"Done. Best attempt: {best.name}, score: {best_score:.9f}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument(
        "--experiments", type=int, default=8, help="Total experiments, excluding baseline"
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--task", choices=list(TASKS), default="circle")
    parser.add_argument("--max-tokens", type=int, default=32768)
    parser.add_argument(
        "--base-url", help="Chat API base URL, including /v1; HTTPS or loopback HTTP"
    )
    parser.add_argument("--model", help="Model ID served at --base-url; defaults to Flint via rec")
    parser.add_argument("--api-key-env", help="Environment variable holding the API key")
    run(parser.parse_args())
