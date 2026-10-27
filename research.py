"""A model proposes; the local evaluator measures; improvements become the next parent."""

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from runner.core import candidate_source, feedback, make_prompt, save
from runner.evaluate import CPU_SECONDS, ROOT, WALL_SECONDS, evaluate
from runner.inference import api_key, complete, model_server
from runner.teams import WORKERS
from tasks import TASKS


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
            "runner/core.py",
            "runner/inference.py",
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
            fixed_files += ["tools/__init__.py", "tools/final_test.py", "tools/final_protocol.py"]
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
    if not (attempts[-1] / "decision.json").exists():
        interrupted = attempts.pop()
        earlier = len(list(root.glob(f"{interrupted.name}-interrupted-*")))
        interrupted.rename(root / f"{interrupted.name}-interrupted-{earlier + 1}")
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


def main(argv=None):
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
    parser.add_argument("--hours", type=float, help="Run a team of workers for this long instead")
    parser.add_argument("--workers", type=int, default=WORKERS, help="Team size")
    parser.add_argument("--replicate", type=int, default=0, help="Team direction offset")
    args = parser.parse_args(argv)
    if args.hours is None:
        run(args)
    else:
        from runner.parallel import run as run_team

        run_team(args)


if __name__ == "__main__":
    main()
