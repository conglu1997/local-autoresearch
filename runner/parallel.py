"""Run asynchronous research workers, eight by default, against one model server."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from urllib.error import URLError

from runner.core import candidate_source, feedback, make_team_prompt, save
from runner.evaluate import CPU_SECONDS, ROOT, WALL_SECONDS, digest, evaluate
from runner.inference import ServerError, api_key, complete, model_server
from runner.search import structural_hash
from runner.teams import REQUESTS, Experiment, Team, configuration
from tasks import TASKS

RETRY_SECONDS = 5


class Stopped(Exception):
    pass


def raise_stopped(signum, frame):
    raise Stopped


def prompt_for(root, brief, selection, ledger):
    return (
        make_team_prompt(brief, root / selection["parent"], ledger)
        + "\n"
        + REQUESTS[selection["request_variant"]]
    )


def attempt(directory):
    """Generate and evaluate one candidate, stopping at the deadline or on SIGTERM."""
    job = json.loads((directory / "admission.json").read_text())
    manifest = json.loads((directory.parent / "manifest.json").read_text())
    inference = manifest["inference"]
    signal.signal(signal.SIGTERM, raise_stopped)
    signal.signal(signal.SIGALRM, raise_stopped)
    signal.setitimer(signal.ITIMER_REAL, max(0.001, job["deadline"] - time.time()))
    try:
        body = json.loads((directory / "request.json").read_text())
        # The team scheduler owns retries and the worker's deadline.
        response = complete(
            body, inference["endpoint"], api_key(inference), directory, max_retries=0
        )
        try:
            source = candidate_source(response)
        except (ValueError, SyntaxError) as error:
            result = {"valid": False, "mean_score": None, "error": str(error)}
        else:
            (directory / "solver.py").write_text(source)
            result = evaluate(source, manifest["seeds"], manifest["task"])
    except Stopped:
        save(directory / "interrupted.json", {"time": time.time()})
        return
    except (URLError, TimeoutError, ConnectionError, ServerError) as error:
        result = {"valid": False, "mean_score": None, "error": f"Inference failed: {error}"}
    signal.setitimer(signal.ITIMER_REAL, 0)
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    save(directory / "evaluation.json", result)
    save(directory / "completed.json", {"time": time.time()})


def append_event(root, event):
    with (root / "journal.jsonl").open("a") as stream:
        stream.write(json.dumps(event) + "\n")


def stop(process):
    if process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()


def run(args):
    root = args.run.resolve()
    workers = args.workers
    status = model_server(args.base_url, args.model, args.api_key_env)
    budget_seconds = args.hours * 3600
    root.mkdir(parents=True)
    paths = [ROOT / "research.py", ROOT / "uv.lock"]
    for folder in ("runner", "tasks", "tools"):
        paths.extend(sorted((ROOT / folder).rglob("*.py")))
    paths += [ROOT / f"tasks/{args.task}/{name}" for name in ("program.md", "baseline.py")]
    paths.extend((ROOT / f"tasks/{args.task}").glob("*.csv"))
    sampling = {"temperature": 1.0, "top_p": 0.95, "max_tokens": args.max_tokens}
    if args.base_url is None:
        sampling["top_k"] = 20
    started = time.time()
    deadline = started + budget_seconds
    manifest = {
        "task": args.task,
        "parallel": configuration(workers, args.replicate),
        "model": status["model"],
        "inference": status,
        "budget_seconds": budget_seconds,
        "started_at": datetime.fromtimestamp(started, timezone.utc).isoformat(),
        "problem": TASKS[args.task].PROBLEM,
        "seeds": [0],
        "cpu_seconds_per_seed": CPU_SECONDS,
        "wall_seconds_per_seed": WALL_SECONDS,
        "sampling": sampling,
        "fixed_files": {
            str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths
        },
    }
    if hasattr(TASKS[args.task], "FINAL_PROTOCOL"):
        manifest["final_test"] = TASKS[args.task].final_protocol(budget_seconds)
    save(root / "manifest.json", manifest)

    source = (ROOT / f"tasks/{args.task}/baseline.py").read_text()
    (root / "000").mkdir()
    (root / "000/solver.py").write_text(source)
    result = evaluate(source, manifest["seeds"], args.task)
    save(root / "000/evaluation.json", result)
    if not result["valid"]:
        raise RuntimeError("The baseline failed; see 000/evaluation.json")
    save(root / "000/decision.json", {"decision": "baseline", "parent": None})
    (root / "best.py").write_text(source)

    team = Team(Experiment("000", -1, result["mean_score"], True), workers, args.replicate)
    baseline_entry = {"attempt": "000", "decision": "baseline", **feedback(result)}
    ledgers = [[baseline_entry] for _ in range(workers)]
    brief = (ROOT / f"tasks/{args.task}/program.md").read_text()
    seen = {structural_hash(source): "000"}
    active, failures, next_ready = {}, [0] * workers, [0.0] * workers
    number, completed = 1, 0

    def write_summary():
        save(
            root / "summary.json",
            {
                "best_attempt": team.best.attempt,
                "best_score": team.best.score,
                "completed_experiments": completed,
                "incumbents": [e.attempt for e in team.incumbents],
            },
        )

    def collect(agent):
        nonlocal completed
        process, directory, log = active.pop(agent)
        log.close()
        if (directory / "interrupted.json").exists():
            return
        if (directory / "completed.json").exists():
            failures[agent] = 0
        else:
            failures[agent] += 1
            crashed = {
                "valid": False,
                "mean_score": None,
                "error": f"Worker exited {process.returncode}; see worker.log",
            }
            if (directory / "solver.py").exists():
                crashed["source_sha256"] = digest((directory / "solver.py").read_text())
            save(directory / "evaluation.json", crashed)
            save(directory / "completed.json", {"time": time.time()})
        when = json.loads((directory / "completed.json").read_text())["time"]
        if when > deadline:
            save(directory / "interrupted.json", {"time": when})
            return
        result = json.loads((directory / "evaluation.json").read_text())
        record = team.observe(
            Experiment(directory.name, agent, result["mean_score"], result["valid"])
        )
        admission = json.loads((directory / "admission.json").read_text())
        record.update(agent=agent, parent=admission["parent"], elapsed_seconds=when - started)
        if (directory / "solver.py").exists():
            fingerprint = structural_hash((directory / "solver.py").read_text())
            record["structural_sha256"] = fingerprint
            if fingerprint in seen:
                record["duplicate_of"] = seen[fingerprint]
            seen.setdefault(fingerprint, directory.name)
        save(directory / "decision.json", record)
        append_event(root, {"type": "complete", "attempt": directory.name, "agent": agent})
        completed += 1
        ledgers[agent].append(
            {"attempt": directory.name, "decision": record["decision"], **feedback(result)}
        )
        (root / "best.py").write_text((root / team.best.attempt / "solver.py").read_text())
        write_summary()
        next_ready[agent] = time.time() + (0 if result["valid"] else RETRY_SECONDS)
        print(directory.name, agent, record["decision"], result["mean_score"], flush=True)

    stopping = False

    def request_stop(signum, frame):
        nonlocal stopping
        stopping = True

    previous = {sig: signal.signal(sig, request_stop) for sig in (signal.SIGINT, signal.SIGTERM)}
    reason = "time budget"
    try:
        while time.time() < deadline and not stopping:
            for agent in [a for a, (process, _, _) in active.items() if process.poll() is not None]:
                collect(agent)
            if min(failures) >= 3:
                reason = "every worker failed three times in a row"
                break
            for agent in range(workers):
                if agent in active or failures[agent] >= 3 or time.time() < next_ready[agent]:
                    continue
                selection = team.select(agent)
                directory = root / f"{number:03d}"
                directory.mkdir()
                save(
                    directory / "admission.json",
                    {**selection, "elapsed_seconds": time.time() - started, "deadline": deadline},
                )
                prompt = prompt_for(root, brief, selection, ledgers[agent])
                body = {
                    "model": status["model"],
                    "messages": [{"role": "user", "content": prompt}],
                    "stream": True,
                    "stream_options": {"include_usage": True},
                    **sampling,
                }
                save(directory / "request.json", body)
                append_event(root, {"type": "admit", "attempt": directory.name, "agent": agent})
                log = (directory / "worker.log").open("w")
                process = subprocess.Popen(
                    [sys.executable, "-m", "runner.parallel", "--attempt", str(directory)],
                    cwd=ROOT,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
                active[agent] = (process, directory, log)
                number += 1
            time.sleep(0.1)
        if stopping:
            reason = "stopped"
    finally:
        research_seconds = time.time() - started
        for process, _, _ in active.values():
            process.terminate()
        for agent in list(active):
            stop(active[agent][0])
            collect(agent)
        for sig, handler in previous.items():
            signal.signal(sig, handler)
        write_summary()
        save(
            root / "finished.json",
            {
                "reason": "error" if sys.exception() else reason,
                "research_seconds": research_seconds,
                "completed_experiments": completed,
                "admitted_experiments": number - 1,
            },
        )
    print(f"Finished ({reason}). Best attempt {team.best.attempt}: {team.best.score}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run one team attempt")
    parser.add_argument("--attempt", type=Path, required=True)
    attempt(parser.parse_args().attempt)
