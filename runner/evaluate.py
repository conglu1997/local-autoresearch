"""Run an isolated solver and independently score its returned solution."""

import hashlib
import json
import math
import os
import resource
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from tasks import TASKS

ROOT = Path(__file__).resolve().parents[1]
CPU_SECONDS = 15
WALL_SECONDS = 25
SEEDS = (0,)


def digest(text):
    return hashlib.sha256(text.encode()).hexdigest()


def sandbox_profile(work):
    # Block user data and mounted volumes; allow Python, dependencies and the candidate workspace.
    reads = [
        work,
        Path(sys.prefix),
        Path(sys.base_prefix),
        Path(sys.executable).resolve().parent.parent,
    ]
    rules = [
        "(version 1)",
        "(allow default)",
        "(deny network*)",
        "(deny process-fork)",
        '(deny file-read-data (subpath "/Users") (subpath "/private") (subpath "/Volumes") (subpath "/opt") (subpath "/Library"))',
        "(deny file-write*)",
    ]
    rules += [f"(allow file-read-data (subpath {json.dumps(str(p.resolve()))}))" for p in reads]
    rules += [
        f'(allow file-write* (subpath {json.dumps(str(work.resolve()))}) (literal "/dev/null"))'
    ]
    return "\n".join(rules)


def run_seed(source, seed, task="circle", *, final=False):
    if final and not hasattr(TASKS[task], "final_problem"):
        raise ValueError(f"Task {task} has no held-out test split")
    if sys.platform != "darwin":
        raise RuntimeError("The isolated runner currently requires macOS sandbox-exec")
    with tempfile.TemporaryDirectory(prefix="flint-packing-") as directory:
        work = Path(directory).resolve()
        (work / "solver.py").write_text(source)
        (work / "worker.py").write_text(Path(__file__).with_name("worker.py").read_text())
        problem = TASKS[task].PROBLEM
        if final:
            problem = TASKS[task].final_problem()
        (work / "input.json").write_text(json.dumps({"problem": problem, "seed": seed}))
        (work / "sandbox.sb").write_text(sandbox_profile(work))
        env = {
            "PATH": "/usr/bin:/bin",
            "HOME": str(work),
            "TMPDIR": str(work),
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONNOUSERSITE": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "OMP_NUM_THREADS": "1",
            "VECLIB_MAXIMUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
        }
        start = time.monotonic()
        cpu_before = resource.getrusage(resource.RUSAGE_CHILDREN)
        # Keep the original file handles: candidates can replace names inside their workspace.
        with (
            (work / "stdout.txt").open("w+b") as stdout,
            (work / "stderr.txt").open("w+b") as stderr,
            (work / "solution.json").open("w+b") as output,
        ):
            process = subprocess.Popen(
                [
                    "/usr/bin/sandbox-exec",
                    "-f",
                    str(work / "sandbox.sb"),
                    sys.executable,
                    "-I",
                    str(work / "worker.py"),
                    str(CPU_SECONDS),
                ],
                cwd=work,
                env=env,
                stdout=stdout,
                stderr=stderr,
                start_new_session=True,
            )
            timeout = False
            try:
                process.wait(timeout=WALL_SECONDS)
            except subprocess.TimeoutExpired:
                timeout = True
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            finally:
                if process.poll() is None:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
            stdout.seek(0)
            stderr.seek(0)
            output.seek(0)
            stdout_text = stdout.read().decode("utf-8", errors="backslashreplace")
            stderr_text = stderr.read().decode("utf-8", errors="backslashreplace")
            solution_bytes = output.read()
        cpu_after = resource.getrusage(resource.RUSAGE_CHILDREN)
        result = {
            "seed": seed,
            "wall_seconds": time.monotonic() - start,
            "cpu_seconds": cpu_after.ru_utime
            + cpu_after.ru_stime
            - cpu_before.ru_utime
            - cpu_before.ru_stime,
            "exit_code": process.returncode,
            "timeout": timeout,
            "stdout": stdout_text,
            "stderr": stderr_text,
            "valid": False,
        }
        if process.returncode != 0 or timeout:
            cpu_limit = process.returncode == -signal.SIGXCPU or (
                process.returncode == -signal.SIGKILL and result["cpu_seconds"] >= CPU_SECONDS - 0.1
            )
            result["failure_kind"] = "timeout" if timeout or cpu_limit else "runtime_error"
            result["error"] = (
                f"Wall time limit exceeded ({WALL_SECONDS} seconds)"
                if timeout
                else f"CPU time limit exceeded ({CPU_SECONDS} seconds)"
                if cpu_limit
                else f"Solver exited with code {process.returncode}: "
                + (stderr_text.strip().splitlines()[-1] if stderr_text.strip() else "no stderr")
            )
            if solution_bytes:
                result["failed_output"] = solution_bytes.decode("utf-8", errors="backslashreplace")
            return result
        try:
            solution = json.loads(solution_bytes)
        except (ValueError, UnicodeDecodeError, RecursionError) as error:
            result.update(
                failure_kind="invalid_output",
                error=str(error),
                failed_output=solution_bytes.decode("utf-8", errors="backslashreplace"),
            )
            return result
        try:
            checker = TASKS[task].check_final if final else TASKS[task].check
            result.update(checker(solution))
        except (ValueError, OverflowError, RecursionError) as error:
            result.update(
                failure_kind="constraint_violation",
                error=str(error),
                failed_output=solution_bytes.decode("utf-8", errors="backslashreplace"),
            )
            return result
        result["valid"] = True
        return result


def evaluate(source, seeds=SEEDS, task="circle", *, final=False):
    trials = [run_seed(source, seed, task, final=final) for seed in seeds]
    for trial in trials:
        if trial["cpu_seconds"] > CPU_SECONDS or trial["wall_seconds"] > WALL_SECONDS:
            trial["valid"] = False
            trial["failure_kind"] = "timeout"
            trial["error"] = "Measured execution time exceeded the frozen limit"
    valid = all(trial["valid"] for trial in trials)
    return {
        "source_sha256": digest(source),
        "valid": valid,
        "mean_score": math.fsum(t["score"] for t in trials) / len(trials) if valid else None,
        "trials": trials,
    }
