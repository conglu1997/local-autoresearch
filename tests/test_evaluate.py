import json
import math
import sys
from pathlib import Path

import pytest

from runner.evaluate import run_seed
from tasks.circle import certify, validate
from runner.core import candidate_source
from tasks.circle.baseline import solve
from runner import evaluate as evaluator


def test_grid_baseline():
    circles = solve(26, 0, 15)
    assert validate(circles)["score"] == pytest.approx(26 / 12)
    assert certify(circles)["exact_feasible"]


@pytest.mark.parametrize(
    "circles",
    [
        [],
        [[0.5, 0.5, 0.6]],
        [[0.5, 0.5, -0.1]],
        [[0.5, 0.5, 0]],
        [[math.nan, 0.5, 0.1]],
        [[0.5, 0.5, math.inf]],
        [[True, 0.5, 0.1]],
        [[0.5, 0.5]],
        [[0.5, 0.5, 0.1]] * 2,
    ],
)
def test_invalid_geometry(circles):
    with pytest.raises(ValueError):
        validate(circles, n=2 if len(circles) == 2 else 1)


def test_tolerance_is_not_exact_feasibility():
    circles = [[0.5, 0.5, 0.5 + 1e-10]]
    validate(circles, n=1)
    assert not certify(circles)["exact_feasible"]


def test_truncated_model_output_is_never_executed():
    with pytest.raises(ValueError, match="did not finish"):
        candidate_source({"finish_reason": "length", "content": "```python\nx=1\n```"})


@pytest.mark.skipif(sys.platform != "darwin", reason="macOS isolated execution")
def test_real_sandbox_baseline():
    result = run_seed(Path("tasks/circle/baseline.py").read_text(), 0)
    assert result["valid"], result
    assert result["certificate"]["exact_feasible"]


@pytest.mark.skipif(sys.platform != "darwin", reason="macOS isolated execution")
def test_sandbox_blocks_external_reads_and_network(tmp_path):
    secret = tmp_path / "outside-workspace.txt"
    secret.write_text("not research input")
    source = f"""from pathlib import Path
import socket
def solve(n, seed, budget):
    try:
        Path({str(secret)!r}).read_text()
    except PermissionError:
        pass
    else:
        raise AssertionError("External file read allowed")
    try:
        socket.socket().connect(("127.0.0.1", 11500))
    except PermissionError:
        pass
    else:
        raise AssertionError("Network connection allowed")
    return [[(i % 6 + .5) / 6, (i // 6 + .5) / 6, .08] for i in range(n)]
"""
    result = run_seed(source, 0)
    assert result["valid"], result


@pytest.mark.skipif(sys.platform != "darwin", reason="macOS isolated execution")
def test_numerical_libraries_work_inside_sandbox():
    source = """import numpy as np
from scipy.special import expit
def solve(n, seed, budget):
    assert expit(0) == .5
    return np.array([[(i % 6 + .5) / 6, (i // 6 + .5) / 6, .08] for i in range(n)]).tolist()
"""
    result = run_seed(source, 0)
    assert result["valid"], result


@pytest.mark.skipif(sys.platform != "darwin", reason="macOS isolated execution")
def test_cpu_budget_is_enforced(monkeypatch):
    monkeypatch.setattr(evaluator, "CPU_SECONDS", 1)
    result = run_seed("while True: pass\n", 0)
    assert not result["valid"]
    assert result["exit_code"] != 0
    assert result["wall_seconds"] < 5


@pytest.mark.skipif(sys.platform != "darwin", reason="macOS isolated execution")
def test_wall_budget_is_enforced(monkeypatch):
    monkeypatch.setattr(evaluator, "WALL_SECONDS", 0.1)
    result = run_seed("import time\ntime.sleep(10)\n", 0)
    assert not result["valid"]
    assert result["timeout"]


@pytest.mark.parametrize("name", ["stdout.txt", "stderr.txt"])
def test_candidate_cannot_redirect_parent_log_reads(tmp_path, name):
    secret = tmp_path / "sentinel.txt"
    secret.write_text("synthetic private data")
    source = f"""from pathlib import Path
def solve(n, seed, budget):
    Path({name!r}).unlink()
    Path({name!r}).symlink_to({str(secret)!r})
    print("candidate output")
    return [[(i % 6 + .5) / 6, (i // 6 + .5) / 6, .08] for i in range(n)]
"""
    result = run_seed(source, 0)
    assert result["valid"], result
    assert result["stdout"] == "candidate output\n"
    assert result["stderr"] == ""


def test_candidate_cannot_redirect_parent_solution_read(tmp_path):
    secret = tmp_path / "geometry.json"
    secret.write_text(json.dumps(solve(26, 0, 15)))
    source = f"""import os
from pathlib import Path
def solve(n, seed, budget):
    Path("solution.json").unlink(missing_ok=True)
    Path("solution.json").symlink_to({str(secret)!r})
    os._exit(0)
"""
    result = run_seed(source, 0)
    assert result["exit_code"] == 0
    assert not result["valid"]
    assert "error" in result
    assert "circles" not in result


def test_binary_logs_do_not_crash_the_evaluator():
    source = """import os
def solve(n, seed, budget):
    os.write(1, b"\\xff")
    os.write(2, b"\\xfe")
    return [[(i % 6 + .5) / 6, (i // 6 + .5) / 6, .08] for i in range(n)]
"""
    result = run_seed(source, 0)
    assert result["valid"], result
    assert result["stdout"] == r"\xff"
    assert result["stderr"] == r"\xfe"


def test_oversized_coordinates_are_an_invalid_candidate():
    result = run_seed("def solve(n, seed, budget):\n    return [[10**400, .5, .1]] * n\n", 0)
    assert not result["valid"]
    assert "error" in result


def test_deeply_nested_json_is_an_invalid_candidate():
    source = """import os
from pathlib import Path
def solve(n, seed, budget):
    Path("solution.json").write_text("[" * 10000 + "0" + "]" * 10000)
    os._exit(0)
"""
    result = run_seed(source, 0)
    assert result["exit_code"] == 0
    assert not result["valid"]
    assert "error" in result
