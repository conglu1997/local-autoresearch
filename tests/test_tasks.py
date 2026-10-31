import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from runner.evaluate import digest, evaluate, run_seed
from tasks import TASKS, route, schedule, sequence
from tasks.route.baseline import solve as route_baseline
from tasks.schedule.baseline import solve as schedule_baseline
import research
from runner import inference


def test_route_recomputes_closed_tour():
    order = route_baseline(route.PROBLEM, 0, 15)
    result = route.check(order)
    assert result["length"] > 0
    assert result["score"] == -result["length"]
    assert route.check(order[::-1])["score"] == pytest.approx(result["score"])


@pytest.mark.parametrize("order", [[], [0] * 128, list(range(127)), [True, *range(1, 128)]])
def test_route_rejects_missing_or_repeated_points(order):
    with pytest.raises(ValueError):
        route.check(order)


def test_schedule_recomputes_loads():
    assignment = [0] * 80
    checked = schedule.check(assignment)
    assert checked["makespan"] == sum(row[0] for row in schedule.PROBLEM)
    assert checked["score"] == -checked["makespan"]
    assert schedule.check(schedule_baseline(schedule.PROBLEM, 0, 15))["score"] > checked["score"]


@pytest.mark.parametrize("assignment", [[], [8] * 80, [-1] * 80, [True] * 80, [0.0] * 80])
def test_schedule_rejects_invalid_assignments(assignment):
    with pytest.raises(ValueError):
        schedule.check(assignment)


@pytest.mark.parametrize("task", TASKS)
def test_task_baselines_run_in_real_sandbox(task):
    problem = copy.deepcopy(TASKS[task].PROBLEM)
    result = evaluate(Path(f"tasks/{task}/baseline.py").read_text(), task=task)
    assert result["valid"], result
    assert len(result["trials"]) == 1
    assert TASKS[task].PROBLEM == problem


def test_candidate_cannot_set_its_own_score():
    result = run_seed("def solve(problem, seed, budget):\n    return {'score': 1e99}\n", 0, "route")
    assert not result["valid"]
    assert "exactly once" in result["error"]


@pytest.mark.parametrize("task", [name for name in TASKS if name != "bike"])
def test_final_mode_rejects_tasks_without_a_test_split(task):
    with pytest.raises(ValueError, match="no held-out test split"):
        run_seed("", 0, task, final=True)


def test_sequence_uses_aperiodic_correlation_and_exact_energy():
    result = sequence.check([1] * 256)
    assert result["correlations"] == list(range(255, 0, -1))
    assert result["energy"] == sum(k * k for k in range(1, 256))
    assert result["score"] == -result["energy"]
    assert sequence.check([-1] * 256) == {**result, "solution": [-1] * 256}


@pytest.mark.parametrize("bits", [[1] * 255, [0] * 256, [True] * 256, [1.0] * 256])
def test_sequence_rejects_invalid_bits(bits):
    with pytest.raises(ValueError):
        sequence.check(bits)


@pytest.mark.parametrize("task", ["circle", "route"])
def test_runner_records_single_trial_and_rejects_changed_evaluator(tmp_path, monkeypatch, task):
    status = {
        "ready": True,
        "model": "flint",
        "endpoint": "http://127.0.0.1:1/v1",
        "engine": "test",
        "version": "test",
    }
    monkeypatch.setattr(
        inference.subprocess,
        "check_output",
        lambda command, **kwargs: json.dumps(status) if command[1] == "status" else "test-key",
    )
    baseline = Path(f"tasks/{task}/baseline.py").read_text()
    monkeypatch.setattr(
        inference,
        "complete",
        lambda *a: {"finish_reason": "stop", "content": f"```python\n{baseline}```"},
    )
    args = SimpleNamespace(
        base_url=None,
        model=None,
        api_key_env=None,
        run=tmp_path / "run",
        resume=False,
        task=task,
        max_tokens=16384,
        experiments=1,
    )
    research.run(args)
    manifest_path = args.run / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    assert manifest["seeds"] == [0]
    assert manifest["problem"] == TASKS[task].PROBLEM
    assert manifest["sampling"]["temperature"] == 1
    assert manifest["sampling"]["top_k"] == 20
    assert manifest["model"] == "flint"
    for name in (
        "runner/__init__.py",
        "runner/evaluate.py",
        "runner/worker.py",
        f"tasks/{task}/baseline.py",
        f"tasks/{task}/program.md",
        f"tasks/{task}/__init__.py",
    ):
        assert manifest["fixed_files"][name] == digest(Path(name).read_text())
    assert json.loads((args.run / "001/decision.json").read_text())["decision"] == "discard"
    args.resume = True
    status["version"] = "changed"
    with pytest.raises(RuntimeError, match="Model configuration changed"):
        research.run(args)
    status["version"] = "test"
    manifest["fixed_files"][f"tasks/{task}/__init__.py"] = "changed"
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(RuntimeError, match="Fixed file changed"):
        research.run(args)
