import gzip
import json
from pathlib import Path

import pytest

from runner.evaluate import digest
from tasks.circle import certify, validate
from tasks.circle.baseline import solve
from tasks import TASKS
from tasks.route.baseline import solve as route_baseline
from tasks.schedule.baseline import solve as schedule_baseline
from tools.verify_run import verify, verify_stream


@pytest.fixture
def recorded_response(tmp_path):
    usage = {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30}
    chunks = [
        {"choices": [{"delta": {"reasoning": "Try a swap."}, "finish_reason": None}]},
        {"choices": [{"delta": {"content": "Candidate."}, "finish_reason": None}]},
        {"choices": [{"delta": {}, "finish_reason": "stop"}], "usage": usage},
    ]
    lines = [f"data: {json.dumps(chunk)}" for chunk in chunks] + ["data: [DONE]"]
    stream = "".join(
        json.dumps({"seconds": i, "line": line}) + "\n" for i, line in enumerate(lines, 1)
    )
    (tmp_path / "stream.jsonl.gz").write_bytes(gzip.compress(stream.encode()))
    response = {
        "content": "Candidate.",
        "reasoning": "Try a swap.",
        "usage": usage,
        "finish_reason": "stop",
        "first_token_seconds": 1,
        "seconds": 5,
    }
    (tmp_path / "response.json").write_text(json.dumps(response))
    (tmp_path / "response.txt").write_text(response["content"])
    return response


def test_verifies_response_against_raw_stream(tmp_path, recorded_response):
    verify_stream(tmp_path)


@pytest.mark.parametrize(
    "field,value",
    [
        ("content", "Rewritten candidate."),
        ("reasoning", "Rewritten reasoning."),
        ("usage", {"prompt_tokens": 100, "completion_tokens": 200, "total_tokens": 300}),
        ("finish_reason", "length"),
        ("first_token_seconds", 2),
        ("seconds", 2),
    ],
)
def test_rejects_response_that_differs_from_stream(tmp_path, recorded_response, field, value):
    recorded_response[field] = value
    (tmp_path / "response.json").write_text(json.dumps(recorded_response))
    with pytest.raises(AssertionError):
        verify_stream(tmp_path)


def test_rejects_response_with_missing_stream_usage_and_finish(tmp_path, recorded_response):
    path = tmp_path / "stream.jsonl.gz"
    lines = gzip.decompress(path.read_bytes()).decode().splitlines(keepends=True)
    path.write_bytes(gzip.compress("".join(lines[:-2]).encode()))
    with pytest.raises(AssertionError):
        verify_stream(tmp_path)


def test_preserves_consistently_recorded_incomplete_response(tmp_path, recorded_response):
    path = tmp_path / "stream.jsonl.gz"
    lines = gzip.decompress(path.read_bytes()).decode().splitlines(keepends=True)
    path.write_bytes(gzip.compress("".join(lines[:-2]).encode()))
    recorded_response["finish_reason"] = None
    recorded_response["usage"] = None
    (tmp_path / "response.json").write_text(json.dumps(recorded_response))
    verify_stream(tmp_path)


def write_bundle(root, validation, completed=0):
    source = "# Recorded baseline\n"
    circles = solve(26, 0, 15)
    measured = validate(circles)
    trial = {
        "seed": 0,
        "valid": True,
        "circles": circles,
        **measured,
        "certificate": certify(circles),
    }
    baseline = {
        "source_sha256": digest(source),
        "valid": True,
        "mean_score": measured["score"],
        "trials": [trial],
    }
    files = {
        "manifest.json": json.dumps({"task": "circle", "seeds": [0]}),
        "000/solver.py": source,
        "000/evaluation.json": json.dumps(baseline),
        "000/decision.json": json.dumps({"decision": "baseline"}),
        "best.py": source,
        "summary.json": json.dumps(
            {
                "best_attempt": "000",
                "best_score": measured["score"],
                "completed_experiments": completed,
            }
        ),
        "validation.json": json.dumps(validation),
    }
    entries = []
    for name, text in files.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
        entries.append(
            {"file": name, "original_sha256": digest(text), "export_sha256": digest(text)}
        )
    (root / "export.json").write_text(json.dumps({"files": entries}))


@pytest.fixture
def validation():
    circles = solve(26, 0, 15)
    measured = validate(circles)
    return {
        "source_sha256": digest("# Recorded baseline\n"),
        "valid": True,
        "mean_score": measured["score"],
        "trials": [
            {
                "seed": seed,
                "valid": True,
                "circles": circles,
                **measured,
                "certificate": certify(circles),
            }
            for seed in range(3, 8)
        ],
    }


def test_verifies_post_run_geometry(tmp_path, validation):
    write_bundle(tmp_path, validation)
    verify(tmp_path)


def test_rejects_inflated_attempt_count(tmp_path, validation):
    write_bundle(tmp_path, validation, completed=1)
    with pytest.raises(AssertionError):
        verify(tmp_path)


@pytest.mark.parametrize("corruption", ["source", "seed", "score", "certificate", "mean", "status"])
def test_rejects_inconsistent_post_run_evidence_even_with_valid_hashes(
    tmp_path, validation, corruption
):
    if corruption == "source":
        validation["source_sha256"] = digest("different candidate")
    elif corruption == "seed":
        validation["trials"][0]["seed"] = 0
    elif corruption == "score":
        validation["trials"][0]["score"] += 0.1
    elif corruption == "certificate":
        validation["trials"][0]["certificate"]["exact_feasible"] = False
    elif corruption == "mean":
        validation["mean_score"] += 0.1
    else:
        validation["valid"] = False
    write_bundle(tmp_path, validation)
    with pytest.raises(AssertionError):
        verify(tmp_path)


def test_preserves_failed_post_run_trial(tmp_path, validation):
    validation["trials"][0] = {"seed": 3, "valid": False, "error": "Constraint violation"}
    validation["valid"] = False
    validation["mean_score"] = None
    write_bundle(tmp_path, validation)
    verify(tmp_path)


def test_rejects_invalid_post_run_geometry(tmp_path, validation):
    validation["trials"][0]["circles"][0][0] = -1.0
    write_bundle(tmp_path, validation)
    with pytest.raises(ValueError, match="Constraint violation"):
        verify(tmp_path)


@pytest.mark.parametrize(
    "task,baseline", [("route", route_baseline), ("schedule", schedule_baseline)]
)
def test_task_export_recomputes_scores_even_if_hashes_are_replaced(tmp_path, task, baseline):
    source = Path(f"tasks/{task}/baseline.py").read_text()
    problem = TASKS[task].PROBLEM
    trial = {"valid": True, "seed": 0, **TASKS[task].check(baseline(problem, 0, 15))}
    evaluation = {
        "source_sha256": digest(source),
        "valid": True,
        "mean_score": trial["score"],
        "trials": [trial],
    }
    files = {
        "manifest.json": json.dumps({"task": task, "problem": problem, "seeds": [0]}),
        "000/solver.py": source,
        "000/evaluation.json": json.dumps(evaluation),
        "000/decision.json": json.dumps({"decision": "baseline", "parent": None}),
    }

    def write():
        entries = []
        for name, text in files.items():
            path = tmp_path / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text)
            entries.append(
                {"file": name, "original_sha256": digest(text), "export_sha256": digest(text)}
            )
        (tmp_path / "export.json").write_text(json.dumps({"files": entries}))

    write()
    verify(tmp_path)
    evaluation["mean_score"] += 1
    evaluation["trials"][0]["score"] += 1
    files["000/evaluation.json"] = json.dumps(evaluation)
    write()
    with pytest.raises(AssertionError):
        verify(tmp_path)
