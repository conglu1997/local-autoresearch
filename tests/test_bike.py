import copy
import gzip
import json
import math
import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest

from runner.evaluate import digest, run_seed
from tasks import bike
from tools import final_test as final_runner
import research
from runner import inference
from report import report
from tools.verify_run import verify


def test_chronological_split_and_candidate_inputs():
    assert [len(rows) for rows in (bike.TRAIN, bike.VALIDATION, bike.TEST)] == [8645, 4358, 4376]
    assert max(row["dteday"] for row in bike.TRAIN) < min(row["dteday"] for row in bike.VALIDATION)
    assert max(row["dteday"] for row in bike.VALIDATION) < min(row["dteday"] for row in bike.TEST)
    assert set(bike.PROBLEM) == {"features", "train_x", "train_y", "predict_x"}
    assert not {"cnt", "casual", "registered"} & set(bike.FEATURES)
    assert len(bike.PROBLEM["train_y"]) == 8645
    assert len(bike.PROBLEM["predict_x"]) == 4358
    final = bike.final_problem()
    assert len(final["train_y"]) == 13003
    assert len(final["predict_x"]) == 4376
    assert final["train_y"] == [int(row["cnt"]) for row in bike.TRAIN + bike.VALIDATION]
    assert set(final) == set(bike.PROBLEM)


def test_rmsle_against_hand_computed_values():
    measured = bike.score([0, 3], [{"cnt": "0"}, {"cnt": "1"}])
    assert measured["rmsle"] == pytest.approx(math.log(2) / math.sqrt(2))
    assert measured["score"] == -measured["rmsle"]
    assert measured["total_bias_percent"] == 200
    assert measured["mean_log_error"] == pytest.approx(math.log(2) / 2)


def test_weekend_and_elapsed_time_are_correct_and_label_free():
    weekend = bike.FEATURES.index("weekend")
    elapsed = bike.FEATURES.index("elapsed_days")
    rows = {r["dteday"]: r for r in bike.TRAIN if r["hr"] == "12"}
    assert bike.features(rows["2011-01-01"])[weekend] == 1
    assert bike.features(rows["2011-01-02"])[weekend] == 1
    assert bike.features(rows["2011-01-07"])[weekend] == 0
    assert bike.features(rows["2011-01-01"])[elapsed] == 0
    assert bike.features(rows["2011-12-31"])[elapsed] == 364
    row = rows["2011-01-07"]
    assert bike.features(row) == bike.features(
        {**row, "cnt": "999999", "casual": "0", "registered": "0"}
    )


def test_count_bias_and_rmsle_report_different_properties():
    measured = bike.score([1, 1], [{"cnt": "1"}, {"cnt": "7"}])
    assert measured["total_bias_percent"] == -75
    assert measured["rmsle"] > 0
    assert measured["mean_log_error"] < 0


@pytest.mark.parametrize("predictions", [[], [True], [-1], [math.nan], [math.inf], ["1"]])
def test_invalid_predictions_are_rejected(predictions):
    with pytest.raises(ValueError):
        bike.score(predictions, [{"cnt": "1"}])


def test_labels_and_original_csv_cannot_be_read_in_sandbox():
    source = f"""from pathlib import Path
def solve(problem, seed, budget):
    assert set(problem) == {{'features', 'train_x', 'train_y', 'predict_x'}}
    for name in [{str(bike.DATA.resolve())!r}, {str(Path(bike.__file__).resolve())!r}]:
        try:
            Path(name).read_text()
        except PermissionError:
            continue
        raise AssertionError('Scoring data readable from candidate')
    return [0] * len(problem['predict_x'])
"""
    result = run_seed(source, 0, "bike")
    assert result["valid"], result


def test_run_freezes_csv_and_final_protocol(tmp_path, monkeypatch):
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
    args = SimpleNamespace(
        base_url=None,
        model=None,
        api_key_env=None,
        run=tmp_path / "run",
        resume=False,
        task="bike",
        max_tokens=16384,
        experiments=0,
    )
    research.run(args)
    manifest = json.loads((args.run / "manifest.json").read_text())
    assert "tasks/bike/hour.csv" in manifest["fixed_files"]
    assert manifest["fixed_files"]["tools/final_test.py"] == digest(
        Path("tools/final_test.py").read_text()
    )
    assert manifest["final_test"] == bike.FINAL_PROTOCOL
    assert manifest["problem"] == bike.PROBLEM


def test_final_test_refits_unchanged_winner_once(tmp_path, monkeypatch):
    source = "def solve(problem, seed, budget):\n    return [0] * len(problem['predict_x'])\n"
    winner = tmp_path / "017"
    winner.mkdir()
    (winner / "solver.py").write_text(source)
    (winner / "evaluation.json").write_text(
        json.dumps({"source_sha256": digest(source), "mean_score": -0.5})
    )
    baseline_source = source.replace("return [0]", "return [1]")
    (tmp_path / "000").mkdir()
    (tmp_path / "000/solver.py").write_text(baseline_source)
    (tmp_path / "000/evaluation.json").write_text(
        json.dumps({"source_sha256": digest(baseline_source)})
    )
    (tmp_path / "summary.json").write_text(
        json.dumps({"completed_experiments": 100, "best_attempt": "017", "best_score": -0.5})
    )
    manifest = {
        "task": "bike",
        "seeds": [0],
        "fixed_files": {},
        "final_test": copy.deepcopy(bike.FINAL_PROTOCOL),
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    calls = []

    def evaluate(program, seeds, task, *, final):
        calls.append((program, seeds, task, final))
        return {"valid": False, "mean_score": None, "trials": []}

    monkeypatch.setattr(final_runner, "evaluate", evaluate)
    final_runner.final_test(tmp_path)
    assert calls == [(baseline_source, [0], "bike", True), (source, [0], "bike", True)]
    assert (tmp_path / "final-test.json").exists()
    assert (tmp_path / "baseline-final-test.json").exists()
    with pytest.raises(FileExistsError):
        final_runner.final_test(tmp_path)
    assert len(calls) == 2


def test_final_test_requires_completed_search(tmp_path, monkeypatch):
    manifest = {
        "task": "bike",
        "seeds": [0],
        "fixed_files": {},
        "final_test": copy.deepcopy(bike.FINAL_PROTOCOL),
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    (tmp_path / "summary.json").write_text(json.dumps({"completed_experiments": 99}))
    before = {path.name: path.read_bytes() for path in tmp_path.iterdir()}
    monkeypatch.setattr(
        final_runner,
        "evaluate",
        lambda *args, **kwargs: pytest.fail("Incomplete search reached held-out evaluation"),
    )
    with pytest.raises(ValueError, match="100-attempt"):
        final_runner.final_test(tmp_path)
    assert not (tmp_path / "final-test-started.json").exists()
    assert {path.name: path.read_bytes() for path in tmp_path.iterdir()} == before


def test_final_test_command_rejects_non_ml_task(tmp_path):
    (tmp_path / "manifest.json").write_text(json.dumps({"task": "route"}))
    (tmp_path / "summary.json").write_text(json.dumps({"completed_experiments": 100}))
    with pytest.raises(ValueError, match="no held-out test split"):
        final_runner.final_test(tmp_path)


@pytest.mark.parametrize(
    "corruption",
    [
        None,
        "prediction",
        "baseline_missing",
    ],
)
def test_verifier_checks_final_predictions_against_separate_labels(
    tmp_path, monkeypatch, corruption
):
    monkeypatch.setattr(bike, "PROBLEM", {"synthetic_fixture": True})
    monkeypatch.setattr(bike, "VALIDATION", [{"cnt": "1"}])
    monkeypatch.setattr(bike, "TEST", [{"cnt": "3"}])
    source = "# recorded fixture\n"
    content = f"```python\n{source}```"
    response = {
        "content": content,
        "reasoning": "",
        "usage": None,
        "finish_reason": "stop",
        "first_token_seconds": 1,
        "seconds": 2,
    }
    chunk = {"choices": [{"delta": {"content": content}, "finish_reason": "stop"}]}
    stream = json.dumps({"seconds": 1, "line": "data: " + json.dumps(chunk)}) + "\n"
    trial = {"seed": 0, "valid": True, **bike.check([1])}
    result = {"source_sha256": digest(source), "valid": True, "mean_score": 0, "trials": [trial]}
    files = {
        "manifest.json": json.dumps(
            {
                "task": "bike",
                "problem": bike.PROBLEM,
                "seeds": [0],
                "final_test": bike.FINAL_PROTOCOL,
            }
        ),
        "final-test-started.json": json.dumps(
            {"best_attempt": "000", "source_sha256": digest(source)}
        ),
    }
    for number in range(101):
        name = f"{number:03d}"
        files[f"{name}/solver.py"] = source
        files[f"{name}/evaluation.json"] = json.dumps(result)
        files[f"{name}/decision.json"] = json.dumps(
            {
                "decision": "baseline" if number == 0 else "discard",
                "parent": None if number == 0 else "000",
            }
        )
        if number:
            files[f"{name}/response.json"] = json.dumps(response)
            files[f"{name}/response.txt"] = content
            files[f"{name}/stream.jsonl.gz"] = stream
    final = {
        "source_sha256": digest(source),
        "valid": True,
        "mean_score": 0,
        "trials": [{"seed": 0, "valid": True, **bike.check_final([3])}],
    }
    if corruption == "rmsle":
        final["trials"][0]["rmsle"] = 0.1
    elif corruption == "prediction":
        final["trials"][0]["solution"] = [1]
    elif corruption == "source":
        final["source_sha256"] = digest("another program")
    elif corruption == "seed":
        final["trials"][0]["seed"] = 1
    elif corruption == "mean":
        final["mean_score"] = 0.1
    files["final-test.json"] = json.dumps(final)
    baseline_final = {
        "source_sha256": digest(source),
        "valid": True,
        "mean_score": 0,
        "trials": [{"seed": 0, "valid": True, **bike.check_final([3])}],
    }
    if corruption == "baseline_prediction":
        baseline_final["trials"][0]["solution"] = [1]
    if corruption == "baseline_source":
        baseline_final["source_sha256"] = digest("another baseline")
    files["baseline-final-test.json"] = json.dumps(baseline_final)
    if corruption == "baseline_missing":
        del files["baseline-final-test.json"]
    if corruption == "protocol":
        manifest = json.loads(files["manifest.json"])
        manifest["final_test"]["version"] = "unknown"
        files["manifest.json"] = json.dumps(manifest)
    entries = []
    for name, text in files.items():
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(gzip.compress(text.encode()) if path.suffix == ".gz" else text.encode())
        entries.append(
            {"file": name, "original_sha256": digest(text), "export_sha256": digest(text)}
        )
    (tmp_path / "export.json").write_text(json.dumps({"files": entries}))
    if corruption is None:
        verify(tmp_path)
    elif corruption == "baseline_missing":
        with pytest.raises(FileNotFoundError):
            verify(tmp_path)
    else:
        with pytest.raises(AssertionError):
            verify(tmp_path)


def test_bike_example_can_be_verified_and_rendered(tmp_path):
    root = Path(__file__).resolve().parents[1]
    example = tmp_path / "bike"
    shutil.copytree(root / "examples/2026-10-21-bike", example)
    verify(example)
    report(example)
    document = (example / "report.html").read_text()
    assert "0.535904 → 0.370806" in document
    assert "Held-out final test" in document
    assert "Matched baseline final test" in document


def test_final_evaluation_uses_refit_inputs_and_final_checker(monkeypatch):
    monkeypatch.setattr(bike, "final_problem", lambda: {"train_y": [1, 2], "predict_x": [[0]]})
    monkeypatch.setattr(bike, "TEST", [{"cnt": "3"}])
    source = """def solve(problem, seed, budget):
    assert problem == {'train_y': [1, 2], 'predict_x': [[0]]}
    return [3]
"""
    result = run_seed(source, 0, "bike", final=True)
    assert result["valid"] and result["rmsle"] == 0
