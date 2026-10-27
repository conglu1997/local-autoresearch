"""Held-out test checks shared by the final-test command and the run verifiers."""

import json
import math

from tasks import TASKS


def budget(manifest):
    return manifest["budget_seconds"] if "parallel" in manifest else None


def check_completion(root, manifest, completed):
    if "parallel" in manifest:
        if json.loads((root / "finished.json").read_text())["reason"] != "time budget":
            raise ValueError("Finish the full time budget before the final test")
    elif completed != 100:
        raise ValueError("Finish the 100-attempt search before the final test")


def verify_final_evaluation(result, task, seeds, source_hash):
    assert result["source_sha256"] == source_hash
    assert [trial["seed"] for trial in result["trials"]] == seeds
    assert result["valid"] == all(trial["valid"] for trial in result["trials"])
    scores = []
    for trial in result["trials"]:
        if trial["valid"]:
            checked = TASKS[task].check_final(trial["solution"])
            assert all(trial[key] == value for key, value in checked.items())
            scores.append(checked["score"])
    if result["valid"]:
        assert math.isclose(
            result["mean_score"], math.fsum(scores) / len(scores), rel_tol=0, abs_tol=1e-12
        )
    else:
        assert result["mean_score"] is None


def verify_final_artifacts(root, manifest, best, completed, integrity):
    """Check a saved held-out test, if one was started, without executing programs."""
    if not (root / "final-test-started.json").exists():
        return "not started"
    task = manifest["task"]
    assert manifest["final_test"] == TASKS[task].final_protocol(budget(manifest))
    check_completion(root, manifest, completed)
    marker = json.loads((root / "final-test-started.json").read_text())
    assert marker["best_attempt"] == best
    assert marker["source_sha256"] == integrity.source_hash(f"{best}/solver.py")
    if not (root / "final-test.json").exists():
        return "started"
    for name, attempt in (("baseline-final-test.json", "000"), ("final-test.json", best)):
        verify_final_evaluation(
            json.loads((root / name).read_text()),
            task,
            manifest["seeds"],
            integrity.source_hash(f"{attempt}/solver.py"),
        )
    return "complete"
