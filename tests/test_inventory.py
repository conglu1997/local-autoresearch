import json

import pytest

from runner.search import structural_hash
from tools.verify_run import inventory
from runner.evaluate import digest


BASELINE = "def solve(n, seed, budget):\n    return []\n"


def make_run(root):
    (root / "000").mkdir()
    (root / "000/solver.py").write_text(BASELINE)
    (root / "manifest.json").write_text(json.dumps({"seeds": [0, 1, 2]}))


def attempt(root, number, source, valid=True):
    directory = root / f"{number:03d}"
    directory.mkdir()
    (directory / "solver.py").write_text(source)
    (directory / "response.json").write_text(
        json.dumps({"finish_reason": "stop", "content": f"```python\n{source}```"})
    )
    (directory / "evaluation.json").write_text(
        json.dumps(
            {
                "valid": valid,
                "mean_score": 1.0 if valid else None,
                "source_sha256": digest(source),
                "trials": [{"seed": seed, "score": 1.0, "valid": valid} for seed in range(3)],
            }
        )
    )
    (directory / "decision.json").write_text(json.dumps({"decision": "discard"}))


def test_structural_identity_ignores_presentation_changes():
    decorated = '"""Module description."""\n# Comment\ndef solve(n, seed, budget):\n    """Description."""\n    return [ ]\n'
    assert structural_hash(BASELINE) == structural_hash(decorated)
    assert structural_hash(BASELINE) != structural_hash(BASELINE.replace("[]", "[1]"))


def test_inventory_counts_only_distinct_successful_candidates(tmp_path):
    make_run(tmp_path)
    variant = BASELINE.replace("[]", "[1]")
    inputs = [
        (BASELINE, False),
        (BASELINE, True),
        (variant, True),
        ("# comment\n" + variant, True),
        (BASELINE.replace("[]", "[2]"), True),
    ]
    for number, (source, valid) in enumerate(inputs, 1):
        attempt(tmp_path, number, source, valid)

    progress = inventory(tmp_path)
    assert progress["completed_attempts"] == 5
    assert progress["distinct_scored_candidates"] == 2
    assert progress["records"][1]["duplicate_of"] == "000"
    assert progress["records"][3]["duplicate_of"] == "003"


def test_counter_rejects_source_not_authored_in_recorded_response(tmp_path):
    make_run(tmp_path)
    attempt(tmp_path, 1, BASELINE.replace("[]", "[1]"))
    (tmp_path / "001/solver.py").write_text(BASELINE.replace("[]", "[2]"))
    with pytest.raises(AssertionError):
        inventory(tmp_path)


def test_counter_requires_the_entire_seed_suite(tmp_path):
    make_run(tmp_path)
    attempt(tmp_path, 1, BASELINE.replace("[]", "[1]"))
    path = tmp_path / "001/evaluation.json"
    result = json.loads(path.read_text())
    result["trials"].pop()
    path.write_text(json.dumps(result))
    with pytest.raises(AssertionError):
        inventory(tmp_path)
