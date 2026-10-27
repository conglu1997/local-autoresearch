"""Verify exported traces, candidate provenance, scores and keep/discard decisions."""

import argparse
import gzip
import json
import math
from pathlib import Path

from runner.evaluate import digest
from tools.artifact_integrity import ArtifactIntegrity
from tasks.circle import certify, validate
from runner.core import candidate_source
from runner.inference import stream_text
from tasks import TASKS
from tools.final_protocol import verify_final_artifacts


def inventory(root):
    """Count valid, distinct implementations; comments and docstrings do not distinguish them."""
    manifest = json.loads((root / "manifest.json").read_text())
    integrity = ArtifactIntegrity(root)
    integrity.verify_files()
    seen = {integrity.fingerprint("000/solver.py"): "000"}
    records = []
    for directory in sorted(p for p in root.iterdir() if p.is_dir() and p.name.isdigit()):
        if directory.name == "000" or not (directory / "decision.json").exists():
            continue
        evaluation = json.loads((directory / "evaluation.json").read_text())
        record = {"attempt": directory.name, "qualifies": False}
        if not evaluation["valid"]:
            record["exclusion"] = "unsuccessful evaluation"
        else:
            source = (directory / "solver.py").read_text()
            response = json.loads((directory / "response.json").read_text())
            assert candidate_source(response) == source, directory.name
            assert (
                integrity.source_hash(f"{directory.name}/solver.py") == evaluation["source_sha256"]
            ), directory.name
            assert [t["seed"] for t in evaluation["trials"]] == manifest["seeds"]
            assert all(t["valid"] for t in evaluation["trials"])
            score = evaluation["mean_score"]
            assert math.isfinite(score)
            assert math.isclose(
                score,
                math.fsum(t["score"] for t in evaluation["trials"]) / len(manifest["seeds"]),
                rel_tol=0,
                abs_tol=1e-12,
            )
            fingerprint = integrity.fingerprint(f"{directory.name}/solver.py")
            record.update(score=score, structural_sha256=fingerprint)
            if fingerprint in seen:
                record.update(exclusion="duplicate implementation", duplicate_of=seen[fingerprint])
            else:
                seen[fingerprint] = directory.name
                record["qualifies"] = True
        records.append(record)
    return {
        "completed_attempts": len(records),
        "distinct_scored_candidates": sum(r["qualifies"] for r in records),
        "records": records,
    }


def verify_stream(directory):
    response = json.loads((directory / "response.json").read_text())
    parts = {"content": [], "reasoning": []}
    usage, finish, first_token = None, None, None
    previous_seconds = 0
    path = directory / "stream.jsonl.gz"
    stream = gzip.open(path, "rt") if path.exists() else (directory / "stream.jsonl").open()
    with stream:
        for record in stream:
            event = json.loads(record)
            assert previous_seconds <= event["seconds"] <= response["seconds"]
            previous_seconds = event["seconds"]
            line = event["line"]
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            chunk = json.loads(line[6:])
            if chunk.get("usage") is not None:
                usage = chunk["usage"]
            for choice in chunk["choices"]:
                for key, value in stream_text(choice["delta"]).items():
                    if value:
                        parts[key].append(value)
                        if first_token is None:
                            first_token = event["seconds"]
                if choice["finish_reason"] is not None:
                    finish = choice["finish_reason"]
    for key, fragments in parts.items():
        assert "".join(fragments) == response[key], f"{directory.name}: {key} differs from stream"
    assert usage == response["usage"]
    assert finish == response["finish_reason"]
    assert first_token == response["first_token_seconds"]
    assert (directory / "response.txt").read_text() == response["content"]


def verify(root):
    manifest = json.loads((root / "manifest.json").read_text())
    # Exports predating task selection contain only circle-packing records.
    task = manifest["task"] if "task" in manifest else "circle"
    if task != "circle":
        assert manifest["problem"] == TASKS[task].PROBLEM
    integrity = ArtifactIntegrity(root)
    integrity.verify_files()
    best, score, count = "000", -math.inf, 0
    for directory in sorted(p for p in root.iterdir() if p.is_dir() and p.name.isdigit()):
        if (directory / "response.json").exists():
            verify_stream(directory)
        if not (directory / "decision.json").exists():
            continue  # Interrupted calls retain their raw stream but have no decision.
        count += 1
        evaluation = json.loads((directory / "evaluation.json").read_text())
        decision = json.loads((directory / "decision.json").read_text())
        source = directory / "solver.py"
        if source.exists():
            assert (
                integrity.source_hash(f"{directory.name}/solver.py") == evaluation["source_sha256"]
            )
            if directory.name != "000":
                response = json.loads((directory / "response.json").read_text())
                assert digest(candidate_source(response)) == digest(source.read_text())
        expected = "invalid"
        if evaluation["valid"]:
            assert [trial["seed"] for trial in evaluation["trials"]] == manifest["seeds"]
            measured = []
            for trial in evaluation["trials"]:
                assert trial["valid"]
                if task == "circle":
                    checked = validate(trial["circles"])
                    assert certify(trial["circles"]) == trial["certificate"]
                else:
                    checked = TASKS[task].check(trial["solution"])
                    assert all(trial[key] == value for key, value in checked.items())
                assert math.isclose(checked["score"], trial["score"], rel_tol=0, abs_tol=1e-12)
                measured.append(checked["score"])
            mean = math.fsum(measured) / len(measured)
            assert math.isclose(mean, evaluation["mean_score"], rel_tol=0, abs_tol=1e-12)
            expected = "keep" if mean > score + 1e-9 else "discard"
        if directory.name == "000":
            assert evaluation["valid"]
            expected = "baseline"
        else:
            assert decision["parent"] == best
        assert decision["decision"] == expected, directory.name
        if expected in ("keep", "baseline"):
            best, score = directory.name, evaluation["mean_score"]
    if (root / "summary.json").exists():
        summary = json.loads((root / "summary.json").read_text())
        assert summary["best_attempt"] == best
        assert summary["best_score"] == score
        assert summary["completed_experiments"] == count - 1
        assert (root / "best.py").read_text() == (root / best / "solver.py").read_text()
    if (root / "progress.json").exists():
        progress = json.loads((root / "progress.json").read_text())
        for key, value in inventory(root).items():
            assert progress[key] == value, key
        if progress["status"] == "complete":
            assert progress["distinct_scored_candidates"] >= progress["target"]
    final_state = verify_final_artifacts(root, manifest, best, count - 1, integrity)
    if (root / "validation.json").exists():
        validation = json.loads((root / "validation.json").read_text())
        assert validation["source_sha256"] == integrity.source_hash(f"{best}/solver.py")
        assert [trial["seed"] for trial in validation["trials"]] == [3, 4, 5, 6, 7]
        assert validation["valid"] == all(trial["valid"] for trial in validation["trials"])
        measured = []
        for trial in validation["trials"]:
            if trial["valid"]:
                checked = validate(trial["circles"])
                assert math.isclose(checked["score"], trial["score"], rel_tol=0, abs_tol=1e-12)
                assert checked["max_violation"] == trial["max_violation"]
                assert certify(trial["circles"]) == trial["certificate"]
                measured.append(checked["score"])
        if validation["valid"]:
            assert math.isclose(
                math.fsum(measured) / len(measured),
                validation["mean_score"],
                rel_tol=0,
                abs_tol=1e-12,
            )
        else:
            assert validation["mean_score"] is None
    print(
        f"Verified {count} completed records; best {best}: {score:.9f}; "
        f"held-out test: {final_state}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    verify(parser.parse_args().run)
