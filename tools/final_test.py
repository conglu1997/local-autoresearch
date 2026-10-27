"""Refit the validation winner once and score the held-out period."""

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from runner.evaluate import ROOT, digest, evaluate
from runner.core import save
from tasks import TASKS
from tools.final_protocol import budget, check_completion


def final_test(root):
    manifest = json.loads((root / "manifest.json").read_text())
    summary = json.loads((root / "summary.json").read_text())
    task = manifest["task"]
    if not hasattr(TASKS[task], "FINAL_PROTOCOL"):
        raise ValueError(f"Task {task} has no held-out test split")
    check_completion(root, manifest, summary["completed_experiments"])
    for name, expected in manifest["fixed_files"].items():
        if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f"Frozen file changed: {name}")
    if manifest["final_test"] != TASKS[task].final_protocol(budget(manifest)):
        raise ValueError("Final-test procedure differs from the frozen protocol")
    best = root / summary["best_attempt"]
    source = (best / "solver.py").read_text()
    result = json.loads((best / "evaluation.json").read_text())
    if digest(source) != result["source_sha256"] or result["mean_score"] != summary["best_score"]:
        raise ValueError("Selected program differs from its recorded validation result")
    if task == "bike":
        baseline_source = (root / "000/solver.py").read_text()
        baseline_result = json.loads((root / "000/evaluation.json").read_text())
        if digest(baseline_source) != baseline_result["source_sha256"]:
            raise ValueError("Baseline differs from its recorded validation program")
    with (root / "final-test-started.json").open("x") as marker:
        json.dump(
            {
                "started_at": datetime.now(timezone.utc).isoformat(),
                "best_attempt": best.name,
                "source_sha256": digest(source),
            },
            marker,
            indent=2,
        )
    if task == "bike":
        baseline_final = evaluate(baseline_source, manifest["seeds"], task, final=True)
        save(root / "baseline-final-test.json", baseline_final)
    measured = (
        baseline_final
        if task == "bike" and source == baseline_source
        else evaluate(source, manifest["seeds"], task, final=True)
    )
    save(root / "final-test.json", measured)
    print(json.dumps({key: value for key, value in measured.items() if key != "trials"}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    final_test(parser.parse_args().run.resolve())
