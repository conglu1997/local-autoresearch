"""Render a team run as HTML without executing its programs."""

from importlib import import_module
import html
import json
import math

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from tools.artifact_integrity import ArtifactIntegrity

EVIDENCE = (
    "request.json",
    "response.json",
    "solver.py",
    "evaluation.json",
    "decision.json",
    "worker.log",
    "stream.jsonl",
    "stream.jsonl.gz",
)


def load(path):
    return json.loads(path.read_text())


def measure(root):
    """Count outcomes and trace the best score over research time, in completion order."""
    manifest = load(root / "manifest.json")
    integrity = ArtifactIntegrity(root)
    best = load(root / "000/evaluation.json")["mean_score"]
    seen = {integrity.fingerprint("000/solver.py")}
    points, invalid, repeats, tokens, unfinished = [], 0, 0, 0, 0
    for directory in sorted(p for p in root.iterdir() if p.is_dir() and p.name.isdigit()):
        if directory.name == "000":
            continue
        if (directory / "response.json").exists():
            usage = load(directory / "response.json")["usage"]
            tokens += usage["completion_tokens"] if usage else 0
        if not (directory / "decision.json").exists():
            unfinished += 1
            continue
        evaluation = load(directory / "evaluation.json")
        invalid += not evaluation["valid"]
        if (directory / "solver.py").exists():
            fingerprint = integrity.fingerprint(f"{directory.name}/solver.py")
            repeats += fingerprint in seen
            seen.add(fingerprint)
        score = evaluation["mean_score"] if evaluation["valid"] else -math.inf
        points.append((load(directory / "decision.json")["elapsed_seconds"] / 3600, score))
    hours, best_scores = [0], [best]
    for elapsed, score in sorted(points):
        best = max(best, score)
        hours.append(elapsed)
        best_scores.append(best)
    if (root / "finished.json").exists():
        research_seconds = load(root / "finished.json")["research_seconds"]
        hours.append(min(research_seconds, manifest["budget_seconds"]) / 3600)
        best_scores.append(best)
    return {
        "hours": hours,
        "best_scores": best_scores,
        "completed": len(points),
        "invalid": invalid,
        "repeats": repeats,
        "unfinished": unfinished,
        "completion_tokens": tokens,
    }


def parallel_report(root, manifest):
    record = measure(root)
    task = manifest["task"]
    workers = manifest["parallel"]["agents"]
    team = f"{workers} worker" if workers == 1 else f"{workers} workers"
    sign = -1 if task != "circle" and import_module(f"tasks.{task}.plot").MINIMIZE else 1
    fig, ax = plt.subplots(figsize=(9, 4.5), facecolor="#F7F8FA")
    best = [sign * s for s in record["best_scores"]]
    ax.step(record["hours"], best, where="post", color="#D57542", linewidth=2.5)
    ax.set(
        xlabel="Research hours",
        ylabel="Best validation score" if task == "bike" else "Best score",
        title=f"{task} · {team}",
    )
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.15)
    fig.tight_layout()
    fig.savefig(root / "progress.png", dpi=160)
    plt.close(fig)

    def links(directory):
        return " · ".join(
            f'<a href="{directory.name}/{name}">{name}</a>'
            for name in EVIDENCE
            if (directory / name).exists()
        )

    attempts = [p for p in root.iterdir() if p.is_dir() and p.name.isdigit() and p.name != "000"]
    completed = sorted(
        (p for p in attempts if (p / "decision.json").exists()),
        key=lambda p: load(p / "decision.json")["elapsed_seconds"],
    )
    rows = []
    for directory in completed:
        decision = load(directory / "decision.json")
        evaluation = load(directory / "evaluation.json")
        score = f"{sign * evaluation['mean_score']:.6g}" if evaluation["valid"] else "invalid"
        outcome = decision["decision"] + (" · new best" if decision["global_improvement"] else "")
        rows.append(
            f"<tr><td>{directory.name}</td><td>{decision['agent']}</td>"
            f"<td>{decision['elapsed_seconds'] / 3600:.3f}</td><td>{score}</td>"
            f"<td>{html.escape(outcome)}</td><td>{links(directory)}</td></tr>"
        )
    unfinished = "".join(
        f"<li>{p.name}: {links(p)}</li>"
        for p in sorted(attempts)
        if not (p / "decision.json").exists()
    )
    title = f"{html.escape(task)} · {team}"
    document = f"""<!doctype html><html lang="en"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{title}</title><style>
body{{max-width:1200px;margin:40px auto;padding:0 24px;font:16px/1.6 system-ui;color:#233343}}
img{{max-width:100%}}table{{border-collapse:collapse;width:100%;font-size:13px}}
td,th{{text-align:left;border-bottom:1px solid #e3e8ef;padding:8px}}a{{color:#3473A8}}
</style><h1>{title}</h1>
<p>{record["completed"]} completed experiments · {record["invalid"]} invalid ·
{record["repeats"]} repeats · {record["unfinished"]} unfinished ·
{record["completion_tokens"]:,} output tokens</p>
<img src="progress.png" alt="Best score over research hours">
<p><a href="manifest.json">Settings</a> · <a href="summary.json">Summary</a> ·
<a href="journal.jsonl">Event log</a> · <a href="best.py">Best program</a></p>
<h2>Experiments in completion order</h2><table><tr><th>Attempt</th><th>Worker</th>
<th>Hours</th><th>Score</th><th>Decision</th><th>Files</th></tr>{"".join(rows)}</table>
<h2>Unfinished at the deadline</h2><ul>{unfinished}</ul></html>"""
    (root / "report.html").write_text(document)
    print(root / "report.html")
