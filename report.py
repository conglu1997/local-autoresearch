"""Render recorded solutions, scores and annotated model responses."""

import argparse
import difflib
import html
import json
from pathlib import Path
from importlib import import_module

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.ticker import MaxNLocator


def load(path):
    return json.loads(path.read_text())


def report(root):
    manifest = load(root / "manifest.json")
    if "parallel" in manifest:
        from tools.team_report import parallel_report

        return parallel_report(root, manifest)
    model = "Flint" if manifest["model"] == "flint" else manifest["model"]
    task = manifest["task"] if "task" in manifest else "circle"
    score_label = "Σr" if len(manifest["seeds"]) == 1 else "mean Σr"
    measurement_note = (
        "Packing and score from one fixed seed."
        if len(manifest["seeds"]) == 1
        else "First trial shown; score averaged over the recorded search seeds."
    )
    annotations = load(root / "annotations.json") if (root / "annotations.json").exists() else {}
    attempts = sorted(p for p in root.iterdir() if p.is_dir() and p.name.isdigit())
    completed = [p for p in attempts if (p / "decision.json").exists()]
    best = root / "000"
    rows, cards, scores, best_scores = [], [], [], []
    baseline = load(best / "evaluation.json")["mean_score"]
    best_score = baseline
    exact_best, exact_score = best, baseline
    total_seconds, tokens, usage_complete = 0, 0, True
    for directory in completed:
        result = load(directory / "evaluation.json")
        decision = load(directory / "decision.json")
        outcome = decision["decision"]
        if outcome in ("keep", "baseline"):
            best, best_score = directory, result["mean_score"]
        if (
            task == "circle"
            and result["valid"]
            and all(trial["certificate"]["exact_feasible"] for trial in result["trials"])
            and result["mean_score"] > exact_score
        ):
            exact_best, exact_score = directory, result["mean_score"]
        score = f"{result['mean_score']:.8f}" if result["valid"] else "invalid"
        scores.append(result["mean_score"])
        best_scores.append(best_score)
        attempt_link = (
            "000 (baseline)"
            if directory.name == "000"
            else f'<a href="#attempt-{directory.name}">{directory.name}</a>'
        )
        rows.append(f"<tr><td>{attempt_link}</td><td>{score}</td><td>{outcome}</td></tr>")
        if directory.name == "000":
            continue
        response = load(directory / "response.json")
        total_seconds += response["seconds"]
        if response["usage"] is None:
            usage_complete = False
        else:
            tokens += response["usage"]["completion_tokens"]
        parent = root / decision["parent"] / "solver.py"
        code_file = directory / "solver.py"
        diff = "No executable candidate was extracted."
        if code_file.exists():
            diff = "".join(
                difflib.unified_diff(
                    parent.read_text().splitlines(True),
                    code_file.read_text().splitlines(True),
                    fromfile=f"{decision['parent']}/solver.py",
                    tofile=f"{directory.name}/solver.py",
                )
            )
        trial_details = {key: value for key, value in result.items() if key != "trials"}
        if "trials" in result:
            trial_details["trials"] = [
                {key: value for key, value in trial.items() if key not in ("circles", "solution")}
                for trial in result["trials"]
            ]
        annotation = (
            f'<p class="note"><strong>Note:</strong> {html.escape(annotations[directory.name])}</p>'
            if directory.name in annotations
            else ""
        )
        diagnostic = (
            f'<p><a href="{directory.name}/diagnostic.json">Separate diagnostics (not scored)</a></p>'
            if (directory / "diagnostic.json").exists()
            else ""
        )
        stream = "stream.jsonl.gz" if (directory / "stream.jsonl.gz").exists() else "stream.jsonl"
        cards.append(f'''<article id="attempt-{directory.name}">
<h2>{directory.name} <span class="{outcome}">{outcome}</span> <span class="score">{score}</span></h2>
<p class="note">Parent {decision["parent"]} · {response["seconds"]:.1f}s request time · finish: {html.escape(str(response["finish_reason"]))}</p>
{annotation}
{diagnostic}
<details open><summary>Model response — verbatim</summary><pre>{html.escape(response["content"])}</pre></details>
<details><summary>Recorded reasoning</summary><pre>{html.escape(response["reasoning"])}</pre></details>
<details><summary>Change from parent</summary><pre>{html.escape(diff)}</pre></details>
<details><summary>Evaluator output</summary><pre>{html.escape(json.dumps(trial_details, indent=2))}</pre></details>
<p><a href="{directory.name}/request.json">Full request</a> · <a href="{directory.name}/{stream}">Raw stream</a> · <a href="{directory.name}/evaluation.json">Solution &amp; measurements</a></p>
</article>''')
    if task != "circle":
        task_report(root, task, model, completed, best, scores, best_scores, cards)
        return
    plt.rcParams.update({"font.family": "DejaVu Sans", "svg.fonttype": "none"})
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.3), facecolor="#faf9f6")
    best_title = (
        "Grid baseline (no improvement yet)"
        if best.name == "000"
        else f"{model} · attempt {best.name}"
    )
    for axis, directory, title in zip(axes, [root / "000", best], ["Grid baseline", best_title]):
        result = load(directory / "evaluation.json")
        circles = result["trials"][0]["circles"]
        axis.set(xlim=(0, 1), ylim=(0, 1), aspect="equal")
        axis.set_facecolor("#faf9f6")
        axis.set_xticks([])
        axis.set_yticks([])
        for i, (x, y, radius) in enumerate(circles):
            color = plt.cm.viridis(i / len(circles))
            axis.add_patch(
                Circle(
                    (x, y), radius, facecolor=color, edgecolor="white", linewidth=0.7, alpha=0.85
                )
            )
        axis.set_title(f"{title}\n{score_label} = {result['mean_score']:.6f}", loc="left", pad=16)
    fig.suptitle("Circle packing", x=0.07, ha="left", fontsize=16)
    fig.text(
        0.07,
        0.015,
        measurement_note,
        fontsize=9,
        color="#555555",
    )
    fig.tight_layout(rect=(0, 0.04, 1, 0.94))
    fig.savefig(root / "packings.svg", facecolor=fig.get_facecolor())
    fig.savefig(root / "packings.png", dpi=160, facecolor=fig.get_facecolor())
    plt.close(fig)
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), facecolor="#faf9f6")
    valid_x = [i for i, value in enumerate(scores) if value is not None]
    for axis in axes:
        axis.set_facecolor("#faf9f6")
        axis.step(
            range(len(scores)),
            best_scores,
            where="post",
            color="#5c4094",
            label="Best retained",
            linewidth=2,
        )
        axis.scatter(
            valid_x,
            [scores[i] for i in valid_x],
            color="#5c4094",
            s=16,
            alpha=0.65,
            zorder=3,
            label="Scored attempt",
        )
        axis.set(xlabel="Attempt (0 = baseline)", ylabel=score_label)
        axis.grid(alpha=0.15)
    invalid_x = [i for i, value in enumerate(scores) if value is None]
    axes[0].scatter(
        invalid_x,
        [baseline] * len(invalid_x),
        marker="x",
        color="#b5473a",
        zorder=4,
        label="Invalid (placed at baseline; no score)",
    )
    axes[0].set_title("Every attempt, including regressions and failures", loc="left")
    axes[0].legend(fontsize=8, loc="lower left")
    improvement = best_score - baseline
    zoom_floor = best_score - max(improvement * 0.15, 0.001)
    zoom_start = next(i for i, value in enumerate(best_scores) if value >= zoom_floor)
    axes[1].set(
        xlim=(max(0, zoom_start - 1), max(len(scores) - 0.5, zoom_start + 1)),
        ylim=(zoom_floor, best_score + max(improvement * 0.025, 0.0002)),
    )
    axes[1].set_title("Detail near the best score · lower results outside this view", loc="left")
    fig.tight_layout()
    fig.savefig(root / "progress.svg", facecolor=fig.get_facecolor())
    fig.savefig(root / "progress.png", dpi=180, facecolor=fig.get_facecolor())
    plt.close(fig)
    best_result = load(best / "evaluation.json")
    fig, axes = plt.subplots(1, len(best_result["trials"]), figsize=(12, 4.5), squeeze=False)
    fig.set_facecolor("#faf9f6")
    for axis, trial in zip(axes[0], best_result["trials"]):
        axis.set(xlim=(0, 1), ylim=(0, 1), aspect="equal", xticks=[], yticks=[])
        axis.set_facecolor("#faf9f6")
        for i, (x, y, radius) in enumerate(trial["circles"]):
            axis.add_patch(
                Circle(
                    (x, y),
                    radius,
                    facecolor=plt.cm.viridis(i / 26),
                    edgecolor="white",
                    linewidth=0.6,
                )
            )
        axis.set_title(f"Seed {trial['seed']} · Σr = {trial['score']:.6f}", loc="left", pad=12)
    fig.suptitle(
        f"Retained candidate {best.name} · every search seed", x=0.04, ha="left", fontsize=16
    )
    fig.text(
        0.04,
        0.02,
        "Saved coordinates.",
        fontsize=9,
        color="#555555",
    )
    fig.tight_layout(rect=(0, 0.06, 1, 0.93))
    fig.savefig(root / "best-seeds.svg", facecolor=fig.get_facecolor())
    fig.savefig(root / "best-seeds.png", dpi=180, facecolor=fig.get_facecolor())
    plt.close(fig)
    exact = sum(trial["certificate"]["exact_feasible"] for trial in best_result["trials"])
    exact_link = (
        "000 (baseline)"
        if exact_best.name == "000"
        else f'<a href="#attempt-{exact_best.name}">attempt {exact_best.name}</a>'
    )
    validation = ""
    if (root / "notes.json").exists():
        validation += f"<details><summary>Recorded run metadata</summary><pre>{html.escape((root / 'notes.json').read_text())}</pre></details>"
    if (root / "validation.json").exists():
        check = load(root / "validation.json")
        validation += f'<details><summary>Post-run seeds 3–7</summary><pre>{html.escape(json.dumps({key: value for key, value in check.items() if key != "trials"}, indent=2))}</pre><a href="validation.json">Full validation</a></details>'
    document = f"""<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{html.escape(model)} / circle packing</title><style>
body{{margin:0;background:#faf9f6;color:#24212a;font:16px/1.6 system-ui,sans-serif}}main{{max-width:1040px;margin:auto;padding:48px 24px}}h1{{font-size:clamp(36px,6vw,64px);line-height:1.1;letter-spacing:-2px;font-weight:550;margin:18px 0}}h2{{font-size:22px}}.eyebrow{{text-transform:uppercase;letter-spacing:.16em;color:#705494;font-size:12px}}.intro{{max-width:720px;color:#625c69}}img{{width:100%;height:auto}}.metrics{{display:flex;gap:40px;flex-wrap:wrap;margin:28px 0}}.metrics b{{display:block;font-size:30px;font-weight:500}}.note,.metrics span{{color:#746c7d;font-size:13px}}article{{border-top:1px solid #ded9e4;padding:22px 0}}pre{{white-space:pre-wrap;overflow-wrap:anywhere;background:#f0edf3;padding:20px;border-radius:8px;font:12px/1.6 ui-monospace,monospace}}summary{{cursor:pointer;padding:8px 0;font-weight:550}}table{{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}}th,td{{text-align:left;border-bottom:1px solid #e3deea;padding:9px}}a{{color:#604193}}.keep{{color:#347353}}.invalid{{color:#ab493a}}.discard{{color:#8b7147}}.score{{float:right;font:20px ui-monospace,monospace}}footer{{margin-top:32px;font-size:13px;color:#746c7d}}
</style><main><div class="eyebrow">{html.escape(model)} · Circle packing</div><h1>26 circles in a square</h1>
<p class="intro">Maximize the sum of radii while keeping every circle inside the square and avoiding overlaps.</p>
<div class="metrics"><div><b>{best_score:.6f}</b><span>best {score_label}</span></div><div><b>+{(best_score / baseline - 1) * 100:.1f}%</b><span>over the simple grid</span></div><div><b>{len(completed) - 1}</b><span>attempts</span></div><div><b>{total_seconds / 60:.1f} min</b><span>request time</span></div></div>
<img src="packings.svg" alt="Baseline and best packing, shown side by side"><img src="progress.svg" alt="All measured experiment scores and the retained best score">
<details><summary>Best candidate across all search seeds</summary><img src="best-seeds.svg" alt="The retained candidate's unchanged packing for each search seed"></details>
<p class="note">Red crosses mark invalid attempts at the baseline level, not a measured score. Best candidate exact decimal feasibility: {exact}/{len(best_result["trials"])} search seeds. Generated tokens: {tokens if usage_complete else "unknown (API usage incomplete)"}.</p>
<p class="note">Highest mean score with exact decimal feasibility on every search seed: {exact_link}, Σr = {exact_score:.12f}.</p>
<details><summary>Experiment ledger</summary><table><tr><th>Attempt</th><th>{score_label}</th><th>Decision</th></tr>{"".join(rows)}</table></details>{validation}
{"".join(cards)}<footer>{html.escape(model)} proposed the solver changes; the evaluator and baseline stayed fixed. <a href="manifest.json">Run manifest</a>.</footer></main></html>"""
    (root / "report.html").write_text(document)
    print(root / "report.html")


def task_report(root, task, model, completed, best, scores, best_scores, cards):
    view = import_module(f"tasks.{task}.plot")
    trials = [
        load(directory / "evaluation.json")["trials"][0] for directory in (root / "000", best)
    ]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), facecolor="white")
    fig.suptitle(f"{model} · {view.TITLE}", x=0.05, ha="left", fontsize=16)
    view.draw(axes[:2], trials)
    titles = ["Baseline", model if best.name != "000" else "Best · baseline"]
    for axis, trial, title in zip(axes, trials, titles):
        axis.set_title(f"{title} · {trial[view.METRIC]:,.6g}", loc="left")
    values = [-s for s in best_scores] if view.MINIMIZE else best_scores
    axes[2].step(
        range(len(values)),
        values,
        where="post",
        color="#397fba",
        marker="o",
        markersize=4,
        label="Best",
    )
    valid = [i for i, score in enumerate(scores) if score is not None]
    attempt_values = [-scores[i] if view.MINIMIZE else scores[i] for i in valid]
    axes[2].scatter(valid, attempt_values, color="#7c8794", s=18, label="Attempt", zorder=3)
    scored = sum(score is not None for score in scores[1:])
    axes[2].set_title(f"Progress · {scored}/{len(scores) - 1} scored", loc="left")
    axes[2].legend(frameon=False, fontsize=9)
    axes[2].set(xlabel="Attempt", ylabel=view.LABEL, xlim=(-0.1, max(1, len(values) - 1) + 0.1))
    axes[2].xaxis.set_major_locator(MaxNLocator(integer=True))
    axes[2].yaxis.set_major_locator(MaxNLocator(nbins=5))
    for axis in axes:
        axis.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(root / "progress.png", dpi=120)
    plt.close(fig)
    direction = "lower is better" if view.MINIMIZE else "higher is better"
    measured = f"{trials[0][view.METRIC]:,.6g} → {trials[1][view.METRIC]:,.6g}"
    final_note = ""
    if (root / "final-test.json").exists():
        final = load(root / "final-test.json")
        measured_final = (
            f"{final['trials'][0][view.METRIC]:.6g}" if final["valid"] else "failed evaluation"
        )
        final_note = f'<p><a href="final-test.json">Held-out final test</a>: {measured_final}.</p>'
        if task == "bike":
            baseline_final = load(root / "baseline-final-test.json")
            baseline_value = (
                f"{baseline_final['trials'][0][view.METRIC]:.6g}"
                if baseline_final["valid"]
                else "failed evaluation"
            )
            final_note += (
                '<p><a href="baseline-final-test.json">Matched baseline final test</a>: '
                f"{baseline_value}. Both programs refit on the same data and budget.</p>"
            )
    document = f"""<!doctype html><html lang="en"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{html.escape(model)} · {html.escape(view.TITLE)}</title><style>
body{{max-width:1100px;margin:40px auto;padding:0 24px;color:#2c3642;font:16px/1.6 system-ui}}
img{{width:100%}}pre{{white-space:pre-wrap;overflow-wrap:anywhere;background:#f5f7fa;padding:20px;font:13px/1.5 monospace}}
article{{border-top:1px solid #dbe3eb;padding:24px 0}}a{{color:#397fba}}.note{{color:#6f7e90}}summary{{cursor:pointer}}
</style><h1>{html.escape(model)} · {html.escape(view.TITLE)}</h1><p>{len(completed) - 1} attempts · {scored} scored · {html.escape(view.LABEL)}: {measured} ({direction})</p>
<img src="progress.png" alt="Baseline, retained solution and measured progress">
<p>The evaluator, data and baseline stayed fixed throughout the search.</p>{final_note}
{"".join(cards)}</html>"""
    (root / "report.html").write_text(document)
    print(root / "report.html")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    report(parser.parse_args().run.resolve())
