"""Render local research galleries from recorded experiments."""

import argparse
import hashlib
import json
import subprocess
import tempfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
from matplotlib.patches import Circle, FancyBboxPatch
from matplotlib.ticker import MaxNLocator
from PIL import Image

from tasks.jobshop import schedule
from tasks.bike import VALIDATION

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "docs/images"

PAPER = "#FAF9F5"
INK = "#263B43"
MUTED = "#687A7F"
COLORS = ["#B7B0DB", "#BDDCD7", "#F1D6AE", "#C9D7EB", "#E6BEC6"]
RUNS = [
    {
        "key": "circle",
        "folder": "2026-10-20-final",
        "title": "Circle packing",
        "subtitle": "26 circles in a square",
        "headline": "Circle\npacking",
        "description": "Fit 26 circles into a square.\nMake their radii add up to more.",
        "accent": "#7961A7",
        "tint": "#F0EDF7",
        "metric": "Mean sum of radii",
        "gain": "larger radius sum",
        "score_format": ".3f",
        "milestones": [
            (1, "Stagger the rows", "Offset equal-sized circles\nin alternating rows."),
            (
                22,
                "Let every radius vary",
                "Optimize positions and individual\nradii together with SLSQP.",
            ),
            (
                109,
                "Try fresh starting layouts",
                "Place circles sequentially at random,\nthen optimize the new packing.",
            ),
        ],
    },
    {
        "key": "jobshop",
        "folder": "2026-10-20-jobshop",
        "title": "Factory scheduling",
        "subtitle": "40 jobs across 20 machines",
        "headline": "Factory\nscheduling",
        "description": "Order 800 operations across 20 machines.\nGet the last job finished sooner.",
        "accent": "#277D82",
        "tint": "#EAF3F0",
        "metric": "Finishing time",
        "gain": "shorter finishing time",
        "score_format": ",.0f",
        "milestones": [
            (
                28,
                "Try 64 starting schedules",
                "Randomize greedy choices;\nkeep the fastest schedule.",
            ),
            (
                34,
                "Make longer schedule moves",
                "Reinsert operations; also change\ninitialization and annealing.",
            ),
            (
                71,
                "Look ahead at machine work",
                "Rank partial schedules by workload;\nallow moves earlier or later.",
            ),
        ],
    },
    {
        "key": "bike",
        "folder": "2026-10-21-bike",
        "title": "Machine learning",
        "subtitle": "Bike rental prediction",
        "headline": "Machine learning\nfor bike rentals",
        "description": "Train a model to predict hourly rentals\nfrom calendar and weather features.",
        "accent": "#B36B3D",
        "tint": "#FAF0E4",
        "metric": "Validation RMSLE",
        "gain": "lower validation error",
        "score_format": ".5f",
        "milestones": [],
    },
]


def load(path):
    return json.loads(path.read_text())


def records(spec, root=None):
    if root is None:
        root = ROOT / "examples" / spec["folder"]
    summary = load(root / "summary.json")
    baseline = load(root / "000/evaluation.json")
    winner = load(root / summary["best_attempt"] / "evaluation.json")
    values, states, seconds = [], [], [0]
    best = baseline["mean_score"]
    for number in range(summary["completed_experiments"] + 1):
        directory = root / f"{number:03d}"
        if number:
            seconds.append(load(directory / "response.json")["seconds"])
        decision = load(directory / "decision.json")["decision"]
        if decision in {"baseline", "keep"}:
            evaluation = load(directory / "evaluation.json")
            best = evaluation["mean_score"]
            states.append(
                {
                    "attempt": number,
                    "trial": evaluation["trials"][0],
                    "best": abs(best),
                    "gain": (best - baseline["mean_score"]) / abs(baseline["mean_score"]) * 100,
                }
            )
        values.append(best if spec["key"] == "circle" else -best)
    hours = np.cumsum(seconds) / 3600
    return {
        "root": root,
        "states": states,
        "baseline": abs(baseline["mean_score"]),
        "best": abs(winner["mean_score"]),
        "gain": (winner["mean_score"] - baseline["mean_score"]) / abs(baseline["mean_score"]) * 100,
        "trial": winner["trials"][0],
        "values": values,
        "hours": hours,
    }


def panel(fig, bounds, color):
    fig.add_artist(
        FancyBboxPatch(
            bounds[:2],
            bounds[2],
            bounds[3],
            boxstyle="round,pad=0,rounding_size=0.025",
            transform=fig.transFigure,
            facecolor=color,
            edgecolor="none",
            zorder=-1,
        )
    )


def clean_axis(axis):
    axis.spines[["top", "right"]].set_visible(False)
    axis.spines[["bottom", "left"]].set_color("#D3DEDD")
    axis.tick_params(colors=MUTED, length=0, labelsize=9, pad=6)
    axis.set_facecolor("none")
    axis.grid(axis="y", color="white", linewidth=1)
    axis.set_axisbelow(True)


def packing(axis, data):
    axis.set(xlim=(-0.025, 1.025), ylim=(-0.025, 1.025), aspect="equal")
    axis.add_patch(
        FancyBboxPatch(
            (0, 0),
            1,
            1,
            boxstyle="round,pad=0,rounding_size=0.01",
            facecolor=PAPER,
            edgecolor="#C6BBDC",
            linewidth=1.5,
        )
    )
    for i, (x, y, radius) in enumerate(data["trial"]["circles"]):
        axis.add_patch(
            Circle(
                (x, y), radius, facecolor=COLORS[i % len(COLORS)], edgecolor=PAPER, linewidth=1.8
            )
        )
    axis.axis("off")


def factory(axis, data):
    for job, _, machine, start, finish in schedule(data["trial"]["solution"]):
        axis.broken_barh(
            [(start, finish - start)],
            (machine + 0.15, 0.73),
            facecolors=COLORS[job % len(COLORS)],
            edgecolors=PAPER,
            linewidth=0.35,
        )
    axis.axvline(data["best"], color="#277D82", linewidth=1.2, ls=":")
    axis.set(xlim=(0, data["baseline"]), ylim=(-0.3, 20.3), yticks=[], xlabel="Schedule time")
    axis.xaxis.set_major_locator(MaxNLocator(4))
    clean_axis(axis)
    axis.spines["left"].set_visible(False)
    axis.grid(False)


def bike_predictions(axis, data):
    days, indices = np.unique([row["dteday"] for row in VALIDATION], return_inverse=True)
    observed = np.bincount(indices, weights=[int(row["cnt"]) for row in VALIDATION])
    predicted = np.bincount(indices, weights=data["trial"]["solution"])
    limit = max(
        observed.max(),
        *(np.bincount(indices, weights=s["trial"]["solution"]).max() for s in data["states"]),
    )
    dates = np.asarray(days, dtype="datetime64[D]")
    axis.plot(dates, observed, color=MUTED, lw=1.3, alpha=0.65, label="Actual")
    axis.plot(dates, predicted, color="#B36B3D", lw=1.5, label="Predicted")
    axis.set(xlim=(dates[0], dates[-1]), ylim=(0, limit * 1.06), ylabel="Daily rentals")
    axis.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    axis.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    axis.yaxis.set_major_locator(MaxNLocator(3))
    clean_axis(axis)
    axis.legend(loc="upper left", fontsize=8, frameon=False, ncol=2)


def progress_plot(axis, spec, data, end_hours=None):
    end = data["hours"][-1] if end_hours is None else min(end_hours, data["hours"][-1])
    index = int(np.searchsorted(data["hours"], end, side="right") - 1)
    axis.step(
        data["hours"][: index + 1],
        data["values"][: index + 1],
        where="post",
        color=spec["accent"],
        linewidth=2.5,
    )
    axis.scatter(
        [data["hours"][index]], [data["values"][index]], s=24, color=spec["accent"], zorder=3
    )
    axis.set(
        xlabel="Model time (hours)",
        ylabel=spec["metric"],
        xlim=(-0.03 * end, 1.03 * end),
    )
    axis.xaxis.set_major_locator(MaxNLocator(4))
    axis.yaxis.set_major_locator(MaxNLocator(3))
    clean_axis(axis)
    axis.grid(axis="y", color="#E4E7E3")


def bike_annotations(axis, spec, data, end_hours):
    if "discoveries" in spec:
        annotations = []
        for item in spec["discoveries"]:
            attempt = item["attempt"]
            if data["hours"][attempt] > end_hours:
                continue
            annotation = axis.annotate(
                item["label"],
                xy=(data["hours"][attempt], data["values"][attempt]),
                xytext=item["position"],
                textcoords="axes fraction",
                fontsize=12,
                va="bottom",
                linespacing=1.3,
                bbox=dict(facecolor=PAPER, edgecolor="none", pad=3),
                arrowprops=dict(
                    arrowstyle="-",
                    color=spec["accent"],
                    lw=1.1,
                    shrinkA=5,
                    shrinkB=3,
                    relpos=(0, 0),
                ),
            )
            annotations.append((attempt, annotation))
        return annotations
    return []


def bike_card(spec, data, social=False):
    end = min(spec["hours"], data["hours"][-1])
    index = int(np.searchsorted(data["hours"], end, side="right") - 1)
    best = data["values"][index]
    gain = (data["baseline"] - best) / data["baseline"] * 100
    fig = plt.figure(figsize=(8, 10) if social else (12, 8), facecolor=PAPER)
    fig.text(
        0.075, 0.94, "FLINT · LOCAL ON A MACBOOK", fontsize=12, weight="bold", color=spec["accent"]
    )
    title = "Learning to predict\nbike rentals" if social else "Learning to predict bike rentals"
    fig.text(0.075, 0.89, title, fontsize=29, va="top", weight="medium")
    fig.text(
        0.075,
        0.76,
        f"{end:g} model hours · {index} completed attempts"
        + (" · in progress" if data["in_progress"] else ""),
        fontsize=13,
        color=MUTED,
    )
    axis = fig.add_axes((0.13, 0.24, 0.80, 0.46))
    progress_plot(axis, spec, data, end_hours=end)
    axis.set_xticks(np.linspace(0, end, 4))
    low, high = min(data["values"][: index + 1]), max(data["values"][: index + 1])
    axis.set(
        ylim=(low - (high - low) * 0.12, high + (high - low) * 0.2),
        ylabel="Best validation RMSLE ↓",
    )
    axis.tick_params(labelsize=11)
    axis.xaxis.label.set_size(12)
    axis.yaxis.label.set_size(12)
    bike_annotations(axis, spec, data, end_hours=end)
    fig.text(
        0.075,
        0.13,
        f"{data['baseline']:.4f} → {best:.4f}",
        fontsize=24,
        color=spec["accent"],
    )
    fig.text(
        0.075,
        0.09,
        f"{gain:.2f}% lower validation error",
        fontsize=14,
        color=spec["accent"],
    )
    if social:
        output = OUTPUT / "social"
        output.mkdir(exist_ok=True)
        fig.savefig(output / "bike.png", dpi=150, facecolor=PAPER)
        plt.close(fig)
    else:
        save(fig, "bike")


def art(fig, spec, data, bounds):
    axis = fig.add_axes(bounds)
    if spec["key"] == "circle":
        packing(axis, data)
    else:
        factory(axis, data)


def save(fig, name):
    fig.savefig(OUTPUT / f"{name}.png", dpi=150, facecolor=PAPER)
    plt.close(fig)


def cover(spec, data):
    if spec["key"] == "bike":
        bike_card(spec, data)
        return
    fig = plt.figure(figsize=(12, 8), facecolor=PAPER)
    panel(fig, (0.55, 0.49, 0.405, 0.45), spec["tint"])
    fig.text(
        0.065,
        0.93,
        "FLINT · LOCAL ON A MACBOOK",
        color=spec["accent"],
        fontsize=11,
        weight="bold",
    )
    fig.text(
        0.065, 0.885, spec["headline"], fontsize=31, linespacing=1.02, va="top", weight="medium"
    )
    fig.text(0.065, 0.71, spec["description"], fontsize=11, linespacing=1.5, va="top", color=MUTED)
    fig.text(
        0.065,
        0.555,
        f"{data['gain']:.{2 if spec['key'] == 'bike' else 1}f}%",
        fontsize=29,
        color=spec["accent"],
    )
    fig.text(0.065, 0.52, spec["gain"], fontsize=11, color=spec["accent"])
    if spec["key"] == "circle":
        art(fig, spec, data, (0.61, 0.555, 0.285, 0.36))
        caption = "26 circles · one recorded packing"
    else:
        art(fig, spec, data, (0.585, 0.60, 0.335, 0.30))
        caption = "40 jobs · 20 machines · 800 operations"
    fig.text(0.7525, 0.525, caption, ha="center", fontsize=9, color=MUTED)

    axis = fig.add_axes((0.105, 0.12, 0.495, 0.29))
    progress_plot(axis, spec, data)
    for index, (attempt, title, description) in enumerate(spec["milestones"], 1):
        axis.annotate(
            str(index),
            (data["hours"][attempt], data["values"][attempt]),
            xytext=(0, 19),
            textcoords="offset points",
            ha="center",
            va="center",
            fontsize=9,
            color="white",
            bbox=dict(boxstyle="circle,pad=0.3", fc=spec["accent"], ec=PAPER, lw=1.5),
            arrowprops=dict(arrowstyle="-", color=spec["accent"], lw=1),
        )
        y = 0.405 - (index - 1) * 0.115
        fig.text(
            0.655,
            y,
            str(index),
            ha="center",
            va="center",
            fontsize=9,
            color="white",
            bbox=dict(boxstyle="circle,pad=0.3", fc=spec["accent"], ec="none"),
        )
        fig.text(0.68, y, title, va="center", fontsize=11.5, weight="medium")
        fig.text(0.68, y - 0.025, description, va="top", fontsize=10, linespacing=1.4, color=MUTED)
        before, after = data["values"][attempt - 1], data["values"][attempt]
        fig.text(
            0.68,
            y - 0.084,
            f"{before:{spec['score_format']}} → {after:{spec['score_format']}}",
            fontsize=11,
            color=spec["accent"],
        )
    save(fig, spec["key"])


def social_card(spec, data):
    if spec["key"] == "bike":
        bike_card(spec, data, social=True)
        return
    fig = plt.figure(figsize=(8, 10), facecolor=PAPER)
    fig.text(
        0.075,
        0.945,
        "FLINT · LOCAL ON A MACBOOK",
        fontsize=15,
        weight="bold",
        color=spec["accent"],
    )
    fig.text(0.075, 0.885, spec["title"], fontsize=30, weight="medium")
    fig.text(0.075, 0.845, spec["subtitle"], fontsize=18, color=MUTED)
    panel(fig, (0.06, 0.43, 0.88, 0.37), spec["tint"])
    axis = fig.add_axes((0.15, 0.525, 0.76, 0.235))
    if spec["key"] == "circle":
        axis.set_position((0.25, 0.485, 0.5, 0.30))
        packing(axis, data)
        caption = "26 circles · one recorded packing"
    else:
        factory(axis, data)
        caption = "40 jobs · 20 machines · 800 operations"
    axis.tick_params(labelsize=14)
    axis.xaxis.label.set_size(14)
    fig.text(0.5, 0.455, caption, ha="center", fontsize=14, color=MUTED)
    fig.text(
        0.075,
        0.355,
        f"{data['gain']:.{2 if spec['key'] == 'bike' else 1}f}%",
        fontsize=38,
        color=spec["accent"],
    )
    fig.text(0.37, 0.355, spec["gain"], fontsize=18, color=spec["accent"])
    fig.text(0.17, 0.305, "Best result so far", fontsize=15, color=MUTED)
    axis = fig.add_axes((0.17, 0.125, 0.74, 0.155))
    progress_plot(axis, spec, data)
    axis.tick_params(labelsize=14)
    axis.xaxis.label.set_size(15)
    axis.yaxis.label.set_size(15)
    output = OUTPUT / "social"
    output.mkdir(exist_ok=True)
    fig.savefig(output / f"{spec['key']}.png", dpi=150, facecolor=PAPER)
    plt.close(fig)


def scene(spec, data):
    fig = plt.figure(figsize=(4.8, 2.7), dpi=150, facecolor=spec["tint"])
    bounds = {
        "circle": (0.03, 0.02, 0.94, 0.96),
        "jobshop": (0.11, 0.20, 0.85, 0.78),
        "bike": (0.19, 0.20, 0.77, 0.75),
    }[spec["key"]]
    axis = fig.add_axes(bounds)
    if spec["key"] == "circle":
        packing(axis, data)
    elif spec["key"] == "bike":
        bike_predictions(axis, data)
    else:
        factory(axis, data)
    fig.canvas.draw()
    image = Image.fromarray(np.asarray(fig.canvas.buffer_rgba())).convert("RGB")
    plt.close(fig)
    return image


def overview(experiments, bike_hours=3.0):
    fig = plt.figure(figsize=(16, 9), dpi=80, facecolor=PAPER)
    fig.text(0.055, 0.92, "Flint autoresearch on my MacBook", fontsize=33, weight="medium")
    fig.text(
        0.055,
        0.865,
        "Geometry, scheduling and machine learning. Entirely local.",
        fontsize=16,
        color=MUTED,
    )
    tracks = []
    for index, (spec, data) in enumerate(experiments):
        x = 0.045 + index * 0.315
        panel(fig, (x, 0.075, 0.285, 0.72), spec["tint"])
        fig.text(x + 0.022, 0.745, spec["title"], fontsize=19, weight="medium")
        subtitle = spec["subtitle"]
        if spec["key"] == "bike" and data["in_progress"]:
            subtitle = "Bike rentals · run in progress"
        fig.text(x + 0.022, 0.705, subtitle, fontsize=11, color=MUTED)
        end = min(data["hours"][-1], bike_hours) if spec["key"] == "bike" else data["hours"][-1]
        track = dict(spec=spec, data=data, end=end)
        images = [scene(spec, data | state) for state in data["states"]]
        picture = fig.add_axes((x + 0.015, 0.435, 0.255, 0.255))
        image = picture.imshow(images[-1], interpolation="lanczos", aspect="equal")
        picture.axis("off")
        gain = fig.text(x + 0.022, 0.365, "", fontsize=27, color=spec["accent"])
        fig.text(x + 0.022, 0.33, spec["gain"], fontsize=12, color=spec["accent"])
        clock = fig.text(x + 0.263, 0.29, "", ha="right", fontsize=10, color=spec["accent"])
        axis = fig.add_axes((x + 0.05, 0.155, 0.213, 0.12))
        track.update(images=images, image=image)
        low, high = min(data["values"]), max(data["values"])
        padding = (high - low) * 0.12
        axis.set(
            xlim=(0, end * 1.025),
            ylim=(low - padding, high + padding),
            xlabel="Model time (hours)",
            ylabel=spec["metric"],
        )
        axis.xaxis.set_major_locator(MaxNLocator(4))
        axis.yaxis.set_major_locator(MaxNLocator(3))
        clean_axis(axis)
        axis.axhline(data["baseline"], color=spec["accent"], alpha=0.18, lw=1, ls=":")
        (line,) = axis.step([], [], where="post", color=spec["accent"], lw=2.5)
        (halo,) = axis.plot([], [], "o", color=spec["accent"], markersize=13, alpha=0.15)
        (dot,) = axis.plot([], [], "o", color=spec["accent"], markersize=5)
        cursor = axis.axvline(0, color=spec["accent"], alpha=0.2, lw=1)
        track.update(gain=gain, clock=clock, line=line, halo=halo, dot=dot, cursor=cursor)
        tracks.append(track)
    update_overview(tracks, 1)
    return fig, tracks


def update_overview(tracks, progress):
    for track in tracks:
        data, spec = track["data"], track["spec"]
        positions = data["hours"]
        elapsed = min(progress * track["end"], positions[-1])
        index = int(np.searchsorted(positions, elapsed, side="right") - 1)
        selected = next(
            i for i in reversed(range(len(data["states"]))) if data["states"][i]["attempt"] <= index
        )
        state = data["states"][selected]
        picture = track["images"][selected]
        if selected and index < len(positions) - 1:
            blend = min(1, (elapsed - positions[state["attempt"]]) / (track["end"] * 0.015))
            eased = blend * blend * (3 - 2 * blend)
            picture = Image.blend(track["images"][selected - 1], picture, eased)
        track["image"].set_data(picture)
        track["line"].set_data(
            [*positions[: index + 1], elapsed],
            [*data["values"][: index + 1], data["values"][index]],
        )
        track["cursor"].set_xdata([elapsed, elapsed])
        for marker in (track["halo"], track["dot"]):
            marker.set_data([elapsed], [data["values"][index]])
        track["gain"].set_text(f"{state['gain']:.{2 if spec['key'] == 'bike' else 1}f}%")
        minutes = round(elapsed * 60)
        track["clock"].set_text(f"{minutes // 60}h {minutes % 60:02d}m")


def animate_overview(fig, tracks, mp4_only=False):
    # 324 frames at 1280×720 stay within X's 300-million-pixel GIF budget.
    fps, travel_seconds = 15, 17
    update_overview(tracks, 0)
    fig.canvas.draw()
    first = Image.fromarray(np.asarray(fig.canvas.buffer_rgba())).convert("RGB")
    width, height = first.size
    with tempfile.TemporaryDirectory(prefix="flint-gallery-") as temporary:
        video = Path(temporary) / "frames.mkv"
        palette = Path(temporary) / "palette.png"
        command = [
            "ffmpeg",
            "-v",
            "error",
            "-y",
            "-f",
            "rawvideo",
            "-pixel_format",
            "rgb24",
            "-video_size",
            f"{width}x{height}",
            "-framerate",
            str(fps),
            "-i",
            "pipe:0",
            "-c:v",
            "ffv1",
            str(video),
        ]
        with subprocess.Popen(command, stdin=subprocess.PIPE) as encoder:
            for _ in range(fps):
                encoder.stdin.write(first.tobytes())
            for progress in np.linspace(0, 1, travel_seconds * fps + 1):
                update_overview(tracks, progress)
                fig.canvas.draw()
                frame = Image.fromarray(np.asarray(fig.canvas.buffer_rgba())).convert("RGB")
                encoder.stdin.write(frame.tobytes())
            for _ in range(3 * fps):
                encoder.stdin.write(frame.tobytes())
            for blend in np.linspace(0, 1, 8):
                eased = blend * blend * (3 - 2 * blend)
                encoder.stdin.write(Image.blend(frame, first, eased).tobytes())
            encoder.stdin.close()
            if encoder.wait() != 0:
                raise subprocess.CalledProcessError(encoder.returncode, command)
        if mp4_only:
            subprocess.run(
                [
                    "ffmpeg",
                    "-v",
                    "error",
                    "-y",
                    "-i",
                    str(video),
                    "-an",
                    "-c:v",
                    "libx264",
                    "-preset",
                    "medium",
                    "-crf",
                    "18",
                    "-pix_fmt",
                    "yuv420p",
                    "-movflags",
                    "+faststart",
                    str(OUTPUT / "overview.mp4"),
                ],
                check=True,
            )
            return
        subprocess.run(
            [
                "ffmpeg",
                "-v",
                "error",
                "-y",
                "-i",
                str(video),
                "-vf",
                "palettegen",
                "-frames:v",
                "1",
                str(palette),
            ],
            check=True,
        )
        subprocess.run(
            [
                "ffmpeg",
                "-v",
                "error",
                "-y",
                "-i",
                str(video),
                "-i",
                str(palette),
                "-filter_complex",
                "[0:v][1:v]paletteuse=dither=none:diff_mode=rectangle",
                "-loop",
                "0",
                str(OUTPUT / "overview.gif"),
            ],
            check=True,
        )


def load_discoveries(path, run, completed):
    discoveries = load(path)
    for item in discoveries:
        attempt = item["attempt"]
        if not 1 <= attempt <= completed:
            raise ValueError(f"Discovery {attempt} is outside the completed run")
        directory = run / f"{attempt:03d}"
        if load(directory / "decision.json")["decision"] != "keep":
            raise ValueError(f"Discovery {attempt} is not a retained improvement")
        actual = hashlib.sha256((directory / "solver.py").read_bytes()).hexdigest()
        if actual != item["source_sha256"]:
            raise ValueError(f"Discovery {attempt} belongs to a different program")
    return discoveries


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--bike-run",
        type=Path,
        help="Bike run to visualize; supply matching discovery labels separately",
    )
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    parser.add_argument(
        "--bike-hours",
        type=float,
        default=3.0,
        help="Bike chart and playback window in recorded model hours",
    )
    parser.add_argument(
        "--bike-discoveries", type=Path, help="Verified labels for the selected bike run"
    )
    parser.add_argument("--in-progress", action="store_true", help="Label a partial bike run")
    format_group = parser.add_mutually_exclusive_group()
    format_group.add_argument(
        "--still-only", action="store_true", help="Skip overview GIF encoding"
    )
    format_group.add_argument(
        "--mp4-only", action="store_true", help="Render only a 1920×1080 overview MP4"
    )
    args = parser.parse_args()
    if args.bike_discoveries is not None and args.bike_run is None:
        parser.error("--bike-discoveries requires --bike-run")
    if args.in_progress and args.bike_run is None:
        parser.error("--in-progress requires --bike-run")
    OUTPUT = args.output_dir.resolve()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "text.color": INK,
            "axes.labelcolor": MUTED,
            "axes.labelsize": 9,
        }
    )
    experiments = []
    for spec in RUNS:
        if spec["key"] == "bike" and args.bike_run is not None:
            spec = spec | {"milestones": []}
            data = records(spec, args.bike_run.resolve())
        else:
            data = records(spec)
        if spec["key"] == "bike":
            spec = spec | {"hours": args.bike_hours}
            if args.bike_run is not None and len(data["values"]) < 101 and not args.in_progress:
                parser.error("A partial bike run requires --in-progress")
            data["in_progress"] = args.in_progress
            if args.bike_discoveries is not None:
                spec = spec | {
                    "discoveries": load_discoveries(
                        args.bike_discoveries, data["root"], len(data["values"]) - 1
                    )
                }
            elif args.bike_run is None:
                spec = spec | {
                    "discoveries": load_discoveries(
                        data["root"] / "discoveries.json", data["root"], len(data["values"]) - 1
                    )
                }
        experiments.append((spec, data))
    if not args.mp4_only:
        for spec, data in experiments:
            cover(spec, data)
            social_card(spec, data)
    fig, tracks = overview(experiments, bike_hours=args.bike_hours)
    if args.mp4_only:
        fig.set_dpi(120)
    if not args.still_only:
        animate_overview(fig, tracks, mp4_only=args.mp4_only)
