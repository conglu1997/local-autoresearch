"""Replay saved packings beside their score history without executing candidate code."""

import argparse
import json
import subprocess
import tempfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle
from PIL import Image

PAPER = "#ffffff"
INK = "#2c3642"
MUTED = "#6f7e90"
BLUE = "#397fba"


def load(path):
    return json.loads(path.read_text())


def render(root, output):
    manifest = load(root / "manifest.json")
    model = "Flint" if manifest["model"] == "flint" else manifest["model"]
    summary = load(root / "summary.json")
    attempts = [root / f"{number:03d}" for number in range(summary["completed_experiments"] + 1)]
    results = [load(attempt / "evaluation.json") for attempt in attempts]
    decisions = [load(attempt / "decision.json") for attempt in attempts]
    seconds = [
        0,
        *[load(attempt / "response.json")["seconds"] for attempt in attempts[1:]],
    ]
    hours = np.cumsum(seconds) / 3600
    # Follow the recorded keep decisions, showing the score of the displayed packing.
    # Do not reselect historical candidates using this single-trial view.
    baseline = results[0]["trials"][0]["score"]
    kept, best_scores = [], []
    best_index = 0
    for index, decision in enumerate(decisions):
        if decision["decision"] in {"baseline", "keep"}:
            best_index = index
        kept.append(best_index)
        best_scores.append(results[best_index]["trials"][0]["score"])
    retained = sorted(set(kept))
    total_seconds = sum(seconds)
    fps, travel_seconds, start_hold, end_hold = 14, 20, 1, 3
    frame_count = round((start_hold + travel_seconds + end_hold) * fps) + 1
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 12, "text.color": INK})
    fig = plt.figure(figsize=(12, 6.5), dpi=100, facecolor=PAPER)
    fig.text(
        0.055, 0.965, f"{model} · Circle packing", fontsize=22, color=INK, weight="medium", va="top"
    )
    packing = fig.add_axes((0.055, 0.15, 0.405, 0.748))
    best_axis = fig.add_axes((0.55, 0.24, 0.405, 0.53))
    fig.text(0.55, 0.895, "Score", fontsize=20, color=MUTED)
    score_label = fig.text(0.55, 0.835, "", fontsize=34, color=BLUE)
    gain_label = fig.text(0.955, 0.835, "", ha="right", fontsize=22, color=BLUE)
    clock_label = fig.text(0.955, 0.895, "", ha="right", fontsize=20, color=MUTED)
    output.parent.mkdir(parents=True, exist_ok=True)

    packing.set(xlim=(0, 1), ylim=(0, 1), aspect="equal", xticks=[], yticks=[])
    packing.set_facecolor(PAPER)
    packing_images = {}
    tile, tile_axis = plt.subplots(figsize=(6, 6), dpi=100)
    tile.subplots_adjust(left=0, right=1, bottom=0, top=1)
    for selected in retained:
        tile_axis.clear()
        tile_axis.set(xlim=(0, 1), ylim=(0, 1), aspect="equal", xticks=[], yticks=[])
        tile_axis.set_facecolor(PAPER)
        tile_axis.spines[:].set_visible(False)
        for x, y, radius in results[selected]["trials"][0]["circles"]:
            tile_axis.add_patch(
                Circle(
                    (x, y),
                    radius,
                    facecolor="#6da4d5",
                    edgecolor=PAPER,
                    linewidth=0.7,
                )
            )
        tile.canvas.draw()
        packing_images[selected] = Image.fromarray(np.asarray(tile.canvas.buffer_rgba())).convert(
            "RGB"
        )
    plt.close(tile)
    packing_image = packing.imshow(packing_images[0], extent=(0, 1, 0, 1), interpolation="lanczos")
    best_axis.set_facecolor(PAPER)
    best_axis.set_xlim(0, hours[-1] * 1.025)
    best_axis.set_ylim(baseline - 0.025, max(best_scores) + 0.025)
    best_axis.grid(alpha=0.12)
    best_axis.spines[["top", "right"]].set_visible(False)
    best_axis.spines[["bottom", "left"]].set_color("#dbe3eb")
    best_axis.tick_params(colors=MUTED, labelsize=17, length=0, pad=10)
    best_axis.set_xlabel("Model time (hours)", color=MUTED, labelpad=16, fontsize=18)
    best_axis.set_ylabel("Sum of radii", color=MUTED, labelpad=12, fontsize=18)
    cursor = best_axis.axvline(0, color=BLUE, alpha=0.2, linewidth=1)
    (line,) = best_axis.step([], [], where="post", color=BLUE, linewidth=3)
    (halo,) = best_axis.plot([], [], "o", color=BLUE, markersize=17, alpha=0.12)
    (dot,) = best_axis.plot([], [], "o", color=BLUE, markersize=7)
    for artist in (cursor, line, halo, dot):
        artist.set_snap(False)
    with tempfile.TemporaryDirectory(prefix="flint-animation-") as temporary:
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
            "1200x650",
            "-framerate",
            str(fps),
            "-i",
            "pipe:0",
            "-c:v",
            "ffv1",
            str(video),
        ]
        with subprocess.Popen(command, stdin=subprocess.PIPE) as encoder:
            for frame in range(frame_count):
                elapsed = min(
                    total_seconds, max(0, frame / fps - start_hold) / travel_seconds * total_seconds
                )
                current_hour = elapsed / 3600
                index = int(np.searchsorted(hours, current_hour, side="right") - 1)
                selected = kept[index]
                position = retained.index(selected)
                current_image = packing_images[selected]
                if position:
                    previous = retained[position - 1]
                    until_next = (
                        (hours[retained[position + 1]] - hours[selected])
                        / hours[-1]
                        * travel_seconds
                        if position + 1 < len(retained)
                        else end_hold
                    )
                    fade_seconds = min(0.32, until_next * 0.8)
                    since_change = (current_hour - hours[selected]) / hours[-1] * travel_seconds
                    blend = min(1, since_change / fade_seconds)
                    blend = blend * blend * (3 - 2 * blend)
                    current_image = Image.blend(packing_images[previous], current_image, blend)
                packing_image.set_data(current_image)
                line.set_data(
                    [*hours[: index + 1], current_hour],
                    [*best_scores[: index + 1], best_scores[index]],
                )
                cursor.set_xdata([current_hour, current_hour])
                for marker in (halo, dot):
                    marker.set_data([current_hour], [best_scores[index]])
                score_label.set_text(f"{best_scores[index]:.6f}")
                gain_label.set_text(f"+{(best_scores[index] / baseline - 1) * 100:.1f}%")
                clock_label.set_text(f"{int(elapsed) // 3600}h {(int(elapsed) // 60) % 60:02d}m")
                fig.canvas.draw()
                rgb = Image.fromarray(np.asarray(fig.canvas.buffer_rgba())).convert("RGB")
                encoder.stdin.write(rgb.tobytes())
                if frame % 200 == 0:
                    print(f"Rendered {frame}/{frame_count} frames", flush=True)
            encoder.stdin.close()
            if encoder.wait() != 0:
                raise subprocess.CalledProcessError(encoder.returncode, command)
        subprocess.run(
            [
                "ffmpeg",
                "-v",
                "error",
                "-y",
                "-i",
                str(video),
                "-vf",
                "palettegen=stats_mode=diff",
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
                str(output),
            ],
            check=True,
        )
    plt.close(fig)
    print(
        json.dumps(
            {
                "file": str(output),
                "frames": frame_count,
                "seconds": frame_count / fps,
                "best_candidate": attempts[kept[-1]].name,
                "best_score": best_scores[-1],
                "model_hours": float(hours[-1]),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    render(args.run, args.output)
