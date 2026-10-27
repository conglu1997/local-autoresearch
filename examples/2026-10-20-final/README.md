# Circle packing

<p align="center">
  <img src="../../docs/images/circle.png" width="760" alt="Circle packing overview">
</p>

Fit 26 circles inside a unit square and maximize the sum of their radii. Across **128 attempts**, the mean score rose from **2.166667 to 2.629408**, a **21.4% increase** over the grid baseline. Sixteen changes were retained; 27 attempts failed.

The plot highlights [staggered rows](001/solver.py), [joint optimization of positions and radii](022/solver.py), and [sequential random starting layouts](109/solver.py). The [annotations](annotations.json) explain what changed and how it scored.

## Animation

<p align="center">
  <img src="demo.gif" width="880" alt="Research trajectory">
</p>

The 24-second time-lapse compresses roughly 7.4 hours of model response time, excluding evaluation and pauses. Both panels follow the first recorded trial of each retained candidate. Short dissolves connect packings; the score curve preserves the measured steps.

## Evaluation

This run used seeds 0, 1 and 2. All three outputs had to pass, and the mean score had to improve by more than 1e-9 to keep a candidate. New runs use one seed. Of the 101 valid attempts, 97 were distinct implementations and four were repeats.

Each candidate received 15 CPU seconds per trial, including imports, with a 25-second wall limit. The evaluator checks the circle count, finite coordinates, positive radii, square boundaries and every pair of circles, allowing at most 1e-9 geometric error.

A separate rational check tests exact feasibility of the saved decimal coordinates. The best retained candidate, 118, passes numerical tolerance but fails exact feasibility on all three trials. Candidate 070 is the highest-scoring candidate that passes the exact check on every search trial, with mean **2.622324086**.

An additional check of candidate 118 on seeds 3–7 passed numerical tolerance with mean **2.624840812** and a largest violation of about 5.02e-12. These are extra seeds for the same instance, not other circle counts.

Candidate 118's gain came from a changed starting layout, not the rotation its response described. The [annotations](annotations.json) separate measured changes from the model's claims.

## Inspect the run

Open [report.html](report.html) locally for all prompts, responses, timed streams, code changes, scores and annotations. Every attempt and failure is preserved; [export.json](export.json) records the evidence hashes.

Verify without executing candidates:

```sh
uv run python -m tools.verify_run examples/2026-10-20-final
```

Rebuild the animation from saved measurements (`ffmpeg` required):

```sh
uv run python -m tools.render_progress_gif examples/2026-10-20-final examples/2026-10-20-final/demo.gif
```
