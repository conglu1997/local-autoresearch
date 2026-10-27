# Machine learning: bike rental demand

<p align="center">
  <img src="../../docs/images/bike.png" width="760" alt="Bike demand: 30.81% lower validation RMSLE">
</p>

Flint improved a program predicting hourly bike rentals from calendar and observed weather in the public UCI Bike Sharing dataset. Both the proposing model and candidate evaluations ran locally. [Task and dataset](../../tasks/bike).

Across **100 attempts**, validation RMSLE fell from **0.535904 to 0.370806**, a **30.81% reduction**. Seven improvements were retained. Attempt **34** was selected; the remaining 66 attempts did not improve it beyond the runner's 1e-9 threshold.

The search added [quadratic time](012/solver.py), [separated trend features from the residual trees](024/solver.py), [changed the trend model to Tweedie](030/solver.py), and [combined log and quadratic time](034/solver.py). The trend model is fitted to log rental counts. [Verified chart labels](discoveries.json).

## Final test

Training used 2011, and validation used January–June 2012. After the search, the frozen baseline and winner were separately refit unchanged on 2011 through June 2012 and evaluated once on July–December 2012, with the same seed and budgets.

| Measurement | Baseline RMSLE | Selected RMSLE | Reduction |
| --- | ---: | ---: | ---: |
| Validation | 0.535904 | 0.370806 | 30.81% |
| Held-out test after refitting | 0.395280 | 0.359408 | 9.08% |

The selected program underpredicts total rentals by **4.43% on validation** and **1.16% on the test period**. Test results did not influence selection; neither program was retried. [Measurements](notes.json) · [Baseline test](baseline-final-test.json) · [Selected test](final-test.json).

## Recorded run

There were **77 valid evaluations and 23 failed attempts**: 13 responses hit the output-token limit and 10 candidates failed evaluation. The valid programs have distinct syntax after removing comments and docstrings, though some produce identical scores. All prompts, responses, compressed raw streams, code, predictions and decisions are retained in the [report](report.html).

Model responses took **5.015 hours** and generated **1,128,751 tokens**. The plots use this recorded response time, including failed attempts; they exclude evaluation time. Candidates used seed 0, a 15 CPU-second limit and a 25 wall-second limit.

The overview compares actual and predicted daily totals across the full validation period on a fixed scale. Its small progress curve measures hourly validation RMSLE.

The README gallery shows the first **three model hours**, covering **46 completed attempts** and the selected program. The full report retains all 100 attempts over 5.015 model hours.

## Reproduce the figures

```sh
uv run python -m tools.verify_run examples/2026-10-21-bike
uv run python -m tools.render_gallery
```

## Data

Fanaee-T, H. (2013). *Bike Sharing*. UCI Machine Learning Repository. [doi:10.24432/C5W894](https://doi.org/10.24432/C5W894). Dataset licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). The original CSV is preserved in [the task folder](../../tasks/bike/hour.csv).
