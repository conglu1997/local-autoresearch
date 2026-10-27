# Machine learning: bike rental demand

Train an ML regression model to predict hourly bike rentals from calendar and observed weather. Commuter peaks, weekends and seasons give Flint room to experiment with features and model settings. Weather for the hour being predicted is supplied.

The public UCI Bike Sharing dataset has 17,379 hourly rows. The solver receives fourteen features and training rental counts. The original twelve calendar/weather features are supplemented by a weekend flag (Saturday or Sunday) and elapsed days since January 1, 2011. `casual` and `registered` are excluded because they add up to the target. The evaluator selects by RMSLE (root mean squared logarithmic error), while also reporting signed total-count bias and mean log error. Negative diagnostics indicate underprediction.

| Split | Period | Rows |
| --- | --- | ---: |
| Training | 2011 | 8,645 |
| Validation | January–June 2012 | 4,358 |
| Final test | July–December 2012 | 4,376 |

Validation and test labels stay outside the candidate sandbox. After 100 attempts, both the selected program and frozen baseline are refit unchanged on training plus validation data and tested once on identical rows and budgets. The same limits apply: 15 CPU seconds and 25 wall seconds. The final test cannot be retried, even if an evaluation fails. If the baseline wins the search, its final evaluation is reused for the winner.

The baseline fits a linear time/weather/calendar model with smooth annual features, then 100 histogram-boosted trees to residual log counts. The explicit time component can extrapolate; a constant year flag in the training set cannot learn an annual effect. Flint can replace the full model with temporal trends, calibration, additive/count models, residual models, trees or small ensembles. The brief recommends chronological internal validation. All runs use seed 0.

```sh
uv run python research.py --task bike --run runs/bike-demo --experiments 100 --max-tokens 16384
uv run python -m tools.final_test runs/bike-demo
uv run python report.py runs/bike-demo
```

This chronological split differs from the Kaggle competition. The final test checks for overfitting to the validation period.

## Data

Fanaee-T, H. (2013). *Bike Sharing*. UCI Machine Learning Repository. [doi:10.24432/C5W894](https://doi.org/10.24432/C5W894). Licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). `hour.csv` is copied unchanged from the [UCI download](https://archive.ics.uci.edu/static/public/275/bike+sharing+dataset.zip).
