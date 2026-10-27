"""Predict hourly bike rentals from calendar and observed weather features."""

import csv
import math
from datetime import date
from pathlib import Path

DATA = Path(__file__).with_name("hour.csv")

BASE_FEATURES = [
    "season",
    "yr",
    "mnth",
    "hr",
    "holiday",
    "weekday",
    "workingday",
    "weathersit",
    "temp",
    "atemp",
    "hum",
    "windspeed",
]
with DATA.open() as stream:
    _rows = list(csv.DictReader(stream))
TRAIN = [row for row in _rows if row["dteday"] < "2012-01-01"]
VALIDATION = [row for row in _rows if "2012-01-01" <= row["dteday"] < "2012-07-01"]
TEST = [row for row in _rows if row["dteday"] >= "2012-07-01"]

FEATURES = [*BASE_FEATURES, "weekend", "elapsed_days"]

FINAL_PROTOCOL = {
    "version": "temporal-calibration-2",
    "selection": "Best validation RMSLE after 100 attempts, including failures.",
    "refit": "Refit the selected program and frozen baseline separately on 2011 through June 2012.",
    "test": "July through December 2012, each evaluated once after selection.",
    "limits": "Same 15 CPU seconds, 25 wall seconds and seed 0 as validation.",
    "comparison": "Baseline and selected program use identical training data, test rows and budgets.",
}


def features(row):
    return [
        *[float(row[key]) for key in BASE_FEATURES],
        float(int(row["weekday"]) in (0, 6)),
        float((date.fromisoformat(row["dteday"]) - date(2011, 1, 1)).days),
    ]


def problem(train, predict):
    return {
        "features": FEATURES,
        "train_x": [features(row) for row in train],
        "train_y": [int(row["cnt"]) for row in train],
        "predict_x": [features(row) for row in predict],
    }


PROBLEM = problem(TRAIN, VALIDATION)


def final_problem():
    return problem(TRAIN + VALIDATION, TEST)


def score(predictions, rows):
    if not isinstance(predictions, list) or len(predictions) != len(rows):
        raise ValueError(f"Return {len(rows)} predictions in the supplied order")
    if any(
        type(value) not in (int, float) or not math.isfinite(value) or value < 0
        for value in predictions
    ):
        raise ValueError("Predictions must be finite, nonnegative numbers")
    rmsle = math.sqrt(
        math.fsum(
            (math.log1p(value) - math.log1p(int(row["cnt"]))) ** 2
            for value, row in zip(predictions, rows)
        )
        / len(rows)
    )
    observed_total = sum(int(row["cnt"]) for row in rows)
    return {
        "solution": predictions,
        "rmsle": rmsle,
        "score": -rmsle,
        "total_bias_percent": 100 * (math.fsum(p / observed_total for p in predictions) - 1),
        "mean_log_error": math.fsum(
            math.log1p(p) - math.log1p(int(row["cnt"]))
            for p, row in zip(predictions, rows, strict=True)
        )
        / len(rows),
    }


def check(predictions):
    return score(predictions, VALIDATION)


def check_final(predictions):
    return score(predictions, TEST)


def final_protocol(budget_seconds):
    if budget_seconds is None:
        return FINAL_PROTOCOL
    return {
        **FINAL_PROTOCOL,
        "version": "timed-search-1",
        "selection": "Best validation RMSLE at the end of the frozen time budget, including failures.",
        "budget_seconds": budget_seconds,
    }
