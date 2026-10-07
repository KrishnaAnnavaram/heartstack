"""Preprocessing and the model zoo. Every model is one sklearn Pipeline, so all fitting happens inside a fold.

Preprocessing: median imputation with missing indicators and scaling for numeric columns,
most-frequent imputation for binary columns, one-hot encoding (with a 'missing' level) for nominal columns.
"""

from __future__ import annotations

from typing import Callable

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier, StackingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .sources import BINARY, NOMINAL, NUMERIC


def preprocessor(include_source: bool = False, missing_indicators: bool = True) -> ColumnTransformer:
    parts = [
        ("num", Pipeline([("impute", SimpleImputer(strategy="median", add_indicator=missing_indicators)),
                          ("scale", StandardScaler())]), NUMERIC),
        ("bin", SimpleImputer(strategy="most_frequent"), BINARY),
        ("nom", Pipeline([("impute", SimpleImputer(strategy="constant", fill_value="missing")),
                          ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), NOMINAL),
    ]
    if include_source:
        parts.append(("src", OneHotEncoder(handle_unknown="ignore", sparse_output=False), ["source"]))
    return ColumnTransformer(parts, remainder="drop", verbose_feature_names_out=False)


def _logreg(seed: int) -> LogisticRegression:
    return LogisticRegression(max_iter=2000, random_state=seed)


def _rf(seed: int) -> RandomForestClassifier:
    return RandomForestClassifier(n_estimators=300, min_samples_leaf=3, random_state=seed,
                                  n_jobs=1)


def _hgb(seed: int) -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, random_state=seed)


def _stack(seed: int) -> StackingClassifier:
    return StackingClassifier(
        estimators=[("lr", _logreg(seed)), ("rf", _rf(seed)), ("hgb", _hgb(seed))],
        final_estimator=LogisticRegression(max_iter=2000, random_state=seed),
        cv=5, stack_method="predict_proba", passthrough=False, n_jobs=1,
    )


def _xgb(seed: int):  # pragma: no cover - optional extra "boost"
    from xgboost import XGBClassifier

    return XGBClassifier(n_estimators=300, max_depth=3, learning_rate=0.05, subsample=0.9, random_state=seed,
                         eval_metric="logloss")


MODELS: dict[str, Callable[[int], object]] = {
    "logreg": _logreg,
    "random_forest": _rf,
    "hist_gbm": _hgb,
    "stack": _stack,
}
OPTIONAL_MODELS: dict[str, Callable[[int], object]] = {"xgboost": _xgb}

# Small grids for the inner search. Keys use the Pipeline step name "model".
GRIDS: dict[str, dict[str, list]] = {
    "logreg": {"model__C": [0.1, 1.0, 10.0]},
    "random_forest": {"model__max_depth": [None, 6], "model__min_samples_leaf": [1, 5]},
    "hist_gbm": {"model__max_depth": [3, None], "model__learning_rate": [0.05, 0.1]},
    "stack": {"model__final_estimator__C": [0.1, 1.0]},
    "xgboost": {"model__max_depth": [2, 4]},
}


def available_models() -> list[str]:
    names = list(MODELS)
    try:
        import xgboost  # noqa: F401

        names.append("xgboost")
    except ImportError:
        pass
    return names


def make_pipeline(name: str, seed: int, include_source: bool = False, missing_indicators: bool = True) -> Pipeline:
    factory = MODELS.get(name) or OPTIONAL_MODELS.get(name)
    if factory is None:
        raise ValueError(f"unknown model {name!r}. Known: {sorted(MODELS) + sorted(OPTIONAL_MODELS)}")
    return Pipeline([("prep", preprocessor(include_source, missing_indicators)), ("model", factory(seed))])
