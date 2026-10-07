"""The experiment: split once, select a model with nested CV on the training set, then test it once.

1. Stratified split by source and target. The test set is not used again until step 5.
2. Nested CV on the training set: the inner GridSearchCV tunes (AUROC), the outer folds score the model.
3. Select the model with the best mean outer AUROC.
4. Tune the selected model on the full training set. Choose the threshold on out-of-fold training predictions.
5. Score the test set one time: discrimination, operating point, calibration, decision curve, intervals, subgroups.
6. Leave-one-source-out: train on two sources, test on the third (external validation).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.inspection import permutation_importance
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_predict, train_test_split

from .config import Settings
from .metrics import bootstrap, calibration, discrimination, net_benefit, operating_point, threshold_for_sensitivity
from .models import GRIDS, make_pipeline
from .sources import FEATURES

log = logging.getLogger(__name__)
DECISION_THRESHOLDS = np.round(np.arange(0.05, 0.55, 0.05), 2)


def _strata(df: pd.DataFrame) -> pd.Series:
    return df["source"].astype(str) + "_" + df["target"].astype(str)


def split(df: pd.DataFrame, test_size: float, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    train, test = train_test_split(df, test_size=test_size, random_state=seed, stratify=_strata(df))
    return train.reset_index(drop=True), test.reset_index(drop=True)


def _xy(df: pd.DataFrame, include_source: bool):
    cols = FEATURES + (["source"] if include_source else [])
    return df[cols], df["target"].to_numpy()


def tuned(name: str, s: Settings, inner_seed: int) -> GridSearchCV:
    return GridSearchCV(make_pipeline(name, s.seed, s.include_source_feature, s.missing_indicators), GRIDS.get(name, {}), scoring="roc_auc",
                        cv=StratifiedKFold(s.inner_folds, shuffle=True, random_state=inner_seed), n_jobs=s.n_jobs,
                        refit=True)


def nested_cv(train: pd.DataFrame, models: list[str], s: Settings) -> dict:
    X, y = _xy(train, s.include_source_feature)
    outer = StratifiedKFold(s.outer_folds, shuffle=True, random_state=s.seed)
    out = {}
    for name in models:
        scores = []
        for k, (tr, va) in enumerate(outer.split(X, _strata(train))):
            search = tuned(name, s, s.seed + k + 1).fit(X.iloc[tr], y[tr])
            p = search.predict_proba(X.iloc[va])[:, 1]
            scores.append(discrimination(y[va], p)["auroc"])
        out[name] = {"outer_auroc": [round(v, 4) for v in scores], "mean": round(float(np.mean(scores)), 4),
                     "sd": round(float(np.std(scores)), 4)}
        log.info("nested CV %s: %.3f", name, out[name]["mean"])
    return out


def fit_final(name: str, train: pd.DataFrame, s: Settings):
    """Tune on the full training set, then choose the threshold on out-of-fold predictions."""
    X, y = _xy(train, s.include_source_feature)
    search = tuned(name, s, s.seed).fit(X, y)
    best = search.best_estimator_
    oof = cross_val_predict(clone(best), X, y, cv=StratifiedKFold(s.inner_folds, shuffle=True, random_state=s.seed),
                            method="predict_proba")[:, 1]
    threshold = threshold_for_sensitivity(y, oof, s.target_sensitivity)
    return best, search.best_params_, threshold


def score(model, df: pd.DataFrame, threshold: float, s: Settings, *, with_ci: bool = True) -> dict:
    X, y = _xy(df, s.include_source_feature)
    p = model.predict_proba(X)[:, 1]
    res = {"n": int(len(y)), "prevalence": round(float(y.mean()), 4), **discrimination(y, p),
           "operating_point": operating_point(y, p, threshold), "calibration": calibration(y, p)}
    if with_ci and len(np.unique(y)) == 2:
        res["ci95"] = bootstrap(y, p, threshold=threshold, n_boot=s.n_bootstrap, seed=s.seed)
        res["decision_curve"] = net_benefit(y, p, DECISION_THRESHOLDS)
    return res


def subgroups(model, df: pd.DataFrame, threshold: float, s: Settings) -> dict:
    groups = {
        "sex=male": df["sex_male"] == 1, "sex=female": df["sex_male"] == 0,
        "age<55": df["age"] < 55, "age>=55": df["age"] >= 55,
    }
    for src in sorted(df["source"].unique()):
        groups[f"source={src}"] = df["source"] == src
    out = {}
    for name, mask in groups.items():
        g = df[mask.fillna(False)]
        if len(g) >= 20 and g["target"].nunique() == 2:
            r = score(model, g, threshold, s, with_ci=False)
            out[name] = {"n": r["n"], "prevalence": r["prevalence"], "auroc": r["auroc"],
                         "sensitivity": r["operating_point"]["sensitivity"],
                         "specificity": r["operating_point"]["specificity"]}
        else:
            out[name] = {"n": int(len(g)), "note": "too few rows or one class only"}
    return out


def leave_one_source_out(df: pd.DataFrame, name: str, s: Settings) -> dict:
    out = {}
    for held in sorted(df["source"].unique()):
        train, test = df[df["source"] != held], df[df["source"] == held]
        if test["target"].nunique() < 2 or train["source"].nunique() < 1:
            continue
        model, _, threshold = fit_final(name, train.reset_index(drop=True), s)
        r = score(model, test.reset_index(drop=True), threshold, s, with_ci=False)
        out[held] = {"n": r["n"], "prevalence": r["prevalence"], "auroc": r["auroc"], "auprc": r["auprc"],
                     "brier": r["brier"], "ece": r["calibration"]["ece"],
                     "sensitivity": r["operating_point"]["sensitivity"],
                     "specificity": r["operating_point"]["specificity"]}
    return out


def importance(model, df: pd.DataFrame, s: Settings, n_repeats: int = 10) -> dict:
    X, y = _xy(df, s.include_source_feature)
    r = permutation_importance(model, X, y, scoring="roc_auc", n_repeats=n_repeats, random_state=s.seed)
    order = np.argsort(-r.importances_mean)
    return {X.columns[i]: {"mean": round(float(r.importances_mean[i]), 4), "sd": round(float(r.importances_std[i]), 4)}
            for i in order}


@dataclass
class ExperimentResult:
    selection: dict
    selected: str
    best_params: dict
    threshold: float
    test: dict
    subgroups: dict
    loso: dict
    loso_baseline: dict
    importance: dict
    settings: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items()}


def run_experiment(df: pd.DataFrame, models: list[str], s: Settings, *, baseline: str = "logreg"):
    train, test = split(df, s.test_size, s.seed)
    selection = nested_cv(train, models, s)
    selected = max(selection, key=lambda m: selection[m]["mean"])
    model, params, threshold = fit_final(selected, train, s)
    test_res = score(model, test, threshold, s)  # the only use of the test set
    loso = leave_one_source_out(df, selected, s)
    result = ExperimentResult(
        selection=selection, selected=selected, best_params={k: str(v) for k, v in params.items()},
        threshold=round(threshold, 4), test=test_res, subgroups=subgroups(model, test, threshold, s),
        loso=loso,
        loso_baseline=leave_one_source_out(df, baseline, s) if baseline != selected else loso,
        importance=importance(model, test, s),
        settings={"seed": s.seed, "test_size": s.test_size, "outer_folds": s.outer_folds,
                  "inner_folds": s.inner_folds, "target_sensitivity": s.target_sensitivity,
                  "n_bootstrap": s.n_bootstrap, "include_source_feature": s.include_source_feature,
                  "missing_indicators": s.missing_indicators,
                  "train_rows": int(len(train)), "test_rows": int(len(test))},
    )
    return result, model
