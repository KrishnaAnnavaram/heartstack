import numpy as np
import pytest

from heartstack.experiment import split
from heartstack.metrics import bootstrap, calibration, discrimination, net_benefit, operating_point, threshold_for_sensitivity
from heartstack.models import available_models, make_pipeline
from heartstack.sources import FEATURES


def test_split_is_disjoint_and_stratified_by_source(dataset):
    df, _ = dataset
    train, test = split(df, 0.25, seed=0)
    assert len(train) + len(test) == len(df)
    for src in df["source"].unique():
        share = (test["source"] == src).mean()
        assert abs(share - (df["source"] == src).mean()) < 0.05


def test_preprocessing_is_fit_on_training_rows_only(dataset):
    df, _ = dataset
    train, test = split(df, 0.25, seed=0)
    pipe = make_pipeline("logreg", seed=0).fit(train[FEATURES], train["target"])
    imputer = pipe.named_steps["prep"].named_transformers_["num"].named_steps["impute"]
    assert np.allclose(imputer.statistics_, train[["age", "resting_bp", "cholesterol", "max_hr", "oldpeak"]].median())
    scaler = pipe.named_steps["prep"].named_transformers_["num"].named_steps["scale"]
    assert scaler.n_samples_seen_ == len(train)


def test_nominal_codes_are_one_hot_encoded(dataset):
    df, _ = dataset
    pipe = make_pipeline("logreg", seed=0).fit(df[FEATURES], df["target"])
    names = list(pipe.named_steps["prep"].get_feature_names_out())
    assert "chest_pain_asymptomatic" in names and "st_slope_missing" in names
    assert "missingindicator_cholesterol" in names
    assert "chest_pain" not in names


def test_missing_indicators_can_be_turned_off(dataset):
    df, _ = dataset
    pipe = make_pipeline("logreg", seed=0, missing_indicators=False).fit(df[FEATURES], df["target"])
    assert not any(n.startswith("missingindicator") for n in pipe.named_steps["prep"].get_feature_names_out())


def test_model_zoo(dataset):
    df, _ = dataset
    assert {"logreg", "random_forest", "hist_gbm", "stack"} <= set(available_models())
    p = make_pipeline("stack", seed=0).fit(df[FEATURES], df["target"]).predict_proba(df[FEATURES])[:, 1]
    assert p.shape == (len(df),) and 0 <= p.min() <= p.max() <= 1
    with pytest.raises(ValueError):
        make_pipeline("nope", seed=0)


def test_optional_xgboost_model(dataset):
    pytest.importorskip("xgboost")
    df, _ = dataset
    make_pipeline("xgboost", seed=0).fit(df[FEATURES], df["target"])


def test_threshold_reaches_target_sensitivity():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 400)
    p = np.clip(y * 0.3 + rng.random(400) * 0.7, 0, 1)
    t = threshold_for_sensitivity(y, p, 0.9)
    assert operating_point(y, p, t)["sensitivity"] >= 0.9


def test_operating_point_counts():
    op = operating_point([1, 1, 0, 0], [0.9, 0.2, 0.8, 0.1], 0.5)
    assert (op["tp"], op["fn"], op["fp"], op["tn"]) == (1, 1, 1, 1)
    assert op["sensitivity"] == 0.5 and op["ppv"] == 0.5


def test_calibration_and_discrimination():
    rng = np.random.default_rng(1)
    p = rng.random(5000)
    y = (rng.random(5000) < p).astype(int)
    cal = calibration(y, p)
    assert cal["ece"] < 0.03 and abs(cal["slope"] - 1) < 0.15
    d = discrimination(y, p)
    assert 0.7 < d["auroc"] < 0.9 and d["brier"] < 0.2
    assert np.isnan(discrimination([1, 1], [0.2, 0.3])["auroc"])


def test_net_benefit_formula():
    rows = net_benefit([1, 0, 1, 0], [0.9, 0.1, 0.6, 0.4], [0.5])
    assert rows[0]["model"] == pytest.approx(0.5) and rows[0]["treat_all"] == pytest.approx(0.0)


def test_bootstrap_intervals_contain_point_estimate():
    rng = np.random.default_rng(2)
    y = rng.integers(0, 2, 300)
    p = np.clip(y * 0.4 + rng.random(300) * 0.6, 0, 1)
    ci = bootstrap(y, p, threshold=0.5, n_boot=100, seed=0)
    auc = discrimination(y, p)["auroc"]
    assert ci["auroc"][0] <= auc <= ci["auroc"][1]
    assert bootstrap(y, p, threshold=0.5, n_boot=0, seed=0) == {}
