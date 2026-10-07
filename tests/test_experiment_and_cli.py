import json
import tomllib
from pathlib import Path

import pandas as pd
import pytest

from heartstack import experiment
from heartstack.cli import main
from heartstack.config import ConfigError, Settings
from heartstack.report import model_card


def test_experiment_touches_the_test_set_once_and_selects_by_nested_cv(dataset, fast_settings, monkeypatch):
    df, _ = dataset
    calls = []
    real_score = experiment.score

    def spy(model, data, threshold, s, *, with_ci=True):
        calls.append((len(data), with_ci))
        return real_score(model, data, threshold, s, with_ci=with_ci)

    monkeypatch.setattr(experiment, "score", spy)
    result, _ = experiment.run_experiment(df, ["logreg", "random_forest"], fast_settings)
    sel = result.selection
    assert result.selected == max(sel, key=lambda m: sel[m]["mean"])
    assert len(sel["logreg"]["outer_auroc"]) == fast_settings.outer_folds
    assert sum(1 for _, ci in calls if ci) == 1  # one full evaluation of the held-out test set
    assert set(result.loso) == {"hospital_india", "kaggle_compilation", "uci_cleveland"}
    assert result.loso_baseline and "ci95" in result.test and result.test["decision_curve"]
    assert result.subgroups["sex=male"]["n"] > 0 and result.importance


def test_runs_are_reproducible(dataset, fast_settings):
    df, _ = dataset
    a, _ = experiment.run_experiment(df, ["logreg"], fast_settings)
    b, _ = experiment.run_experiment(df, ["logreg"], fast_settings)
    assert a.test["auroc"] == b.test["auroc"] and a.threshold == b.threshold and a.selection == b.selection


def test_model_card_states_limits(dataset, fast_settings):
    df, rep = dataset
    result, _ = experiment.run_experiment(df, ["logreg"], fast_settings)
    card = model_card(result.to_dict(), rep.to_dict(), synthetic=True)
    assert "Not a diagnostic device" in card and "Retrospective" in card and "Leave-one-source-out" in card
    assert "Synthetic data: yes" in card


def test_settings_from_env():
    s = Settings.from_env({"HEARTSTACK_SEED": "7", "HEARTSTACK_MISSING_INDICATORS": "false"})
    assert s.seed == 7 and s.missing_indicators is False and s.include_source_feature is False
    with pytest.raises(ConfigError):
        Settings.from_env({"HEARTSTACK_TEST_SIZE": "0.9"})
    with pytest.raises(ConfigError):
        Settings.from_env({"HEARTSTACK_INCLUDE_SOURCE": "maybe"})


def test_cli_end_to_end(tmp_path, capsys):
    raw = tmp_path / "raw"
    assert main(["synth", "--out-dir", str(raw), "--seed", "5"]) == 0
    capsys.readouterr()
    assert main(["build-data", "--data-dir", str(raw), "--out", str(tmp_path / "b")]) == 0
    build = json.loads(capsys.readouterr().out)
    assert build["build"]["duplicates_removed"] == 150
    out = tmp_path / "run"
    assert main(["train", "--data-dir", str(raw), "--out", str(out), "--models", "logreg", "--fast"]) == 0
    assert (out / "model_card.md").exists() and (out / "report.json").exists()
    harmonised = pd.read_csv(tmp_path / "b" / "harmonised.csv").head(5)
    harmonised.to_csv(tmp_path / "in.csv", index=False)
    assert main(["predict", "--model", str(out / "model.pkl"), "--input", str(tmp_path / "in.csv"),
                 "--output", str(tmp_path / "pred.csv")]) == 0
    pred = pd.read_csv(tmp_path / "pred.csv")
    assert pred["risk"].between(0, 1).all() and set(pred["flag"]) <= {0, 1}
    assert main(["build-data", "--data-dir", str(tmp_path / "none")]) == 2


def test_core_dependencies_are_light():
    meta = tomllib.loads((Path(__file__).parents[1] / "pyproject.toml").read_text(encoding="utf-8"))
    core = " ".join(meta["project"]["dependencies"])
    for heavy in ("xgboost", "lightgbm", "shap", "torch", "tensorflow"):
        assert heavy not in core
