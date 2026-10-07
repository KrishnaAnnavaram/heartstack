"""The ``heartstack`` command."""

from __future__ import annotations

import argparse
import json
import logging
import os
import pickle
import sys
import warnings
from dataclasses import replace
from pathlib import Path

import pandas as pd

from .config import ConfigError, Settings
from .experiment import run_experiment
from .models import available_models
from .report import write_report
from .sources import FEATURES, DataValidationError, build_dataset, profile
from .synthetic import write_synthetic


def _settings(args) -> Settings:
    s = Settings.from_env()
    if getattr(args, "data_dir", None):
        s = replace(s, data_dir=Path(args.data_dir))
    if getattr(args, "out", None):
        s = replace(s, work_dir=Path(args.out))
    if getattr(args, "fast", False):
        s = replace(s, outer_folds=3, inner_folds=3, n_bootstrap=200, n_jobs=-1)
    if getattr(args, "seed", None) is not None:
        s = replace(s, seed=args.seed)
    return s


def cmd_synth(args) -> int:
    print(f"wrote synthetic source files to {write_synthetic(Path(args.out_dir), seed=args.seed or 7)}")
    return 0


def cmd_build(args) -> int:
    s = _settings(args)
    df, rep = build_dataset(s.data_dir)
    s.work_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(s.work_dir / "harmonised.csv", index=False)
    print(json.dumps({"build": rep.to_dict(), "profile": profile(df)}, indent=2))
    return 0


def cmd_train(args) -> int:
    s = _settings(args)
    df, rep = build_dataset(s.data_dir)
    models = args.models.split(",") if args.models else available_models()
    result, model = run_experiment(df, models, s)
    synthetic = (s.data_dir / "SYNTHETIC").exists()
    j, m = write_report(s.work_dir, result.to_dict(), rep.to_dict(), synthetic)
    with (s.work_dir / "model.pkl").open("wb") as fh:
        pickle.dump({"model": model, "threshold": result.threshold, "features": FEATURES}, fh)
    t = result.test
    print(f"selected {result.selected} (nested CV AUROC {result.selection[result.selected]['mean']:.3f})")
    print(f"test AUROC {t['auroc']:.3f}, sensitivity {t['operating_point']['sensitivity']}, "
          f"specificity {t['operating_point']['specificity']}, ECE {t['calibration']['ece']}")
    for src, v in result.loso.items():
        print(f"leave-one-source-out {src}: AUROC {v['auroc']:.3f}")
    print(f"wrote {j} and {m}")
    return 0


def cmd_predict(args) -> int:
    with Path(args.model).open("rb") as fh:  # only load model files that you made yourself
        bundle = pickle.load(fh)
    df = pd.read_csv(args.input)
    missing = set(bundle["features"]) - set(df.columns)
    if missing:
        raise DataValidationError(f"input lacks harmonised columns {sorted(missing)}")
    p = bundle["model"].predict_proba(df[bundle["features"]])[:, 1]
    out = df.assign(risk=p.round(4), flag=(p >= bundle["threshold"]).astype(int))
    out.to_csv(args.output, index=False)
    print(f"wrote {len(out)} predictions to {args.output} (threshold {bundle['threshold']})")
    return 0


def cmd_demo(args) -> int:
    out = Path(args.out or "runs/demo")
    data = write_synthetic(out / "data")
    ns = argparse.Namespace(data_dir=str(data), out=str(out), fast=True, seed=None, models=args.models)
    return cmd_train(ns)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="heartstack", description="Leakage-safe multi-cohort CAD risk models.")
    p.add_argument("-v", "--verbose", action="store_true")
    sub = p.add_subparsers(dest="command", required=True)

    sp = sub.add_parser("synth", help="write synthetic files in the three source layouts")
    sp.add_argument("--out-dir", default="data/synthetic")
    sp.add_argument("--seed", type=int)
    sp.set_defaults(func=cmd_synth)

    sp = sub.add_parser("build-data", help="validate, harmonise and de-duplicate the sources")
    sp.add_argument("--data-dir")
    sp.add_argument("--out")
    sp.set_defaults(func=cmd_build)

    sp = sub.add_parser("train", help="nested CV selection, one test evaluation, leave-one-source-out, model card")
    sp.add_argument("--data-dir")
    sp.add_argument("--out")
    sp.add_argument("--models", help="comma list (default: all available)")
    sp.add_argument("--seed", type=int)
    sp.add_argument("--fast", action="store_true", help="3 outer folds, 200 bootstrap samples, all CPU cores")
    sp.set_defaults(func=cmd_train)

    sp = sub.add_parser("predict", help="score a CSV file with harmonised columns")
    sp.add_argument("--model", required=True)
    sp.add_argument("--input", required=True)
    sp.add_argument("--output", required=True)
    sp.set_defaults(func=cmd_predict)

    sp = sub.add_parser("demo", help="offline demo on synthetic data")
    sp.add_argument("--out")
    sp.add_argument("--models", default="logreg,random_forest,hist_gbm,stack")
    sp.set_defaults(func=cmd_demo)
    return p


def main(argv: list[str] | None = None) -> int:
    # joblib asks the OS for the physical core count. On some Windows versions that call fails noisily.
    warnings.filterwarnings("ignore", message="Could not find the number of physical cores")
    os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(os.cpu_count() or 1))
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING, format="%(levelname)s %(message)s")
    try:
        return args.func(args)
    except (ConfigError, DataValidationError, FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
