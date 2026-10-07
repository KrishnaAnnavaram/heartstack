"""One loader and one code book for each source, a validation step and the harmonised table.

Harmonised columns (``FEATURES`` + ``target`` + ``source``):

- numeric: ``age``, ``resting_bp``, ``cholesterol``, ``max_hr``, ``oldpeak``
- binary: ``sex_male``, ``fasting_bs``, ``exercise_angina``
- nominal: ``chest_pain``, ``resting_ecg``, ``st_slope``

Rules:
- A value of 0 for ``cholesterol`` or ``resting_bp`` means "not measured". It becomes missing.
- A code that the code book does not know is an error (no silent row loss). A code that the code book
  maps to ``None`` (documented as "not recorded") becomes missing, and the row stays.
- A numeric value outside its plausible range becomes missing and is counted.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

NUMERIC = ["age", "resting_bp", "cholesterol", "max_hr", "oldpeak"]
BINARY = ["sex_male", "fasting_bs", "exercise_angina"]
NOMINAL = ["chest_pain", "resting_ecg", "st_slope"]
FEATURES = NUMERIC + BINARY + NOMINAL
SOURCES = ("hospital_india", "kaggle_compilation", "uci_cleveland")
# When the same patient row is in two sources, keep the copy from the source that comes first here.
DEDUP_PRIORITY = ("uci_cleveland", "hospital_india", "kaggle_compilation")

RANGES = {"age": (18, 100), "resting_bp": (60, 250), "cholesterol": (80, 700), "max_hr": (50, 230), "oldpeak": (-5, 10)}
ZERO_IS_MISSING = ("resting_bp", "cholesterol")

CHEST_PAIN = ("typical_angina", "atypical_angina", "non_anginal", "asymptomatic")
ECG = ("normal", "st_t_abnormality", "lv_hypertrophy")
SLOPE = ("up", "flat", "down")

CODEBOOKS: dict[str, dict[str, dict]] = {
    "hospital_india": {
        "chest_pain": {0: "typical_angina", 1: "atypical_angina", 2: "non_anginal", 3: "asymptomatic"},
        "resting_ecg": {0: "normal", 1: "st_t_abnormality", 2: "lv_hypertrophy"},
        "st_slope": {0: None, 1: "up", 2: "flat", 3: "down"},  # 0 = not recorded in this cohort
        "sex_male": {1: 1, 0: 0},
    },
    "kaggle_compilation": {
        "chest_pain": {"TA": "typical_angina", "ATA": "atypical_angina", "NAP": "non_anginal", "ASY": "asymptomatic"},
        "resting_ecg": {"Normal": "normal", "ST": "st_t_abnormality", "LVH": "lv_hypertrophy"},
        "st_slope": {"Up": "up", "Flat": "flat", "Down": "down"},
        "sex_male": {"M": 1, "F": 0},
        "exercise_angina": {"Y": 1, "N": 0},
    },
    "uci_cleveland": {
        "chest_pain": {1: "typical_angina", 2: "atypical_angina", 3: "non_anginal", 4: "asymptomatic"},
        "resting_ecg": {0: "normal", 1: "st_t_abnormality", 2: "lv_hypertrophy"},
        "st_slope": {1: "up", 2: "flat", 3: "down"},
        "sex_male": {1: 1, 0: 0},
    },
}

RENAMES = {
    "hospital_india": {"age": "age", "gender": "sex_male", "chestpain": "chest_pain", "restingBP": "resting_bp",
                       "serumcholestrol": "cholesterol", "fastingbloodsugar": "fasting_bs",
                       "restingrelectro": "resting_ecg", "maxheartrate": "max_hr", "exerciseangia": "exercise_angina",
                       "oldpeak": "oldpeak", "slope": "st_slope", "target": "target"},
    "kaggle_compilation": {"Age": "age", "Sex": "sex_male", "ChestPainType": "chest_pain", "RestingBP": "resting_bp",
                           "Cholesterol": "cholesterol", "FastingBS": "fasting_bs", "RestingECG": "resting_ecg",
                           "MaxHR": "max_hr", "ExerciseAngina": "exercise_angina", "Oldpeak": "oldpeak",
                           "ST_Slope": "st_slope", "HeartDisease": "target"},
    "uci_cleveland": {"age": "age", "sex": "sex_male", "cp": "chest_pain", "trestbps": "resting_bp", "chol": "cholesterol",
                      "fbs": "fasting_bs", "restecg": "resting_ecg", "thalach": "max_hr", "exang": "exercise_angina",
                      "oldpeak": "oldpeak", "slope": "st_slope", "num": "target"},
}
FILES = {"hospital_india": "Cardiovascular_Disease_Dataset.csv", "kaggle_compilation": "heart.csv",
         "uci_cleveland": "heart_disease.csv"}


class DataValidationError(ValueError):
    pass


@dataclass
class SourceReport:
    source: str
    rows_read: int = 0
    zero_as_missing: dict[str, int] = field(default_factory=dict)
    not_recorded_codes: dict[str, int] = field(default_factory=dict)
    out_of_range: dict[str, int] = field(default_factory=dict)


@dataclass
class BuildReport:
    sources: list[SourceReport]
    rows_before_dedup: int
    duplicates_removed: int
    rows: int
    removed_by_source: dict[str, int]

    def to_dict(self) -> dict:
        return {"sources": [s.__dict__ for s in self.sources], "rows_before_dedup": self.rows_before_dedup,
                "duplicates_removed": self.duplicates_removed, "rows": self.rows,
                "removed_by_source": self.removed_by_source}


def _map_codes(df: pd.DataFrame, source: str, rep: SourceReport) -> pd.DataFrame:
    for col, book in CODEBOOKS[source].items():
        values = df[col]
        unknown = sorted({str(v) for v in values.dropna().unique() if v not in book})
        if unknown:
            raise DataValidationError(f"{source}: column {col} has codes not in the code book: {unknown}")
        mapped = values.map(lambda v, b=book: b.get(v) if pd.notna(v) else None)
        not_recorded = int(values.notna().sum() - mapped.notna().sum())
        if not_recorded:
            rep.not_recorded_codes[col] = not_recorded
        df[col] = mapped.where(mapped.notna(), np.nan)  # None -> NaN, so the imputers see a missing value
    return df


def harmonise(raw: pd.DataFrame, source: str) -> tuple[pd.DataFrame, SourceReport]:
    if source not in RENAMES:
        raise DataValidationError(f"unknown source {source!r}")
    missing = set(RENAMES[source]) - set(raw.columns)
    if missing:
        raise DataValidationError(f"{source}: missing columns {sorted(missing)}")
    rep = SourceReport(source, rows_read=len(raw))
    df = raw[list(RENAMES[source])].rename(columns=RENAMES[source]).copy()
    df = _map_codes(df, source, rep)
    if source == "uci_cleveland":
        df["target"] = (pd.to_numeric(df["target"]) > 0).astype(int)
    for col in NUMERIC + ["fasting_bs", "exercise_angina", "sex_male", "target"]:
        df[col] = pd.to_numeric(df[col], errors="raise").astype(float)
    for col in ZERO_IS_MISSING:
        zeros = int((df[col] == 0).sum())
        if zeros:
            rep.zero_as_missing[col] = zeros
            df.loc[df[col] == 0, col] = np.nan
    for col, (lo, hi) in RANGES.items():
        bad = df[col].notna() & ~df[col].between(lo, hi)
        if bad.any():
            rep.out_of_range[col] = int(bad.sum())
            df.loc[bad, col] = np.nan
    if df["target"].isna().any() or not set(df["target"].unique()) <= {0.0, 1.0}:
        raise DataValidationError(f"{source}: target must be 0 or 1")
    df["target"] = df["target"].astype(int)
    df["source"] = source
    return df[FEATURES + ["target", "source"]], rep


def load_source(data_dir: Path, source: str) -> tuple[pd.DataFrame, SourceReport]:
    path = Path(data_dir) / FILES[source]
    if not path.exists():
        raise FileNotFoundError(f"{path} not found (see data/README.md, or run 'heartstack synth')")
    return harmonise(pd.read_csv(path), source)


def build_dataset(data_dir: Path, sources: tuple[str, ...] = SOURCES) -> tuple[pd.DataFrame, BuildReport]:
    """Load, validate and combine the sources. Remove rows that are in two sources (keep by priority)."""
    frames, reports = [], []
    for s in sources:
        df, rep = load_source(data_dir, s)
        frames.append(df)
        reports.append(rep)
    combined = pd.concat(frames, ignore_index=True)
    before = len(combined)
    combined["_prio"] = combined["source"].map({s: i for i, s in enumerate(DEDUP_PRIORITY)})
    combined = combined.sort_values("_prio", kind="stable")
    key = FEATURES + ["target"]
    dup = combined.duplicated(subset=key, keep="first")
    removed = combined.loc[dup, "source"].value_counts().to_dict()
    combined = combined.loc[~dup].drop(columns="_prio").sort_index().reset_index(drop=True)
    return combined, BuildReport(reports, before, int(dup.sum()), len(combined), {k: int(v) for k, v in removed.items()})


def profile(df: pd.DataFrame) -> dict:
    """Rows, prevalence and missing share by source."""
    out = {}
    for s, g in df.groupby("source"):
        out[s] = {"rows": int(len(g)), "prevalence": round(float(g["target"].mean()), 3),
                  "missing": {c: round(float(g[c].isna().mean()), 3) for c in FEATURES if g[c].isna().any()}}
    return out
