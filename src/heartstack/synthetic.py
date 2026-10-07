"""Synthetic raw files in the exact layouts of the three public sources, with their known problems.

- ``Cardiovascular_Disease_Dataset.csv``: integer codes, slope code 0 for some rows, some cholesterol 0.
- ``heart.csv``: text codes, many cholesterol 0 (with a higher disease rate) and a copy of every
  Cleveland-like row (so the de-duplication has work to do).
- ``heart_disease.csv``: codes 1-4 for chest pain, a 0-4 severity target, ``ca`` and ``thal`` columns.

The patients are invented. A logistic model with a different intercept for each source makes the labels.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

CP = np.array(["typical_angina", "atypical_angina", "non_anginal", "asymptomatic"])
ECG = np.array(["normal", "st_t_abnormality", "lv_hypertrophy"])
SLOPE = np.array(["up", "flat", "down"])


def _patients(rng: np.random.Generator, n: int, intercept: float) -> pd.DataFrame:
    age = np.clip(rng.normal(54, 9, n), 28, 80).round()
    male = (rng.random(n) < 0.72).astype(int)
    cp = rng.choice(4, n, p=[0.08, 0.18, 0.24, 0.50])
    bp = np.clip(rng.normal(132, 17, n), 94, 200).round()
    chol = np.clip(rng.normal(240, 50, n), 120, 560).round()
    fbs = (rng.random(n) < 0.2).astype(int)
    ecg = rng.choice(3, n, p=[0.6, 0.2, 0.2])
    max_hr = np.clip(210 - 0.8 * age + rng.normal(0, 18, n), 70, 202).round()
    angina = (rng.random(n) < 0.35).astype(int)
    oldpeak = np.clip(rng.gamma(1.2, 0.9, n), 0, 6.2).round(1)
    slope = rng.choice(3, n, p=[0.45, 0.45, 0.10])
    logit = (intercept + 0.045 * (age - 54) + 0.9 * male + 1.4 * (cp == 3) - 0.6 * (cp == 1) + 0.012 * (bp - 132)
             + 0.004 * (chol - 240) + 0.5 * fbs - 0.025 * (max_hr - 150) + 1.0 * angina + 0.55 * oldpeak
             + 1.1 * (slope == 1) + 0.4 * (ecg == 1))
    target = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
    return pd.DataFrame({"age": age, "male": male, "cp": cp, "bp": bp, "chol": chol, "fbs": fbs, "ecg": ecg,
                         "max_hr": max_hr, "angina": angina, "oldpeak": oldpeak, "slope": slope, "target": target})


def _to_kaggle(p: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({
        "Age": p["age"].astype(int), "Sex": np.where(p["male"] == 1, "M", "F"),
        "ChestPainType": np.array(["TA", "ATA", "NAP", "ASY"])[p["cp"]], "RestingBP": p["bp"].astype(int),
        "Cholesterol": p["chol"].astype(int), "FastingBS": p["fbs"], "RestingECG": np.array(["Normal", "ST", "LVH"])[p["ecg"]],
        "MaxHR": p["max_hr"].astype(int), "ExerciseAngina": np.where(p["angina"] == 1, "Y", "N"),
        "Oldpeak": p["oldpeak"], "ST_Slope": np.array(["Up", "Flat", "Down"])[p["slope"]], "HeartDisease": p["target"],
    })


def write_synthetic(out: Path, n_hospital: int = 400, n_kaggle: int = 300, n_cleveland: int = 150, seed: int = 7) -> Path:
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)

    h = _patients(rng, n_hospital, intercept=-2.2)
    hosp = pd.DataFrame({
        "patientid": rng.choice(np.arange(100000, 9999999), n_hospital, replace=False),
        "age": h["age"].astype(int), "gender": h["male"], "chestpain": h["cp"], "restingBP": h["bp"].astype(int),
        "serumcholestrol": h["chol"].astype(int), "fastingbloodsugar": h["fbs"], "restingrelectro": h["ecg"],
        "maxheartrate": h["max_hr"].astype(int), "exerciseangia": h["angina"], "oldpeak": h["oldpeak"],
        "slope": h["slope"] + 1, "noofmajorvessels": rng.integers(0, 4, n_hospital), "target": h["target"],
    })
    hosp.loc[rng.random(n_hospital) < 0.18, "slope"] = 0           # code 0 = not recorded
    hosp.loc[rng.random(n_hospital) < 0.05, "serumcholestrol"] = 0  # 0 = not measured
    hosp.to_csv(out / "Cardiovascular_Disease_Dataset.csv", index=False)

    c = _patients(rng, n_cleveland, intercept=-2.7)
    severity = np.where(c["target"] == 1, rng.integers(1, 5, n_cleveland), 0)
    cle = pd.DataFrame({
        "age": c["age"].astype(int), "sex": c["male"], "cp": c["cp"] + 1, "trestbps": c["bp"].astype(int),
        "chol": c["chol"].astype(int), "fbs": c["fbs"], "restecg": c["ecg"], "thalach": c["max_hr"].astype(int),
        "exang": c["angina"], "oldpeak": c["oldpeak"], "slope": c["slope"] + 1,
        "ca": np.where(rng.random(n_cleveland) < 0.02, np.nan, rng.integers(0, 4, n_cleveland).astype(float)),
        "thal": np.where(rng.random(n_cleveland) < 0.02, np.nan, rng.choice([3.0, 6.0, 7.0], n_cleveland)),
        "num": severity,
    })
    cle.to_csv(out / "heart_disease.csv", index=False)

    k = _patients(rng, n_kaggle, intercept=-2.3)
    missing = rng.random(n_kaggle) < 0.19
    k.loc[missing, "target"] = (rng.random(int(missing.sum())) < 0.85).astype(int)  # missing chol rows: high rate
    kag = _to_kaggle(k)
    kag.loc[missing, "Cholesterol"] = 0
    kag.loc[kag.index[0], "RestingBP"] = 0
    kag = pd.concat([kag, _to_kaggle(c)], ignore_index=True).sample(frac=1.0, random_state=seed)
    kag.to_csv(out / "heart.csv", index=False)
    (out / "SYNTHETIC").write_text("Synthetic files written by heartstack. Not patient data.\n", encoding="utf-8")
    return out
