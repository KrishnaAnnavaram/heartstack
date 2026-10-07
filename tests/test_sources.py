import numpy as np
import pandas as pd
import pytest

from heartstack.sources import FEATURES, DataValidationError, harmonise, profile


def _hospital_row(**kw):
    row = dict(patientid=1, age=50, gender=1, chestpain=3, restingBP=130, serumcholestrol=220, fastingbloodsugar=0,
               restingrelectro=0, maxheartrate=150, exerciseangia=0, oldpeak=1.0, slope=2, noofmajorvessels=0, target=1)
    row.update(kw)
    return row


def test_hospital_codes_map_to_shared_labels():
    df, rep = harmonise(pd.DataFrame([_hospital_row()]), "hospital_india")
    r = df.iloc[0]
    assert r["chest_pain"] == "asymptomatic" and r["st_slope"] == "flat" and r["resting_ecg"] == "normal"
    assert r["sex_male"] == 1 and r["source"] == "hospital_india" and list(df.columns) == FEATURES + ["target", "source"]


def test_slope_code_zero_becomes_missing_and_the_row_stays():
    df, rep = harmonise(pd.DataFrame([_hospital_row(slope=0), _hospital_row()]), "hospital_india")
    assert len(df) == 2 and pd.isna(df.loc[0, "st_slope"]) and rep.not_recorded_codes == {"st_slope": 1}


def test_unknown_code_fails_loudly():
    with pytest.raises(DataValidationError, match="not in the code book"):
        harmonise(pd.DataFrame([_hospital_row(chestpain=7)]), "hospital_india")


def test_zero_cholesterol_and_bp_are_missing_not_values():
    df, rep = harmonise(pd.DataFrame([_hospital_row(serumcholestrol=0), _hospital_row(restingBP=0)]), "hospital_india")
    assert pd.isna(df.loc[0, "cholesterol"]) and pd.isna(df.loc[1, "resting_bp"])
    assert rep.zero_as_missing == {"cholesterol": 1, "resting_bp": 1}


def test_out_of_range_values_are_counted():
    df, rep = harmonise(pd.DataFrame([_hospital_row(maxheartrate=400)]), "hospital_india")
    assert pd.isna(df.loc[0, "max_hr"]) and rep.out_of_range == {"max_hr": 1}


def test_kaggle_and_cleveland_layouts():
    kag = pd.DataFrame([dict(Age=40, Sex="F", ChestPainType="ATA", RestingBP=140, Cholesterol=289, FastingBS=0,
                             RestingECG="LVH", MaxHR=172, ExerciseAngina="Y", Oldpeak=-0.5, ST_Slope="Down",
                             HeartDisease=0)])
    k, _ = harmonise(kag, "kaggle_compilation")
    assert k.loc[0, "chest_pain"] == "atypical_angina" and k.loc[0, "exercise_angina"] == 1 and k.loc[0, "sex_male"] == 0
    cle = pd.DataFrame([dict(age=63, sex=1, cp=1, trestbps=145, chol=233, fbs=1, restecg=2, thalach=150, exang=0,
                             oldpeak=2.3, slope=3, ca=0.0, thal=6.0, num=3)])
    c, _ = harmonise(cle, "uci_cleveland")
    assert c.loc[0, "target"] == 1 and c.loc[0, "chest_pain"] == "typical_angina" and c.loc[0, "st_slope"] == "down"


def test_missing_columns_and_bad_target():
    with pytest.raises(DataValidationError, match="missing columns"):
        harmonise(pd.DataFrame([{"age": 1}]), "uci_cleveland")
    with pytest.raises(DataValidationError, match="target"):
        harmonise(pd.DataFrame([_hospital_row(target=2)]), "hospital_india")


def test_build_removes_cross_source_duplicates_and_keeps_source(dataset):
    df, rep = dataset
    assert rep.duplicates_removed == 70 and rep.removed_by_source == {"kaggle_compilation": 70}
    assert set(df["source"]) == {"hospital_india", "kaggle_compilation", "uci_cleveland"}
    assert (df["source"] == "uci_cleveland").sum() == 70
    assert not df.duplicated(subset=FEATURES + ["target"]).any()
    prof = profile(df)
    assert prof["kaggle_compilation"]["missing"]["cholesterol"] > 0
    assert np.isclose(sum(p["rows"] for p in prof.values()), len(df))


def test_missing_file_is_reported(tmp_path):
    from heartstack.sources import build_dataset

    with pytest.raises(FileNotFoundError):
        build_dataset(tmp_path)
