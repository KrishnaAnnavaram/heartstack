# Data for heartstack

The repository contains no data files. The demo and the tests use synthetic files that
`heartstack synth` or `heartstack demo` writes in the same layouts as the three public sources.

## 1. Sources

| Source name in heartstack | File | Rows | Origin | Licence |
|---|---|---|---|---|
| `hospital_india` | `Cardiovascular_Disease_Dataset.csv` | 1,000 | Mendeley Data, "Cardiovascular Disease Dataset" (hospital cohort, India), doi:10.17632/dzz48mvjht.1 | CC BY 4.0 |
| `kaggle_compilation` | `heart.csv` | 918 | Kaggle, "Heart Failure Prediction Dataset" (fedesoriano). A compilation of the Cleveland, Hungarian, Switzerland, Long Beach VA and Statlog sets | ODbL 1.0 |
| `uci_cleveland` | `heart_disease.csv` | 303 | UCI Machine Learning Repository, Heart Disease, id 45 (processed Cleveland file with a header row) | CC BY 4.0 |

URLs:

- https://data.mendeley.com/datasets/dzz48mvjht/1
- https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction
- https://archive.ics.uci.edu/dataset/45/heart+disease

Download the three files, rename them to the names in the table and put them in `data/raw/`
(or set `HEARTSTACK_DATA_DIR`). Cite each source as its licence asks.

## 2. Expected columns

| File | Columns |
|---|---|
| `Cardiovascular_Disease_Dataset.csv` | `patientid, age, gender, chestpain, restingBP, serumcholestrol, fastingbloodsugar, restingrelectro, maxheartrate, exerciseangia, oldpeak, slope, noofmajorvessels, target` |
| `heart.csv` | `Age, Sex, ChestPainType, RestingBP, Cholesterol, FastingBS, RestingECG, MaxHR, ExerciseAngina, Oldpeak, ST_Slope, HeartDisease` |
| `heart_disease.csv` | `age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal, num` |

`patientid`, `noofmajorvessels`, `ca` and `thal` are not used, because the other sources do not have them.

## 3. Known properties of the real files

`heartstack build-data` on the real files reports these values:

| Property | Value |
|---|---|
| `serumcholestrol = 0` in `hospital_india` | 53 rows, now missing |
| `Cholesterol = 0` in `kaggle_compilation` | 172 rows, now missing |
| `RestingBP = 0` in `kaggle_compilation` | 1 row, now missing |
| `slope = 0` in `hospital_india` | 180 rows, now missing (the rows stay) |
| Cleveland rows that are also in `heart.csv` | 303, removed from `kaggle_compilation` |
| Rows after de-duplication | 1,918 (1,000 + 615 + 303) |

## 4. Synthetic files

`heartstack synth --out-dir data/synthetic` writes the three files with invented patients: 400 hospital rows,
300 compilation rows plus a copy of the 150 Cleveland-like rows, and 150 Cleveland-like rows. The files have the
same problems as the real files: zero cholesterol, slope code 0 and the cross-source copies.
