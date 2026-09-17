import json
from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

CONFIG = json.loads("{\"target\": \"loan_status\", \"features\": [\"cb_person_cred_hist_length\", \"credit_score\", \"loan_amnt\", \"loan_int_rate\", \"loan_intent\", \"loan_percent_income\", \"person_age\", \"person_education\", \"person_emp_exp\", \"person_gender\", \"person_home_ownership\", \"person_income\", \"previous_loan_defaults_on_file\"], \"numeric\": [\"cb_person_cred_hist_length\", \"credit_score\", \"loan_amnt\", \"loan_int_rate\", \"loan_percent_income\", \"person_age\", \"person_emp_exp\", \"person_income\"], \"classes\": [\"0\", \"1\"], \"metric\": \"accuracy\", \"task_type\": \"classification\", \"prediction_kind\": \"class_code\", \"estimator\": \"logistic_regression\", \"seed\": 42}")


def solve(data_dir=Path("/app/data"), output=Path("/app/submission.csv")):
    numeric = CONFIG["numeric"]
    categorical = [c for c in CONFIG["features"] if c not in numeric]
    dtypes = {"row_id": str, **{c: str for c in categorical}}
    train = pd.read_csv(data_dir / "train.csv", dtype=dtypes, keep_default_na=False, na_values=[""])
    test = pd.read_csv(data_dir / "test.csv", dtype=dtypes, keep_default_na=False, na_values=[""])
    preprocessing = ColumnTransformer([
        ("numeric", make_pipeline(SimpleImputer(strategy="median", keep_empty_features=True), StandardScaler()), numeric),
        ("categorical", make_pipeline(SimpleImputer(strategy="most_frequent", keep_empty_features=True), OneHotEncoder(handle_unknown="ignore")), categorical),
    ])
    estimator = (
        RandomForestRegressor(n_estimators=40, max_depth=12, random_state=CONFIG["seed"], n_jobs=1)
        if CONFIG["estimator"] == "random_forest_regressor"
        else LogisticRegression(max_iter=500, random_state=CONFIG["seed"])
    )
    model = make_pipeline(preprocessing, estimator)
    model.fit(train[CONFIG["features"]], train[CONFIG["target"]])
    predictions = model.predict(test[CONFIG["features"]])
    pd.DataFrame({"row_id": test["row_id"], "prediction": predictions}).to_csv(output, index=False, lineterminator="\n")


if __name__ == "__main__":
    solve()
