# src/models/train_model.py

import json, logging, joblib, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

from src.data.utils.helpers import load_yaml, setup_logging
from src.features.build_features import build_features

setup_logging()
log = logging.getLogger(__name__)

def train(processed_parquet: str, params_path: str = "config/params.yaml") -> dict:
    params = load_yaml(params_path)
    df = pd.read_parquet(processed_parquet)
    df = build_features(df, params_path)

    target = params["target"]
    df = df.dropna(subset=[target])  # remove rows with missing target
    y = df[target]
    X = df.drop(columns=[target])

    cat_cols = X.select_dtypes(include=["object", "category"]).columns
    num_cols = X.select_dtypes(exclude=["object", "category"]).columns

    preprocessor = ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]), num_cols),
        ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                          ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), cat_cols)
    ])

    if params["model"]["algo"] == "LinearRegression":
        model = LinearRegression()
    else:
        model = RandomForestRegressor(**params["model"].get("rf_params", {}))

    pipe = Pipeline([("pre", preprocessor), ("model", model)])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=params["model"]["test_size"], random_state=params["random_state"]
    )

    pipe.fit(X_train, y_train)
    preds = pipe.predict(X_test)

    mse = mean_squared_error(y_test, preds)
    metrics = {
        "r2": r2_score(y_test, preds),
        "mae": mean_absolute_error(y_test, preds),
        "rmse": mse ** 0.5,
    }

    joblib.dump(pipe, "data/processed/model.pkl")
    with open("data/processed/metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    log.info("Model trained. Metrics: %s", metrics)
    return metrics

if __name__ == "__main__":
    params = load_yaml("config/params.yaml")
    train(params["processed_parquet"])
