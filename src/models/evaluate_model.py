from __future__ import annotations
import json
import pandas as pd
import joblib

def load_metrics(path: str = "data/processed/metrics.json") -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def predict_sample(rows: pd.DataFrame, model_path: str = "data/processed/model.pkl") -> pd.Series:
    model = joblib.load(model_path)
    return pd.Series(model.predict(rows), index=rows.index)
