from __future__ import annotations
import pandas as pd
from src.data.utils.helpers import load_yaml

def build_features(df: pd.DataFrame, params_path: str = "config/params.yaml") -> pd.DataFrame:
    params = load_yaml(params_path)
    if params["features"]["build_price_per_m2"]:
        df["price_per_m2"] = df["Price"] / df["Living space size (m2)"]
    if params["features"]["build_age"]:
        # conservative: unknown years -> NaN
        df["age"] = pd.NA
        if "Build year" in df.columns:
            df["age"] = pd.to_datetime("today").year - df["Build year"]
    return df
