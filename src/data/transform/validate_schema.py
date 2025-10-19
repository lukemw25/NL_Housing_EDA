from __future__ import annotations
import logging, os, yaml, pandas as pd
from src.data.utils.helpers import setup_logging

setup_logging()
log = logging.getLogger(__name__)

SCHEMA_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "../../../config/schema.yaml"
)
SCHEMA_PATH = os.path.normpath(SCHEMA_PATH)

with open(SCHEMA_PATH, "r", encoding="utf-8") as f:
    SCHEMA = yaml.safe_load(f)["columns"]

def validate_schema(df: pd.DataFrame) -> pd.DataFrame:
    for col, expected in SCHEMA.items():
        if col not in df.columns:
            raise ValueError(f"Missing column: {col}")
        if expected == "float":
            if not pd.api.types.is_float_dtype(df[col]):
                df[col] = pd.to_numeric(df[col], errors="coerce")
        elif expected == "int":
            if not pd.api.types.is_integer_dtype(df[col]):
                df[col] = pd.to_numeric(df[col], errors="coerce", downcast="integer")
        elif expected == "category":
            df[col] = df[col].astype("category")
        else:
            raise ValueError(f"Unknown expected type '{expected}' for column '{col}'")
    log.info("Schema validated on %d rows, %d columns", df.shape[0], df.shape[1])
    return df
