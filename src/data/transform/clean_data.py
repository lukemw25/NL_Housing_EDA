from __future__ import annotations
import re, logging
import pandas as pd
from src.data.utils.helpers import load_yaml, ensure_dirs, setup_logging
from src.data.transform.validate_schema import validate_schema

setup_logging()
log = logging.getLogger(__name__)

_EURO_RE = re.compile(r"[^\d.,]")
_M2_RE   = re.compile(r"[^\d.,]")
_DOT_FIX = re.compile(r"(?<=\d)\.(?=\d{3}\b)")  # thousands dot

def _to_float(s: object) -> float | None:
    if s is None or (isinstance(s, float) and pd.isna(s)):
        return None
    x = str(s).strip()
    x = _EURO_RE.sub("", x)         # drop € and spaces
    x = x.replace("m²", "").strip()
    x = _DOT_FIX.sub("", x)         # "1.050.000" -> "1050000"
    x = x.replace(",", ".")         # decimal comma -> dot if present
    try:
        return float(x)
    except ValueError:
        return None

def _rooms_from_text(s: object) -> int | None:
    # examples: "5 kamers (4 slaapkamers)" -> 5
    if s is None or (isinstance(s, float) and pd.isna(s)):
        return None
    m = re.search(r"(\d+)", str(s))
    return int(m.group(1)) if m else None

def clean_frame(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    if params["cleaning"].get("standardize_casing", True):
        if "City" in df.columns:
            df["City"] = df["City"].astype(str).str.strip()

    # parse numeric-like strings
    num_cols = ["Price", "Lot size (m2)", "Living space size (m2)", "Estimated neighbourhood price per m2"]
    for c in num_cols:
        if c in df.columns:
            df[c] = df[c].map(_to_float)

    # rooms parsing robust
    if "Rooms" in df.columns and not pd.api.types.is_numeric_dtype(df["Rooms"]):
        df["Rooms"] = df["Rooms"].map(_rooms_from_text)

    # outlier handling / rescaling
    rooms_max = int(params["cleaning"]["rooms_max"])
    df.loc[df["Rooms"] > rooms_max, "Rooms"] = pd.NA

    thr = float(params["cleaning"]["neigh_price_per_m2_rescale_threshold"])
    mask = df["Estimated neighbourhood price per m2"] > thr
    df.loc[mask, "Estimated neighbourhood price per m2"] = df.loc[mask, "Estimated neighbourhood price per m2"] / 100.0

    # build year coercion
    if "Build year" in df.columns:
        df["Build year"] = pd.to_numeric(df["Build year"], errors="coerce", downcast="integer")

    # category coercion (defer final casting to validator)
    return df

def main() -> None:
    setup_logging()
    params = load_yaml("config/params.yaml")
    raw_csv = params["raw_csv"]
    interim_csv = params["interim_csv"]
    processed_parquet = params["processed_parquet"]

    df = pd.read_csv(raw_csv)
    log.info("Loaded raw: %s", df.shape)

    df = clean_frame(df, params)
    df = validate_schema(df)

    ensure_dirs(interim_csv, processed_parquet)
    df.to_csv(interim_csv, index=False)
    df.to_parquet(processed_parquet, engine="pyarrow", index=False)
    log.info("Saved interim: %s", interim_csv)
    log.info("Saved processed: %s", processed_parquet)

if __name__ == "__main__":
    main()
