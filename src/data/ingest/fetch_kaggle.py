from __future__ import annotations
import os, logging
import pandas as pd
from src.data.utils.helpers import load_yaml, ensure_dirs, setup_logging

log = logging.getLogger(__name__)

def load_raw(params_path: str = "config/params.yaml") -> pd.DataFrame:
    setup_logging()
    params = load_yaml(params_path)
    raw_csv = params["raw_csv"]
    if not os.path.exists(raw_csv):
        raise FileNotFoundError(f"Raw CSV not found at {raw_csv}")
    df = pd.read_csv(raw_csv)
    log.info("Raw shape: %s", df.shape)
    return df

def main() -> None:
    df = load_raw()
    # optional: quick integrity echo
    print("Shape:", df.shape)
    print(df.head(3).to_string(index=False))

if __name__ == "__main__":
    main()
