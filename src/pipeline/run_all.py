# src/pipeline/run_all.py
from __future__ import annotations
import logging
from src.data.utils.helpers import load_yaml, setup_logging
from src.data.ingest.fetch_kaggle import load_raw
from src.data.transform.clean_data import clean_frame
from src.data.transform.validate_schema import validate_schema
from src.features.build_features import build_features

setup_logging()
log = logging.getLogger(__name__)

def run_all(params_path: str = "config/params.yaml") -> None:
    params = load_yaml(params_path)

    # 1. ingest
    df = load_raw(params_path)

    # 2. clean
    df = clean_frame(df, params)

    # 3. validate
    df = validate_schema(df)

    # 4. save processed
    interim_csv = params["interim_csv"]
    processed_parquet = params["processed_parquet"]
    from src.data.utils.helpers import ensure_dirs
    ensure_dirs(interim_csv, processed_parquet)
    df.to_csv(interim_csv, index=False)
    df_feat = build_features(df, params_path)
    df_feat.to_parquet(processed_parquet, engine="pyarrow", index=False)
    log.info("Pipeline complete: %s", processed_parquet)

    # 5. model training
    from src.models.train_model import train
    train(processed_parquet, params_path)

def main() -> None:
    run_all()

if __name__ == "__main__":
    main()
