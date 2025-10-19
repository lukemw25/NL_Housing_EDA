from __future__ import annotations
import os, yaml, logging, logging.config

def load_yaml(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def ensure_dirs(*paths: str) -> None:
    for p in paths:
        os.makedirs(os.path.dirname(p), exist_ok=True)

def setup_logging(cfg_path: str = "config/logging.yaml") -> None:
    if os.path.exists(cfg_path):
        import yaml
        with open(cfg_path, "r", encoding="utf-8") as f:
            logging.config.dictConfig(yaml.safe_load(f))
    else:
        logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
