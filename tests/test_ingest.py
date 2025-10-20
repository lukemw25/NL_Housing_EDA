import pandas as pd
from src.data.ingest.fetch_kaggle import load_raw

def test_load_raw_shape():
    df = load_raw()
    assert isinstance(df, pd.DataFrame)
    assert df.shape[0] > 0
