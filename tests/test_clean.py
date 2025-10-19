import pandas as pd
from src.data.transform.clean_data import clean_frame
from src.data.utils.helpers import load_yaml

def test_rooms_outlier_removed():
    params = load_yaml("config/params.yaml")
    df = pd.DataFrame({"Rooms":[5, 999], "Price":[100.0, 200.0],
                       "Lot size (m2)":[100,100], "Living space size (m2)":[50,50],
                       "Estimated neighbourhood price per m2":[3.0,3.0],
                       "Build year":[2000,2000], "City":["A","A"], "House type":["X","X"], "Energy label":["B","B"]})
    out = clean_frame(df, params)
    assert pd.isna(out.loc[1,"Rooms"])
