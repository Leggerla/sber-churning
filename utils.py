import pandas as pd

def perform_date(df, cols):
    for col in cols:
        df[col] = pd.to_datetime(df[col])
    return df
