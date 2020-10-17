import pandas as pd
import os 

def perform_date(df, cols):
    for col in cols:
        df[col] = pd.to_datetime(df[col])
    return df

def concat_several(dir_name='./shipments'):
    li = []
    for filename in sorted(os.listdir(dir_name)):
        df = pd.read_csv(os.path.join(dir_name, filename))
        li.append(df)
    return pd.concat(li,axis=0)