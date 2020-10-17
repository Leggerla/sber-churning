import os
import pandas as pd
from utils import perform_date

def join_train_addresses(path='./'):
  train = pd.read_csv(os.path.join(path, 'train/train.csv'))
  addresses = pd.read_csv(os.path.join(path, 'misc/addresses.csv'))
  train_new = pd.merge(train, addresses, on=['phone_id'], how='left')
  return train_new
  
def return_shipments(path='./'):
  shipments2020_3_1 = pd.read_csv(os.path.join(path, 'shipments/shipments2020-03-01.csv'))
  shipments2020_4_30 = pd.read_csv(os.path.join(path, 'shipments/shipments2020-04-30.csv'))
  shipments2020_6_29 = pd.read_csv(os.path.join(path, 'shipments/shipments2020-06-29.csv'))
  shipments2020_1_1 = pd.read_csv(os.path.join(path, 'shipments/shipments2020-01-01.csv'))

  shipments = pd.concat([shipments2020_1_1, shipments2020_3_1, shipments2020_4_30, shipments2020_6_29])
  
  shipments = perform_date(shipments,date_cols)
  
  return shipments
