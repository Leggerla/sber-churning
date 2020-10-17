import os
import pandas as pd
from utils import *

  
def main(path='./'):
  train = pd.read_csv(os.path.join(path, 'train/train.csv'))
  addresses = pd.read_csv(os.path.join(path, 'misc/addresses.csv'))
  train_new = pd.merge(train, addresses, on=['phone_id'], how='left')

  shipments = concat_several()
  date_cols_shipments = ['order_created_at','order_completed_at','shipment_starts_at','shipped_at']
  shipments = perform_date(shipments,date_cols_shipments)

  # join by month
  shipments['order_completed_at_month'] = shipments['order_completed_at'].apply(lambda x: x.month)

  train_new['order_completed_at'] = pd.to_datetime(train_new['order_completed_at'])
  train_new['order_completed_at_month'] = train_new['order_completed_at'].apply(lambda x: x.month)

  train_all = pd.merge(train_new,shipments,left_on=['id','order_completed_at_month'],right_on=['ship_address_id','order_completed_at_month'],how='inner')
  
  train_all.to_hdf('train_all.h5',key='df',mode='w')
