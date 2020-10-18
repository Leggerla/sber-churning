import os
import pandas as pd


def get_train(path='./'):
    train = pd.read_csv(os.path.join(path, 'train/train.csv'))
    train = train.rename(columns={'order_completed_at': 'month'})

    train['year'] = pd.to_datetime(train['month']).dt.year
    train['month'] = pd.to_datetime(train['month']).dt.month
    # train['y'] = train['month'] - 1

    return train


def train_test_split(df):
    test_phone_id = pd.DataFrame(
        df[df['month'] == 7]['phone_id'].unique(),
        columns=['phone_id']).sample(frac=0.01).values[:, 0]

    hold_out = df[df['phone_id'].isin(test_phone_id)]
    train = df[~df['phone_id'].isin(test_phone_id)]

    return train, hold_out


def get_shipments(path='./'):
    li = []
    ship_dir = os.path.join(path, 'shipments')
    for filename in sorted(os.listdir(ship_dir)):
        df = pd.read_csv(os.path.join(ship_dir, filename))
        li.append(df)
    shipments = pd.concat(li, axis=0)

    shipments['month'] = pd.to_datetime(shipments['order_created_at']).dt.month
    shipments['year'] = pd.to_datetime(shipments['order_created_at']).dt.year

    shipments = shipments[(shipments['year'] == 2020) | ((shipments['year'] == 2019) & (shipments['month'] == 12))]

    return shipments
