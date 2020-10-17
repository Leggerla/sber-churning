import pandas as pd
from joining_tables import get_train, get_shipments
import os


class FeatureExtractor:
    def __init__(self, path='./'):
        self.path = path
        self.numerical = ['total_weight', 'total_cost', 'rate', 'shipped_time', 'order_time', 'promo_total']
        self.categorical = ['platform', 'os', 'retailer', 's.order_state', 'shipment_state', 's.city_name', 'is_rated',
                            'dw_kind']
        self.other = ['s.store_id', 'ship_address_id', 'user_id', 'shipment_id', 'order_id']

        self.orders = self.collect_orders()

    def collect_orders(self):
        train = get_train(self.path)
        addresses = pd.read_csv(os.path.join(self.path, 'misc/addresses.csv'))
        train = train.merge(addresses, on='phone_id', how='left')

        orders = get_shipments(self.path)
        orders = train[['phone_id', 'id']].drop_duplicates().merge(orders,
                                                                   left_on='id', right_on='ship_address_id',
                                                                   how='left')

        orders['is_rated'] = orders['rate'].apply(lambda x: 1 if x == 0 else 0)

        orders['shipped_time'] = (
                                         pd.to_datetime(orders['shipped_at']) -
                                         pd.to_datetime(orders['shipment_starts_at'])
                                 ).dt.total_seconds() // 60
        orders['order_time'] = (
                                       pd.to_datetime(orders['order_completed_at']) -
                                       pd.to_datetime(orders['order_created_at'])
                               ).dt.total_seconds() // 60

        return orders

    def exract_all(self):
        features_tables = []
        for field in self.numerical + self.categorical + self.other:
            features_tables.append(self.extract_feature(field))

        return pd.concat(features_tables, axis=1)

    def extract_feature(self, field):
        groupby_field = self.orders.groupby(['phone_id', 'month'])[field]
        stats = None
        if field in self.numerical:
            stats = groupby_field.agg(['min', 'max', 'mean', 'median'])
            stats.columns = [x + '_' + field for x in stats.columns]
            return stats

        elif field in self.categorical:
            stats = pd.DataFrame(groupby_field.value_counts()).unstack(level=2).fillna(0)
            stats.columns = stats.columns.droplevel()
            if field in ['s.order_state', 'shipment_state']:
                stats.columns = [x + '_' + field for x in stats.columns]
        elif field in self.other:
            stats = pd.DataFrame(groupby_field.nunique())

        return stats
