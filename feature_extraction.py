import pandas as pd
import os
from joining_tables import get_train, get_shipments, get_messages


class FeatureExtractor:
    def __init__(self, path='./'):
        self.path = path
        self.numerical = ['total_weight', 'total_cost', 'rate', 'shipped_time', 'order_time', 'promo_total']
        self.categorical = ['platform', 'os', 'retailer', 's.order_state', 'shipment_state',
                            's.city_name', 'is_rated', 'dw_kind']
        self.other = ['s.store_id', 'ship_address_id', 'user_id', 'shipment_id', 'order_id']
        self.week = ['week']
        self.shop_cart = ['price', 'discount', 'quantity_x','quantity_y', 'cancelled',
                            'Pricer::PerKilo', 'Pricer::PerItem', 'Pricer::PerPackage', 'Pricer::PerPack',
                            'dis/price', 'replaced', 'ratio_dics']

    def collect_orders(self, train):
        orders = get_shipments(self.path)

        orders = train[['phone_id', 'id']].drop_duplicates().merge(orders,
                                                                   left_on='id', right_on='ship_address_id',
                                                                   how='left')
        messages = get_messages()
        orders = orders.merge(messages, on='user_id', how='left')

        orders = orders[~orders['ship_address_id'].isna()]

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

    def exract_all(self, orders):
        features_tables = []
        for field in self.numerical + self.categorical + self.other + self.week:
            features_tables.append(self.extract_feature(orders, field))

        features_tables.extend(self.extract_shop_cart_features())

        return pd.concat(features_tables, axis=1)

    def extract_feature(self, orders, field):
        groupby_field = orders.groupby(['phone_id', 'month'])[field]
        stats = None
        if field in self.numerical:
            stats = groupby_field.agg(['min', 'max', 'mean', 'median'])
            stats.columns = [x + '_' + field for x in stats.columns]
        elif field in self.categorical:
            stats = pd.DataFrame(groupby_field.value_counts()).unstack(level=2).fillna(0)
            stats.columns = stats.columns.droplevel()
            if field in ['s.order_state', 'shipment_state']:
                stats.columns = [x + '_' + field for x in stats.columns]
        elif field in self.other:
            stats = pd.DataFrame(groupby_field.nunique())
        elif field == 'week':
            day_of_month = pd.to_datetime(orders['order_created_at']).dt.day
            orders['week'] = (day_of_month - 1) // 7 + 1
            groupby_field = orders.groupby(['phone_id', 'month', 'week'])['shipment_id']
            stats = pd.DataFrame(groupby_field.nunique()).unstack(level=2).fillna(0)
            stats.columns = stats.columns.droplevel()
            stats.columns = [str(int(x)) + '_' + 'week' for x in stats.columns]

        return stats

    def extract_shop_cart_features(self):
        merge_ship = pd.read_csv(os.join.path(self.path, 'merge_ship.csv'))
        phone_user = pd.read_csv(os.join.path(self.path, 'phone_user_train.csv'))
        mm = merge_ship.merge(phone_user, on='user_id', how='inner')

        features_tables = []
        for field in self.shop_cart:

            features_tables.append(mm.groupby(['phone_id', 'month'])[field].sum())

        return features_tables