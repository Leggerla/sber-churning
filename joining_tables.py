import os
import pandas as pd
from utils import months_processing


def get_train(path):
    train = pd.read_csv(os.path.join(path, 'train/train.csv'))
    train = train.rename(columns={'order_completed_at': 'month'})

    train['year'] = pd.to_datetime(train['month']).dt.year
    train['month'] = pd.to_datetime(train['month']).dt.month
    train['prev_month'] = train['month'] - 1

    return train


def train_test_split(df):
    test_phone_id = pd.DataFrame(
        df[df['month'] == 7]['phone_id'].unique(),
        columns=['phone_id']).sample(frac=0.01).values[:, 0]

    hold_out = df[df['phone_id'].isin(test_phone_id)]
    train = df[~df['phone_id'].isin(test_phone_id)]

    return train, hold_out


def get_shipments(path):
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

def combine_processing_actions_and_msgs(path):
    inner_path = os.path.join(path, 'messages')
    # открываем данные
    actions = pd.read_csv(inner_path + 'actions.csv')
    messages = pd.read_csv(inner_path + 'messages.csv')

    # переводим в общим формат с actions id
    messages = messages.astype({'action_id': 'int64'})

    # работаем со временем
    messages['sent'] = pd.to_datetime(messages['sent'], unit='s')
    ActM_combine = messages.merge(
        actions, left_on=['action_id'], right_on=['id'])
    ActM_combine['date'] = ActM_combine['sent'].dt.date
    date = pd.to_datetime(ActM_combine['date'])
    ActM_combine['date'] = date
    year = ActM_combine.date.dt.year
    month = ActM_combine.date.dt.month
    day = ActM_combine.date.dt.day
    wday = ActM_combine.date.dt.dayofweek

    # вытаскиваем дни недели и тд
    ActM_combine['date'] = date
    ActM_combine['year'] = year
    ActM_combine['month'] = month
    ActM_combine['day'] = day
    ActM_combine['wday'] = wday

    hours = ActM_combine['sent'].dt.hour
    minutes = ActM_combine['sent'].dt.minute
    ActM_combine['hours'] = hours
    ActM_combine['minutes'] = minutes

    # дроп ненужное
    ActM_combine = ActM_combine.drop(['date'], axis=1)
    ActM_combine = ActM_combine.drop(['sent'], axis=1)
    ActM_combine = ActM_combine.drop(['id'], axis=1)

    ActM_combine.rename(columns={"year": "year_act",
                                 "month": "month_act",
                                 "day": "day_act",
                                 "wday": "wday_act",
                                 "hours": "hours_act",
                                 "minutes": "minutes_act", })

    # переводим все в нумпай, чтобы быстрее обрабатывать данные
    array = ActM_combine.values

    # заменим все пропущенные значения в пушах на 0
    ActM_combine.subject.fillna(0, inplace=True)

    # поиск "промокод" в списке + дата до какого действует промокод
    array_promocode = []
    array_promocode_data = []

    for i in range(len(array)):
        each_str = array[i][3]

        try:
            each_str = each_str.replace('#', '').replace('@', '').replace(
                '!', '').replace(',', '').replace('.', '')
            each_str = each_str.lower().split(' ')

            if 'промокод' in each_str:
                array_promocode.append(1)
                if 'до' in each_str:
                    for j in range(len(each_str)):
                        if each_str[j] == 'до':
                            array_promocode_data.append(each_str[j + 1])
                else:
                    array_promocode_data.append(0)
            else:
                array_promocode.append(0)
                array_promocode_data.append(0)

        except:
            array_promocode.append(0)
            array_promocode_data.append(0)

    subject_promotion = []  # акции
    subject_other = []  # другое
    subject_shopping_cart = []  # корзина
    subject_discount = []  # скидки
    subject_bonus = []  # бонусы

    for i in range(len(array)):
        each_str = array[i][2]

        try:
            each_str = each_str.replace('#', '').replace('@', '').replace(
                '!', '').replace(',', '').replace('.', '').replace('?', '')
            each_str = each_str.lower().split(' ')

            cart = 0
            discount = 0
            bonus = 0
            promotion = 0
            other = 0

            if 'корзине' in each_str:
                cart = 1
            elif 'скидки' in each_str or 'скидка' in each_str or 'скидку' in each_str:
                discount = 1
            elif 'бонусов' in each_str or 'бонусы' in each_str:
                bonus = 1
            elif 'акция' in each_str:
                promotion = 1
            else:
                other = 1

            subject_shopping_cart.append(cart)
            subject_discount.append(discount)
            subject_bonus.append(bonus)
            subject_promotion.append(promotion)
            subject_other.append(other)

        except:
            subject_promotion.append(0)
            subject_other.append(0)
            subject_shopping_cart.append(0)
            subject_discount.append(0)
            subject_bonus.append(0)

    # записываем новые колонки в датасет
    ActM_combine['promocode'] = array_promocode
    ActM_combine['promocode_data'] = array_promocode_data
    ActM_combine['shopping_cart'] = subject_shopping_cart
    ActM_combine['discount'] = subject_discount
    ActM_combine['bonus'] = subject_bonus
    ActM_combine['promotion'] = subject_promotion
    ActM_combine['other'] = subject_other

    ActM_combine['promo_data_day'] = 0
    ActM_combine['promo_data_month'] = 0
    for i in range(len(ActM_combine)):
        if ActM_combine['promocode_data'][i] != 0:
            ActM_combine['promo_data_day'][i] = int(ActM_combine['promocode_data'][i][0] +
                                                    ActM_combine['promocode_data'][i][1])
            ActM_combine['promo_data_month'][i] = int(ActM_combine['promocode_data'][i][2] +
                                                      ActM_combine['promocode_data'][i][3])

    ActM_combine = ActM_combine.drop(['promocode_data'], axis=1)

    # сколько push уведомлений приходит человеку

    a = ActM_combine.user_id.value_counts()
    a = dict(a)
    a = pd.DataFrame({'count': a}, columns=['count']).reset_index()
    ActM_combine = ActM_combine.merge(a, left_on='user_id', right_on='index')
    ActM_combine = ActM_combine.drop(['index'], axis=1)

    return months_processing(ActM_combine)