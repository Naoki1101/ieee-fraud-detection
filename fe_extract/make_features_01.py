import numpy as np
import pandas as pd
import re
import yaml
import unicodedata
import datetime

# from sklearn.manifold import TSNE
# from bhtsne import tsne
import umap
from scipy.sparse.csgraph import connected_components
import matplotlib.cm as cm
from sklearn.model_selection import StratifiedKFold

from base import Feature, get_arguments, generate_features, sigmoid, minmaxscale, target_encoding, SinCos

# import MeCab
import lda
from sklearn.feature_extraction.text import CountVectorizer

with open('./configs/default.yaml', 'r') as yf:
    config = yaml.load(yf)

with open('./configs/release_date.yaml', 'r') as yf:
    release_info = yaml.load(yf)

with open('./configs/email.yaml', 'r') as yf:
    email_info = yaml.load(yf)

SEED = config['seed']
START_DATE = config['start_date']

os2release_date = release_info['os']
brawser2release_date = release_info['brawser']
email_transformer = email_info['emails']

Feature.dir = 'features'


class transaction_dt(Feature):
    def create_features(self):
        self.train[self.__class__.__name__] = train_transaction['TransactionDT']
        self.test[self.__class__.__name__] = test_transaction['TransactionDT']


class transaction_dt_last(Feature):
    def create_features(self):
        col = 'card1'
        df_last_dt = train_transaction.groupby('card1')['TransactionDT'].max().to_frame(self.__class__.__name__).reset_index()
        self.train[self.__class__.__name__] = pd.merge(train_transaction[col], df_last_dt, on='card1', how='left')[self.__class__.__name__]
        self.test[self.__class__.__name__] = pd.merge(test_transaction[col], df_last_dt, on='card1', how='left')[self.__class__.__name__]


class transaction_dt_month(Feature):
    def create_features(self):
        col = 'TransactionDT'
        whole = pd.concat([train_transaction[col], test_transaction[col]], axis=0).to_frame()
        whole['datetime_'] = whole[col].apply(lambda x: (START_DATE + datetime.timedelta(seconds=x)))
        whole['month'] = whole['datetime_'].apply(lambda x: x.month)
        self.train[self.__class__.__name__] = whole['month'][:len(train_transaction)]
        self.test[self.__class__.__name__] = whole['month'][len(train_transaction):]


class transaction_dt_day(Feature):
    def create_features(self):
        col = 'TransactionDT'
        whole = pd.concat([train_transaction[col], test_transaction[col]], axis=0).to_frame()
        whole['datetime_'] = whole[col].apply(lambda x: (START_DATE + datetime.timedelta(seconds=x)))
        whole['day'] = whole['datetime_'].apply(lambda x: x.day)
        self.train[self.__class__.__name__] = whole['day'][:len(train_transaction)]
        self.test[self.__class__.__name__] = whole['day'][len(train_transaction):]


class transaction_dt_hour(Feature):
    def create_features(self):
        col = 'TransactionDT'
        self.train[self.__class__.__name__] = train_transaction[col] // 3600 % 24
        self.test[self.__class__.__name__] = test_transaction[col] // 3600 % 24


class transaction_dt_weekday(Feature):
    def create_features(self):
        col = 'TransactionDT'
        whole = pd.concat([train_transaction[col], test_transaction[col]], axis=0).to_frame()
        whole['datetime_'] = whole[col].apply(lambda x: (START_DATE + datetime.timedelta(seconds=x)))
        whole['weekday'] = whole['datetime_'].apply(lambda x: datetime.date.weekday(x))
        self.train[self.__class__.__name__] = whole['weekday'][:len(train_transaction)]
        self.test[self.__class__.__name__] = whole['weekday'][len(train_transaction):]


class transaction_dt_weekday_sin(Feature):
    def create_features(self):
        col = 'TransactionDT'
        whole = pd.concat([train_transaction[col], test_transaction[col]], axis=0).to_frame()
        whole['datetime_'] = whole[col].apply(lambda x: (START_DATE + datetime.timedelta(seconds=x)))
        whole['weekday'] = whole['datetime_'].apply(lambda x: datetime.date.weekday(x))

        sincos = SinCos(feature_name="weekday", period=7)
        whole, _ = sincos.create_features(whole)
        self.train[self.__class__.__name__] = whole['weekday_sin'][:len(train_transaction)]
        self.test[self.__class__.__name__] = whole['weekday_sin'][len(train_transaction):]


class count_transaction_each_card1_and_day(Feature):
    def create_features(self):
        feats = ['TransactionDT', 'card1']
        len_ = len(train_transaction)
        whole = pd.concat([train_transaction[feats], test_transaction[feats]], axis=0)
        whole['date'] = whole[feats[0]].apply(lambda x: (START_DATE + datetime.timedelta(seconds=x)))
        whole['card1_date'] = whole['card1'].astype(str) + '_' + whole['date'].astype(str)
        le = whole['card1_date'].value_counts().to_dict()
        whole[self.__class__.__name__] = whole['card1_date'].map(le)
        self.train[self.__class__.__name__] = whole[self.__class__.__name__].values[:len_]
        self.test[self.__class__.__name__] = whole[self.__class__.__name__].values[len_:]


class count_transaction_each_card1_and_card5_and_day(Feature):
    def create_features(self):
        feats = ['TransactionDT', 'card1', 'card5']
        len_ = len(train_transaction)
        whole = pd.concat([train_transaction[feats], test_transaction[feats]], axis=0)
        whole['date'] = whole[feats[0]].apply(lambda x: (START_DATE + datetime.timedelta(seconds=x)))
        whole['card1_card5_date'] = whole['card1'].astype(str) + '_' + whole['card5'].astype(str) + '_' + whole['date'].astype(str)
        le = whole['card1_card5_date'].value_counts().to_dict()
        whole[self.__class__.__name__] = whole['card1_card5_date'].map(le)
        self.train[self.__class__.__name__] = whole[self.__class__.__name__].values[:len_]
        self.test[self.__class__.__name__] = whole[self.__class__.__name__].values[len_:]


class transaction_dt_bin_26(Feature):
    def create_features(self):
        interval = 3600 * 24 * 7
        col = 'TransactionDT'
        len_ = len(train_transaction)

        whole = pd.concat([train_transaction[col], test_transaction[col]], axis=0).to_frame()
        whole['level_'] = (whole['TransactionDT'] - 86400) // interval
        df_count = whole['level_'].value_counts().reset_index()
        df_count.rename(columns={'index': 'level_', 'level_': 'count_'}, inplace=True)
        whole = pd.merge(whole, df_count, on='level_', how='left')

        self.train['transaction_dt_bin_26'] = whole['count_'][:len_].values
        self.test['transaction_dt_bin_26'] = whole['count_'][len_:].values


class transaction_dt_bin_91(Feature):
    def create_features(self):
        interval = 3600 * 24 * 2
        col = 'TransactionDT'
        len_ = len(train_transaction)

        whole = pd.concat([train_transaction[col], test_transaction[col]], axis=0).to_frame()
        whole['level_'] = (whole['TransactionDT'] - 86400) // interval
        df_count = whole['level_'].value_counts().reset_index()
        df_count.rename(columns={'index': 'level_', 'level_': 'count_'}, inplace=True)
        whole = pd.merge(whole, df_count, on='level_', how='left')

        self.train['transaction_dt_bin_91'] = whole['count_'][:len_].values
        self.test['transaction_dt_bin_91'] = whole['count_'][len_:].values


class transaction_dt_bin_182(Feature):
    def create_features(self):
        interval = 3600 * 24
        col = 'TransactionDT'
        len_ = len(train_transaction)

        whole = pd.concat([train_transaction[col], test_transaction[col]], axis=0).to_frame()
        whole['level_'] = (whole['TransactionDT'] - 86400) // interval
        df_count = whole['level_'].value_counts().reset_index()
        df_count.rename(columns={'index': 'level_', 'level_': 'count_'}, inplace=True)
        whole = pd.merge(whole, df_count, on='level_', how='left')

        self.train['transaction_dt_bin_182'] = whole['count_'][:len_].values
        self.test['transaction_dt_bin_182'] = whole['count_'][len_:].values


class transaction_dt_bin_182_div_500(Feature):
    def create_features(self):
        interval = 3600 * 24
        col = 'TransactionDT'
        len_ = len(train_transaction)

        whole = pd.concat([train_transaction[col], test_transaction[col]], axis=0).to_frame()
        whole['level_'] = (whole['TransactionDT'] - 86400) // interval
        df_count = whole['level_'].value_counts().reset_index()
        df_count.rename(columns={'index': 'level_', 'level_': 'count_'}, inplace=True)
        whole = pd.merge(whole, df_count, on='level_', how='left')

        self.train[self.__class__.__name__] = whole['count_'][:len_].values // 500
        self.test[self.__class__.__name__] = whole['count_'][len_:].values // 500


class transaction_dt_bin_364(Feature):
    def create_features(self):
        interval = 3600 * 12
        col = 'TransactionDT'
        len_ = len(train_transaction)

        whole = pd.concat([train_transaction[col], test_transaction[col]], axis=0).to_frame()
        whole['level_'] = (whole['TransactionDT'] - 86400) // interval
        df_count = whole['level_'].value_counts().reset_index()
        df_count.rename(columns={'index': 'level_', 'level_': 'count_'}, inplace=True)
        whole = pd.merge(whole, df_count, on='level_', how='left')

        self.train['transaction_dt_bin_364'] = whole['count_'][:len_].values
        self.test['transaction_dt_bin_364'] = whole['count_'][len_:].values


class transaction_dt_bin_728(Feature):
    def create_features(self):
        interval = 3600 * 6
        col = 'TransactionDT'
        len_ = len(train_transaction)

        whole = pd.concat([train_transaction[col], test_transaction[col]], axis=0).to_frame()
        whole['level_'] = (whole['TransactionDT'] - 86400) // interval
        df_count = whole['level_'].value_counts().reset_index()
        df_count.rename(columns={'index': 'level_', 'level_': 'count_'}, inplace=True)
        whole = pd.merge(whole, df_count, on='level_', how='left')

        self.train['transaction_dt_bin_728'] = whole['count_'][:len_].values
        self.test['transaction_dt_bin_728'] = whole['count_'][len_:].values


class transaction_dt_bin_1456(Feature):
    def create_features(self):
        interval = 3600 * 3
        col = 'TransactionDT'
        len_ = len(train_transaction)

        whole = pd.concat([train_transaction[col], test_transaction[col]], axis=0).to_frame()
        whole['level_'] = (whole['TransactionDT'] - 86400) // interval
        df_count = whole['level_'].value_counts().reset_index()
        df_count.rename(columns={'index': 'level_', 'level_': 'count_'}, inplace=True)
        whole = pd.merge(whole, df_count, on='level_', how='left')

        self.train['transaction_dt_bin_1456'] = whole['count_'][:len_].values
        self.test['transaction_dt_bin_1456'] = whole['count_'][len_:].values


class transaction_dt_bin_4368(Feature):
    def create_features(self):
        interval = 3600
        col = 'TransactionDT'
        len_ = len(train_transaction)

        whole = pd.concat([train_transaction[col], test_transaction[col]], axis=0).to_frame()
        whole['level_'] = (whole['TransactionDT'] - 86400) // interval
        df_count = whole['level_'].value_counts().reset_index()
        df_count.rename(columns={'index': 'level_', 'level_': 'count_'}, inplace=True)
        whole = pd.merge(whole, df_count, on='level_', how='left')

        self.train['transaction_dt_bin_4368'] = whole['count_'][:len_].values
        self.test['transaction_dt_bin_4368'] = whole['count_'][len_:].values


class transaction_dt_bin_8736(Feature):
    def create_features(self):
        interval = 3600 / 2
        col = 'TransactionDT'
        len_ = len(train_transaction)

        whole = pd.concat([train_transaction[col], test_transaction[col]], axis=0).to_frame()
        whole['level_'] = (whole['TransactionDT'] - 86400) // interval
        df_count = whole['level_'].value_counts().reset_index()
        df_count.rename(columns={'index': 'level_', 'level_': 'count_'}, inplace=True)
        whole = pd.merge(whole, df_count, on='level_', how='left')

        self.train['transaction_dt_bin_8736'] = whole['count_'][:len_].values
        self.test['transaction_dt_bin_8736'] = whole['count_'][len_:].values


class transaction_dt_bin_52416(Feature):
    def create_features(self):
        interval = 3600 / 12  # 5分
        col = 'TransactionDT'
        len_ = len(train_transaction)

        whole = pd.concat([train_transaction[col], test_transaction[col]], axis=0).to_frame()
        whole['level_'] = (whole['TransactionDT'] - 86400) // interval
        df_count = whole['level_'].value_counts().reset_index()
        df_count.rename(columns={'index': 'level_', 'level_': 'count_'}, inplace=True)
        whole = pd.merge(whole, df_count, on='level_', how='left')

        self.train[self.__class__.__name__] = whole['count_'][:len_].values
        self.test[self.__class__.__name__] = whole['count_'][len_:].values


class count_each_day_div_month_mean(Feature):
    def create_features(self):
        feats = ['TransactionDT']
        len_ = len(train_transaction)
        whole = pd.concat([train_transaction[feats], test_transaction[feats]], axis=0)
        whole['date'] = whole['TransactionDT'].apply(lambda x: (START_DATE + datetime.timedelta(seconds=x)))

        count_transaction_each_day = whole['date'].value_counts().reset_index()
        count_transaction_each_day.rename(columns={'index': 'date', 'date': 'count_'}, inplace=True)
        count_transaction_each_day['month'] = count_transaction_each_day['date'].apply(lambda x: str(x)[:7])
        mean_transaction_each_month = count_transaction_each_day.groupby('month')['count_'].mean().to_frame('mean_')
        count_transaction_each_day = pd.merge(count_transaction_each_day, mean_transaction_each_month, on='month', how='left')
        count_transaction_each_day['count_each_day_div_month_mean'] = count_transaction_each_day['count_'] / count_transaction_each_day['mean_']
        le = dict(count_transaction_each_day[['date', 'count_each_day_div_month_mean']].values)

        whole['count_each_day_div_month_mean'] = whole['date'].map(le)
        self.train[self.__class__.__name__] = whole['count_each_day_div_month_mean'].values[:len_]
        self.test[self.__class__.__name__] = whole['count_each_day_div_month_mean'].values[len_:]


class transaction_mean_amt_each_month(Feature):
    def create_features(self):
        feats = ['card1', 'TransactionDT', 'TransactionAmt']
        whole = pd.concat([train_transaction[feats], test_transaction[feats]], axis=0).reset_index(drop=True)
        whole['datetime_'] = whole['TransactionDT'].apply(lambda x: (START_DATE + datetime.timedelta(seconds=x)))
        whole['month'] = whole['datetime_'].apply(lambda x: x.month)

        df_month_sum = whole.pivot_table(index='card1', columns='month', values='TransactionAmt', aggfunc='sum', fill_value=0)
        df_month_sum[self.__class__.__name__] = np.mean(df_month_sum.values, axis=1)

        self.train[self.__class__.__name__] = pd.merge(train_transaction['card1'], df_month_sum, on='card1', how='left')[self.__class__.__name__]
        self.test[self.__class__.__name__] = pd.merge(test_transaction['card1'], df_month_sum, on='card1', how='left')[self.__class__.__name__]


class transaction_std_amt_each_month(Feature):
    def create_features(self):
        feats = ['card1', 'TransactionDT', 'TransactionAmt']
        whole = pd.concat([train_transaction[feats], test_transaction[feats]], axis=0).reset_index(drop=True)
        whole['datetime_'] = whole['TransactionDT'].apply(lambda x: (START_DATE + datetime.timedelta(seconds=x)))
        whole['month'] = whole['datetime_'].apply(lambda x: x.month)

        df_month_sum = whole.pivot_table(index='card1', columns='month', values='TransactionAmt', aggfunc='sum', fill_value=0)
        df_month_sum[self.__class__.__name__] = np.std(df_month_sum.values, axis=1)

        self.train[self.__class__.__name__] = pd.merge(train_transaction['card1'], df_month_sum, on='card1', how='left')[self.__class__.__name__]
        self.test[self.__class__.__name__] = pd.merge(test_transaction['card1'], df_month_sum, on='card1', how='left')[self.__class__.__name__]


class transaction_amt(Feature):
    def create_features(self):
        self.train['TransactionAmt'] = train_transaction['TransactionAmt']
        self.test['TransactionAmt'] = test_transaction['TransactionAmt']


class transaction_amt_1(Feature):
    def create_features(self):
        self.train['transaction_amt_1'] = train_transaction['TransactionAmt'].apply(lambda x: x % 1)
        self.test['transaction_amt_1'] = test_transaction['TransactionAmt'].apply(lambda x: x % 1)


class transaction_amt_10(Feature):
    def create_features(self):
        self.train['transaction_amt_10'] = train_transaction['TransactionAmt'].apply(lambda x: x * 10 % 1)
        self.test['transaction_amt_10'] = test_transaction['TransactionAmt'].apply(lambda x: x * 10 % 1)


class transaction_amt_100(Feature):
    def create_features(self):
        self.train['transaction_amt_100'] = train_transaction['TransactionAmt'].apply(lambda x: x * 100 % 1)
        self.test['transaction_amt_100'] = test_transaction['TransactionAmt'].apply(lambda x: x * 100 % 1)


class transaction_amt_div_card1_mean(Feature):
    def create_features(self):
        feats = ['card1', 'TransactionAmt']
        whole = pd.concat([train_transaction[feats], test_transaction[feats]], axis=0)
        df_mean = whole.groupby(feats[0])[feats[1]].mean().reset_index()
        df_mean.columns = ['card1', 'mean_card1']
        train = pd.merge(train_transaction[feats], df_mean, on=feats[0], how='left')
        test = pd.merge(test_transaction[feats], df_mean, on=feats[0], how='left')
        self.train[self.__class__.__name__] = train['TransactionAmt'] / train['mean_card1']
        self.test[self.__class__.__name__] = test['TransactionAmt'] / test['mean_card1']


class transaction_amt_div_card1_median(Feature):
    def create_features(self):
        feats = ['card1', 'TransactionAmt']
        whole = pd.concat([train_transaction[feats], test_transaction[feats]], axis=0)
        df_median = whole.groupby(feats[0])[feats[1]].median().reset_index()
        df_median.columns = ['card1', 'median_card1']
        train = pd.merge(train_transaction[feats], df_median, on=feats[0], how='left')
        test = pd.merge(test_transaction[feats], df_median, on=feats[0], how='left')
        self.train[self.__class__.__name__] = train['TransactionAmt'] / train['median_card1']
        self.test[self.__class__.__name__] = test['TransactionAmt'] / test['median_card1']


class transaction_amt_div_card5_mean(Feature):
    def create_features(self):
        feats = ['card5', 'TransactionAmt']
        whole = pd.concat([train_transaction[feats], test_transaction[feats]], axis=0)
        df_mean = whole.groupby(feats[0])[feats[1]].mean().reset_index()
        df_mean.columns = ['card5', 'mean_card5']
        train = pd.merge(train_transaction[feats], df_mean, on=feats[0], how='left')
        test = pd.merge(test_transaction[feats], df_mean, on=feats[0], how='left')
        self.train[self.__class__.__name__] = train['TransactionAmt'] / train['mean_card5']
        self.test[self.__class__.__name__] = test['TransactionAmt'] / test['mean_card5']


class transaction_amt_div_card5_median(Feature):
    def create_features(self):
        feats = ['card5', 'TransactionAmt']
        whole = pd.concat([train_transaction[feats], test_transaction[feats]], axis=0)
        df_median = whole.groupby(feats[0])[feats[1]].median().reset_index()
        df_median.columns = ['card5', 'median_card5']
        train = pd.merge(train_transaction[feats], df_median, on=feats[0], how='left')
        test = pd.merge(test_transaction[feats], df_median, on=feats[0], how='left')
        self.train[self.__class__.__name__] = train['TransactionAmt'] / train['median_card5']
        self.test[self.__class__.__name__] = test['TransactionAmt'] / test['median_card5']


class transaction_amt_div_addr1_mean(Feature):
    def create_features(self):
        feats = ['addr1', 'TransactionAmt']
        whole = pd.concat([train_transaction[feats], test_transaction[feats]], axis=0)
        df_mean = whole.groupby(feats[0])[feats[1]].mean().reset_index()
        df_mean.columns = ['addr1', 'mean_addr1']
        train = pd.merge(train_transaction[feats], df_mean, on=feats[0], how='left')
        test = pd.merge(test_transaction[feats], df_mean, on=feats[0], how='left')
        self.train[self.__class__.__name__] = train['TransactionAmt'] / train['mean_addr1']
        self.test[self.__class__.__name__] = test['TransactionAmt'] / test['mean_addr1']


class transaction_amt_div_addr2_mean(Feature):
    def create_features(self):
        feats = ['addr2', 'TransactionAmt']
        whole = pd.concat([train_transaction[feats], test_transaction[feats]], axis=0)
        df_mean = whole.groupby(feats[0])[feats[1]].mean().reset_index()
        df_mean.columns = ['addr2', 'mean_addr2']
        train = pd.merge(train_transaction[feats], df_mean, on=feats[0], how='left')
        test = pd.merge(test_transaction[feats], df_mean, on=feats[0], how='left')
        self.train[self.__class__.__name__] = train['TransactionAmt'] / train['mean_addr2']
        self.test[self.__class__.__name__] = test['TransactionAmt'] / test['mean_addr2']


class product_cd(Feature):
    def create_features(self):
        le = {'W': 0, 'C': 1, 'R': 2, 'H': 3, 'S': 4}
        self.train['ProductCD'] = train_transaction['ProductCD'].apply(lambda x: le[x])
        self.test['ProductCD'] = test_transaction['ProductCD'].apply(lambda x: le[x])


class card1(Feature):
    def create_features(self):
        self.train['card1'] = train_transaction['card1']
        self.test['card1'] = test_transaction['card1']


class card1_selected(Feature):
    def create_features(self):
        feats = ['card1', 'TransactionDT']
        whole = pd.concat([train_transaction[feats], test_transaction[feats]], axis=0)
        whole['datetime_'] = whole['TransactionDT'].apply(lambda x: (START_DATE + datetime.timedelta(seconds=x)))
        whole['month'] = whole['datetime_'].apply(lambda x: x.month)

        df_pivot = whole.pivot_table(index='card1',
                                     columns='month',
                                     values='TransactionDT',
                                     aggfunc='count',
                                     fill_value=-9999999)
        use_idx = np.where(np.sum(df_pivot.values, axis=1) > 0)[0]
        card1_le = {df_pivot.index[idx]: i for i, idx in enumerate(use_idx)}

        self.train[self.__class__.__name__] = train_transaction['card1'].map(card1_le).fillna(999)
        self.test[self.__class__.__name__] = test_transaction['card1'].map(card1_le).fillna(999)


class card2_selected(Feature):
    def create_features(self):
        feats = ['card2', 'TransactionDT']
        whole = pd.concat([train_transaction[feats], test_transaction[feats]], axis=0)
        whole['datetime_'] = whole['TransactionDT'].apply(lambda x: (START_DATE + datetime.timedelta(seconds=x)))
        whole['month'] = whole['datetime_'].apply(lambda x: x.month)

        df_pivot = whole.pivot_table(index='card2',
                                     columns='month',
                                     values='TransactionDT',
                                     aggfunc='count',
                                     fill_value=-9999999)
        use_idx = np.where(np.sum(df_pivot.values, axis=1) > 0)[0]
        card1_le = {df_pivot.index[idx]: i for i, idx in enumerate(use_idx)}

        self.train[self.__class__.__name__] = train_transaction['card2'].map(card1_le).fillna(999)
        self.test[self.__class__.__name__] = test_transaction['card2'].map(card1_le).fillna(999)


class card2(Feature):
    def create_features(self):
        self.train['card2'] = train_transaction['card2']
        self.test['card2'] = test_transaction['card2']


class card3(Feature):
    def create_features(self):
        self.train['card3'] = train_transaction['card3']
        self.test['card3'] = test_transaction['card3']


class card4(Feature):
    def create_features(self):
        le = {'visa': 0, 'mastercard': 1, 'american express': 2, 'discover': 3}
        self.train['card4'] = train_transaction['card4'].apply(lambda x: le[x] if type(x) == str else x)
        self.test['card4'] = test_transaction['card4'].apply(lambda x: le[x] if type(x) == str else x)


class card5(Feature):
    def create_features(self):
        self.train['card5'] = train_transaction['card5']
        self.test['card5'] = test_transaction['card5']


class card5_selected(Feature):
    def create_features(self):
        feats = ['card5', 'TransactionDT']
        whole = pd.concat([train_transaction[feats], test_transaction[feats]], axis=0)
        whole['datetime_'] = whole['TransactionDT'].apply(lambda x: (START_DATE + datetime.timedelta(seconds=x)))
        whole['month'] = whole['datetime_'].apply(lambda x: x.month)

        df_pivot = whole.pivot_table(index='card5',
                                     columns='month',
                                     values='TransactionDT',
                                     aggfunc='count',
                                     fill_value=-9999999)
        use_idx = np.where(np.sum(df_pivot.values, axis=1) > 0)[0]
        card5_le = {df_pivot.index[idx]: i for i, idx in enumerate(use_idx)}

        self.train[self.__class__.__name__] = train_transaction['card5'].map(card5_le)
        self.test[self.__class__.__name__] = test_transaction['card5'].map(card5_le)


class card6(Feature):
    def create_features(self):
        le = {'debit': 0, 'credit': 1, 'debit or credit': 2, 'charge card': 3}
        self.train['card6'] = train_transaction['card6'].apply(lambda x: le[x] if type(x) == str else x)
        self.test['card6'] = test_transaction['card6'].apply(lambda x: le[x] if type(x) == str else x)


class addr1(Feature):
    def create_features(self):
        self.train['addr1'] = train_transaction['addr1']
        self.test['addr1'] = test_transaction['addr1']


class addr1_selected(Feature):
    def create_features(self):
        feats = ['addr1', 'TransactionDT']
        whole = pd.concat([train_transaction[feats], test_transaction[feats]], axis=0)
        whole['datetime_'] = whole['TransactionDT'].apply(lambda x: (START_DATE + datetime.timedelta(seconds=x)))
        whole['month'] = whole['datetime_'].apply(lambda x: x.month)

        df_pivot = whole.pivot_table(index='addr1',
                                     columns='month',
                                     values='TransactionDT',
                                     aggfunc='count',
                                     fill_value=-9999999)
        use_idx = np.where(np.sum(df_pivot.values, axis=1) > 0)[0]
        addr1_le = {df_pivot.index[idx]: i for i, idx in enumerate(use_idx)}

        self.train[self.__class__.__name__] = train_transaction['addr1'].map(addr1_le)
        self.test[self.__class__.__name__] = test_transaction['addr1'].map(addr1_le)


class addr1_round1(Feature):
    def create_features(self):
        self.train['addr1'] = np.round(train_transaction['addr1'].values, -1)
        self.test['addr1'] = np.round(test_transaction['addr1'].values, -1)


class addr1_round2(Feature):
    def create_features(self):
        self.train['addr1'] = np.round(train_transaction['addr1'].values, -2)
        self.test['addr1'] = np.round(test_transaction['addr1'].values, -2)


class addr2(Feature):
    def create_features(self):
        self.train['addr2'] = train_transaction['addr2']
        self.test['addr2'] = test_transaction['addr2']


def addr2_clf(v):
    if v == 87:
        return 0
    elif v == 60:
        return 1
    elif v == 96:
        return 2
    else:
        return 3


class addr2_flg(Feature):
    def create_features(self):
        self.train['addr2_flg'] = train_transaction['addr2'].apply(addr2_clf)
        self.test['addr2_flg'] = test_transaction['addr2'].apply(addr2_clf)


class addr2_round1(Feature):
    def create_features(self):
        self.train['addr2'] = np.round(train_transaction['addr2'].values, -1)
        self.test['addr2'] = np.round(test_transaction['addr2'].values, -1)


class dist1(Feature):
    def create_features(self):
        self.train['dist1'] = train_transaction['dist1']
        self.test['dist1'] = test_transaction['dist1']


class dist1_flg(Feature):
    def create_features(self):
        self.train['dist1_flg'] = train_transaction['dist1'].fillna(-1).apply(lambda x: x if x < 100 else 100)
        self.test['dist1_flg'] = test_transaction['dist1'].fillna(-1).apply(lambda x: x if x < 100 else 100)


class dist2(Feature):
    def create_features(self):
        self.train['dist2'] = train_transaction['dist2']
        self.test['dist2'] = test_transaction['dist2']


class p_emaildomain(Feature):
    def create_features(self):
        whole = pd.concat([train_transaction['P_emaildomain'], test_transaction['P_emaildomain']], axis=0).to_frame()
        le = {v: i for i, v in enumerate(whole['P_emaildomain'].unique())}
        self.train['P_emaildomain'] = train_transaction['P_emaildomain'].apply(lambda x: le[x] if type(x) == str else x)
        self.test['P_emaildomain'] = test_transaction['P_emaildomain'].apply(lambda x: le[x] if type(x) == str else x)


class p_emaildomain_simple(Feature):
    def create_features(self):
        len_ = len(train_transaction)
        whole = pd.concat([train_transaction['P_emaildomain'], test_transaction['P_emaildomain']], axis=0).to_frame()
        whole[self.__class__.__name__] = whole['P_emaildomain'].map(email_transformer)
        le = whole[self.__class__.__name__].value_counts().to_dict()
        self.train[self.__class__.__name__] = whole[self.__class__.__name__].map(le).values[:len_]
        self.test[self.__class__.__name__] = whole[self.__class__.__name__].map(le).values[len_:]


class p_emaildomain_top(Feature):
    def create_features(self):
        whole = pd.concat([train_transaction['P_emaildomain'], test_transaction['P_emaildomain']], axis=0).to_frame()
        whole['p_emaildomain_top'] = whole['P_emaildomain'].apply(lambda x: x.split(".")[0] if type(x) == str else x)
        le = {v: i for i, v in enumerate(whole['p_emaildomain_top'].unique())}
        whole['p_emaildomain_top'] = whole['p_emaildomain_top'].apply(lambda x: le[x] if type(x) == str else x)
        self.train['p_emaildomain_top'] = whole['p_emaildomain_top'][:len(train_transaction)]
        self.test['p_emaildomain_top'] = whole['p_emaildomain_top'][len(train_transaction):]


class p_emaildomain_last(Feature):
    def create_features(self):
        whole = pd.concat([train_transaction['P_emaildomain'], test_transaction['P_emaildomain']], axis=0).to_frame()
        whole['p_emaildomain_last'] = whole['P_emaildomain'].apply(lambda x: x.split(".")[-1] if type(x) == str else x)
        le = {v: i for i, v in enumerate(whole['p_emaildomain_last'].unique())}
        whole['p_emaildomain_last'] = whole['p_emaildomain_last'].apply(lambda x: le[x] if type(x) == str else x)
        self.train['p_emaildomain_last'] = whole['p_emaildomain_last'][:len(train_transaction)]
        self.test['p_emaildomain_last'] = whole['p_emaildomain_last'][len(train_transaction):]


class r_emaildomain(Feature):
    def create_features(self):
        whole = pd.concat([train_transaction['R_emaildomain'], test_transaction['R_emaildomain']], axis=0).to_frame()
        le = {v: i for i, v in enumerate(whole['R_emaildomain'].unique())}
        self.train['R_emaildomain'] = train_transaction['R_emaildomain'].apply(lambda x: le[x] if type(x) == str else x)
        self.test['R_emaildomain'] = test_transaction['R_emaildomain'].apply(lambda x: le[x] if type(x) == str else x)


class r_emaildomain_simple(Feature):
    def create_features(self):
        len_ = len(train_transaction)
        whole = pd.concat([train_transaction['R_emaildomain'], test_transaction['R_emaildomain']], axis=0).to_frame()
        whole[self.__class__.__name__] = whole['R_emaildomain'].map(email_transformer)
        le = whole[self.__class__.__name__].value_counts().to_dict()
        self.train[self.__class__.__name__] = whole[self.__class__.__name__].map(le).values[:len_]
        self.test[self.__class__.__name__] = whole[self.__class__.__name__].map(le).values[len_:]


class r_emaildomain_top(Feature):
    def create_features(self):
        whole = pd.concat([train_transaction['R_emaildomain'], test_transaction['R_emaildomain']], axis=0).to_frame()
        whole['r_emaildomain_top'] = whole['R_emaildomain'].apply(lambda x: x.split(".")[0] if type(x) == str else x)
        le = {v: i for i, v in enumerate(whole['r_emaildomain_top'].unique())}
        whole['r_emaildomain_top'] = whole['r_emaildomain_top'].apply(lambda x: le[x] if type(x) == str else x)
        self.train['r_emaildomain_top'] = whole['r_emaildomain_top'][:len(train_transaction)]
        self.test['r_emaildomain_top'] = whole['r_emaildomain_top'][len(train_transaction):]


class r_emaildomain_last(Feature):
    def create_features(self):
        whole = pd.concat([train_transaction['R_emaildomain'], test_transaction['R_emaildomain']], axis=0).to_frame()
        whole['r_emaildomain_last'] = whole['R_emaildomain'].apply(lambda x: x.split(".")[-1] if type(x) == str else x)
        le = {v: i for i, v in enumerate(whole['r_emaildomain_last'].unique())}
        whole['r_emaildomain_last'] = whole['r_emaildomain_last'].apply(lambda x: le[x] if type(x) == str else x)
        self.train['r_emaildomain_last'] = whole['r_emaildomain_last'][:len(train_transaction)]
        self.test['r_emaildomain_last'] = whole['r_emaildomain_last'][len(train_transaction):]


class c1(Feature):
    def create_features(self):
        self.train['C1'] = train_transaction['C1']
        self.test['C1'] = test_transaction['C1']


class c2(Feature):
    def create_features(self):
        self.train['C2'] = train_transaction['C2']
        self.test['C2'] = test_transaction['C2']


class c3(Feature):
    def create_features(self):
        self.train['C3'] = train_transaction['C3']
        self.test['C3'] = test_transaction['C3']


class c4(Feature):
    def create_features(self):
        self.train['C4'] = train_transaction['C4']
        self.test['C4'] = test_transaction['C4']


class c5(Feature):
    def create_features(self):
        self.train['C5'] = train_transaction['C5']
        self.test['C5'] = test_transaction['C5']


class c6(Feature):
    def create_features(self):
        self.train['C6'] = train_transaction['C6']
        self.test['C6'] = test_transaction['C6']


class c7(Feature):
    def create_features(self):
        self.train['C7'] = train_transaction['C7']
        self.test['C7'] = test_transaction['C7']


class c8(Feature):
    def create_features(self):
        self.train['C8'] = train_transaction['C8']
        self.test['C8'] = test_transaction['C8']


class c9(Feature):
    def create_features(self):
        self.train['C9'] = train_transaction['C9']
        self.test['C9'] = test_transaction['C9']


class c10(Feature):
    def create_features(self):
        self.train['C10'] = train_transaction['C10']
        self.test['C10'] = test_transaction['C10']


class c11(Feature):
    def create_features(self):
        self.train['C11'] = train_transaction['C11']
        self.test['C11'] = test_transaction['C11']


class c12(Feature):
    def create_features(self):
        self.train['C12'] = train_transaction['C12']
        self.test['C12'] = test_transaction['C12']


class c13(Feature):
    def create_features(self):
        self.train['C13'] = train_transaction['C13']
        self.test['C13'] = test_transaction['C13']


class c14(Feature):
    def create_features(self):
        self.train['C14'] = train_transaction['C14']
        self.test['C14'] = test_transaction['C14']


class sum_c(Feature):
    def create_features(self):
        feats = [f'C{i}' for i in range(1, 15)]
        self.train['sum_c'] = np.sum(train_transaction[feats].values, axis=1)
        self.test['sum_c'] = np.sum(test_transaction[feats].values, axis=1)


class d1(Feature):
    def create_features(self):
        self.train['D1'] = train_transaction['D1']
        self.test['D1'] = test_transaction['D1']


class d2(Feature):
    def create_features(self):
        self.train['D2'] = train_transaction['D2']
        self.test['D2'] = test_transaction['D2']


class d3(Feature):
    def create_features(self):
        self.train['D3'] = train_transaction['D3']
        self.test['D3'] = test_transaction['D3']


class d4(Feature):
    def create_features(self):
        self.train['D4'] = train_transaction['D4']
        self.test['D4'] = test_transaction['D4']


class d5(Feature):
    def create_features(self):
        self.train['D5'] = train_transaction['D5']
        self.test['D5'] = test_transaction['D5']


class d6(Feature):
    def create_features(self):
        self.train['D6'] = train_transaction['D6']
        self.test['D6'] = test_transaction['D6']


class d7(Feature):
    def create_features(self):
        self.train['D7'] = train_transaction['D7']
        self.test['D7'] = test_transaction['D7']


class d8(Feature):
    def create_features(self):
        self.train['D8'] = train_transaction['D8']
        self.test['D8'] = test_transaction['D8']


class d9(Feature):
    def create_features(self):
        self.train['D9'] = train_transaction['D9']
        self.test['D9'] = test_transaction['D9']


class d10(Feature):
    def create_features(self):
        self.train['D10'] = train_transaction['D10']
        self.test['D10'] = test_transaction['D10']


class d11(Feature):
    def create_features(self):
        self.train['D11'] = train_transaction['D11']
        self.test['D11'] = test_transaction['D11']


class d12(Feature):
    def create_features(self):
        self.train['D12'] = train_transaction['D12']
        self.test['D12'] = test_transaction['D12']


class d13(Feature):
    def create_features(self):
        self.train['D13'] = train_transaction['D13']
        self.test['D13'] = test_transaction['D13']


class d14(Feature):
    def create_features(self):
        self.train['D14'] = train_transaction['D14']
        self.test['D14'] = test_transaction['D14']


class d15(Feature):
    def create_features(self):
        self.train['D15'] = train_transaction['D15']
        self.test['D15'] = test_transaction['D15']


class m1(Feature):
    def create_features(self):
        whole = pd.concat([train_transaction['M1'], test_transaction['M1']], axis=0).to_frame()
        le = {v: i for i, v in enumerate(whole['M1'].unique())}
        self.train['M1'] = train_transaction['M1'].apply(lambda x: le[x] if type(x) == str else x)
        self.test['M1'] = test_transaction['M1'].apply(lambda x: le[x] if type(x) == str else x)


class m2(Feature):
    def create_features(self):
        whole = pd.concat([train_transaction['M2'], test_transaction['M2']], axis=0).to_frame()
        le = {v: i for i, v in enumerate(whole['M2'].unique())}
        self.train['M2'] = train_transaction['M2'].apply(lambda x: le[x] if type(x) == str else x)
        self.test['M2'] = test_transaction['M2'].apply(lambda x: le[x] if type(x) == str else x)


class m3(Feature):
    def create_features(self):
        whole = pd.concat([train_transaction['M3'], test_transaction['M3']], axis=0).to_frame()
        le = {v: i for i, v in enumerate(whole['M3'].unique())}
        self.train['M3'] = train_transaction['M3'].apply(lambda x: le[x] if type(x) == str else x)
        self.test['M3'] = test_transaction['M3'].apply(lambda x: le[x] if type(x) == str else x)


class m4(Feature):
    def create_features(self):
        whole = pd.concat([train_transaction['M4'], test_transaction['M4']], axis=0).to_frame()
        le = {v: i for i, v in enumerate(whole['M4'].unique())}
        self.train['M4'] = train_transaction['M4'].apply(lambda x: le[x] if type(x) == str else x)
        self.test['M4'] = test_transaction['M4'].apply(lambda x: le[x] if type(x) == str else x)


class m5(Feature):
    def create_features(self):
        whole = pd.concat([train_transaction['M5'], test_transaction['M5']], axis=0).to_frame()
        le = {v: i for i, v in enumerate(whole['M5'].unique())}
        self.train['M5'] = train_transaction['M5'].apply(lambda x: le[x] if type(x) == str else x)
        self.test['M5'] = test_transaction['M5'].apply(lambda x: le[x] if type(x) == str else x)


class m6(Feature):
    def create_features(self):
        whole = pd.concat([train_transaction['M6'], test_transaction['M6']], axis=0).to_frame()
        le = {v: i for i, v in enumerate(whole['M6'].unique())}
        self.train['M6'] = train_transaction['M6'].apply(lambda x: le[x] if type(x) == str else x)
        self.test['M6'] = test_transaction['M6'].apply(lambda x: le[x] if type(x) == str else x)


class m7(Feature):
    def create_features(self):
        whole = pd.concat([train_transaction['M7'], test_transaction['M7']], axis=0).to_frame()
        le = {v: i for i, v in enumerate(whole['M7'].unique())}
        self.train['M7'] = train_transaction['M7'].apply(lambda x: le[x] if type(x) == str else x)
        self.test['M7'] = test_transaction['M7'].apply(lambda x: le[x] if type(x) == str else x)


class m8(Feature):
    def create_features(self):
        whole = pd.concat([train_transaction['M8'], test_transaction['M8']], axis=0).to_frame()
        le = {v: i for i, v in enumerate(whole['M8'].unique())}
        self.train['M8'] = train_transaction['M8'].apply(lambda x: le[x] if type(x) == str else x)
        self.test['M8'] = test_transaction['M8'].apply(lambda x: le[x] if type(x) == str else x)


class m9(Feature):
    def create_features(self):
        whole = pd.concat([train_transaction['M9'], test_transaction['M9']], axis=0).to_frame()
        le = {v: i for i, v in enumerate(whole['M9'].unique())}
        self.train['M9'] = train_transaction['M9'].apply(lambda x: le[x] if type(x) == str else x)
        self.test['M9'] = test_transaction['M9'].apply(lambda x: le[x] if type(x) == str else x)


class v1(Feature):
    def create_features(self):
        self.train['V1'] = train_transaction['V1']
        self.test['V1'] = test_transaction['V1']


class v2(Feature):
    def create_features(self):
        self.train['V2'] = train_transaction['V2']
        self.test['V2'] = test_transaction['V2']


class v3(Feature):
    def create_features(self):
        self.train['V3'] = train_transaction['V3']
        self.test['V3'] = test_transaction['V3']


class v4(Feature):
    def create_features(self):
        self.train['V4'] = train_transaction['V4']
        self.test['V4'] = test_transaction['V4']


class v5(Feature):
    def create_features(self):
        self.train['V5'] = train_transaction['V5']
        self.test['V5'] = test_transaction['V5']


class v6(Feature):
    def create_features(self):
        self.train['V6'] = train_transaction['V6']
        self.test['V6'] = test_transaction['V6']


class v7(Feature):
    def create_features(self):
        self.train['V7'] = train_transaction['V7']
        self.test['V7'] = test_transaction['V7']


class v8(Feature):
    def create_features(self):
        self.train['V8'] = train_transaction['V8']
        self.test['V8'] = test_transaction['V8']


class v9(Feature):
    def create_features(self):
        self.train['V9'] = train_transaction['V9']
        self.test['V9'] = test_transaction['V9']


class v10(Feature):
    def create_features(self):
        self.train['V10'] = train_transaction['V10']
        self.test['V10'] = test_transaction['V10']


class v11(Feature):
    def create_features(self):
        self.train['V11'] = train_transaction['V11']
        self.test['V11'] = test_transaction['V11']


class v12(Feature):
    def create_features(self):
        self.train['V12'] = train_transaction['V12']
        self.test['V12'] = test_transaction['V12']


class v13(Feature):
    def create_features(self):
        self.train['V13'] = train_transaction['V13']
        self.test['V13'] = test_transaction['V13']


class v14(Feature):
    def create_features(self):
        self.train['V14'] = train_transaction['V14']
        self.test['V14'] = test_transaction['V14']


class v15(Feature):
    def create_features(self):
        self.train['V15'] = train_transaction['V15']
        self.test['V15'] = test_transaction['V15']


class v16(Feature):
    def create_features(self):
        self.train['V16'] = train_transaction['V16']
        self.test['V16'] = test_transaction['V16']


class v17(Feature):
    def create_features(self):
        self.train['V17'] = train_transaction['V17']
        self.test['V17'] = test_transaction['V17']


class v18(Feature):
    def create_features(self):
        self.train['V18'] = train_transaction['V18']
        self.test['V18'] = test_transaction['V18']


class v19(Feature):
    def create_features(self):
        self.train['V19'] = train_transaction['V19']
        self.test['V19'] = test_transaction['V19']


class v20(Feature):
    def create_features(self):
        self.train['V20'] = train_transaction['V20']
        self.test['V20'] = test_transaction['V20']


class v21(Feature):
    def create_features(self):
        self.train['V21'] = train_transaction['V21']
        self.test['V21'] = test_transaction['V21']


class v22(Feature):
    def create_features(self):
        self.train['V22'] = train_transaction['V22']
        self.test['V22'] = test_transaction['V22']


class v23(Feature):
    def create_features(self):
        self.train['V23'] = train_transaction['V23']
        self.test['V23'] = test_transaction['V23']


class v24(Feature):
    def create_features(self):
        self.train['V24'] = train_transaction['V24']
        self.test['V24'] = test_transaction['V24']


class v25(Feature):
    def create_features(self):
        self.train['V25'] = train_transaction['V25']
        self.test['V25'] = test_transaction['V25']


class v26(Feature):
    def create_features(self):
        self.train['V26'] = train_transaction['V26']
        self.test['V26'] = test_transaction['V26']


class v27(Feature):
    def create_features(self):
        self.train['V27'] = train_transaction['V27']
        self.test['V27'] = test_transaction['V27']


class v28(Feature):
    def create_features(self):
        self.train['V28'] = train_transaction['V28']
        self.test['V28'] = test_transaction['V28']


class v29(Feature):
    def create_features(self):
        self.train['V29'] = train_transaction['V29']
        self.test['V29'] = test_transaction['V29']


class v30(Feature):
    def create_features(self):
        self.train['V30'] = train_transaction['V30']
        self.test['V30'] = test_transaction['V30']


class v31(Feature):
    def create_features(self):
        self.train['V31'] = train_transaction['V31']
        self.test['V31'] = test_transaction['V31']


class v32(Feature):
    def create_features(self):
        self.train['V32'] = train_transaction['V32']
        self.test['V32'] = test_transaction['V32']


class v33(Feature):
    def create_features(self):
        self.train['V33'] = train_transaction['V33']
        self.test['V33'] = test_transaction['V33']


class v34(Feature):
    def create_features(self):
        self.train['V34'] = train_transaction['V34']
        self.test['V34'] = test_transaction['V34']


class v35(Feature):
    def create_features(self):
        self.train['V35'] = train_transaction['V35']
        self.test['V35'] = test_transaction['V35']


class v36(Feature):
    def create_features(self):
        self.train['V36'] = train_transaction['V36']
        self.test['V36'] = test_transaction['V36']


class v37(Feature):
    def create_features(self):
        self.train['V37'] = train_transaction['V37']
        self.test['V37'] = test_transaction['V37']


class v38(Feature):
    def create_features(self):
        self.train['V38'] = train_transaction['V38']
        self.test['V38'] = test_transaction['V38']


class v39(Feature):
    def create_features(self):
        self.train['V39'] = train_transaction['V39']
        self.test['V39'] = test_transaction['V39']


class v40(Feature):
    def create_features(self):
        self.train['V40'] = train_transaction['V40']
        self.test['V40'] = test_transaction['V40']


class v41(Feature):
    def create_features(self):
        self.train['V41'] = train_transaction['V41']
        self.test['V41'] = test_transaction['V41']


class v42(Feature):
    def create_features(self):
        self.train['V42'] = train_transaction['V42']
        self.test['V42'] = test_transaction['V42']


class v43(Feature):
    def create_features(self):
        self.train['V43'] = train_transaction['V43']
        self.test['V43'] = test_transaction['V43']


class v44(Feature):
    def create_features(self):
        self.train['V44'] = train_transaction['V44']
        self.test['V44'] = test_transaction['V44']


class v45(Feature):
    def create_features(self):
        self.train['V45'] = train_transaction['V45']
        self.test['V45'] = test_transaction['V45']


class v46(Feature):
    def create_features(self):
        self.train['V46'] = train_transaction['V46']
        self.test['V46'] = test_transaction['V46']


class v47(Feature):
    def create_features(self):
        self.train['V47'] = train_transaction['V47']
        self.test['V47'] = test_transaction['V47']


class v48(Feature):
    def create_features(self):
        self.train['V48'] = train_transaction['V48']
        self.test['V48'] = test_transaction['V48']


class v49(Feature):
    def create_features(self):
        self.train['V49'] = train_transaction['V49']
        self.test['V49'] = test_transaction['V49']


class v50(Feature):
    def create_features(self):
        self.train['V50'] = train_transaction['V50']
        self.test['V50'] = test_transaction['V50']


class v51(Feature):
    def create_features(self):
        self.train['V51'] = train_transaction['V51']
        self.test['V51'] = test_transaction['V51']


class v52(Feature):
    def create_features(self):
        self.train['V52'] = train_transaction['V52']
        self.test['V52'] = test_transaction['V52']


class v53(Feature):
    def create_features(self):
        self.train['V53'] = train_transaction['V53']
        self.test['V53'] = test_transaction['V53']


class v54(Feature):
    def create_features(self):
        self.train['V54'] = train_transaction['V54']
        self.test['V54'] = test_transaction['V54']


class v55(Feature):
    def create_features(self):
        self.train['V55'] = train_transaction['V55']
        self.test['V55'] = test_transaction['V55']


class v56(Feature):
    def create_features(self):
        self.train['V56'] = train_transaction['V56']
        self.test['V56'] = test_transaction['V56']


class v57(Feature):
    def create_features(self):
        self.train['V57'] = train_transaction['V57']
        self.test['V57'] = test_transaction['V57']


class v58(Feature):
    def create_features(self):
        self.train['V58'] = train_transaction['V58']
        self.test['V58'] = test_transaction['V58']


class v59(Feature):
    def create_features(self):
        self.train['V59'] = train_transaction['V59']
        self.test['V59'] = test_transaction['V59']


class v60(Feature):
    def create_features(self):
        self.train['V60'] = train_transaction['V60']
        self.test['V60'] = test_transaction['V60']


class v61(Feature):
    def create_features(self):
        self.train['V61'] = train_transaction['V61']
        self.test['V61'] = test_transaction['V61']


class v62(Feature):
    def create_features(self):
        self.train['V62'] = train_transaction['V62']
        self.test['V62'] = test_transaction['V62']


class v63(Feature):
    def create_features(self):
        self.train['V63'] = train_transaction['V63']
        self.test['V63'] = test_transaction['V63']


class v64(Feature):
    def create_features(self):
        self.train['V64'] = train_transaction['V64']
        self.test['V64'] = test_transaction['V64']


class v65(Feature):
    def create_features(self):
        self.train['V65'] = train_transaction['V65']
        self.test['V65'] = test_transaction['V65']


class v66(Feature):
    def create_features(self):
        self.train['V66'] = train_transaction['V66']
        self.test['V66'] = test_transaction['V66']


class v67(Feature):
    def create_features(self):
        self.train['V67'] = train_transaction['V67']
        self.test['V67'] = test_transaction['V67']


class v68(Feature):
    def create_features(self):
        self.train['V68'] = train_transaction['V68']
        self.test['V68'] = test_transaction['V68']


class v69(Feature):
    def create_features(self):
        self.train['V69'] = train_transaction['V69']
        self.test['V69'] = test_transaction['V69']


class v70(Feature):
    def create_features(self):
        self.train['V70'] = train_transaction['V70']
        self.test['V70'] = test_transaction['V70']


class v71(Feature):
    def create_features(self):
        self.train['V71'] = train_transaction['V71']
        self.test['V71'] = test_transaction['V71']


class v72(Feature):
    def create_features(self):
        self.train['V72'] = train_transaction['V72']
        self.test['V72'] = test_transaction['V72']


class v73(Feature):
    def create_features(self):
        self.train['V73'] = train_transaction['V73']
        self.test['V73'] = test_transaction['V73']


class v74(Feature):
    def create_features(self):
        self.train['V74'] = train_transaction['V74']
        self.test['V74'] = test_transaction['V74']


class v75(Feature):
    def create_features(self):
        self.train['V75'] = train_transaction['V75']
        self.test['V75'] = test_transaction['V75']


class v76(Feature):
    def create_features(self):
        self.train['V76'] = train_transaction['V76']
        self.test['V76'] = test_transaction['V76']


class v77(Feature):
    def create_features(self):
        self.train['V77'] = train_transaction['V77']
        self.test['V77'] = test_transaction['V77']


class v78(Feature):
    def create_features(self):
        self.train['V78'] = train_transaction['V78']
        self.test['V78'] = test_transaction['V78']


class v79(Feature):
    def create_features(self):
        self.train['V79'] = train_transaction['V79']
        self.test['V79'] = test_transaction['V79']


class v80(Feature):
    def create_features(self):
        self.train['V80'] = train_transaction['V80']
        self.test['V80'] = test_transaction['V80']


class v81(Feature):
    def create_features(self):
        self.train['V81'] = train_transaction['V81']
        self.test['V81'] = test_transaction['V81']


class v82(Feature):
    def create_features(self):
        self.train['V82'] = train_transaction['V82']
        self.test['V82'] = test_transaction['V82']


class v83(Feature):
    def create_features(self):
        self.train['V83'] = train_transaction['V83']
        self.test['V83'] = test_transaction['V83']


class v84(Feature):
    def create_features(self):
        self.train['V84'] = train_transaction['V84']
        self.test['V84'] = test_transaction['V84']


class v85(Feature):
    def create_features(self):
        self.train['V85'] = train_transaction['V85']
        self.test['V85'] = test_transaction['V85']


class v86(Feature):
    def create_features(self):
        self.train['V86'] = train_transaction['V86']
        self.test['V86'] = test_transaction['V86']


class v87(Feature):
    def create_features(self):
        self.train['V87'] = train_transaction['V87']
        self.test['V87'] = test_transaction['V87']


class v88(Feature):
    def create_features(self):
        self.train['V88'] = train_transaction['V88']
        self.test['V88'] = test_transaction['V88']


class v89(Feature):
    def create_features(self):
        self.train['V89'] = train_transaction['V89']
        self.test['V89'] = test_transaction['V89']


class v90(Feature):
    def create_features(self):
        self.train['V90'] = train_transaction['V90']
        self.test['V90'] = test_transaction['V90']


class v91(Feature):
    def create_features(self):
        self.train['V91'] = train_transaction['V91']
        self.test['V91'] = test_transaction['V91']


class v92(Feature):
    def create_features(self):
        self.train['V92'] = train_transaction['V92']
        self.test['V92'] = test_transaction['V92']


class v93(Feature):
    def create_features(self):
        self.train['V93'] = train_transaction['V93']
        self.test['V93'] = test_transaction['V93']


class v94(Feature):
    def create_features(self):
        self.train['V94'] = train_transaction['V94']
        self.test['V94'] = test_transaction['V94']


class v95(Feature):
    def create_features(self):
        self.train['V95'] = train_transaction['V95']
        self.test['V95'] = test_transaction['V95']


class v96(Feature):
    def create_features(self):
        self.train['V96'] = train_transaction['V96']
        self.test['V96'] = test_transaction['V96']


class v97(Feature):
    def create_features(self):
        self.train['V97'] = train_transaction['V97']
        self.test['V97'] = test_transaction['V97']


class v98(Feature):
    def create_features(self):
        self.train['V98'] = train_transaction['V98']
        self.test['V98'] = test_transaction['V98']


class v99(Feature):
    def create_features(self):
        self.train['V99'] = train_transaction['V99']
        self.test['V99'] = test_transaction['V99']


class v100(Feature):
    def create_features(self):
        self.train['V100'] = train_transaction['V100']
        self.test['V100'] = test_transaction['V100']


class v101(Feature):
    def create_features(self):
        self.train['V101'] = train_transaction['V101']
        self.test['V101'] = test_transaction['V101']


class v102(Feature):
    def create_features(self):
        self.train['V102'] = train_transaction['V102']
        self.test['V102'] = test_transaction['V102']


class v103(Feature):
    def create_features(self):
        self.train['V103'] = train_transaction['V103']
        self.test['V103'] = test_transaction['V103']


class v104(Feature):
    def create_features(self):
        self.train['V104'] = train_transaction['V104']
        self.test['V104'] = test_transaction['V104']


class v105(Feature):
    def create_features(self):
        self.train['V105'] = train_transaction['V105']
        self.test['V105'] = test_transaction['V105']


class v106(Feature):
    def create_features(self):
        self.train['V106'] = train_transaction['V106']
        self.test['V106'] = test_transaction['V106']


class v107(Feature):
    def create_features(self):
        self.train['V107'] = train_transaction['V107']
        self.test['V107'] = test_transaction['V107']


class v108(Feature):
    def create_features(self):
        self.train['V108'] = train_transaction['V108']
        self.test['V108'] = test_transaction['V108']


class v109(Feature):
    def create_features(self):
        self.train['V109'] = train_transaction['V109']
        self.test['V109'] = test_transaction['V109']


class v110(Feature):
    def create_features(self):
        self.train['V110'] = train_transaction['V110']
        self.test['V110'] = test_transaction['V110']


class v111(Feature):
    def create_features(self):
        self.train['V111'] = train_transaction['V111']
        self.test['V111'] = test_transaction['V111']


class v112(Feature):
    def create_features(self):
        self.train['V112'] = train_transaction['V112']
        self.test['V112'] = test_transaction['V112']


class v113(Feature):
    def create_features(self):
        self.train['V113'] = train_transaction['V113']
        self.test['V113'] = test_transaction['V113']


class v114(Feature):
    def create_features(self):
        self.train['V114'] = train_transaction['V114']
        self.test['V114'] = test_transaction['V114']


class v115(Feature):
    def create_features(self):
        self.train['V115'] = train_transaction['V115']
        self.test['V115'] = test_transaction['V115']


class v116(Feature):
    def create_features(self):
        self.train['V116'] = train_transaction['V116']
        self.test['V116'] = test_transaction['V116']


class v117(Feature):
    def create_features(self):
        self.train['V117'] = train_transaction['V117']
        self.test['V117'] = test_transaction['V117']


class v118(Feature):
    def create_features(self):
        self.train['V118'] = train_transaction['V118']
        self.test['V118'] = test_transaction['V118']


class v119(Feature):
    def create_features(self):
        self.train['V119'] = train_transaction['V119']
        self.test['V119'] = test_transaction['V119']


class v120(Feature):
    def create_features(self):
        self.train['V120'] = train_transaction['V120']
        self.test['V120'] = test_transaction['V120']


class v121(Feature):
    def create_features(self):
        self.train['V121'] = train_transaction['V121']
        self.test['V121'] = test_transaction['V121']


class v122(Feature):
    def create_features(self):
        self.train['V122'] = train_transaction['V122']
        self.test['V122'] = test_transaction['V122']


class v123(Feature):
    def create_features(self):
        self.train['V123'] = train_transaction['V123']
        self.test['V123'] = test_transaction['V123']


class v124(Feature):
    def create_features(self):
        self.train['V124'] = train_transaction['V124']
        self.test['V124'] = test_transaction['V124']


class v125(Feature):
    def create_features(self):
        self.train['V125'] = train_transaction['V125']
        self.test['V125'] = test_transaction['V125']


class v126(Feature):
    def create_features(self):
        self.train['V126'] = train_transaction['V126']
        self.test['V126'] = test_transaction['V126']


class v127(Feature):
    def create_features(self):
        self.train['V127'] = train_transaction['V127']
        self.test['V127'] = test_transaction['V127']


class v128(Feature):
    def create_features(self):
        self.train['V128'] = train_transaction['V128']
        self.test['V128'] = test_transaction['V128']


class v129(Feature):
    def create_features(self):
        self.train['V129'] = train_transaction['V129']
        self.test['V129'] = test_transaction['V129']


class v130(Feature):
    def create_features(self):
        self.train['V130'] = train_transaction['V130']
        self.test['V130'] = test_transaction['V130']


class v131(Feature):
    def create_features(self):
        self.train['V131'] = train_transaction['V131']
        self.test['V131'] = test_transaction['V131']


class v132(Feature):
    def create_features(self):
        self.train['V132'] = train_transaction['V132']
        self.test['V132'] = test_transaction['V132']


class v133(Feature):
    def create_features(self):
        self.train['V133'] = train_transaction['V133']
        self.test['V133'] = test_transaction['V133']


class v134(Feature):
    def create_features(self):
        self.train['V134'] = train_transaction['V134']
        self.test['V134'] = test_transaction['V134']


class v135(Feature):
    def create_features(self):
        self.train['V135'] = train_transaction['V135']
        self.test['V135'] = test_transaction['V135']


class v136(Feature):
    def create_features(self):
        self.train['V136'] = train_transaction['V136']
        self.test['V136'] = test_transaction['V136']


class v137(Feature):
    def create_features(self):
        self.train['V137'] = train_transaction['V137']
        self.test['V137'] = test_transaction['V137']


class v138(Feature):
    def create_features(self):
        self.train['V138'] = train_transaction['V138']
        self.test['V138'] = test_transaction['V138']


class v139(Feature):
    def create_features(self):
        self.train['V139'] = train_transaction['V139']
        self.test['V139'] = test_transaction['V139']


class v140(Feature):
    def create_features(self):
        self.train['V140'] = train_transaction['V140']
        self.test['V140'] = test_transaction['V140']


class v141(Feature):
    def create_features(self):
        self.train['V141'] = train_transaction['V141']
        self.test['V141'] = test_transaction['V141']


class v142(Feature):
    def create_features(self):
        self.train['V142'] = train_transaction['V142']
        self.test['V142'] = test_transaction['V142']


class v143(Feature):
    def create_features(self):
        self.train['V143'] = train_transaction['V143']
        self.test['V143'] = test_transaction['V143']


class v144(Feature):
    def create_features(self):
        self.train['V144'] = train_transaction['V144']
        self.test['V144'] = test_transaction['V144']


class v145(Feature):
    def create_features(self):
        self.train['V145'] = train_transaction['V145']
        self.test['V145'] = test_transaction['V145']


class v146(Feature):
    def create_features(self):
        self.train['V146'] = train_transaction['V146']
        self.test['V146'] = test_transaction['V146']


class v147(Feature):
    def create_features(self):
        self.train['V147'] = train_transaction['V147']
        self.test['V147'] = test_transaction['V147']


class v148(Feature):
    def create_features(self):
        self.train['V148'] = train_transaction['V148']
        self.test['V148'] = test_transaction['V148']


class v149(Feature):
    def create_features(self):
        self.train['V149'] = train_transaction['V149']
        self.test['V149'] = test_transaction['V149']


class v150(Feature):
    def create_features(self):
        self.train['V150'] = train_transaction['V150']
        self.test['V150'] = test_transaction['V150']


class v151(Feature):
    def create_features(self):
        self.train['V151'] = train_transaction['V151']
        self.test['V151'] = test_transaction['V151']


class v152(Feature):
    def create_features(self):
        self.train['V152'] = train_transaction['V152']
        self.test['V152'] = test_transaction['V152']


class v153(Feature):
    def create_features(self):
        self.train['V153'] = train_transaction['V153']
        self.test['V153'] = test_transaction['V153']


class v154(Feature):
    def create_features(self):
        self.train['V154'] = train_transaction['V154']
        self.test['V154'] = test_transaction['V154']


class v155(Feature):
    def create_features(self):
        self.train['V155'] = train_transaction['V155']
        self.test['V155'] = test_transaction['V155']


class v156(Feature):
    def create_features(self):
        self.train['V156'] = train_transaction['V156']
        self.test['V156'] = test_transaction['V156']


class v157(Feature):
    def create_features(self):
        self.train['V157'] = train_transaction['V157']
        self.test['V157'] = test_transaction['V157']


class v158(Feature):
    def create_features(self):
        self.train['V158'] = train_transaction['V158']
        self.test['V158'] = test_transaction['V158']


class v159(Feature):
    def create_features(self):
        self.train['V159'] = train_transaction['V159']
        self.test['V159'] = test_transaction['V159']


class v160(Feature):
    def create_features(self):
        self.train['V160'] = train_transaction['V160']
        self.test['V160'] = test_transaction['V160']


class v161(Feature):
    def create_features(self):
        self.train['V161'] = train_transaction['V161']
        self.test['V161'] = test_transaction['V161']


class v162(Feature):
    def create_features(self):
        self.train['V162'] = train_transaction['V162']
        self.test['V162'] = test_transaction['V162']


class v163(Feature):
    def create_features(self):
        self.train['V163'] = train_transaction['V163']
        self.test['V163'] = test_transaction['V163']


class v164(Feature):
    def create_features(self):
        self.train['V164'] = train_transaction['V164']
        self.test['V164'] = test_transaction['V164']


class v165(Feature):
    def create_features(self):
        self.train['V165'] = train_transaction['V165']
        self.test['V165'] = test_transaction['V165']


class v166(Feature):
    def create_features(self):
        self.train['V166'] = train_transaction['V166']
        self.test['V166'] = test_transaction['V166']


class v167(Feature):
    def create_features(self):
        self.train['V167'] = train_transaction['V167']
        self.test['V167'] = test_transaction['V167']


class v168(Feature):
    def create_features(self):
        self.train['V168'] = train_transaction['V168']
        self.test['V168'] = test_transaction['V168']


class v169(Feature):
    def create_features(self):
        self.train['V169'] = train_transaction['V169']
        self.test['V169'] = test_transaction['V169']


class v170(Feature):
    def create_features(self):
        self.train['V170'] = train_transaction['V170']
        self.test['V170'] = test_transaction['V170']


class v171(Feature):
    def create_features(self):
        self.train['V171'] = train_transaction['V171']
        self.test['V171'] = test_transaction['V171']


class v172(Feature):
    def create_features(self):
        self.train['V172'] = train_transaction['V172']
        self.test['V172'] = test_transaction['V172']


class v173(Feature):
    def create_features(self):
        self.train['V173'] = train_transaction['V173']
        self.test['V173'] = test_transaction['V173']


class v174(Feature):
    def create_features(self):
        self.train['V174'] = train_transaction['V174']
        self.test['V174'] = test_transaction['V174']


class v175(Feature):
    def create_features(self):
        self.train['V175'] = train_transaction['V175']
        self.test['V175'] = test_transaction['V175']


class v176(Feature):
    def create_features(self):
        self.train['V176'] = train_transaction['V176']
        self.test['V176'] = test_transaction['V176']


class v177(Feature):
    def create_features(self):
        self.train['V177'] = train_transaction['V177']
        self.test['V177'] = test_transaction['V177']


class v178(Feature):
    def create_features(self):
        self.train['V178'] = train_transaction['V178']
        self.test['V178'] = test_transaction['V178']


class v179(Feature):
    def create_features(self):
        self.train['V179'] = train_transaction['V179']
        self.test['V179'] = test_transaction['V179']


class v180(Feature):
    def create_features(self):
        self.train['V180'] = train_transaction['V180']
        self.test['V180'] = test_transaction['V180']


class v181(Feature):
    def create_features(self):
        self.train['V181'] = train_transaction['V181']
        self.test['V181'] = test_transaction['V181']


class v182(Feature):
    def create_features(self):
        self.train['V182'] = train_transaction['V182']
        self.test['V182'] = test_transaction['V182']


class v183(Feature):
    def create_features(self):
        self.train['V183'] = train_transaction['V183']
        self.test['V183'] = test_transaction['V183']


class v184(Feature):
    def create_features(self):
        self.train['V184'] = train_transaction['V184']
        self.test['V184'] = test_transaction['V184']


class v185(Feature):
    def create_features(self):
        self.train['V185'] = train_transaction['V185']
        self.test['V185'] = test_transaction['V185']


class v186(Feature):
    def create_features(self):
        self.train['V186'] = train_transaction['V186']
        self.test['V186'] = test_transaction['V186']


class v187(Feature):
    def create_features(self):
        self.train['V187'] = train_transaction['V187']
        self.test['V187'] = test_transaction['V187']


class v188(Feature):
    def create_features(self):
        self.train['V188'] = train_transaction['V188']
        self.test['V188'] = test_transaction['V188']


class v189(Feature):
    def create_features(self):
        self.train['V189'] = train_transaction['V189']
        self.test['V189'] = test_transaction['V189']


class v190(Feature):
    def create_features(self):
        self.train['V190'] = train_transaction['V190']
        self.test['V190'] = test_transaction['V190']


class v191(Feature):
    def create_features(self):
        self.train['V191'] = train_transaction['V191']
        self.test['V191'] = test_transaction['V191']


class v192(Feature):
    def create_features(self):
        self.train['V192'] = train_transaction['V192']
        self.test['V192'] = test_transaction['V192']


class v193(Feature):
    def create_features(self):
        self.train['V193'] = train_transaction['V193']
        self.test['V193'] = test_transaction['V193']


class v194(Feature):
    def create_features(self):
        self.train['V194'] = train_transaction['V194']
        self.test['V194'] = test_transaction['V194']


class v195(Feature):
    def create_features(self):
        self.train['V195'] = train_transaction['V195']
        self.test['V195'] = test_transaction['V195']


class v196(Feature):
    def create_features(self):
        self.train['V196'] = train_transaction['V196']
        self.test['V196'] = test_transaction['V196']


class v197(Feature):
    def create_features(self):
        self.train['V197'] = train_transaction['V197']
        self.test['V197'] = test_transaction['V197']


class v198(Feature):
    def create_features(self):
        self.train['V198'] = train_transaction['V198']
        self.test['V198'] = test_transaction['V198']


class v199(Feature):
    def create_features(self):
        self.train['V199'] = train_transaction['V199']
        self.test['V199'] = test_transaction['V199']


class v200(Feature):
    def create_features(self):
        self.train['V200'] = train_transaction['V200']
        self.test['V200'] = test_transaction['V200']


class v201(Feature):
    def create_features(self):
        self.train['V201'] = train_transaction['V201']
        self.test['V201'] = test_transaction['V201']


class v202(Feature):
    def create_features(self):
        self.train['V202'] = train_transaction['V202']
        self.test['V202'] = test_transaction['V202']


class v203(Feature):
    def create_features(self):
        self.train['V203'] = train_transaction['V203']
        self.test['V203'] = test_transaction['V203']


class v204(Feature):
    def create_features(self):
        self.train['V204'] = train_transaction['V204']
        self.test['V204'] = test_transaction['V204']


class v205(Feature):
    def create_features(self):
        self.train['V205'] = train_transaction['V205']
        self.test['V205'] = test_transaction['V205']


class v206(Feature):
    def create_features(self):
        self.train['V206'] = train_transaction['V206']
        self.test['V206'] = test_transaction['V206']


class v207(Feature):
    def create_features(self):
        self.train['V207'] = train_transaction['V207']
        self.test['V207'] = test_transaction['V207']


class v208(Feature):
    def create_features(self):
        self.train['V208'] = train_transaction['V208']
        self.test['V208'] = test_transaction['V208']


class v209(Feature):
    def create_features(self):
        self.train['V209'] = train_transaction['V209']
        self.test['V209'] = test_transaction['V209']


class v210(Feature):
    def create_features(self):
        self.train['V210'] = train_transaction['V210']
        self.test['V210'] = test_transaction['V210']


class v211(Feature):
    def create_features(self):
        self.train['V211'] = train_transaction['V211']
        self.test['V211'] = test_transaction['V211']


class v212(Feature):
    def create_features(self):
        self.train['V212'] = train_transaction['V212']
        self.test['V212'] = test_transaction['V212']


class v213(Feature):
    def create_features(self):
        self.train['V213'] = train_transaction['V213']
        self.test['V213'] = test_transaction['V213']


class v214(Feature):
    def create_features(self):
        self.train['V214'] = train_transaction['V214']
        self.test['V214'] = test_transaction['V214']


class v215(Feature):
    def create_features(self):
        self.train['V215'] = train_transaction['V215']
        self.test['V215'] = test_transaction['V215']


class v216(Feature):
    def create_features(self):
        self.train['V216'] = train_transaction['V216']
        self.test['V216'] = test_transaction['V216']


class v217(Feature):
    def create_features(self):
        self.train['V217'] = train_transaction['V217']
        self.test['V217'] = test_transaction['V217']


class v218(Feature):
    def create_features(self):
        self.train['V218'] = train_transaction['V218']
        self.test['V218'] = test_transaction['V218']


class v219(Feature):
    def create_features(self):
        self.train['V219'] = train_transaction['V219']
        self.test['V219'] = test_transaction['V219']


class v220(Feature):
    def create_features(self):
        self.train['V220'] = train_transaction['V220']
        self.test['V220'] = test_transaction['V220']


class v221(Feature):
    def create_features(self):
        self.train['V221'] = train_transaction['V221']
        self.test['V221'] = test_transaction['V221']


class v222(Feature):
    def create_features(self):
        self.train['V222'] = train_transaction['V222']
        self.test['V222'] = test_transaction['V222']


class v223(Feature):
    def create_features(self):
        self.train['V223'] = train_transaction['V223']
        self.test['V223'] = test_transaction['V223']


class v224(Feature):
    def create_features(self):
        self.train['V224'] = train_transaction['V224']
        self.test['V224'] = test_transaction['V224']


class v225(Feature):
    def create_features(self):
        self.train['V225'] = train_transaction['V225']
        self.test['V225'] = test_transaction['V225']


class v226(Feature):
    def create_features(self):
        self.train['V226'] = train_transaction['V226']
        self.test['V226'] = test_transaction['V226']


class v227(Feature):
    def create_features(self):
        self.train['V227'] = train_transaction['V227']
        self.test['V227'] = test_transaction['V227']


class v228(Feature):
    def create_features(self):
        self.train['V228'] = train_transaction['V228']
        self.test['V228'] = test_transaction['V228']


class v229(Feature):
    def create_features(self):
        self.train['V229'] = train_transaction['V229']
        self.test['V229'] = test_transaction['V229']


class v230(Feature):
    def create_features(self):
        self.train['V230'] = train_transaction['V230']
        self.test['V230'] = test_transaction['V230']


class v231(Feature):
    def create_features(self):
        self.train['V231'] = train_transaction['V231']
        self.test['V231'] = test_transaction['V231']


class v232(Feature):
    def create_features(self):
        self.train['V232'] = train_transaction['V232']
        self.test['V232'] = test_transaction['V232']


class v233(Feature):
    def create_features(self):
        self.train['V233'] = train_transaction['V233']
        self.test['V233'] = test_transaction['V233']


class v234(Feature):
    def create_features(self):
        self.train['V234'] = train_transaction['V234']
        self.test['V234'] = test_transaction['V234']


class v235(Feature):
    def create_features(self):
        self.train['V235'] = train_transaction['V235']
        self.test['V235'] = test_transaction['V235']


class v236(Feature):
    def create_features(self):
        self.train['V236'] = train_transaction['V236']
        self.test['V236'] = test_transaction['V236']


class v237(Feature):
    def create_features(self):
        self.train['V237'] = train_transaction['V237']
        self.test['V237'] = test_transaction['V237']


class v238(Feature):
    def create_features(self):
        self.train['V238'] = train_transaction['V238']
        self.test['V238'] = test_transaction['V238']


class v239(Feature):
    def create_features(self):
        self.train['V239'] = train_transaction['V239']
        self.test['V239'] = test_transaction['V239']


class v240(Feature):
    def create_features(self):
        self.train['V240'] = train_transaction['V240']
        self.test['V240'] = test_transaction['V240']


class v241(Feature):
    def create_features(self):
        self.train['V241'] = train_transaction['V241']
        self.test['V241'] = test_transaction['V241']


class v242(Feature):
    def create_features(self):
        self.train['V242'] = train_transaction['V242']
        self.test['V242'] = test_transaction['V242']


class v243(Feature):
    def create_features(self):
        self.train['V243'] = train_transaction['V243']
        self.test['V243'] = test_transaction['V243']


class v244(Feature):
    def create_features(self):
        self.train['V244'] = train_transaction['V244']
        self.test['V244'] = test_transaction['V244']


class v245(Feature):
    def create_features(self):
        self.train['V245'] = train_transaction['V245']
        self.test['V245'] = test_transaction['V245']


class v246(Feature):
    def create_features(self):
        self.train['V246'] = train_transaction['V246']
        self.test['V246'] = test_transaction['V246']


class v247(Feature):
    def create_features(self):
        self.train['V247'] = train_transaction['V247']
        self.test['V247'] = test_transaction['V247']


class v248(Feature):
    def create_features(self):
        self.train['V248'] = train_transaction['V248']
        self.test['V248'] = test_transaction['V248']


class v249(Feature):
    def create_features(self):
        self.train['V249'] = train_transaction['V249']
        self.test['V249'] = test_transaction['V249']


class v250(Feature):
    def create_features(self):
        self.train['V250'] = train_transaction['V250']
        self.test['V250'] = test_transaction['V250']


class v251(Feature):
    def create_features(self):
        self.train['V251'] = train_transaction['V251']
        self.test['V251'] = test_transaction['V251']


class v252(Feature):
    def create_features(self):
        self.train['V252'] = train_transaction['V252']
        self.test['V252'] = test_transaction['V252']


class v253(Feature):
    def create_features(self):
        self.train['V253'] = train_transaction['V253']
        self.test['V253'] = test_transaction['V253']


class v254(Feature):
    def create_features(self):
        self.train['V254'] = train_transaction['V254']
        self.test['V254'] = test_transaction['V254']


class v255(Feature):
    def create_features(self):
        self.train['V255'] = train_transaction['V255']
        self.test['V255'] = test_transaction['V255']


class v256(Feature):
    def create_features(self):
        self.train['V256'] = train_transaction['V256']
        self.test['V256'] = test_transaction['V256']


class v257(Feature):
    def create_features(self):
        self.train['V257'] = train_transaction['V257']
        self.test['V257'] = test_transaction['V257']


class v258(Feature):
    def create_features(self):
        self.train['V258'] = train_transaction['V258']
        self.test['V258'] = test_transaction['V258']


class v259(Feature):
    def create_features(self):
        self.train['V259'] = train_transaction['V259']
        self.test['V259'] = test_transaction['V259']


class v260(Feature):
    def create_features(self):
        self.train['V260'] = train_transaction['V260']
        self.test['V260'] = test_transaction['V260']


class v261(Feature):
    def create_features(self):
        self.train['V261'] = train_transaction['V261']
        self.test['V261'] = test_transaction['V261']


class v262(Feature):
    def create_features(self):
        self.train['V262'] = train_transaction['V262']
        self.test['V262'] = test_transaction['V262']


class v263(Feature):
    def create_features(self):
        self.train['V263'] = train_transaction['V263']
        self.test['V263'] = test_transaction['V263']


class v264(Feature):
    def create_features(self):
        self.train['V264'] = train_transaction['V264']
        self.test['V264'] = test_transaction['V264']


class v265(Feature):
    def create_features(self):
        self.train['V265'] = train_transaction['V265']
        self.test['V265'] = test_transaction['V265']


class v266(Feature):
    def create_features(self):
        self.train['V266'] = train_transaction['V266']
        self.test['V266'] = test_transaction['V266']


class v267(Feature):
    def create_features(self):
        self.train['V267'] = train_transaction['V267']
        self.test['V267'] = test_transaction['V267']


class v268(Feature):
    def create_features(self):
        self.train['V268'] = train_transaction['V268']
        self.test['V268'] = test_transaction['V268']


class v269(Feature):
    def create_features(self):
        self.train['V269'] = train_transaction['V269']
        self.test['V269'] = test_transaction['V269']


class v270(Feature):
    def create_features(self):
        self.train['V270'] = train_transaction['V270']
        self.test['V270'] = test_transaction['V270']


class v271(Feature):
    def create_features(self):
        self.train['V271'] = train_transaction['V271']
        self.test['V271'] = test_transaction['V271']


class v272(Feature):
    def create_features(self):
        self.train['V272'] = train_transaction['V272']
        self.test['V272'] = test_transaction['V272']


class v273(Feature):
    def create_features(self):
        self.train['V273'] = train_transaction['V273']
        self.test['V273'] = test_transaction['V273']


class v274(Feature):
    def create_features(self):
        self.train['V274'] = train_transaction['V274']
        self.test['V274'] = test_transaction['V274']


class v275(Feature):
    def create_features(self):
        self.train['V275'] = train_transaction['V275']
        self.test['V275'] = test_transaction['V275']


class v276(Feature):
    def create_features(self):
        self.train['V276'] = train_transaction['V276']
        self.test['V276'] = test_transaction['V276']


class v277(Feature):
    def create_features(self):
        self.train['V277'] = train_transaction['V277']
        self.test['V277'] = test_transaction['V277']


class v278(Feature):
    def create_features(self):
        self.train['V278'] = train_transaction['V278']
        self.test['V278'] = test_transaction['V278']


class v279(Feature):
    def create_features(self):
        self.train['V279'] = train_transaction['V279']
        self.test['V279'] = test_transaction['V279']


class v280(Feature):
    def create_features(self):
        self.train['V280'] = train_transaction['V280']
        self.test['V280'] = test_transaction['V280']


class v281(Feature):
    def create_features(self):
        self.train['V281'] = train_transaction['V281']
        self.test['V281'] = test_transaction['V281']


class v282(Feature):
    def create_features(self):
        self.train['V282'] = train_transaction['V282']
        self.test['V282'] = test_transaction['V282']


class v283(Feature):
    def create_features(self):
        self.train['V283'] = train_transaction['V283']
        self.test['V283'] = test_transaction['V283']


class v284(Feature):
    def create_features(self):
        self.train['V284'] = train_transaction['V284']
        self.test['V284'] = test_transaction['V284']


class v285(Feature):
    def create_features(self):
        self.train['V285'] = train_transaction['V285']
        self.test['V285'] = test_transaction['V285']


class v286(Feature):
    def create_features(self):
        self.train['V286'] = train_transaction['V286']
        self.test['V286'] = test_transaction['V286']


class v287(Feature):
    def create_features(self):
        self.train['V287'] = train_transaction['V287']
        self.test['V287'] = test_transaction['V287']


class v288(Feature):
    def create_features(self):
        self.train['V288'] = train_transaction['V288']
        self.test['V288'] = test_transaction['V288']


class v289(Feature):
    def create_features(self):
        self.train['V289'] = train_transaction['V289']
        self.test['V289'] = test_transaction['V289']


class v290(Feature):
    def create_features(self):
        self.train['V290'] = train_transaction['V290']
        self.test['V290'] = test_transaction['V290']


class v291(Feature):
    def create_features(self):
        self.train['V291'] = train_transaction['V291']
        self.test['V291'] = test_transaction['V291']


class v292(Feature):
    def create_features(self):
        self.train['V292'] = train_transaction['V292']
        self.test['V292'] = test_transaction['V292']


class v293(Feature):
    def create_features(self):
        self.train['V293'] = train_transaction['V293']
        self.test['V293'] = test_transaction['V293']


class v294(Feature):
    def create_features(self):
        self.train['V294'] = train_transaction['V294']
        self.test['V294'] = test_transaction['V294']


class v295(Feature):
    def create_features(self):
        self.train['V295'] = train_transaction['V295']
        self.test['V295'] = test_transaction['V295']


class v296(Feature):
    def create_features(self):
        self.train['V296'] = train_transaction['V296']
        self.test['V296'] = test_transaction['V296']


class v297(Feature):
    def create_features(self):
        self.train['V297'] = train_transaction['V297']
        self.test['V297'] = test_transaction['V297']


class v298(Feature):
    def create_features(self):
        self.train['V298'] = train_transaction['V298']
        self.test['V298'] = test_transaction['V298']


class v299(Feature):
    def create_features(self):
        self.train['V299'] = train_transaction['V299']
        self.test['V299'] = test_transaction['V299']


class v300(Feature):
    def create_features(self):
        self.train['V300'] = train_transaction['V300']
        self.test['V300'] = test_transaction['V300']


class v301(Feature):
    def create_features(self):
        self.train['V301'] = train_transaction['V301']
        self.test['V301'] = test_transaction['V301']


class v302(Feature):
    def create_features(self):
        self.train['V302'] = train_transaction['V302']
        self.test['V302'] = test_transaction['V302']


class v303(Feature):
    def create_features(self):
        self.train['V303'] = train_transaction['V303']
        self.test['V303'] = test_transaction['V303']


class v304(Feature):
    def create_features(self):
        self.train['V304'] = train_transaction['V304']
        self.test['V304'] = test_transaction['V304']


class v305(Feature):
    def create_features(self):
        self.train['V305'] = train_transaction['V305']
        self.test['V305'] = test_transaction['V305']


class v306(Feature):
    def create_features(self):
        self.train['V306'] = train_transaction['V306']
        self.test['V306'] = test_transaction['V306']


class v307(Feature):
    def create_features(self):
        self.train['V307'] = train_transaction['V307']
        self.test['V307'] = test_transaction['V307']


class v308(Feature):
    def create_features(self):
        self.train['V308'] = train_transaction['V308']
        self.test['V308'] = test_transaction['V308']


class v309(Feature):
    def create_features(self):
        self.train['V309'] = train_transaction['V309']
        self.test['V309'] = test_transaction['V309']


class v310(Feature):
    def create_features(self):
        self.train['V310'] = train_transaction['V310']
        self.test['V310'] = test_transaction['V310']


class v311(Feature):
    def create_features(self):
        self.train['V311'] = train_transaction['V311']
        self.test['V311'] = test_transaction['V311']


class v312(Feature):
    def create_features(self):
        self.train['V312'] = train_transaction['V312']
        self.test['V312'] = test_transaction['V312']


class v313(Feature):
    def create_features(self):
        self.train['V313'] = train_transaction['V313']
        self.test['V313'] = test_transaction['V313']


class v314(Feature):
    def create_features(self):
        self.train['V314'] = train_transaction['V314']
        self.test['V314'] = test_transaction['V314']


class v315(Feature):
    def create_features(self):
        self.train['V315'] = train_transaction['V315']
        self.test['V315'] = test_transaction['V315']


class v316(Feature):
    def create_features(self):
        self.train['V316'] = train_transaction['V316']
        self.test['V316'] = test_transaction['V316']


class v317(Feature):
    def create_features(self):
        self.train['V317'] = train_transaction['V317']
        self.test['V317'] = test_transaction['V317']


class v318(Feature):
    def create_features(self):
        self.train['V318'] = train_transaction['V318']
        self.test['V318'] = test_transaction['V318']


class v319(Feature):
    def create_features(self):
        self.train['V319'] = train_transaction['V319']
        self.test['V319'] = test_transaction['V319']


class v320(Feature):
    def create_features(self):
        self.train['V320'] = train_transaction['V320']
        self.test['V320'] = test_transaction['V320']


class v321(Feature):
    def create_features(self):
        self.train['V321'] = train_transaction['V321']
        self.test['V321'] = test_transaction['V321']


class v322(Feature):
    def create_features(self):
        self.train['V322'] = train_transaction['V322']
        self.test['V322'] = test_transaction['V322']


class v323(Feature):
    def create_features(self):
        self.train['V323'] = train_transaction['V323']
        self.test['V323'] = test_transaction['V323']


class v324(Feature):
    def create_features(self):
        self.train['V324'] = train_transaction['V324']
        self.test['V324'] = test_transaction['V324']


class v325(Feature):
    def create_features(self):
        self.train['V325'] = train_transaction['V325']
        self.test['V325'] = test_transaction['V325']


class v326(Feature):
    def create_features(self):
        self.train['V326'] = train_transaction['V326']
        self.test['V326'] = test_transaction['V326']


class v327(Feature):
    def create_features(self):
        self.train['V327'] = train_transaction['V327']
        self.test['V327'] = test_transaction['V327']


class v328(Feature):
    def create_features(self):
        self.train['V328'] = train_transaction['V328']
        self.test['V328'] = test_transaction['V328']


class v329(Feature):
    def create_features(self):
        self.train['V329'] = train_transaction['V329']
        self.test['V329'] = test_transaction['V329']


class v330(Feature):
    def create_features(self):
        self.train['V330'] = train_transaction['V330']
        self.test['V330'] = test_transaction['V330']


class v331(Feature):
    def create_features(self):
        self.train['V331'] = train_transaction['V331']
        self.test['V331'] = test_transaction['V331']


class v332(Feature):
    def create_features(self):
        self.train['V332'] = train_transaction['V332']
        self.test['V332'] = test_transaction['V332']


class v333(Feature):
    def create_features(self):
        self.train['V333'] = train_transaction['V333']
        self.test['V333'] = test_transaction['V333']


class v334(Feature):
    def create_features(self):
        self.train['V334'] = train_transaction['V334']
        self.test['V334'] = test_transaction['V334']


class v335(Feature):
    def create_features(self):
        self.train['V335'] = train_transaction['V335']
        self.test['V335'] = test_transaction['V335']


class v336(Feature):
    def create_features(self):
        self.train['V336'] = train_transaction['V336']
        self.test['V336'] = test_transaction['V336']


class v337(Feature):
    def create_features(self):
        self.train['V337'] = train_transaction['V337']
        self.test['V337'] = test_transaction['V337']


class v338(Feature):
    def create_features(self):
        self.train['V338'] = train_transaction['V338']
        self.test['V338'] = test_transaction['V338']


class v339(Feature):
    def create_features(self):
        self.train['V339'] = train_transaction['V339']
        self.test['V339'] = test_transaction['V339']


class sum_v1_v11(Feature):
    def create_features(self):
        v_cols = [f'V{i}' for i in range(1, 12)]
        self.train[self.__class__.__name__] = np.sum(train_transaction[v_cols].values, axis=1)
        self.test[self.__class__.__name__] = np.sum(test_transaction[v_cols].values, axis=1)


class sum_v12_v34(Feature):
    def create_features(self):
        v_cols = [f'V{i}' for i in range(12, 35)]
        self.train[self.__class__.__name__] = np.sum(train_transaction[v_cols].values, axis=1)
        self.test[self.__class__.__name__] = np.sum(test_transaction[v_cols].values, axis=1)


class sum_v35_v52(Feature):
    def create_features(self):
        v_cols = [f'V{i}' for i in range(35, 53)]
        self.train[self.__class__.__name__] = np.sum(train_transaction[v_cols].values, axis=1)
        self.test[self.__class__.__name__] = np.sum(test_transaction[v_cols].values, axis=1)


class sum_v53_v74(Feature):
    def create_features(self):
        v_cols = [f'V{i}' for i in range(53, 75)]
        self.train[self.__class__.__name__] = np.sum(train_transaction[v_cols].values, axis=1)
        self.test[self.__class__.__name__] = np.sum(test_transaction[v_cols].values, axis=1)


class sum_v75_v94(Feature):
    def create_features(self):
        v_cols = [f'V{i}' for i in range(75, 95)]
        self.train[self.__class__.__name__] = np.sum(train_transaction[v_cols].values, axis=1)
        self.test[self.__class__.__name__] = np.sum(test_transaction[v_cols].values, axis=1)


class sum_v95_v137(Feature):
    def create_features(self):
        v_cols = [f'V{i}' for i in range(95, 138)]
        self.train[self.__class__.__name__] = np.sum(train_transaction[v_cols].values, axis=1)
        self.test[self.__class__.__name__] = np.sum(test_transaction[v_cols].values, axis=1)


class sum_v138_v166(Feature):
    def create_features(self):
        v_cols = [f'V{i}' for i in range(138, 167)]
        self.train[self.__class__.__name__] = np.sum(train_transaction[v_cols].values, axis=1)
        self.test[self.__class__.__name__] = np.sum(test_transaction[v_cols].values, axis=1)


class sum_v167_v278(Feature):
    def create_features(self):
        v_cols = [f'V{i}' for i in range(167, 279)]
        self.train[self.__class__.__name__] = np.sum(train_transaction[v_cols].values, axis=1)
        self.test[self.__class__.__name__] = np.sum(test_transaction[v_cols].values, axis=1)


class sum_v279_v320(Feature):
    def create_features(self):
        v_cols =  [
            'V279', 'V280', 'V284', 'V285', 'V286', 'V287', 'V290', 'V291', 'V292', 'V293', 'V294',
            'V295', 'V297', 'V298', 'V299','V302', 'V303', 'V304', 'V305', 'V306', 'V307', 'V308',
            'V309', 'V310', 'V311', 'V312', 'V316', 'V317', 'V318', 'V319','V320', 'V320'
        ]
        self.train[self.__class__.__name__] = np.sum(train_transaction[v_cols].values, axis=1)
        self.test[self.__class__.__name__] = np.sum(test_transaction[v_cols].values, axis=1)


class sum_v281_v315(Feature):
    def create_features(self):
        v_cols = [
            'V281', 'V282', 'V283', 'V288', 'V289', 'V296', 'V300', 'V301', 'V312', 'V313', 'V314', 'V315'
        ]
        self.train[self.__class__.__name__] = np.sum(train_transaction[v_cols].values, axis=1)
        self.test[self.__class__.__name__] = np.sum(test_transaction[v_cols].values, axis=1)


class sum_v322_v339(Feature):
    def create_features(self):
        v_cols = [f'V{i}' for i in range(322, 340)]
        self.train[self.__class__.__name__] = np.sum(train_transaction[v_cols].values, axis=1)
        self.test[self.__class__.__name__] = np.sum(test_transaction[v_cols].values, axis=1)


class P_emaildomain__C2(Feature):
    def create_features(self):
        feats = self.__class__.__name__
        col1, col2 = feats.split('__')
        whole = pd.concat([train_transaction[[col1, col2]], test_transaction[[col1, col2]]], axis=0)
        whole[feats] = whole[col1].astype(str) + '_' + whole[col2].astype(str)
        le = whole[feats].value_counts().to_dict()
        self.train[feats] = (train_transaction[col1].astype(str) + '_' + train_transaction[col2].astype(str)).map(le)
        self.test[feats] = (test_transaction[col1].astype(str) + '_' + test_transaction[col2].astype(str)).map(le)


class card2__dist1(Feature):
    def create_features(self):
        feats = self.__class__.__name__
        col1, col2 = feats.split('__')
        whole = pd.concat([train_transaction[[col1, col2]], test_transaction[[col1, col2]]], axis=0)
        whole[feats] = whole[col1].astype(str) + '_' + whole[col2].astype(str)
        le = whole[feats].value_counts().to_dict()
        self.train[feats] = (train_transaction[col1].astype(str) + '_' + train_transaction[col2].astype(str)).map(le)
        self.test[feats] = (test_transaction[col1].astype(str) + '_' + test_transaction[col2].astype(str)).map(le)


class card5__dist1(Feature):
    def create_features(self):
        feats = self.__class__.__name__
        col1, col2 = feats.split('__')
        whole = pd.concat([train_transaction[[col1, col2]], test_transaction[[col1, col2]]], axis=0)
        whole[feats] = whole[col1].astype(str) + '_' + whole[col2].astype(str)
        le = whole[feats].value_counts().to_dict()
        self.train[feats] = (train_transaction[col1].astype(str) + '_' + train_transaction[col2].astype(str)).map(le)
        self.test[feats] = (test_transaction[col1].astype(str) + '_' + test_transaction[col2].astype(str)).map(le)


class card1__card5(Feature):
    def create_features(self):
        feats = self.__class__.__name__
        col1, col2 = feats.split('__')
        whole = pd.concat([train_transaction[[col1, col2]], test_transaction[[col1, col2]]], axis=0)
        whole[feats] = whole[col1].astype(str) + '_' + whole[col2].astype(str)
        le = whole[feats].value_counts().to_dict()
        self.train[feats] = (train_transaction[col1].astype(str) + '_' + train_transaction[col2].astype(str)).map(le)
        self.test[feats] = (test_transaction[col1].astype(str) + '_' + test_transaction[col2].astype(str)).map(le)


class card1__card5__V283(Feature):
    def create_features(self):
        len_ = len(train_transaction)
        feats = self.__class__.__name__
        col1, col2, col3 = feats.split('__')
        whole = pd.concat([train_transaction[[col1, col2, col3]], test_transaction[[col1, col2, col3]]], axis=0)
        whole[feats] = whole[col1].astype(str) + '_' + whole[col2].astype(str) + '_' + whole[col3].astype(str)
        le = whole[feats].value_counts().to_dict()
        self.train[feats] = whole[feats].map(le).values[:len_]
        self.test[feats] = whole[feats].map(le).values[len_:]


class card5__P_emaildomain(Feature):
    def create_features(self):
        feats = self.__class__.__name__
        col1, col2 = feats.split('__')
        whole = pd.concat([train_transaction[[col1, col2]], test_transaction[[col1, col2]]], axis=0)
        whole[feats] = whole[col1].astype(str) + '_' + whole[col2].astype(str)
        le = whole[feats].value_counts().to_dict()
        self.train[feats] = (train_transaction[col1].astype(str) + '_' + train_transaction[col2].astype(str)).map(le)
        self.test[feats] = (test_transaction[col1].astype(str) + '_' + test_transaction[col2].astype(str)).map(le)


class addr1__card1(Feature):
    def create_features(self):
        feats = self.__class__.__name__
        col1, col2 = feats.split('__')
        whole = pd.concat([train_transaction[[col1, col2]], test_transaction[[col1, col2]]], axis=0)
        whole[feats] = whole[col1].astype(str) + '_' + whole[col2].astype(str)
        le = whole[feats].value_counts().to_dict()
        self.train[feats] = (train_transaction[col1].astype(str) + '_' + train_transaction[col2].astype(str)).map(le)
        self.test[feats] = (test_transaction[col1].astype(str) + '_' + test_transaction[col2].astype(str)).map(le)


class TransactionAmt__card5(Feature):
    def create_features(self):
        feats = self.__class__.__name__
        col1, col2 = feats.split('__')
        whole = pd.concat([train_transaction[[col1, col2]], test_transaction[[col1, col2]]], axis=0)
        whole[feats] = whole[col1].astype(str) + '_' + whole[col2].astype(str)
        le = whole[feats].value_counts().to_dict()
        self.train[feats] = (train_transaction[col1].astype(str) + '_' + train_transaction[col2].astype(str)).map(le)
        self.test[feats] = (test_transaction[col1].astype(str) + '_' + test_transaction[col2].astype(str)).map(le)


class TransactionAmt__card5__addr1(Feature):
    def create_features(self):
        feats = self.__class__.__name__
        col1, col2, col3 = feats.split('__')
        whole = pd.concat([train_transaction[[col1, col2, col3]], test_transaction[[col1, col2, col3]]], axis=0)
        whole[feats] = whole[col1].astype(str) + '_' + whole[col2].astype(str) + '_' + whole[col3].astype(str)
        le = whole[feats].value_counts().to_dict()
        self.train[feats] = (train_transaction[col1].astype(str) + '_' + train_transaction[col2].astype(str) + '_' + train_transaction[col3].astype(str)).map(le)
        self.test[feats] = (test_transaction[col1].astype(str) + '_' + test_transaction[col2].astype(str) + '_' + test_transaction[col3].astype(str)).map(le)


class TransactionAmt__card5__dist2(Feature):
    def create_features(self):
        feats = self.__class__.__name__
        col1, col2, col3 = feats.split('__')
        whole = pd.concat([train_transaction[[col1, col2, col3]], test_transaction[[col1, col2, col3]]], axis=0)
        whole[feats] = whole[col1].astype(str) + '_' + whole[col2].astype(str) + '_' + whole[col3].astype(str)
        le = whole[feats].value_counts().to_dict()
        self.train[feats] = (train_transaction[col1].astype(str) + '_' + train_transaction[col2].astype(str) + '_' + train_transaction[col3].astype(str)).map(le)
        self.test[feats] = (test_transaction[col1].astype(str) + '_' + test_transaction[col2].astype(str) + '_' + test_transaction[col3].astype(str)).map(le)


class card3__card5(Feature):
    def create_features(self):
        feats = self.__class__.__name__
        col1, col2 = feats.split('__')
        whole = pd.concat([train_transaction[[col1, col2]], test_transaction[[col1, col2]]], axis=0)
        whole[feats] = whole[col1].astype(str) + '_' + whole[col2].astype(str)
        le = whole[feats].value_counts().to_dict()
        self.train[feats] = (train_transaction[col1].astype(str) + '_' + train_transaction[col2].astype(str)).map(le)
        self.test[feats] = (test_transaction[col1].astype(str) + '_' + test_transaction[col2].astype(str)).map(le)


class C1_div_C2(Feature):
    def create_features(self):
        train_transaction['C1'] += 0.01
        train_transaction['C2'] += 0.01

        test_transaction['C1'] += 0.01
        test_transaction['C2'] += 0.01
        self.train[self.__class__.__name__] = train_transaction['C1'] / train_transaction['C2']
        self.test[self.__class__.__name__] = test_transaction['C1'] / test_transaction['C2']


class C1_div_C3(Feature):
    def create_features(self):
        train_transaction['C1'] += 0.01
        train_transaction['C3'] += 0.01

        test_transaction['C1'] += 0.01
        test_transaction['C3'] += 0.01
        self.train[self.__class__.__name__] = train_transaction['C1'] / train_transaction['C3']
        self.test[self.__class__.__name__] = test_transaction['C1'] / test_transaction['C3']


class C1_div_C4(Feature):
    def create_features(self):
        train_transaction['C1'] += 0.01
        train_transaction['C4'] += 0.01

        test_transaction['C1'] += 0.01
        test_transaction['C4'] += 0.01
        self.train[self.__class__.__name__] = train_transaction['C1'] / train_transaction['C4']
        self.test[self.__class__.__name__] = test_transaction['C1'] / test_transaction['C4']


class C1_div_C5(Feature):
    def create_features(self):
        train_transaction['C1'] += 0.01
        train_transaction['C5'] += 0.01

        test_transaction['C1'] += 0.01
        test_transaction['C5'] += 0.01
        self.train[self.__class__.__name__] = train_transaction['C1'] / train_transaction['C5']
        self.test[self.__class__.__name__] = test_transaction['C1'] / test_transaction['C5']


class C1_div_C6(Feature):
    def create_features(self):
        train_transaction['C1'] += 0.01
        train_transaction['C6'] += 0.01

        test_transaction['C1'] += 0.01
        test_transaction['C6'] += 0.01
        self.train[self.__class__.__name__] = train_transaction['C1'] / train_transaction['C6']
        self.test[self.__class__.__name__] = test_transaction['C1'] / test_transaction['C6']


class C1_div_C7(Feature):
    def create_features(self):
        train_transaction['C1'] += 0.01
        train_transaction['C7'] += 0.01

        test_transaction['C1'] += 0.01
        test_transaction['C7'] += 0.01
        self.train[self.__class__.__name__] = train_transaction['C1'] / train_transaction['C7']
        self.test[self.__class__.__name__] = test_transaction['C1'] / test_transaction['C7']


class C1_div_C8(Feature):
    def create_features(self):
        train_transaction['C1'] += 0.01
        train_transaction['C8'] += 0.01

        test_transaction['C1'] += 0.01
        test_transaction['C8'] += 0.01
        self.train[self.__class__.__name__] = train_transaction['C1'] / train_transaction['C8']
        self.test[self.__class__.__name__] = test_transaction['C1'] / test_transaction['C8']


class C1_div_C9(Feature):
    def create_features(self):
        train_transaction['C1'] += 0.01
        train_transaction['C9'] += 0.01

        test_transaction['C1'] += 0.01
        test_transaction['C9'] += 0.01
        self.train[self.__class__.__name__] = train_transaction['C1'] / train_transaction['C9']
        self.test[self.__class__.__name__] = test_transaction['C1'] / test_transaction['C9']


class C1_div_C10(Feature):
    def create_features(self):
        train_transaction['C1'] += 0.01
        train_transaction['C10'] += 0.01

        test_transaction['C1'] += 0.01
        test_transaction['C10'] += 0.01
        self.train[self.__class__.__name__] = train_transaction['C1'] / train_transaction['C10']
        self.test[self.__class__.__name__] = test_transaction['C1'] / test_transaction['C10']


class C1_div_C11(Feature):
    def create_features(self):
        train_transaction['C1'] += 0.01
        train_transaction['C11'] += 0.01

        test_transaction['C1'] += 0.01
        test_transaction['C11'] += 0.01
        self.train[self.__class__.__name__] = train_transaction['C1'] / train_transaction['C11']
        self.test[self.__class__.__name__] = test_transaction['C1'] / test_transaction['C11']


class C1_div_C12(Feature):
    def create_features(self):
        train_transaction['C1'] += 0.01
        train_transaction['C12'] += 0.01

        test_transaction['C1'] += 0.01
        test_transaction['C12'] += 0.01
        self.train[self.__class__.__name__] = train_transaction['C1'] / train_transaction['C12']
        self.test[self.__class__.__name__] = test_transaction['C1'] / test_transaction['C12']


class C1_div_C13(Feature):
    def create_features(self):
        train_transaction['C1'] += 0.01
        train_transaction['C13'] += 0.01

        test_transaction['C1'] += 0.01
        test_transaction['C13'] += 0.01
        self.train[self.__class__.__name__] = train_transaction['C1'] / train_transaction['C13']
        self.test[self.__class__.__name__] = test_transaction['C1'] / test_transaction['C13']


class C1_div_C14(Feature):
    def create_features(self):
        train_transaction['C1'] += 0.01
        train_transaction['C14'] += 0.01

        test_transaction['C1'] += 0.01
        test_transaction['C14'] += 0.01
        self.train[self.__class__.__name__] = train_transaction['C1'] / train_transaction['C14']
        self.test[self.__class__.__name__] = test_transaction['C1'] / test_transaction['C14']


class C2_div_C3(Feature):
    def create_features(self):
        train_transaction['C2'] += 0.01
        train_transaction['C3'] += 0.01

        test_transaction['C2'] += 0.01
        test_transaction['C3'] += 0.01
        self.train[self.__class__.__name__] = train_transaction['C2'] / train_transaction['C3']
        self.test[self.__class__.__name__] = test_transaction['C2'] / test_transaction['C3']


class C2_div_C4(Feature):
    def create_features(self):
        train_transaction['C2'] += 0.01
        train_transaction['C4'] += 0.01

        test_transaction['C2'] += 0.01
        test_transaction['C4'] += 0.01
        self.train[self.__class__.__name__] = train_transaction['C2'] / train_transaction['C4']
        self.test[self.__class__.__name__] = test_transaction['C2'] / test_transaction['C4']


class C2_div_C5(Feature):
    def create_features(self):
        train_transaction['C2'] += 0.01
        train_transaction['C5'] += 0.01

        test_transaction['C2'] += 0.01
        test_transaction['C5'] += 0.01
        self.train[self.__class__.__name__] = train_transaction['C2'] / train_transaction['C5']
        self.test[self.__class__.__name__] = test_transaction['C2'] / test_transaction['C5']


class C2_div_C6(Feature):
    def create_features(self):
        train_transaction['C2'] += 0.01
        train_transaction['C6'] += 0.01

        test_transaction['C2'] += 0.01
        test_transaction['C6'] += 0.01
        self.train[self.__class__.__name__] = train_transaction['C2'] / train_transaction['C6']
        self.test[self.__class__.__name__] = test_transaction['C2'] / test_transaction['C6']


class C2_div_C7(Feature):
    def create_features(self):
        train_transaction['C2'] += 0.01
        train_transaction['C7'] += 0.01

        test_transaction['C2'] += 0.01
        test_transaction['C7'] += 0.01
        self.train[self.__class__.__name__] = train_transaction['C2'] / train_transaction['C7']
        self.test[self.__class__.__name__] = test_transaction['C2'] / test_transaction['C7']


class C2_div_C8(Feature):
    def create_features(self):
        train_transaction['C2'] += 0.01
        train_transaction['C8'] += 0.01

        test_transaction['C2'] += 0.01
        test_transaction['C8'] += 0.01
        self.train[self.__class__.__name__] = train_transaction['C2'] / train_transaction['C8']
        self.test[self.__class__.__name__] = test_transaction['C2'] / test_transaction['C8']


class C2_div_C9(Feature):
    def create_features(self):
        train_transaction['C2'] += 0.01
        train_transaction['C9'] += 0.01

        test_transaction['C2'] += 0.01
        test_transaction['C9'] += 0.01
        self.train[self.__class__.__name__] = train_transaction['C2'] / train_transaction['C9']
        self.test[self.__class__.__name__] = test_transaction['C2'] / test_transaction['C9']


class C2_div_C10(Feature):
    def create_features(self):
        train_transaction['C2'] += 0.01
        train_transaction['C10'] += 0.01

        test_transaction['C2'] += 0.01
        test_transaction['C10'] += 0.01
        self.train[self.__class__.__name__] = train_transaction['C2'] / train_transaction['C10']
        self.test[self.__class__.__name__] = test_transaction['C2'] / test_transaction['C10']


class C2_div_C11(Feature):
    def create_features(self):
        train_transaction['C2'] += 0.01
        train_transaction['C11'] += 0.01

        test_transaction['C2'] += 0.01
        test_transaction['C11'] += 0.01
        self.train[self.__class__.__name__] = train_transaction['C2'] / train_transaction['C11']
        self.test[self.__class__.__name__] = test_transaction['C2'] / test_transaction['C11']


class C2_div_C12(Feature):
    def create_features(self):
        train_transaction['C2'] += 0.01
        train_transaction['C12'] += 0.01

        test_transaction['C2'] += 0.01
        test_transaction['C12'] += 0.01
        self.train[self.__class__.__name__] = train_transaction['C2'] / train_transaction['C12']
        self.test[self.__class__.__name__] = test_transaction['C2'] / test_transaction['C12']


class C2_div_C13(Feature):
    def create_features(self):
        train_transaction['C2'] += 0.01
        train_transaction['C13'] += 0.01

        test_transaction['C2'] += 0.01
        test_transaction['C13'] += 0.01
        self.train[self.__class__.__name__] = train_transaction['C2'] / train_transaction['C13']
        self.test[self.__class__.__name__] = test_transaction['C2'] / test_transaction['C13']


class C2_div_C14(Feature):
    def create_features(self):
        train_transaction['C2'] += 0.01
        train_transaction['C14'] += 0.01

        test_transaction['C2'] += 0.01
        test_transaction['C14'] += 0.01
        self.train[self.__class__.__name__] = train_transaction['C2'] / train_transaction['C14']
        self.test[self.__class__.__name__] = test_transaction['C2'] / test_transaction['C14']


class C3_div_C4(Feature):
    def create_features(self):
        train_transaction['C3'] += 0.01
        train_transaction['C4'] += 0.01

        test_transaction['C3'] += 0.01
        test_transaction['C4'] += 0.01
        self.train[self.__class__.__name__] = train_transaction['C3'] / train_transaction['C4']
        self.test[self.__class__.__name__] = test_transaction['C3'] / test_transaction['C4']


class C3_div_C5(Feature):
    def create_features(self):
        train_transaction['C3'] += 0.01
        train_transaction['C5'] += 0.01

        test_transaction['C3'] += 0.01
        test_transaction['C5'] += 0.01
        self.train[self.__class__.__name__] = train_transaction['C3'] / train_transaction['C5']
        self.test[self.__class__.__name__] = test_transaction['C3'] / test_transaction['C5']


class C3_div_C6(Feature):
    def create_features(self):
        train_transaction['C3'] += 0.01
        train_transaction['C6'] += 0.01

        test_transaction['C3'] += 0.01
        test_transaction['C6'] += 0.01
        self.train[self.__class__.__name__] = train_transaction['C3'] / train_transaction['C6']
        self.test[self.__class__.__name__] = test_transaction['C3'] / test_transaction['C6']


class C3_div_C7(Feature):
    def create_features(self):
        train_transaction['C3'] += 0.01
        train_transaction['C7'] += 0.01

        test_transaction['C3'] += 0.01
        test_transaction['C7'] += 0.01
        self.train[self.__class__.__name__] = train_transaction['C3'] / train_transaction['C7']
        self.test[self.__class__.__name__] = test_transaction['C3'] / test_transaction['C7']


class C3_div_C8(Feature):
    def create_features(self):
        train_transaction['C3'] += 0.01
        train_transaction['C8'] += 0.01

        test_transaction['C3'] += 0.01
        test_transaction['C8'] += 0.01
        self.train[self.__class__.__name__] = train_transaction['C3'] / train_transaction['C8']
        self.test[self.__class__.__name__] = test_transaction['C3'] / test_transaction['C8']


class C3_div_C9(Feature):
    def create_features(self):
        train_transaction['C3'] += 0.01
        train_transaction['C9'] += 0.01

        test_transaction['C3'] += 0.01
        test_transaction['C9'] += 0.01
        self.train[self.__class__.__name__] = train_transaction['C3'] / train_transaction['C9']
        self.test[self.__class__.__name__] = test_transaction['C3'] / test_transaction['C9']


class C3_div_C10(Feature):
    def create_features(self):
        train_transaction['C3'] += 0.01
        train_transaction['C10'] += 0.01

        test_transaction['C3'] += 0.01
        test_transaction['C10'] += 0.01
        self.train[self.__class__.__name__] = train_transaction['C3'] / train_transaction['C10']
        self.test[self.__class__.__name__] = test_transaction['C3'] / test_transaction['C10']


class C3_div_C11(Feature):
    def create_features(self):
        train_transaction['C3'] += 0.01
        train_transaction['C11'] += 0.01

        test_transaction['C3'] += 0.01
        test_transaction['C11'] += 0.01
        self.train[self.__class__.__name__] = train_transaction['C3'] / train_transaction['C11']
        self.test[self.__class__.__name__] = test_transaction['C3'] / test_transaction['C11']


class C3_div_C12(Feature):
    def create_features(self):
        train_transaction['C3'] += 0.01
        train_transaction['C12'] += 0.01

        test_transaction['C3'] += 0.01
        test_transaction['C12'] += 0.01
        self.train[self.__class__.__name__] = train_transaction['C3'] / train_transaction['C12']
        self.test[self.__class__.__name__] = test_transaction['C3'] / test_transaction['C12']


class C3_div_C13(Feature):
    def create_features(self):
        train_transaction['C3'] += 0.01
        train_transaction['C13'] += 0.01

        test_transaction['C3'] += 0.01
        test_transaction['C13'] += 0.01
        self.train[self.__class__.__name__] = train_transaction['C3'] / train_transaction['C13']
        self.test[self.__class__.__name__] = test_transaction['C3'] / test_transaction['C13']


class C3_div_C14(Feature):
    def create_features(self):
        train_transaction['C3'] += 0.01
        train_transaction['C14'] += 0.01

        test_transaction['C3'] += 0.01
        test_transaction['C14'] += 0.01
        self.train[self.__class__.__name__] = train_transaction['C3'] / train_transaction['C14']
        self.test[self.__class__.__name__] = test_transaction['C3'] / test_transaction['C14']


class C4_div_C5(Feature):
    def create_features(self):
        train_transaction['C4'] += 0.01
        train_transaction['C5'] += 0.01

        test_transaction['C4'] += 0.01
        test_transaction['C5'] += 0.01
        self.train[self.__class__.__name__] = train_transaction['C4'] / train_transaction['C5']
        self.test[self.__class__.__name__] = test_transaction['C4'] / test_transaction['C5']


class C4_div_C6(Feature):
    def create_features(self):
        train_transaction['C4'] += 0.01
        train_transaction['C6'] += 0.01

        test_transaction['C4'] += 0.01
        test_transaction['C6'] += 0.01
        self.train[self.__class__.__name__] = train_transaction['C4'] / train_transaction['C6']
        self.test[self.__class__.__name__] = test_transaction['C4'] / test_transaction['C6']


class C4_div_C7(Feature):
    def create_features(self):
        train_transaction['C4'] += 0.01
        train_transaction['C7'] += 0.01

        test_transaction['C4'] += 0.01
        test_transaction['C7'] += 0.01
        self.train[self.__class__.__name__] = train_transaction['C4'] / train_transaction['C7']
        self.test[self.__class__.__name__] = test_transaction['C4'] / test_transaction['C7']


class C4_div_C8(Feature):
    def create_features(self):
        train_transaction['C4'] += 0.01
        train_transaction['C8'] += 0.01

        test_transaction['C4'] += 0.01
        test_transaction['C8'] += 0.01
        self.train[self.__class__.__name__] = train_transaction['C4'] / train_transaction['C8']
        self.test[self.__class__.__name__] = test_transaction['C4'] / test_transaction['C8']


class C4_div_C9(Feature):
    def create_features(self):
        train_transaction['C4'] += 0.01
        train_transaction['C9'] += 0.01

        test_transaction['C4'] += 0.01
        test_transaction['C9'] += 0.01
        self.train[self.__class__.__name__] = train_transaction['C4'] / train_transaction['C9']
        self.test[self.__class__.__name__] = test_transaction['C4'] / test_transaction['C9']


class C4_div_C10(Feature):
    def create_features(self):
        train_transaction['C4'] += 0.01
        train_transaction['C10'] += 0.01

        test_transaction['C4'] += 0.01
        test_transaction['C10'] += 0.01
        self.train[self.__class__.__name__] = train_transaction['C4'] / train_transaction['C10']
        self.test[self.__class__.__name__] = test_transaction['C4'] / test_transaction['C10']


class C4_div_C11(Feature):
    def create_features(self):
        train_transaction['C4'] += 0.01
        train_transaction['C11'] += 0.01

        test_transaction['C4'] += 0.01
        test_transaction['C11'] += 0.01
        self.train[self.__class__.__name__] = train_transaction['C4'] / train_transaction['C11']
        self.test[self.__class__.__name__] = test_transaction['C4'] / test_transaction['C11']


class C4_div_C12(Feature):
    def create_features(self):
        train_transaction['C4'] += 0.01
        train_transaction['C12'] += 0.01

        test_transaction['C4'] += 0.01
        test_transaction['C12'] += 0.01
        self.train[self.__class__.__name__] = train_transaction['C4'] / train_transaction['C12']
        self.test[self.__class__.__name__] = test_transaction['C4'] / test_transaction['C12']


class C4_div_C13(Feature):
    def create_features(self):
        train_transaction['C4'] += 0.01
        train_transaction['C13'] += 0.01

        test_transaction['C4'] += 0.01
        test_transaction['C13'] += 0.01
        self.train[self.__class__.__name__] = train_transaction['C4'] / train_transaction['C13']
        self.test[self.__class__.__name__] = test_transaction['C4'] / test_transaction['C13']


class C4_div_C14(Feature):
    def create_features(self):
        train_transaction['C4'] += 0.01
        train_transaction['C14'] += 0.01

        test_transaction['C4'] += 0.01
        test_transaction['C14'] += 0.01
        self.train[self.__class__.__name__] = train_transaction['C4'] / train_transaction['C14']
        self.test[self.__class__.__name__] = test_transaction['C4'] / test_transaction['C14']


class C5_div_C6(Feature):
    def create_features(self):
        train_transaction['C5'] += 0.01
        train_transaction['C6'] += 0.01

        test_transaction['C5'] += 0.01
        test_transaction['C6'] += 0.01
        self.train[self.__class__.__name__] = train_transaction['C5'] / train_transaction['C6']
        self.test[self.__class__.__name__] = test_transaction['C5'] / test_transaction['C6']


class C5_div_C7(Feature):
    def create_features(self):
        train_transaction['C5'] += 0.01
        train_transaction['C7'] += 0.01

        test_transaction['C5'] += 0.01
        test_transaction['C7'] += 0.01
        self.train[self.__class__.__name__] = train_transaction['C5'] / train_transaction['C7']
        self.test[self.__class__.__name__] = test_transaction['C5'] / test_transaction['C7']


class C5_div_C8(Feature):
    def create_features(self):
        train_transaction['C5'] += 0.01
        train_transaction['C8'] += 0.01

        test_transaction['C5'] += 0.01
        test_transaction['C8'] += 0.01
        self.train[self.__class__.__name__] = train_transaction['C5'] / train_transaction['C8']
        self.test[self.__class__.__name__] = test_transaction['C5'] / test_transaction['C8']


class C5_div_C9(Feature):
    def create_features(self):
        train_transaction['C5'] += 0.01
        train_transaction['C9'] += 0.01

        test_transaction['C5'] += 0.01
        test_transaction['C9'] += 0.01
        self.train[self.__class__.__name__] = train_transaction['C5'] / train_transaction['C9']
        self.test[self.__class__.__name__] = test_transaction['C5'] / test_transaction['C9']


class C5_div_C10(Feature):
    def create_features(self):
        train_transaction['C5'] += 0.01
        train_transaction['C10'] += 0.01

        test_transaction['C5'] += 0.01
        test_transaction['C10'] += 0.01
        self.train[self.__class__.__name__] = train_transaction['C5'] / train_transaction['C10']
        self.test[self.__class__.__name__] = test_transaction['C5'] / test_transaction['C10']


class C5_div_C11(Feature):
    def create_features(self):
        train_transaction['C5'] += 0.01
        train_transaction['C11'] += 0.01

        test_transaction['C5'] += 0.01
        test_transaction['C11'] += 0.01
        self.train[self.__class__.__name__] = train_transaction['C5'] / train_transaction['C11']
        self.test[self.__class__.__name__] = test_transaction['C5'] / test_transaction['C11']


class C5_div_C12(Feature):
    def create_features(self):
        train_transaction['C5'] += 0.01
        train_transaction['C12'] += 0.01

        test_transaction['C5'] += 0.01
        test_transaction['C12'] += 0.01
        self.train[self.__class__.__name__] = train_transaction['C5'] / train_transaction['C12']
        self.test[self.__class__.__name__] = test_transaction['C5'] / test_transaction['C12']


class C5_div_C13(Feature):
    def create_features(self):
        train_transaction['C5'] += 0.01
        train_transaction['C13'] += 0.01

        test_transaction['C5'] += 0.01
        test_transaction['C13'] += 0.01
        self.train[self.__class__.__name__] = train_transaction['C5'] / train_transaction['C13']
        self.test[self.__class__.__name__] = test_transaction['C5'] / test_transaction['C13']


class C5_div_C14(Feature):
    def create_features(self):
        train_transaction['C5'] += 0.01
        train_transaction['C14'] += 0.01

        test_transaction['C5'] += 0.01
        test_transaction['C14'] += 0.01
        self.train[self.__class__.__name__] = train_transaction['C5'] / train_transaction['C14']
        self.test[self.__class__.__name__] = test_transaction['C5'] / test_transaction['C14']


class C6_div_C7(Feature):
    def create_features(self):
        train_transaction['C6'] += 0.01
        train_transaction['C7'] += 0.01

        test_transaction['C6'] += 0.01
        test_transaction['C7'] += 0.01
        self.train[self.__class__.__name__] = train_transaction['C6'] / train_transaction['C7']
        self.test[self.__class__.__name__] = test_transaction['C6'] / test_transaction['C7']


class C6_div_C8(Feature):
    def create_features(self):
        train_transaction['C6'] += 0.01
        train_transaction['C8'] += 0.01

        test_transaction['C6'] += 0.01
        test_transaction['C8'] += 0.01
        self.train[self.__class__.__name__] = train_transaction['C6'] / train_transaction['C8']
        self.test[self.__class__.__name__] = test_transaction['C6'] / test_transaction['C8']


class C6_div_C9(Feature):
    def create_features(self):
        train_transaction['C6'] += 0.01
        train_transaction['C9'] += 0.01

        test_transaction['C6'] += 0.01
        test_transaction['C9'] += 0.01
        self.train[self.__class__.__name__] = train_transaction['C6'] / train_transaction['C9']
        self.test[self.__class__.__name__] = test_transaction['C6'] / test_transaction['C9']


class C6_div_C10(Feature):
    def create_features(self):
        train_transaction['C6'] += 0.01
        train_transaction['C10'] += 0.01

        test_transaction['C6'] += 0.01
        test_transaction['C10'] += 0.01
        self.train[self.__class__.__name__] = train_transaction['C6'] / train_transaction['C10']
        self.test[self.__class__.__name__] = test_transaction['C6'] / test_transaction['C10']


class C6_div_C11(Feature):
    def create_features(self):
        train_transaction['C6'] += 0.01
        train_transaction['C11'] += 0.01

        test_transaction['C6'] += 0.01
        test_transaction['C11'] += 0.01
        self.train[self.__class__.__name__] = train_transaction['C6'] / train_transaction['C11']
        self.test[self.__class__.__name__] = test_transaction['C6'] / test_transaction['C11']


class C6_div_C12(Feature):
    def create_features(self):
        train_transaction['C6'] += 0.01
        train_transaction['C12'] += 0.01

        test_transaction['C6'] += 0.01
        test_transaction['C12'] += 0.01
        self.train[self.__class__.__name__] = train_transaction['C6'] / train_transaction['C12']
        self.test[self.__class__.__name__] = test_transaction['C6'] / test_transaction['C12']


class C6_div_C13(Feature):
    def create_features(self):
        train_transaction['C6'] += 0.01
        train_transaction['C13'] += 0.01

        test_transaction['C6'] += 0.01
        test_transaction['C13'] += 0.01
        self.train[self.__class__.__name__] = train_transaction['C6'] / train_transaction['C13']
        self.test[self.__class__.__name__] = test_transaction['C6'] / test_transaction['C13']


class C6_div_C14(Feature):
    def create_features(self):
        train_transaction['C6'] += 0.01
        train_transaction['C14'] += 0.01

        test_transaction['C6'] += 0.01
        test_transaction['C14'] += 0.01
        self.train[self.__class__.__name__] = train_transaction['C6'] / train_transaction['C14']
        self.test[self.__class__.__name__] = test_transaction['C6'] / test_transaction['C14']


class C7_div_C8(Feature):
    def create_features(self):
        train_transaction['C7'] += 0.01
        train_transaction['C8'] += 0.01

        test_transaction['C7'] += 0.01
        test_transaction['C8'] += 0.01
        self.train[self.__class__.__name__] = train_transaction['C7'] / train_transaction['C8']
        self.test[self.__class__.__name__] = test_transaction['C7'] / test_transaction['C8']


class C7_div_C9(Feature):
    def create_features(self):
        train_transaction['C7'] += 0.01
        train_transaction['C9'] += 0.01

        test_transaction['C7'] += 0.01
        test_transaction['C9'] += 0.01
        self.train[self.__class__.__name__] = train_transaction['C7'] / train_transaction['C9']
        self.test[self.__class__.__name__] = test_transaction['C7'] / test_transaction['C9']


class C7_div_C10(Feature):
    def create_features(self):
        train_transaction['C7'] += 0.01
        train_transaction['C10'] += 0.01

        test_transaction['C7'] += 0.01
        test_transaction['C10'] += 0.01
        self.train[self.__class__.__name__] = train_transaction['C7'] / train_transaction['C10']
        self.test[self.__class__.__name__] = test_transaction['C7'] / test_transaction['C10']


class C7_div_C11(Feature):
    def create_features(self):
        train_transaction['C7'] += 0.01
        train_transaction['C11'] += 0.01

        test_transaction['C7'] += 0.01
        test_transaction['C11'] += 0.01
        self.train[self.__class__.__name__] = train_transaction['C7'] / train_transaction['C11']
        self.test[self.__class__.__name__] = test_transaction['C7'] / test_transaction['C11']


class C7_div_C12(Feature):
    def create_features(self):
        train_transaction['C7'] += 0.01
        train_transaction['C12'] += 0.01

        test_transaction['C7'] += 0.01
        test_transaction['C12'] += 0.01
        self.train[self.__class__.__name__] = train_transaction['C7'] / train_transaction['C12']
        self.test[self.__class__.__name__] = test_transaction['C7'] / test_transaction['C12']


class C7_div_C13(Feature):
    def create_features(self):
        train_transaction['C7'] += 0.01
        train_transaction['C13'] += 0.01

        test_transaction['C7'] += 0.01
        test_transaction['C13'] += 0.01
        self.train[self.__class__.__name__] = train_transaction['C7'] / train_transaction['C13']
        self.test[self.__class__.__name__] = test_transaction['C7'] / test_transaction['C13']


class C7_div_C14(Feature):
    def create_features(self):
        train_transaction['C7'] += 0.01
        train_transaction['C14'] += 0.01

        test_transaction['C7'] += 0.01
        test_transaction['C14'] += 0.01
        self.train[self.__class__.__name__] = train_transaction['C7'] / train_transaction['C14']
        self.test[self.__class__.__name__] = test_transaction['C7'] / test_transaction['C14']


class C8_div_C9(Feature):
    def create_features(self):
        train_transaction['C8'] += 0.01
        train_transaction['C9'] += 0.01

        test_transaction['C8'] += 0.01
        test_transaction['C9'] += 0.01
        self.train[self.__class__.__name__] = train_transaction['C8'] / train_transaction['C9']
        self.test[self.__class__.__name__] = test_transaction['C8'] / test_transaction['C9']


class C8_div_C10(Feature):
    def create_features(self):
        train_transaction['C8'] += 0.01
        train_transaction['C10'] += 0.01

        test_transaction['C8'] += 0.01
        test_transaction['C10'] += 0.01
        self.train[self.__class__.__name__] = train_transaction['C8'] / train_transaction['C10']
        self.test[self.__class__.__name__] = test_transaction['C8'] / test_transaction['C10']


class C8_div_C11(Feature):
    def create_features(self):
        train_transaction['C8'] += 0.01
        train_transaction['C11'] += 0.01

        test_transaction['C8'] += 0.01
        test_transaction['C11'] += 0.01
        self.train[self.__class__.__name__] = train_transaction['C8'] / train_transaction['C11']
        self.test[self.__class__.__name__] = test_transaction['C8'] / test_transaction['C11']


class C8_div_C12(Feature):
    def create_features(self):
        train_transaction['C8'] += 0.01
        train_transaction['C12'] += 0.01

        test_transaction['C8'] += 0.01
        test_transaction['C12'] += 0.01
        self.train[self.__class__.__name__] = train_transaction['C8'] / train_transaction['C12']
        self.test[self.__class__.__name__] = test_transaction['C8'] / test_transaction['C12']


class C8_div_C13(Feature):
    def create_features(self):
        train_transaction['C8'] += 0.01
        train_transaction['C13'] += 0.01

        test_transaction['C8'] += 0.01
        test_transaction['C13'] += 0.01
        self.train[self.__class__.__name__] = train_transaction['C8'] / train_transaction['C13']
        self.test[self.__class__.__name__] = test_transaction['C8'] / test_transaction['C13']


class C8_div_C14(Feature):
    def create_features(self):
        train_transaction['C8'] += 0.01
        train_transaction['C14'] += 0.01

        test_transaction['C8'] += 0.01
        test_transaction['C14'] += 0.01
        self.train[self.__class__.__name__] = train_transaction['C8'] / train_transaction['C14']
        self.test[self.__class__.__name__] = test_transaction['C8'] / test_transaction['C14']


class C9_div_C10(Feature):
    def create_features(self):
        train_transaction['C9'] += 0.01
        train_transaction['C10'] += 0.01

        test_transaction['C9'] += 0.01
        test_transaction['C10'] += 0.01
        self.train[self.__class__.__name__] = train_transaction['C9'] / train_transaction['C10']
        self.test[self.__class__.__name__] = test_transaction['C9'] / test_transaction['C10']


class C9_div_C11(Feature):
    def create_features(self):
        train_transaction['C9'] += 0.01
        train_transaction['C11'] += 0.01

        test_transaction['C9'] += 0.01
        test_transaction['C11'] += 0.01
        self.train[self.__class__.__name__] = train_transaction['C9'] / train_transaction['C11']
        self.test[self.__class__.__name__] = test_transaction['C9'] / test_transaction['C11']


class C9_div_C12(Feature):
    def create_features(self):
        train_transaction['C9'] += 0.01
        train_transaction['C12'] += 0.01

        test_transaction['C9'] += 0.01
        test_transaction['C12'] += 0.01
        self.train[self.__class__.__name__] = train_transaction['C9'] / train_transaction['C12']
        self.test[self.__class__.__name__] = test_transaction['C9'] / test_transaction['C12']


class C9_div_C13(Feature):
    def create_features(self):
        train_transaction['C9'] += 0.01
        train_transaction['C13'] += 0.01

        test_transaction['C9'] += 0.01
        test_transaction['C13'] += 0.01
        self.train[self.__class__.__name__] = train_transaction['C9'] / train_transaction['C13']
        self.test[self.__class__.__name__] = test_transaction['C9'] / test_transaction['C13']


class C9_div_C14(Feature):
    def create_features(self):
        train_transaction['C9'] += 0.01
        train_transaction['C14'] += 0.01

        test_transaction['C9'] += 0.01
        test_transaction['C14'] += 0.01
        self.train[self.__class__.__name__] = train_transaction['C9'] / train_transaction['C14']
        self.test[self.__class__.__name__] = test_transaction['C9'] / test_transaction['C14']


class C10_div_C11(Feature):
    def create_features(self):
        train_transaction['C10'] += 0.01
        train_transaction['C11'] += 0.01

        test_transaction['C10'] += 0.01
        test_transaction['C11'] += 0.01
        self.train[self.__class__.__name__] = train_transaction['C10'] / train_transaction['C11']
        self.test[self.__class__.__name__] = test_transaction['C10'] / test_transaction['C11']


class C10_div_C12(Feature):
    def create_features(self):
        train_transaction['C10'] += 0.01
        train_transaction['C12'] += 0.01

        test_transaction['C10'] += 0.01
        test_transaction['C12'] += 0.01
        self.train[self.__class__.__name__] = train_transaction['C10'] / train_transaction['C12']
        self.test[self.__class__.__name__] = test_transaction['C10'] / test_transaction['C12']


class C10_div_C13(Feature):
    def create_features(self):
        train_transaction['C10'] += 0.01
        train_transaction['C13'] += 0.01

        test_transaction['C10'] += 0.01
        test_transaction['C13'] += 0.01
        self.train[self.__class__.__name__] = train_transaction['C10'] / train_transaction['C13']
        self.test[self.__class__.__name__] = test_transaction['C10'] / test_transaction['C13']


class C10_div_C14(Feature):
    def create_features(self):
        train_transaction['C10'] += 0.01
        train_transaction['C14'] += 0.01

        test_transaction['C10'] += 0.01
        test_transaction['C14'] += 0.01
        self.train[self.__class__.__name__] = train_transaction['C10'] / train_transaction['C14']
        self.test[self.__class__.__name__] = test_transaction['C10'] / test_transaction['C14']


class C11_div_C12(Feature):
    def create_features(self):
        train_transaction['C11'] += 0.01
        train_transaction['C12'] += 0.01

        test_transaction['C11'] += 0.01
        test_transaction['C12'] += 0.01
        self.train[self.__class__.__name__] = train_transaction['C11'] / train_transaction['C12']
        self.test[self.__class__.__name__] = test_transaction['C11'] / test_transaction['C12']


class C11_div_C13(Feature):
    def create_features(self):
        train_transaction['C11'] += 0.01
        train_transaction['C13'] += 0.01

        test_transaction['C11'] += 0.01
        test_transaction['C13'] += 0.01
        self.train[self.__class__.__name__] = train_transaction['C11'] / train_transaction['C13']
        self.test[self.__class__.__name__] = test_transaction['C11'] / test_transaction['C13']


class C11_div_C14(Feature):
    def create_features(self):
        train_transaction['C11'] += 0.01
        train_transaction['C14'] += 0.01

        test_transaction['C11'] += 0.01
        test_transaction['C14'] += 0.01
        self.train[self.__class__.__name__] = train_transaction['C11'] / train_transaction['C14']
        self.test[self.__class__.__name__] = test_transaction['C11'] / test_transaction['C14']


class C12_div_C13(Feature):
    def create_features(self):
        train_transaction['C12'] += 0.01
        train_transaction['C13'] += 0.01

        test_transaction['C12'] += 0.01
        test_transaction['C13'] += 0.01
        self.train[self.__class__.__name__] = train_transaction['C12'] / train_transaction['C13']
        self.test[self.__class__.__name__] = test_transaction['C12'] / test_transaction['C13']


class C12_div_C14(Feature):
    def create_features(self):
        train_transaction['C12'] += 0.01
        train_transaction['C14'] += 0.01

        test_transaction['C12'] += 0.01
        test_transaction['C14'] += 0.01
        self.train[self.__class__.__name__] = train_transaction['C12'] / train_transaction['C14']
        self.test[self.__class__.__name__] = test_transaction['C12'] / test_transaction['C14']


class C13_div_C14(Feature):
    def create_features(self):
        train_transaction['C13'] += 0.01
        train_transaction['C14'] += 0.01

        test_transaction['C13'] += 0.01
        test_transaction['C14'] += 0.01
        self.train[self.__class__.__name__] = train_transaction['C13'] / train_transaction['C14']
        self.test[self.__class__.__name__] = test_transaction['C13'] / test_transaction['C14']


class D1_div_D2(Feature):
    def create_features(self):
        train_transaction['D1'] = train_transaction['D1'].fillna(-1) + 0.01
        train_transaction['D2'] = train_transaction['D2'].fillna(-1) + 0.01

        test_transaction['D1'] = test_transaction['D1'].fillna(-1) + 0.01
        test_transaction['D2'] = test_transaction['D2'].fillna(-1) + 0.01

        self.train[self.__class__.__name__] = train_transaction['D1'] / train_transaction['D2']
        self.test[self.__class__.__name__] = test_transaction['D1'] / test_transaction['D2']


class D1_div_D3(Feature):
    def create_features(self):
        train_transaction['D1'] = train_transaction['D1'].fillna(-1) + 0.01
        train_transaction['D3'] = train_transaction['D3'].fillna(-1) + 0.01

        test_transaction['D1'] = test_transaction['D1'].fillna(-1) + 0.01
        test_transaction['D3'] = test_transaction['D3'].fillna(-1) + 0.01

        self.train[self.__class__.__name__] = train_transaction['D1'] / train_transaction['D3']
        self.test[self.__class__.__name__] = test_transaction['D1'] / test_transaction['D3']


class D1_div_D4(Feature):
    def create_features(self):
        train_transaction['D1'] = train_transaction['D1'].fillna(-1) + 0.01
        train_transaction['D4'] = train_transaction['D4'].fillna(-1) + 0.01

        test_transaction['D1'] = test_transaction['D1'].fillna(-1) + 0.01
        test_transaction['D4'] = test_transaction['D4'].fillna(-1) + 0.01

        self.train[self.__class__.__name__] = train_transaction['D1'] / train_transaction['D4']
        self.test[self.__class__.__name__] = test_transaction['D1'] / test_transaction['D4']


class D1_div_D5(Feature):
    def create_features(self):
        train_transaction['D1'] = train_transaction['D1'].fillna(-1) + 0.01
        train_transaction['D5'] = train_transaction['D5'].fillna(-1) + 0.01

        test_transaction['D1'] = test_transaction['D1'].fillna(-1) + 0.01
        test_transaction['D5'] = test_transaction['D5'].fillna(-1) + 0.01

        self.train[self.__class__.__name__] = train_transaction['D1'] / train_transaction['D5']
        self.test[self.__class__.__name__] = test_transaction['D1'] / test_transaction['D5']


class D1_div_D6(Feature):
    def create_features(self):
        train_transaction['D1'] = train_transaction['D1'].fillna(-1) + 0.01
        train_transaction['D6'] = train_transaction['D6'].fillna(-1) + 0.01

        test_transaction['D1'] = test_transaction['D1'].fillna(-1) + 0.01
        test_transaction['D6'] = test_transaction['D6'].fillna(-1) + 0.01

        self.train[self.__class__.__name__] = train_transaction['D1'] / train_transaction['D6']
        self.test[self.__class__.__name__] = test_transaction['D1'] / test_transaction['D6']


class D1_div_D7(Feature):
    def create_features(self):
        train_transaction['D1'] = train_transaction['D1'].fillna(-1) + 0.01
        train_transaction['D7'] = train_transaction['D7'].fillna(-1) + 0.01

        test_transaction['D1'] = test_transaction['D1'].fillna(-1) + 0.01
        test_transaction['D7'] = test_transaction['D7'].fillna(-1) + 0.01

        self.train[self.__class__.__name__] = train_transaction['D1'] / train_transaction['D7']
        self.test[self.__class__.__name__] = test_transaction['D1'] / test_transaction['D7']


class D1_div_D8(Feature):
    def create_features(self):
        train_transaction['D1'] = train_transaction['D1'].fillna(-1) + 0.01
        train_transaction['D8'] = train_transaction['D8'].fillna(-1) + 0.01

        test_transaction['D1'] = test_transaction['D1'].fillna(-1) + 0.01
        test_transaction['D8'] = test_transaction['D8'].fillna(-1) + 0.01

        self.train[self.__class__.__name__] = train_transaction['D1'] / train_transaction['D8']
        self.test[self.__class__.__name__] = test_transaction['D1'] / test_transaction['D8']


class D1_div_D9(Feature):
    def create_features(self):
        train_transaction['D1'] = train_transaction['D1'].fillna(-1) + 0.01
        train_transaction['D9'] = train_transaction['D9'].fillna(-1) + 0.01

        test_transaction['D1'] = test_transaction['D1'].fillna(-1) + 0.01
        test_transaction['D9'] = test_transaction['D9'].fillna(-1) + 0.01

        self.train[self.__class__.__name__] = train_transaction['D1'] / train_transaction['D9']
        self.test[self.__class__.__name__] = test_transaction['D1'] / test_transaction['D9']


class D1_div_D10(Feature):
    def create_features(self):
        train_transaction['D1'] = train_transaction['D1'].fillna(-1) + 0.01
        train_transaction['D10'] = train_transaction['D10'].fillna(-1) + 0.01

        test_transaction['D1'] = test_transaction['D1'].fillna(-1) + 0.01
        test_transaction['D10'] = test_transaction['D10'].fillna(-1) + 0.01

        self.train[self.__class__.__name__] = train_transaction['D1'] / train_transaction['D10']
        self.test[self.__class__.__name__] = test_transaction['D1'] / test_transaction['D10']


class D1_div_D11(Feature):
    def create_features(self):
        train_transaction['D1'] = train_transaction['D1'].fillna(-1) + 0.01
        train_transaction['D11'] = train_transaction['D11'].fillna(-1) + 0.01

        test_transaction['D1'] = test_transaction['D1'].fillna(-1) + 0.01
        test_transaction['D11'] = test_transaction['D11'].fillna(-1) + 0.01

        self.train[self.__class__.__name__] = train_transaction['D1'] / train_transaction['D11']
        self.test[self.__class__.__name__] = test_transaction['D1'] / test_transaction['D11']


class D1_div_D12(Feature):
    def create_features(self):
        train_transaction['D1'] = train_transaction['D1'].fillna(-1) + 0.01
        train_transaction['D12'] = train_transaction['D12'].fillna(-1) + 0.01

        test_transaction['D1'] = test_transaction['D1'].fillna(-1) + 0.01
        test_transaction['D12'] = test_transaction['D12'].fillna(-1) + 0.01

        self.train[self.__class__.__name__] = train_transaction['D1'] / train_transaction['D12']
        self.test[self.__class__.__name__] = test_transaction['D1'] / test_transaction['D12']


class D1_div_D13(Feature):
    def create_features(self):
        train_transaction['D1'] = train_transaction['D1'].fillna(-1) + 0.01
        train_transaction['D13'] = train_transaction['D13'].fillna(-1) + 0.01

        test_transaction['D1'] = test_transaction['D1'].fillna(-1) + 0.01
        test_transaction['D13'] = test_transaction['D13'].fillna(-1) + 0.01

        self.train[self.__class__.__name__] = train_transaction['D1'] / train_transaction['D13']
        self.test[self.__class__.__name__] = test_transaction['D1'] / test_transaction['D13']


class D1_div_D14(Feature):
    def create_features(self):
        train_transaction['D1'] = train_transaction['D1'].fillna(-1) + 0.01
        train_transaction['D14'] = train_transaction['D14'].fillna(-1) + 0.01

        test_transaction['D1'] = test_transaction['D1'].fillna(-1) + 0.01
        test_transaction['D14'] = test_transaction['D14'].fillna(-1) + 0.01

        self.train[self.__class__.__name__] = train_transaction['D1'] / train_transaction['D14']
        self.test[self.__class__.__name__] = test_transaction['D1'] / test_transaction['D14']


class D1_div_D15(Feature):
    def create_features(self):
        train_transaction['D1'] = train_transaction['D1'].fillna(-1) + 0.01
        train_transaction['D15'] = train_transaction['D15'].fillna(-1) + 0.01

        test_transaction['D1'] = test_transaction['D1'].fillna(-1) + 0.01
        test_transaction['D15'] = test_transaction['D15'].fillna(-1) + 0.01

        self.train[self.__class__.__name__] = train_transaction['D1'] / train_transaction['D15']
        self.test[self.__class__.__name__] = test_transaction['D1'] / test_transaction['D15']


class D2_div_D3(Feature):
    def create_features(self):
        train_transaction['D2'] = train_transaction['D2'].fillna(-1) + 0.01
        train_transaction['D3'] = train_transaction['D3'].fillna(-1) + 0.01

        test_transaction['D2'] = test_transaction['D2'].fillna(-1) + 0.01
        test_transaction['D3'] = test_transaction['D3'].fillna(-1) + 0.01

        self.train[self.__class__.__name__] = train_transaction['D2'] / train_transaction['D3']
        self.test[self.__class__.__name__] = test_transaction['D2'] / test_transaction['D3']


class D2_div_D4(Feature):
    def create_features(self):
        train_transaction['D2'] = train_transaction['D2'].fillna(-1) + 0.01
        train_transaction['D4'] = train_transaction['D4'].fillna(-1) + 0.01

        test_transaction['D2'] = test_transaction['D2'].fillna(-1) + 0.01
        test_transaction['D4'] = test_transaction['D4'].fillna(-1) + 0.01

        self.train[self.__class__.__name__] = train_transaction['D2'] / train_transaction['D4']
        self.test[self.__class__.__name__] = test_transaction['D2'] / test_transaction['D4']


class D2_div_D5(Feature):
    def create_features(self):
        train_transaction['D2'] = train_transaction['D2'].fillna(-1) + 0.01
        train_transaction['D5'] = train_transaction['D5'].fillna(-1) + 0.01

        test_transaction['D2'] = test_transaction['D2'].fillna(-1) + 0.01
        test_transaction['D5'] = test_transaction['D5'].fillna(-1) + 0.01

        self.train[self.__class__.__name__] = train_transaction['D2'] / train_transaction['D5']
        self.test[self.__class__.__name__] = test_transaction['D2'] / test_transaction['D5']


class D2_div_D6(Feature):
    def create_features(self):
        train_transaction['D2'] = train_transaction['D2'].fillna(-1) + 0.01
        train_transaction['D6'] = train_transaction['D6'].fillna(-1) + 0.01

        test_transaction['D2'] = test_transaction['D2'].fillna(-1) + 0.01
        test_transaction['D6'] = test_transaction['D6'].fillna(-1) + 0.01

        self.train[self.__class__.__name__] = train_transaction['D2'] / train_transaction['D6']
        self.test[self.__class__.__name__] = test_transaction['D2'] / test_transaction['D6']


class D2_div_D7(Feature):
    def create_features(self):
        train_transaction['D2'] = train_transaction['D2'].fillna(-1) + 0.01
        train_transaction['D7'] = train_transaction['D7'].fillna(-1) + 0.01

        test_transaction['D2'] = test_transaction['D2'].fillna(-1) + 0.01
        test_transaction['D7'] = test_transaction['D7'].fillna(-1) + 0.01

        self.train[self.__class__.__name__] = train_transaction['D2'] / train_transaction['D7']
        self.test[self.__class__.__name__] = test_transaction['D2'] / test_transaction['D7']


class D2_div_D8(Feature):
    def create_features(self):
        train_transaction['D2'] = train_transaction['D2'].fillna(-1) + 0.01
        train_transaction['D8'] = train_transaction['D8'].fillna(-1) + 0.01

        test_transaction['D2'] = test_transaction['D2'].fillna(-1) + 0.01
        test_transaction['D8'] = test_transaction['D8'].fillna(-1) + 0.01

        self.train[self.__class__.__name__] = train_transaction['D2'] / train_transaction['D8']
        self.test[self.__class__.__name__] = test_transaction['D2'] / test_transaction['D8']


class D2_div_D9(Feature):
    def create_features(self):
        train_transaction['D2'] = train_transaction['D2'].fillna(-1) + 0.01
        train_transaction['D9'] = train_transaction['D9'].fillna(-1) + 0.01

        test_transaction['D2'] = test_transaction['D2'].fillna(-1) + 0.01
        test_transaction['D9'] = test_transaction['D9'].fillna(-1) + 0.01

        self.train[self.__class__.__name__] = train_transaction['D2'] / train_transaction['D9']
        self.test[self.__class__.__name__] = test_transaction['D2'] / test_transaction['D9']


class D2_div_D10(Feature):
    def create_features(self):
        train_transaction['D2'] = train_transaction['D2'].fillna(-1) + 0.01
        train_transaction['D10'] = train_transaction['D10'].fillna(-1) + 0.01

        test_transaction['D2'] = test_transaction['D2'].fillna(-1) + 0.01
        test_transaction['D10'] = test_transaction['D10'].fillna(-1) + 0.01

        self.train[self.__class__.__name__] = train_transaction['D2'] / train_transaction['D10']
        self.test[self.__class__.__name__] = test_transaction['D2'] / test_transaction['D10']


class D2_div_D11(Feature):
    def create_features(self):
        train_transaction['D2'] = train_transaction['D2'].fillna(-1) + 0.01
        train_transaction['D11'] = train_transaction['D11'].fillna(-1) + 0.01

        test_transaction['D2'] = test_transaction['D2'].fillna(-1) + 0.01
        test_transaction['D11'] = test_transaction['D11'].fillna(-1) + 0.01

        self.train[self.__class__.__name__] = train_transaction['D2'] / train_transaction['D11']
        self.test[self.__class__.__name__] = test_transaction['D2'] / test_transaction['D11']


class D2_div_D12(Feature):
    def create_features(self):
        train_transaction['D2'] = train_transaction['D2'].fillna(-1) + 0.01
        train_transaction['D12'] = train_transaction['D12'].fillna(-1) + 0.01

        test_transaction['D2'] = test_transaction['D2'].fillna(-1) + 0.01
        test_transaction['D12'] = test_transaction['D12'].fillna(-1) + 0.01

        self.train[self.__class__.__name__] = train_transaction['D2'] / train_transaction['D12']
        self.test[self.__class__.__name__] = test_transaction['D2'] / test_transaction['D12']


class D2_div_D13(Feature):
    def create_features(self):
        train_transaction['D2'] = train_transaction['D2'].fillna(-1) + 0.01
        train_transaction['D13'] = train_transaction['D13'].fillna(-1) + 0.01

        test_transaction['D2'] = test_transaction['D2'].fillna(-1) + 0.01
        test_transaction['D13'] = test_transaction['D13'].fillna(-1) + 0.01

        self.train[self.__class__.__name__] = train_transaction['D2'] / train_transaction['D13']
        self.test[self.__class__.__name__] = test_transaction['D2'] / test_transaction['D13']


class D2_div_D14(Feature):
    def create_features(self):
        train_transaction['D2'] = train_transaction['D2'].fillna(-1) + 0.01
        train_transaction['D14'] = train_transaction['D14'].fillna(-1) + 0.01

        test_transaction['D2'] = test_transaction['D2'].fillna(-1) + 0.01
        test_transaction['D14'] = test_transaction['D14'].fillna(-1) + 0.01

        self.train[self.__class__.__name__] = train_transaction['D2'] / train_transaction['D14']
        self.test[self.__class__.__name__] = test_transaction['D2'] / test_transaction['D14']


class D2_div_D15(Feature):
    def create_features(self):
        train_transaction['D2'] = train_transaction['D2'].fillna(-1) + 0.01
        train_transaction['D15'] = train_transaction['D15'].fillna(-1) + 0.01

        test_transaction['D2'] = test_transaction['D2'].fillna(-1) + 0.01
        test_transaction['D15'] = test_transaction['D15'].fillna(-1) + 0.01

        self.train[self.__class__.__name__] = train_transaction['D2'] / train_transaction['D15']
        self.test[self.__class__.__name__] = test_transaction['D2'] / test_transaction['D15']


class D3_div_D4(Feature):
    def create_features(self):
        train_transaction['D3'] = train_transaction['D3'].fillna(-1) + 0.01
        train_transaction['D4'] = train_transaction['D4'].fillna(-1) + 0.01

        test_transaction['D3'] = test_transaction['D3'].fillna(-1) + 0.01
        test_transaction['D4'] = test_transaction['D4'].fillna(-1) + 0.01

        self.train[self.__class__.__name__] = train_transaction['D3'] / train_transaction['D4']
        self.test[self.__class__.__name__] = test_transaction['D3'] / test_transaction['D4']


class D3_div_D5(Feature):
    def create_features(self):
        train_transaction['D3'] = train_transaction['D3'].fillna(-1) + 0.01
        train_transaction['D5'] = train_transaction['D5'].fillna(-1) + 0.01

        test_transaction['D3'] = test_transaction['D3'].fillna(-1) + 0.01
        test_transaction['D5'] = test_transaction['D5'].fillna(-1) + 0.01

        self.train[self.__class__.__name__] = train_transaction['D3'] / train_transaction['D5']
        self.test[self.__class__.__name__] = test_transaction['D3'] / test_transaction['D5']


class D3_div_D6(Feature):
    def create_features(self):
        train_transaction['D3'] = train_transaction['D3'].fillna(-1) + 0.01
        train_transaction['D6'] = train_transaction['D6'].fillna(-1) + 0.01

        test_transaction['D3'] = test_transaction['D3'].fillna(-1) + 0.01
        test_transaction['D6'] = test_transaction['D6'].fillna(-1) + 0.01

        self.train[self.__class__.__name__] = train_transaction['D3'] / train_transaction['D6']
        self.test[self.__class__.__name__] = test_transaction['D3'] / test_transaction['D6']


class D3_div_D7(Feature):
    def create_features(self):
        train_transaction['D3'] = train_transaction['D3'].fillna(-1) + 0.01
        train_transaction['D7'] = train_transaction['D7'].fillna(-1) + 0.01

        test_transaction['D3'] = test_transaction['D3'].fillna(-1) + 0.01
        test_transaction['D7'] = test_transaction['D7'].fillna(-1) + 0.01

        self.train[self.__class__.__name__] = train_transaction['D3'] / train_transaction['D7']
        self.test[self.__class__.__name__] = test_transaction['D3'] / test_transaction['D7']


class D3_div_D8(Feature):
    def create_features(self):
        train_transaction['D3'] = train_transaction['D3'].fillna(-1) + 0.01
        train_transaction['D8'] = train_transaction['D8'].fillna(-1) + 0.01

        test_transaction['D3'] = test_transaction['D3'].fillna(-1) + 0.01
        test_transaction['D8'] = test_transaction['D8'].fillna(-1) + 0.01

        self.train[self.__class__.__name__] = train_transaction['D3'] / train_transaction['D8']
        self.test[self.__class__.__name__] = test_transaction['D3'] / test_transaction['D8']


class D3_div_D9(Feature):
    def create_features(self):
        train_transaction['D3'] = train_transaction['D3'].fillna(-1) + 0.01
        train_transaction['D9'] = train_transaction['D9'].fillna(-1) + 0.01

        test_transaction['D3'] = test_transaction['D3'].fillna(-1) + 0.01
        test_transaction['D9'] = test_transaction['D9'].fillna(-1) + 0.01

        self.train[self.__class__.__name__] = train_transaction['D3'] / train_transaction['D9']
        self.test[self.__class__.__name__] = test_transaction['D3'] / test_transaction['D9']


class D3_div_D10(Feature):
    def create_features(self):
        train_transaction['D3'] = train_transaction['D3'].fillna(-1) + 0.01
        train_transaction['D10'] = train_transaction['D10'].fillna(-1) + 0.01

        test_transaction['D3'] = test_transaction['D3'].fillna(-1) + 0.01
        test_transaction['D10'] = test_transaction['D10'].fillna(-1) + 0.01

        self.train[self.__class__.__name__] = train_transaction['D3'] / train_transaction['D10']
        self.test[self.__class__.__name__] = test_transaction['D3'] / test_transaction['D10']


class D3_div_D11(Feature):
    def create_features(self):
        train_transaction['D3'] = train_transaction['D3'].fillna(-1) + 0.01
        train_transaction['D11'] = train_transaction['D11'].fillna(-1) + 0.01

        test_transaction['D3'] = test_transaction['D3'].fillna(-1) + 0.01
        test_transaction['D11'] = test_transaction['D11'].fillna(-1) + 0.01

        self.train[self.__class__.__name__] = train_transaction['D3'] / train_transaction['D11']
        self.test[self.__class__.__name__] = test_transaction['D3'] / test_transaction['D11']


class D3_div_D12(Feature):
    def create_features(self):
        train_transaction['D3'] = train_transaction['D3'].fillna(-1) + 0.01
        train_transaction['D12'] = train_transaction['D12'].fillna(-1) + 0.01

        test_transaction['D3'] = test_transaction['D3'].fillna(-1) + 0.01
        test_transaction['D12'] = test_transaction['D12'].fillna(-1) + 0.01

        self.train[self.__class__.__name__] = train_transaction['D3'] / train_transaction['D12']
        self.test[self.__class__.__name__] = test_transaction['D3'] / test_transaction['D12']


class D3_div_D13(Feature):
    def create_features(self):
        train_transaction['D3'] = train_transaction['D3'].fillna(-1) + 0.01
        train_transaction['D13'] = train_transaction['D13'].fillna(-1) + 0.01

        test_transaction['D3'] = test_transaction['D3'].fillna(-1) + 0.01
        test_transaction['D13'] = test_transaction['D13'].fillna(-1) + 0.01

        self.train[self.__class__.__name__] = train_transaction['D3'] / train_transaction['D13']
        self.test[self.__class__.__name__] = test_transaction['D3'] / test_transaction['D13']


class D3_div_D14(Feature):
    def create_features(self):
        train_transaction['D3'] = train_transaction['D3'].fillna(-1) + 0.01
        train_transaction['D14'] = train_transaction['D14'].fillna(-1) + 0.01

        test_transaction['D3'] = test_transaction['D3'].fillna(-1) + 0.01
        test_transaction['D14'] = test_transaction['D14'].fillna(-1) + 0.01

        self.train[self.__class__.__name__] = train_transaction['D3'] / train_transaction['D14']
        self.test[self.__class__.__name__] = test_transaction['D3'] / test_transaction['D14']


class D3_div_D15(Feature):
    def create_features(self):
        train_transaction['D3'] = train_transaction['D3'].fillna(-1) + 0.01
        train_transaction['D15'] = train_transaction['D15'].fillna(-1) + 0.01

        test_transaction['D3'] = test_transaction['D3'].fillna(-1) + 0.01
        test_transaction['D15'] = test_transaction['D15'].fillna(-1) + 0.01

        self.train[self.__class__.__name__] = train_transaction['D3'] / train_transaction['D15']
        self.test[self.__class__.__name__] = test_transaction['D3'] / test_transaction['D15']


class D4_div_D5(Feature):
    def create_features(self):
        train_transaction['D4'] = train_transaction['D4'].fillna(-1) + 0.01
        train_transaction['D5'] = train_transaction['D5'].fillna(-1) + 0.01

        test_transaction['D4'] = test_transaction['D4'].fillna(-1) + 0.01
        test_transaction['D5'] = test_transaction['D5'].fillna(-1) + 0.01

        self.train[self.__class__.__name__] = train_transaction['D4'] / train_transaction['D5']
        self.test[self.__class__.__name__] = test_transaction['D4'] / test_transaction['D5']


class D4_div_D6(Feature):
    def create_features(self):
        train_transaction['D4'] = train_transaction['D4'].fillna(-1) + 0.01
        train_transaction['D6'] = train_transaction['D6'].fillna(-1) + 0.01

        test_transaction['D4'] = test_transaction['D4'].fillna(-1) + 0.01
        test_transaction['D6'] = test_transaction['D6'].fillna(-1) + 0.01

        self.train[self.__class__.__name__] = train_transaction['D4'] / train_transaction['D6']
        self.test[self.__class__.__name__] = test_transaction['D4'] / test_transaction['D6']


class D4_div_D7(Feature):
    def create_features(self):
        train_transaction['D4'] = train_transaction['D4'].fillna(-1) + 0.01
        train_transaction['D7'] = train_transaction['D7'].fillna(-1) + 0.01

        test_transaction['D4'] = test_transaction['D4'].fillna(-1) + 0.01
        test_transaction['D7'] = test_transaction['D7'].fillna(-1) + 0.01

        self.train[self.__class__.__name__] = train_transaction['D4'] / train_transaction['D7']
        self.test[self.__class__.__name__] = test_transaction['D4'] / test_transaction['D7']


class D4_div_D8(Feature):
    def create_features(self):
        train_transaction['D4'] = train_transaction['D4'].fillna(-1) + 0.01
        train_transaction['D8'] = train_transaction['D8'].fillna(-1) + 0.01

        test_transaction['D4'] = test_transaction['D4'].fillna(-1) + 0.01
        test_transaction['D8'] = test_transaction['D8'].fillna(-1) + 0.01

        self.train[self.__class__.__name__] = train_transaction['D4'] / train_transaction['D8']
        self.test[self.__class__.__name__] = test_transaction['D4'] / test_transaction['D8']


class D4_div_D9(Feature):
    def create_features(self):
        train_transaction['D4'] = train_transaction['D4'].fillna(-1) + 0.01
        train_transaction['D9'] = train_transaction['D9'].fillna(-1) + 0.01

        test_transaction['D4'] = test_transaction['D4'].fillna(-1) + 0.01
        test_transaction['D9'] = test_transaction['D9'].fillna(-1) + 0.01

        self.train[self.__class__.__name__] = train_transaction['D4'] / train_transaction['D9']
        self.test[self.__class__.__name__] = test_transaction['D4'] / test_transaction['D9']


class D4_div_D10(Feature):
    def create_features(self):
        train_transaction['D4'] = train_transaction['D4'].fillna(-1) + 0.01
        train_transaction['D10'] = train_transaction['D10'].fillna(-1) + 0.01

        test_transaction['D4'] = test_transaction['D4'].fillna(-1) + 0.01
        test_transaction['D10'] = test_transaction['D10'].fillna(-1) + 0.01

        self.train[self.__class__.__name__] = train_transaction['D4'] / train_transaction['D10']
        self.test[self.__class__.__name__] = test_transaction['D4'] / test_transaction['D10']


class D4_div_D11(Feature):
    def create_features(self):
        train_transaction['D4'] = train_transaction['D4'].fillna(-1) + 0.01
        train_transaction['D11'] = train_transaction['D11'].fillna(-1) + 0.01

        test_transaction['D4'] = test_transaction['D4'].fillna(-1) + 0.01
        test_transaction['D11'] = test_transaction['D11'].fillna(-1) + 0.01

        self.train[self.__class__.__name__] = train_transaction['D4'] / train_transaction['D11']
        self.test[self.__class__.__name__] = test_transaction['D4'] / test_transaction['D11']


class D4_div_D12(Feature):
    def create_features(self):
        train_transaction['D4'] = train_transaction['D4'].fillna(-1) + 0.01
        train_transaction['D12'] = train_transaction['D12'].fillna(-1) + 0.01

        test_transaction['D4'] = test_transaction['D4'].fillna(-1) + 0.01
        test_transaction['D12'] = test_transaction['D12'].fillna(-1) + 0.01

        self.train[self.__class__.__name__] = train_transaction['D4'] / train_transaction['D12']
        self.test[self.__class__.__name__] = test_transaction['D4'] / test_transaction['D12']


class D4_div_D13(Feature):
    def create_features(self):
        train_transaction['D4'] = train_transaction['D4'].fillna(-1) + 0.01
        train_transaction['D13'] = train_transaction['D13'].fillna(-1) + 0.01

        test_transaction['D4'] = test_transaction['D4'].fillna(-1) + 0.01
        test_transaction['D13'] = test_transaction['D13'].fillna(-1) + 0.01

        self.train[self.__class__.__name__] = train_transaction['D4'] / train_transaction['D13']
        self.test[self.__class__.__name__] = test_transaction['D4'] / test_transaction['D13']


class D4_div_D14(Feature):
    def create_features(self):
        train_transaction['D4'] = train_transaction['D4'].fillna(-1) + 0.01
        train_transaction['D14'] = train_transaction['D14'].fillna(-1) + 0.01

        test_transaction['D4'] = test_transaction['D4'].fillna(-1) + 0.01
        test_transaction['D14'] = test_transaction['D14'].fillna(-1) + 0.01

        self.train[self.__class__.__name__] = train_transaction['D4'] / train_transaction['D14']
        self.test[self.__class__.__name__] = test_transaction['D4'] / test_transaction['D14']


class D4_div_D15(Feature):
    def create_features(self):
        train_transaction['D4'] = train_transaction['D4'].fillna(-1) + 0.01
        train_transaction['D15'] = train_transaction['D15'].fillna(-1) + 0.01

        test_transaction['D4'] = test_transaction['D4'].fillna(-1) + 0.01
        test_transaction['D15'] = test_transaction['D15'].fillna(-1) + 0.01

        self.train[self.__class__.__name__] = train_transaction['D4'] / train_transaction['D15']
        self.test[self.__class__.__name__] = test_transaction['D4'] / test_transaction['D15']


class D5_div_D6(Feature):
    def create_features(self):
        train_transaction['D5'] = train_transaction['D5'].fillna(-1) + 0.01
        train_transaction['D6'] = train_transaction['D6'].fillna(-1) + 0.01

        test_transaction['D5'] = test_transaction['D5'].fillna(-1) + 0.01
        test_transaction['D6'] = test_transaction['D6'].fillna(-1) + 0.01

        self.train[self.__class__.__name__] = train_transaction['D5'] / train_transaction['D6']
        self.test[self.__class__.__name__] = test_transaction['D5'] / test_transaction['D6']


class D5_div_D7(Feature):
    def create_features(self):
        train_transaction['D5'] = train_transaction['D5'].fillna(-1) + 0.01
        train_transaction['D7'] = train_transaction['D7'].fillna(-1) + 0.01

        test_transaction['D5'] = test_transaction['D5'].fillna(-1) + 0.01
        test_transaction['D7'] = test_transaction['D7'].fillna(-1) + 0.01

        self.train[self.__class__.__name__] = train_transaction['D5'] / train_transaction['D7']
        self.test[self.__class__.__name__] = test_transaction['D5'] / test_transaction['D7']


class D5_div_D8(Feature):
    def create_features(self):
        train_transaction['D5'] = train_transaction['D5'].fillna(-1) + 0.01
        train_transaction['D8'] = train_transaction['D8'].fillna(-1) + 0.01

        test_transaction['D5'] = test_transaction['D5'].fillna(-1) + 0.01
        test_transaction['D8'] = test_transaction['D8'].fillna(-1) + 0.01

        self.train[self.__class__.__name__] = train_transaction['D5'] / train_transaction['D8']
        self.test[self.__class__.__name__] = test_transaction['D5'] / test_transaction['D8']


class D5_div_D9(Feature):
    def create_features(self):
        train_transaction['D5'] = train_transaction['D5'].fillna(-1) + 0.01
        train_transaction['D9'] = train_transaction['D9'].fillna(-1) + 0.01

        test_transaction['D5'] = test_transaction['D5'].fillna(-1) + 0.01
        test_transaction['D9'] = test_transaction['D9'].fillna(-1) + 0.01

        self.train[self.__class__.__name__] = train_transaction['D5'] / train_transaction['D9']
        self.test[self.__class__.__name__] = test_transaction['D5'] / test_transaction['D9']


class D5_div_D10(Feature):
    def create_features(self):
        train_transaction['D5'] = train_transaction['D5'].fillna(-1) + 0.01
        train_transaction['D10'] = train_transaction['D10'].fillna(-1) + 0.01

        test_transaction['D5'] = test_transaction['D5'].fillna(-1) + 0.01
        test_transaction['D10'] = test_transaction['D10'].fillna(-1) + 0.01

        self.train[self.__class__.__name__] = train_transaction['D5'] / train_transaction['D10']
        self.test[self.__class__.__name__] = test_transaction['D5'] / test_transaction['D10']


class D5_div_D11(Feature):
    def create_features(self):
        train_transaction['D5'] = train_transaction['D5'].fillna(-1) + 0.01
        train_transaction['D11'] = train_transaction['D11'].fillna(-1) + 0.01

        test_transaction['D5'] = test_transaction['D5'].fillna(-1) + 0.01
        test_transaction['D11'] = test_transaction['D11'].fillna(-1) + 0.01

        self.train[self.__class__.__name__] = train_transaction['D5'] / train_transaction['D11']
        self.test[self.__class__.__name__] = test_transaction['D5'] / test_transaction['D11']


class D5_div_D12(Feature):
    def create_features(self):
        train_transaction['D5'] = train_transaction['D5'].fillna(-1) + 0.01
        train_transaction['D12'] = train_transaction['D12'].fillna(-1) + 0.01

        test_transaction['D5'] = test_transaction['D5'].fillna(-1) + 0.01
        test_transaction['D12'] = test_transaction['D12'].fillna(-1) + 0.01

        self.train[self.__class__.__name__] = train_transaction['D5'] / train_transaction['D12']
        self.test[self.__class__.__name__] = test_transaction['D5'] / test_transaction['D12']


class D5_div_D13(Feature):
    def create_features(self):
        train_transaction['D5'] = train_transaction['D5'].fillna(-1) + 0.01
        train_transaction['D13'] = train_transaction['D13'].fillna(-1) + 0.01

        test_transaction['D5'] = test_transaction['D5'].fillna(-1) + 0.01
        test_transaction['D13'] = test_transaction['D13'].fillna(-1) + 0.01

        self.train[self.__class__.__name__] = train_transaction['D5'] / train_transaction['D13']
        self.test[self.__class__.__name__] = test_transaction['D5'] / test_transaction['D13']


class D5_div_D14(Feature):
    def create_features(self):
        train_transaction['D5'] = train_transaction['D5'].fillna(-1) + 0.01
        train_transaction['D14'] = train_transaction['D14'].fillna(-1) + 0.01

        test_transaction['D5'] = test_transaction['D5'].fillna(-1) + 0.01
        test_transaction['D14'] = test_transaction['D14'].fillna(-1) + 0.01

        self.train[self.__class__.__name__] = train_transaction['D5'] / train_transaction['D14']
        self.test[self.__class__.__name__] = test_transaction['D5'] / test_transaction['D14']


class D5_div_D15(Feature):
    def create_features(self):
        train_transaction['D5'] = train_transaction['D5'].fillna(-1) + 0.01
        train_transaction['D15'] = train_transaction['D15'].fillna(-1) + 0.01

        test_transaction['D5'] = test_transaction['D5'].fillna(-1) + 0.01
        test_transaction['D15'] = test_transaction['D15'].fillna(-1) + 0.01

        self.train[self.__class__.__name__] = train_transaction['D5'] / train_transaction['D15']
        self.test[self.__class__.__name__] = test_transaction['D5'] / test_transaction['D15']


class D6_div_D7(Feature):
    def create_features(self):
        train_transaction['D6'] = train_transaction['D6'].fillna(-1) + 0.01
        train_transaction['D7'] = train_transaction['D7'].fillna(-1) + 0.01

        test_transaction['D6'] = test_transaction['D6'].fillna(-1) + 0.01
        test_transaction['D7'] = test_transaction['D7'].fillna(-1) + 0.01

        self.train[self.__class__.__name__] = train_transaction['D6'] / train_transaction['D7']
        self.test[self.__class__.__name__] = test_transaction['D6'] / test_transaction['D7']


class D6_div_D8(Feature):
    def create_features(self):
        train_transaction['D6'] = train_transaction['D6'].fillna(-1) + 0.01
        train_transaction['D8'] = train_transaction['D8'].fillna(-1) + 0.01

        test_transaction['D6'] = test_transaction['D6'].fillna(-1) + 0.01
        test_transaction['D8'] = test_transaction['D8'].fillna(-1) + 0.01

        self.train[self.__class__.__name__] = train_transaction['D6'] / train_transaction['D8']
        self.test[self.__class__.__name__] = test_transaction['D6'] / test_transaction['D8']


class D6_div_D9(Feature):
    def create_features(self):
        train_transaction['D6'] = train_transaction['D6'].fillna(-1) + 0.01
        train_transaction['D9'] = train_transaction['D9'].fillna(-1) + 0.01

        test_transaction['D6'] = test_transaction['D6'].fillna(-1) + 0.01
        test_transaction['D9'] = test_transaction['D9'].fillna(-1) + 0.01

        self.train[self.__class__.__name__] = train_transaction['D6'] / train_transaction['D9']
        self.test[self.__class__.__name__] = test_transaction['D6'] / test_transaction['D9']


class D6_div_D10(Feature):
    def create_features(self):
        train_transaction['D6'] = train_transaction['D6'].fillna(-1) + 0.01
        train_transaction['D10'] = train_transaction['D10'].fillna(-1) + 0.01

        test_transaction['D6'] = test_transaction['D6'].fillna(-1) + 0.01
        test_transaction['D10'] = test_transaction['D10'].fillna(-1) + 0.01

        self.train[self.__class__.__name__] = train_transaction['D6'] / train_transaction['D10']
        self.test[self.__class__.__name__] = test_transaction['D6'] / test_transaction['D10']


class D6_div_D11(Feature):
    def create_features(self):
        train_transaction['D6'] = train_transaction['D6'].fillna(-1) + 0.01
        train_transaction['D11'] = train_transaction['D11'].fillna(-1) + 0.01

        test_transaction['D6'] = test_transaction['D6'].fillna(-1) + 0.01
        test_transaction['D11'] = test_transaction['D11'].fillna(-1) + 0.01

        self.train[self.__class__.__name__] = train_transaction['D6'] / train_transaction['D11']
        self.test[self.__class__.__name__] = test_transaction['D6'] / test_transaction['D11']


class D6_div_D12(Feature):
    def create_features(self):
        train_transaction['D6'] = train_transaction['D6'].fillna(-1) + 0.01
        train_transaction['D12'] = train_transaction['D12'].fillna(-1) + 0.01

        test_transaction['D6'] = test_transaction['D6'].fillna(-1) + 0.01
        test_transaction['D12'] = test_transaction['D12'].fillna(-1) + 0.01

        self.train[self.__class__.__name__] = train_transaction['D6'] / train_transaction['D12']
        self.test[self.__class__.__name__] = test_transaction['D6'] / test_transaction['D12']


class D6_div_D13(Feature):
    def create_features(self):
        train_transaction['D6'] = train_transaction['D6'].fillna(-1) + 0.01
        train_transaction['D13'] = train_transaction['D13'].fillna(-1) + 0.01

        test_transaction['D6'] = test_transaction['D6'].fillna(-1) + 0.01
        test_transaction['D13'] = test_transaction['D13'].fillna(-1) + 0.01

        self.train[self.__class__.__name__] = train_transaction['D6'] / train_transaction['D13']
        self.test[self.__class__.__name__] = test_transaction['D6'] / test_transaction['D13']


class D6_div_D14(Feature):
    def create_features(self):
        train_transaction['D6'] = train_transaction['D6'].fillna(-1) + 0.01
        train_transaction['D14'] = train_transaction['D14'].fillna(-1) + 0.01

        test_transaction['D6'] = test_transaction['D6'].fillna(-1) + 0.01
        test_transaction['D14'] = test_transaction['D14'].fillna(-1) + 0.01

        self.train[self.__class__.__name__] = train_transaction['D6'] / train_transaction['D14']
        self.test[self.__class__.__name__] = test_transaction['D6'] / test_transaction['D14']


class D6_div_D15(Feature):
    def create_features(self):
        train_transaction['D6'] = train_transaction['D6'].fillna(-1) + 0.01
        train_transaction['D15'] = train_transaction['D15'].fillna(-1) + 0.01

        test_transaction['D6'] = test_transaction['D6'].fillna(-1) + 0.01
        test_transaction['D15'] = test_transaction['D15'].fillna(-1) + 0.01

        self.train[self.__class__.__name__] = train_transaction['D6'] / train_transaction['D15']
        self.test[self.__class__.__name__] = test_transaction['D6'] / test_transaction['D15']


class D7_div_D8(Feature):
    def create_features(self):
        train_transaction['D7'] = train_transaction['D7'].fillna(-1) + 0.01
        train_transaction['D8'] = train_transaction['D8'].fillna(-1) + 0.01

        test_transaction['D7'] = test_transaction['D7'].fillna(-1) + 0.01
        test_transaction['D8'] = test_transaction['D8'].fillna(-1) + 0.01

        self.train[self.__class__.__name__] = train_transaction['D7'] / train_transaction['D8']
        self.test[self.__class__.__name__] = test_transaction['D7'] / test_transaction['D8']


class D7_div_D9(Feature):
    def create_features(self):
        train_transaction['D7'] = train_transaction['D7'].fillna(-1) + 0.01
        train_transaction['D9'] = train_transaction['D9'].fillna(-1) + 0.01

        test_transaction['D7'] = test_transaction['D7'].fillna(-1) + 0.01
        test_transaction['D9'] = test_transaction['D9'].fillna(-1) + 0.01

        self.train[self.__class__.__name__] = train_transaction['D7'] / train_transaction['D9']
        self.test[self.__class__.__name__] = test_transaction['D7'] / test_transaction['D9']


class D7_div_D10(Feature):
    def create_features(self):
        train_transaction['D7'] = train_transaction['D7'].fillna(-1) + 0.01
        train_transaction['D10'] = train_transaction['D10'].fillna(-1) + 0.01

        test_transaction['D7'] = test_transaction['D7'].fillna(-1) + 0.01
        test_transaction['D10'] = test_transaction['D10'].fillna(-1) + 0.01

        self.train[self.__class__.__name__] = train_transaction['D7'] / train_transaction['D10']
        self.test[self.__class__.__name__] = test_transaction['D7'] / test_transaction['D10']


class D7_div_D11(Feature):
    def create_features(self):
        train_transaction['D7'] = train_transaction['D7'].fillna(-1) + 0.01
        train_transaction['D11'] = train_transaction['D11'].fillna(-1) + 0.01

        test_transaction['D7'] = test_transaction['D7'].fillna(-1) + 0.01
        test_transaction['D11'] = test_transaction['D11'].fillna(-1) + 0.01

        self.train[self.__class__.__name__] = train_transaction['D7'] / train_transaction['D11']
        self.test[self.__class__.__name__] = test_transaction['D7'] / test_transaction['D11']


class D7_div_D12(Feature):
    def create_features(self):
        train_transaction['D7'] = train_transaction['D7'].fillna(-1) + 0.01
        train_transaction['D12'] = train_transaction['D12'].fillna(-1) + 0.01

        test_transaction['D7'] = test_transaction['D7'].fillna(-1) + 0.01
        test_transaction['D12'] = test_transaction['D12'].fillna(-1) + 0.01

        self.train[self.__class__.__name__] = train_transaction['D7'] / train_transaction['D12']
        self.test[self.__class__.__name__] = test_transaction['D7'] / test_transaction['D12']


class D7_div_D13(Feature):
    def create_features(self):
        train_transaction['D7'] = train_transaction['D7'].fillna(-1) + 0.01
        train_transaction['D13'] = train_transaction['D13'].fillna(-1) + 0.01

        test_transaction['D7'] = test_transaction['D7'].fillna(-1) + 0.01
        test_transaction['D13'] = test_transaction['D13'].fillna(-1) + 0.01

        self.train[self.__class__.__name__] = train_transaction['D7'] / train_transaction['D13']
        self.test[self.__class__.__name__] = test_transaction['D7'] / test_transaction['D13']


class D7_div_D14(Feature):
    def create_features(self):
        train_transaction['D7'] = train_transaction['D7'].fillna(-1) + 0.01
        train_transaction['D14'] = train_transaction['D14'].fillna(-1) + 0.01

        test_transaction['D7'] = test_transaction['D7'].fillna(-1) + 0.01
        test_transaction['D14'] = test_transaction['D14'].fillna(-1) + 0.01

        self.train[self.__class__.__name__] = train_transaction['D7'] / train_transaction['D14']
        self.test[self.__class__.__name__] = test_transaction['D7'] / test_transaction['D14']


class D7_div_D15(Feature):
    def create_features(self):
        train_transaction['D7'] = train_transaction['D7'].fillna(-1) + 0.01
        train_transaction['D15'] = train_transaction['D15'].fillna(-1) + 0.01

        test_transaction['D7'] = test_transaction['D7'].fillna(-1) + 0.01
        test_transaction['D15'] = test_transaction['D15'].fillna(-1) + 0.01

        self.train[self.__class__.__name__] = train_transaction['D7'] / train_transaction['D15']
        self.test[self.__class__.__name__] = test_transaction['D7'] / test_transaction['D15']


class D8_div_D9(Feature):
    def create_features(self):
        train_transaction['D8'] = train_transaction['D8'].fillna(-1) + 0.01
        train_transaction['D9'] = train_transaction['D9'].fillna(-1) + 0.01

        test_transaction['D8'] = test_transaction['D8'].fillna(-1) + 0.01
        test_transaction['D9'] = test_transaction['D9'].fillna(-1) + 0.01

        self.train[self.__class__.__name__] = train_transaction['D8'] / train_transaction['D9']
        self.test[self.__class__.__name__] = test_transaction['D8'] / test_transaction['D9']


class D8_div_D10(Feature):
    def create_features(self):
        train_transaction['D8'] = train_transaction['D8'].fillna(-1) + 0.01
        train_transaction['D10'] = train_transaction['D10'].fillna(-1) + 0.01

        test_transaction['D8'] = test_transaction['D8'].fillna(-1) + 0.01
        test_transaction['D10'] = test_transaction['D10'].fillna(-1) + 0.01

        self.train[self.__class__.__name__] = train_transaction['D8'] / train_transaction['D10']
        self.test[self.__class__.__name__] = test_transaction['D8'] / test_transaction['D10']


class D8_div_D11(Feature):
    def create_features(self):
        train_transaction['D8'] = train_transaction['D8'].fillna(-1) + 0.01
        train_transaction['D11'] = train_transaction['D11'].fillna(-1) + 0.01

        test_transaction['D8'] = test_transaction['D8'].fillna(-1) + 0.01
        test_transaction['D11'] = test_transaction['D11'].fillna(-1) + 0.01

        self.train[self.__class__.__name__] = train_transaction['D8'] / train_transaction['D11']
        self.test[self.__class__.__name__] = test_transaction['D8'] / test_transaction['D11']


class D8_div_D12(Feature):
    def create_features(self):
        train_transaction['D8'] = train_transaction['D8'].fillna(-1) + 0.01
        train_transaction['D12'] = train_transaction['D12'].fillna(-1) + 0.01

        test_transaction['D8'] = test_transaction['D8'].fillna(-1) + 0.01
        test_transaction['D12'] = test_transaction['D12'].fillna(-1) + 0.01

        self.train[self.__class__.__name__] = train_transaction['D8'] / train_transaction['D12']
        self.test[self.__class__.__name__] = test_transaction['D8'] / test_transaction['D12']


class D8_div_D13(Feature):
    def create_features(self):
        train_transaction['D8'] = train_transaction['D8'].fillna(-1) + 0.01
        train_transaction['D13'] = train_transaction['D13'].fillna(-1) + 0.01

        test_transaction['D8'] = test_transaction['D8'].fillna(-1) + 0.01
        test_transaction['D13'] = test_transaction['D13'].fillna(-1) + 0.01

        self.train[self.__class__.__name__] = train_transaction['D8'] / train_transaction['D13']
        self.test[self.__class__.__name__] = test_transaction['D8'] / test_transaction['D13']


class D8_div_D14(Feature):
    def create_features(self):
        train_transaction['D8'] = train_transaction['D8'].fillna(-1) + 0.01
        train_transaction['D14'] = train_transaction['D14'].fillna(-1) + 0.01

        test_transaction['D8'] = test_transaction['D8'].fillna(-1) + 0.01
        test_transaction['D14'] = test_transaction['D14'].fillna(-1) + 0.01

        self.train[self.__class__.__name__] = train_transaction['D8'] / train_transaction['D14']
        self.test[self.__class__.__name__] = test_transaction['D8'] / test_transaction['D14']


class D8_div_D15(Feature):
    def create_features(self):
        train_transaction['D8'] = train_transaction['D8'].fillna(-1) + 0.01
        train_transaction['D15'] = train_transaction['D15'].fillna(-1) + 0.01

        test_transaction['D8'] = test_transaction['D8'].fillna(-1) + 0.01
        test_transaction['D15'] = test_transaction['D15'].fillna(-1) + 0.01

        self.train[self.__class__.__name__] = train_transaction['D8'] / train_transaction['D15']
        self.test[self.__class__.__name__] = test_transaction['D8'] / test_transaction['D15']


class D9_div_D10(Feature):
    def create_features(self):
        train_transaction['D9'] = train_transaction['D9'].fillna(-1) + 0.01
        train_transaction['D10'] = train_transaction['D10'].fillna(-1) + 0.01

        test_transaction['D9'] = test_transaction['D9'].fillna(-1) + 0.01
        test_transaction['D10'] = test_transaction['D10'].fillna(-1) + 0.01

        self.train[self.__class__.__name__] = train_transaction['D9'] / train_transaction['D10']
        self.test[self.__class__.__name__] = test_transaction['D9'] / test_transaction['D10']


class D9_div_D11(Feature):
    def create_features(self):
        train_transaction['D9'] = train_transaction['D9'].fillna(-1) + 0.01
        train_transaction['D11'] = train_transaction['D11'].fillna(-1) + 0.01

        test_transaction['D9'] = test_transaction['D9'].fillna(-1) + 0.01
        test_transaction['D11'] = test_transaction['D11'].fillna(-1) + 0.01

        self.train[self.__class__.__name__] = train_transaction['D9'] / train_transaction['D11']
        self.test[self.__class__.__name__] = test_transaction['D9'] / test_transaction['D11']


class D9_div_D12(Feature):
    def create_features(self):
        train_transaction['D9'] = train_transaction['D9'].fillna(-1) + 0.01
        train_transaction['D12'] = train_transaction['D12'].fillna(-1) + 0.01

        test_transaction['D9'] = test_transaction['D9'].fillna(-1) + 0.01
        test_transaction['D12'] = test_transaction['D12'].fillna(-1) + 0.01

        self.train[self.__class__.__name__] = train_transaction['D9'] / train_transaction['D12']
        self.test[self.__class__.__name__] = test_transaction['D9'] / test_transaction['D12']


class D9_div_D13(Feature):
    def create_features(self):
        train_transaction['D9'] = train_transaction['D9'].fillna(-1) + 0.01
        train_transaction['D13'] = train_transaction['D13'].fillna(-1) + 0.01

        test_transaction['D9'] = test_transaction['D9'].fillna(-1) + 0.01
        test_transaction['D13'] = test_transaction['D13'].fillna(-1) + 0.01

        self.train[self.__class__.__name__] = train_transaction['D9'] / train_transaction['D13']
        self.test[self.__class__.__name__] = test_transaction['D9'] / test_transaction['D13']


class D9_div_D14(Feature):
    def create_features(self):
        train_transaction['D9'] = train_transaction['D9'].fillna(-1) + 0.01
        train_transaction['D14'] = train_transaction['D14'].fillna(-1) + 0.01

        test_transaction['D9'] = test_transaction['D9'].fillna(-1) + 0.01
        test_transaction['D14'] = test_transaction['D14'].fillna(-1) + 0.01

        self.train[self.__class__.__name__] = train_transaction['D9'] / train_transaction['D14']
        self.test[self.__class__.__name__] = test_transaction['D9'] / test_transaction['D14']


class D9_div_D15(Feature):
    def create_features(self):
        train_transaction['D9'] = train_transaction['D9'].fillna(-1) + 0.01
        train_transaction['D15'] = train_transaction['D15'].fillna(-1) + 0.01

        test_transaction['D9'] = test_transaction['D9'].fillna(-1) + 0.01
        test_transaction['D15'] = test_transaction['D15'].fillna(-1) + 0.01

        self.train[self.__class__.__name__] = train_transaction['D9'] / train_transaction['D15']
        self.test[self.__class__.__name__] = test_transaction['D9'] / test_transaction['D15']


class D10_div_D11(Feature):
    def create_features(self):
        train_transaction['D10'] = train_transaction['D10'].fillna(-1) + 0.01
        train_transaction['D11'] = train_transaction['D11'].fillna(-1) + 0.01

        test_transaction['D10'] = test_transaction['D10'].fillna(-1) + 0.01
        test_transaction['D11'] = test_transaction['D11'].fillna(-1) + 0.01

        self.train[self.__class__.__name__] = train_transaction['D10'] / train_transaction['D11']
        self.test[self.__class__.__name__] = test_transaction['D10'] / test_transaction['D11']


class D10_div_D12(Feature):
    def create_features(self):
        train_transaction['D10'] = train_transaction['D10'].fillna(-1) + 0.01
        train_transaction['D12'] = train_transaction['D12'].fillna(-1) + 0.01

        test_transaction['D10'] = test_transaction['D10'].fillna(-1) + 0.01
        test_transaction['D12'] = test_transaction['D12'].fillna(-1) + 0.01

        self.train[self.__class__.__name__] = train_transaction['D10'] / train_transaction['D12']
        self.test[self.__class__.__name__] = test_transaction['D10'] / test_transaction['D12']


class D10_div_D13(Feature):
    def create_features(self):
        train_transaction['D10'] = train_transaction['D10'].fillna(-1) + 0.01
        train_transaction['D13'] = train_transaction['D13'].fillna(-1) + 0.01

        test_transaction['D10'] = test_transaction['D10'].fillna(-1) + 0.01
        test_transaction['D13'] = test_transaction['D13'].fillna(-1) + 0.01

        self.train[self.__class__.__name__] = train_transaction['D10'] / train_transaction['D13']
        self.test[self.__class__.__name__] = test_transaction['D10'] / test_transaction['D13']


class D10_div_D14(Feature):
    def create_features(self):
        train_transaction['D10'] = train_transaction['D10'].fillna(-1) + 0.01
        train_transaction['D14'] = train_transaction['D14'].fillna(-1) + 0.01

        test_transaction['D10'] = test_transaction['D10'].fillna(-1) + 0.01
        test_transaction['D14'] = test_transaction['D14'].fillna(-1) + 0.01

        self.train[self.__class__.__name__] = train_transaction['D10'] / train_transaction['D14']
        self.test[self.__class__.__name__] = test_transaction['D10'] / test_transaction['D14']


class D10_div_D15(Feature):
    def create_features(self):
        train_transaction['D10'] = train_transaction['D10'].fillna(-1) + 0.01
        train_transaction['D15'] = train_transaction['D15'].fillna(-1) + 0.01

        test_transaction['D10'] = test_transaction['D10'].fillna(-1) + 0.01
        test_transaction['D15'] = test_transaction['D15'].fillna(-1) + 0.01

        self.train[self.__class__.__name__] = train_transaction['D10'] / train_transaction['D15']
        self.test[self.__class__.__name__] = test_transaction['D10'] / test_transaction['D15']


class D11_div_D12(Feature):
    def create_features(self):
        train_transaction['D11'] = train_transaction['D11'].fillna(-1) + 0.01
        train_transaction['D12'] = train_transaction['D12'].fillna(-1) + 0.01

        test_transaction['D11'] = test_transaction['D11'].fillna(-1) + 0.01
        test_transaction['D12'] = test_transaction['D12'].fillna(-1) + 0.01

        self.train[self.__class__.__name__] = train_transaction['D11'] / train_transaction['D12']
        self.test[self.__class__.__name__] = test_transaction['D11'] / test_transaction['D12']


class D11_div_D13(Feature):
    def create_features(self):
        train_transaction['D11'] = train_transaction['D11'].fillna(-1) + 0.01
        train_transaction['D13'] = train_transaction['D13'].fillna(-1) + 0.01

        test_transaction['D11'] = test_transaction['D11'].fillna(-1) + 0.01
        test_transaction['D13'] = test_transaction['D13'].fillna(-1) + 0.01

        self.train[self.__class__.__name__] = train_transaction['D11'] / train_transaction['D13']
        self.test[self.__class__.__name__] = test_transaction['D11'] / test_transaction['D13']


class D11_div_D14(Feature):
    def create_features(self):
        train_transaction['D11'] = train_transaction['D11'].fillna(-1) + 0.01
        train_transaction['D14'] = train_transaction['D14'].fillna(-1) + 0.01

        test_transaction['D11'] = test_transaction['D11'].fillna(-1) + 0.01
        test_transaction['D14'] = test_transaction['D14'].fillna(-1) + 0.01

        self.train[self.__class__.__name__] = train_transaction['D11'] / train_transaction['D14']
        self.test[self.__class__.__name__] = test_transaction['D11'] / test_transaction['D14']


class D11_div_D15(Feature):
    def create_features(self):
        train_transaction['D11'] = train_transaction['D11'].fillna(-1) + 0.01
        train_transaction['D15'] = train_transaction['D15'].fillna(-1) + 0.01

        test_transaction['D11'] = test_transaction['D11'].fillna(-1) + 0.01
        test_transaction['D15'] = test_transaction['D15'].fillna(-1) + 0.01

        self.train[self.__class__.__name__] = train_transaction['D11'] / train_transaction['D15']
        self.test[self.__class__.__name__] = test_transaction['D11'] / test_transaction['D15']


class D12_div_D13(Feature):
    def create_features(self):
        train_transaction['D12'] = train_transaction['D12'].fillna(-1) + 0.01
        train_transaction['D13'] = train_transaction['D13'].fillna(-1) + 0.01

        test_transaction['D12'] = test_transaction['D12'].fillna(-1) + 0.01
        test_transaction['D13'] = test_transaction['D13'].fillna(-1) + 0.01

        self.train[self.__class__.__name__] = train_transaction['D12'] / train_transaction['D13']
        self.test[self.__class__.__name__] = test_transaction['D12'] / test_transaction['D13']


class D12_div_D14(Feature):
    def create_features(self):
        train_transaction['D12'] = train_transaction['D12'].fillna(-1) + 0.01
        train_transaction['D14'] = train_transaction['D14'].fillna(-1) + 0.01

        test_transaction['D12'] = test_transaction['D12'].fillna(-1) + 0.01
        test_transaction['D14'] = test_transaction['D14'].fillna(-1) + 0.01

        self.train[self.__class__.__name__] = train_transaction['D12'] / train_transaction['D14']
        self.test[self.__class__.__name__] = test_transaction['D12'] / test_transaction['D14']


class D12_div_D15(Feature):
    def create_features(self):
        train_transaction['D12'] = train_transaction['D12'].fillna(-1) + 0.01
        train_transaction['D15'] = train_transaction['D15'].fillna(-1) + 0.01

        test_transaction['D12'] = test_transaction['D12'].fillna(-1) + 0.01
        test_transaction['D15'] = test_transaction['D15'].fillna(-1) + 0.01

        self.train[self.__class__.__name__] = train_transaction['D12'] / train_transaction['D15']
        self.test[self.__class__.__name__] = test_transaction['D12'] / test_transaction['D15']


class D13_div_D14(Feature):
    def create_features(self):
        train_transaction['D13'] = train_transaction['D13'].fillna(-1) + 0.01
        train_transaction['D14'] = train_transaction['D14'].fillna(-1) + 0.01

        test_transaction['D13'] = test_transaction['D13'].fillna(-1) + 0.01
        test_transaction['D14'] = test_transaction['D14'].fillna(-1) + 0.01

        self.train[self.__class__.__name__] = train_transaction['D13'] / train_transaction['D14']
        self.test[self.__class__.__name__] = test_transaction['D13'] / test_transaction['D14']


class D13_div_D15(Feature):
    def create_features(self):
        train_transaction['D13'] = train_transaction['D13'].fillna(-1) + 0.01
        train_transaction['D15'] = train_transaction['D15'].fillna(-1) + 0.01

        test_transaction['D13'] = test_transaction['D13'].fillna(-1) + 0.01
        test_transaction['D15'] = test_transaction['D15'].fillna(-1) + 0.01

        self.train[self.__class__.__name__] = train_transaction['D13'] / train_transaction['D15']
        self.test[self.__class__.__name__] = test_transaction['D13'] / test_transaction['D15']


class D14_div_D15(Feature):
    def create_features(self):
        train_transaction['D14'] = train_transaction['D14'].fillna(-1) + 0.01
        train_transaction['D15'] = train_transaction['D15'].fillna(-1) + 0.01

        test_transaction['D14'] = test_transaction['D14'].fillna(-1) + 0.01
        test_transaction['D15'] = test_transaction['D15'].fillna(-1) + 0.01

        self.train[self.__class__.__name__] = train_transaction['D14'] / train_transaction['D15']
        self.test[self.__class__.__name__] = test_transaction['D14'] / test_transaction['D15']


class sum_V154_V156_V257_V258(Feature):
    def create_features(self):
        feats = ['V154', 'V156', 'V257', 'V258']
        self.train[self.__class__.__name__] = np.sum(train_transaction[feats].fillna(0).values, axis=1)
        self.test[self.__class__.__name__] = np.sum(test_transaction[feats].fillna(0).values, axis=1)


class dec_flg(Feature):
    def create_features(self):
        col = 'TransactionDT'
        whole = pd.concat([train_transaction[col], test_transaction[col]], axis=0).to_frame()
        whole['datetime_'] = whole[col].apply(lambda x: (START_DATE + datetime.timedelta(seconds=x)))
        whole['month'] = whole['datetime_'].apply(lambda x: x.month)
        self.train[self.__class__.__name__] = np.where(whole['month'][:len(train_transaction)].values == 12, 1, 0)
        self.test[self.__class__.__name__] = np.where(whole['month'][len(train_transaction):].values == 12, 1, 0)



class V310_div_V307(Feature):
    def create_features(self):
        self.train[self.__class__.__name__] = train_transaction['V310'] / (train_transaction['V307'] + 1e-3)
        self.test[self.__class__.__name__] = test_transaction['V310'] / (test_transaction['V307'] + 1e-3)


class V310_div_V313(Feature):
    def create_features(self):
        self.train[self.__class__.__name__] = train_transaction['V310'] / (train_transaction['V313'] + 1e-3)
        self.test[self.__class__.__name__] = test_transaction['V310'] / (test_transaction['V313'] + 1e-3)


class V310_div_V130(Feature):
    def create_features(self):
        self.train[self.__class__.__name__] = train_transaction['V310'] / (train_transaction['V130'] + 1e-3)
        self.test[self.__class__.__name__] = test_transaction['V310'] / (test_transaction['V130'] + 1e-3)


class V310_div_V314(Feature):
    def create_features(self):
        self.train[self.__class__.__name__] = train_transaction['V310'] / (train_transaction['V314'] + 1e-3)
        self.test[self.__class__.__name__] = test_transaction['V310'] / (test_transaction['V314'] + 1e-3)


class V310_div_V315(Feature):
    def create_features(self):
        self.train[self.__class__.__name__] = train_transaction['V310'] / (train_transaction['V315'] + 1e-3)
        self.test[self.__class__.__name__] = test_transaction['V310'] / (test_transaction['V315'] + 1e-3)


class V310_div_V317(Feature):
    def create_features(self):
        self.train[self.__class__.__name__] = train_transaction['V310'] / (train_transaction['V317'] + 1e-3)
        self.test[self.__class__.__name__] = test_transaction['V310'] / (test_transaction['V317'] + 1e-3)


class V310_div_V285(Feature):
    def create_features(self):
        self.train[self.__class__.__name__] = train_transaction['V310'] / (train_transaction['V285'] + 1e-3)
        self.test[self.__class__.__name__] = test_transaction['V310'] / (test_transaction['V285'] + 1e-3)


class V310_div_V283(Feature):
    def create_features(self):
        self.train[self.__class__.__name__] = train_transaction['V310'] / (train_transaction['V283'] + 1e-3)
        self.test[self.__class__.__name__] = test_transaction['V310'] / (test_transaction['V283'] + 1e-3)


class V310_div_V312(Feature):
    def create_features(self):
        self.train[self.__class__.__name__] = train_transaction['V310'] / (train_transaction['V312'] + 1e-3)
        self.test[self.__class__.__name__] = test_transaction['V310'] / (test_transaction['V312'] + 1e-3)


class V310_div_V308(Feature):
    def create_features(self):
        self.train[self.__class__.__name__] = train_transaction['V310'] / (train_transaction['V308'] + 1e-3)
        self.test[self.__class__.__name__] = test_transaction['V310'] / (test_transaction['V308'] + 1e-3)


class V310_div_V127(Feature):
    def create_features(self):
        self.train[self.__class__.__name__] = train_transaction['V310'] / (train_transaction['V127'] + 1e-3)
        self.test[self.__class__.__name__] = test_transaction['V310'] / (test_transaction['V127'] + 1e-3)


class V310_div_V282(Feature):
    def create_features(self):
        self.train[self.__class__.__name__] = train_transaction['V310'] / (train_transaction['V282'] + 1e-3)
        self.test[self.__class__.__name__] = test_transaction['V310'] / (test_transaction['V282'] + 1e-3)


class V310_div_V45(Feature):
    def create_features(self):
        self.train[self.__class__.__name__] = train_transaction['V310'] / (train_transaction['V45'] + 1e-3)
        self.test[self.__class__.__name__] = test_transaction['V310'] / (test_transaction['V45'] + 1e-3)


class V310_div_V131(Feature):
    def create_features(self):
        self.train[self.__class__.__name__] = train_transaction['V310'] / (train_transaction['V131'] + 1e-3)
        self.test[self.__class__.__name__] = test_transaction['V310'] / (test_transaction['V131'] + 1e-3)


class V307_div_V313(Feature):
    def create_features(self):
        self.train[self.__class__.__name__] = train_transaction['V307'] / (train_transaction['V313'] + 1e-3)
        self.test[self.__class__.__name__] = test_transaction['V307'] / (test_transaction['V313'] + 1e-3)


class V307_div_V130(Feature):
    def create_features(self):
        self.train[self.__class__.__name__] = train_transaction['V307'] / (train_transaction['V130'] + 1e-3)
        self.test[self.__class__.__name__] = test_transaction['V307'] / (test_transaction['V130'] + 1e-3)


class V307_div_V314(Feature):
    def create_features(self):
        self.train[self.__class__.__name__] = train_transaction['V307'] / (train_transaction['V314'] + 1e-3)
        self.test[self.__class__.__name__] = test_transaction['V307'] / (test_transaction['V314'] + 1e-3)


class V307_div_V315(Feature):
    def create_features(self):
        self.train[self.__class__.__name__] = train_transaction['V307'] / (train_transaction['V315'] + 1e-3)
        self.test[self.__class__.__name__] = test_transaction['V307'] / (test_transaction['V315'] + 1e-3)


class V307_div_V317(Feature):
    def create_features(self):
        self.train[self.__class__.__name__] = train_transaction['V307'] / (train_transaction['V317'] + 1e-3)
        self.test[self.__class__.__name__] = test_transaction['V307'] / (test_transaction['V317'] + 1e-3)


class V307_div_V285(Feature):
    def create_features(self):
        self.train[self.__class__.__name__] = train_transaction['V307'] / (train_transaction['V285'] + 1e-3)
        self.test[self.__class__.__name__] = test_transaction['V307'] / (test_transaction['V285'] + 1e-3)


class V307_div_V283(Feature):
    def create_features(self):
        self.train[self.__class__.__name__] = train_transaction['V307'] / (train_transaction['V283'] + 1e-3)
        self.test[self.__class__.__name__] = test_transaction['V307'] / (test_transaction['V283'] + 1e-3)


class V307_div_V312(Feature):
    def create_features(self):
        self.train[self.__class__.__name__] = train_transaction['V307'] / (train_transaction['V312'] + 1e-3)
        self.test[self.__class__.__name__] = test_transaction['V307'] / (test_transaction['V312'] + 1e-3)


class V307_div_V308(Feature):
    def create_features(self):
        self.train[self.__class__.__name__] = train_transaction['V307'] / (train_transaction['V308'] + 1e-3)
        self.test[self.__class__.__name__] = test_transaction['V307'] / (test_transaction['V308'] + 1e-3)


class V307_div_V127(Feature):
    def create_features(self):
        self.train[self.__class__.__name__] = train_transaction['V307'] / (train_transaction['V127'] + 1e-3)
        self.test[self.__class__.__name__] = test_transaction['V307'] / (test_transaction['V127'] + 1e-3)


class V307_div_V282(Feature):
    def create_features(self):
        self.train[self.__class__.__name__] = train_transaction['V307'] / (train_transaction['V282'] + 1e-3)
        self.test[self.__class__.__name__] = test_transaction['V307'] / (test_transaction['V282'] + 1e-3)


class V307_div_V45(Feature):
    def create_features(self):
        self.train[self.__class__.__name__] = train_transaction['V307'] / (train_transaction['V45'] + 1e-3)
        self.test[self.__class__.__name__] = test_transaction['V307'] / (test_transaction['V45'] + 1e-3)


class V307_div_V131(Feature):
    def create_features(self):
        self.train[self.__class__.__name__] = train_transaction['V307'] / (train_transaction['V131'] + 1e-3)
        self.test[self.__class__.__name__] = test_transaction['V307'] / (test_transaction['V131'] + 1e-3)


class V313_div_V130(Feature):
    def create_features(self):
        self.train[self.__class__.__name__] = train_transaction['V313'] / (train_transaction['V130'] + 1e-3)
        self.test[self.__class__.__name__] = test_transaction['V313'] / (test_transaction['V130'] + 1e-3)


class V313_div_V314(Feature):
    def create_features(self):
        self.train[self.__class__.__name__] = train_transaction['V313'] / (train_transaction['V314'] + 1e-3)
        self.test[self.__class__.__name__] = test_transaction['V313'] / (test_transaction['V314'] + 1e-3)


class V313_div_V315(Feature):
    def create_features(self):
        self.train[self.__class__.__name__] = train_transaction['V313'] / (train_transaction['V315'] + 1e-3)
        self.test[self.__class__.__name__] = test_transaction['V313'] / (test_transaction['V315'] + 1e-3)


class V313_div_V317(Feature):
    def create_features(self):
        self.train[self.__class__.__name__] = train_transaction['V313'] / (train_transaction['V317'] + 1e-3)
        self.test[self.__class__.__name__] = test_transaction['V313'] / (test_transaction['V317'] + 1e-3)


class V313_div_V285(Feature):
    def create_features(self):
        self.train[self.__class__.__name__] = train_transaction['V313'] / (train_transaction['V285'] + 1e-3)
        self.test[self.__class__.__name__] = test_transaction['V313'] / (test_transaction['V285'] + 1e-3)


class V313_div_V283(Feature):
    def create_features(self):
        self.train[self.__class__.__name__] = train_transaction['V313'] / (train_transaction['V283'] + 1e-3)
        self.test[self.__class__.__name__] = test_transaction['V313'] / (test_transaction['V283'] + 1e-3)


class V313_div_V312(Feature):
    def create_features(self):
        self.train[self.__class__.__name__] = train_transaction['V313'] / (train_transaction['V312'] + 1e-3)
        self.test[self.__class__.__name__] = test_transaction['V313'] / (test_transaction['V312'] + 1e-3)


class V313_div_V308(Feature):
    def create_features(self):
        self.train[self.__class__.__name__] = train_transaction['V313'] / (train_transaction['V308'] + 1e-3)
        self.test[self.__class__.__name__] = test_transaction['V313'] / (test_transaction['V308'] + 1e-3)


class V313_div_V127(Feature):
    def create_features(self):
        self.train[self.__class__.__name__] = train_transaction['V313'] / (train_transaction['V127'] + 1e-3)
        self.test[self.__class__.__name__] = test_transaction['V313'] / (test_transaction['V127'] + 1e-3)


class V313_div_V282(Feature):
    def create_features(self):
        self.train[self.__class__.__name__] = train_transaction['V313'] / (train_transaction['V282'] + 1e-3)
        self.test[self.__class__.__name__] = test_transaction['V313'] / (test_transaction['V282'] + 1e-3)


class V313_div_V45(Feature):
    def create_features(self):
        self.train[self.__class__.__name__] = train_transaction['V313'] / (train_transaction['V45'] + 1e-3)
        self.test[self.__class__.__name__] = test_transaction['V313'] / (test_transaction['V45'] + 1e-3)


class V313_div_V131(Feature):
    def create_features(self):
        self.train[self.__class__.__name__] = train_transaction['V313'] / (train_transaction['V131'] + 1e-3)
        self.test[self.__class__.__name__] = test_transaction['V313'] / (test_transaction['V131'] + 1e-3)


class V130_div_V314(Feature):
    def create_features(self):
        self.train[self.__class__.__name__] = train_transaction['V130'] / (train_transaction['V314'] + 1e-3)
        self.test[self.__class__.__name__] = test_transaction['V130'] / (test_transaction['V314'] + 1e-3)


class V130_div_V315(Feature):
    def create_features(self):
        self.train[self.__class__.__name__] = train_transaction['V130'] / (train_transaction['V315'] + 1e-3)
        self.test[self.__class__.__name__] = test_transaction['V130'] / (test_transaction['V315'] + 1e-3)


class V130_div_V317(Feature):
    def create_features(self):
        self.train[self.__class__.__name__] = train_transaction['V130'] / (train_transaction['V317'] + 1e-3)
        self.test[self.__class__.__name__] = test_transaction['V130'] / (test_transaction['V317'] + 1e-3)


class V130_div_V285(Feature):
    def create_features(self):
        self.train[self.__class__.__name__] = train_transaction['V130'] / (train_transaction['V285'] + 1e-3)
        self.test[self.__class__.__name__] = test_transaction['V130'] / (test_transaction['V285'] + 1e-3)


class V130_div_V283(Feature):
    def create_features(self):
        self.train[self.__class__.__name__] = train_transaction['V130'] / (train_transaction['V283'] + 1e-3)
        self.test[self.__class__.__name__] = test_transaction['V130'] / (test_transaction['V283'] + 1e-3)


class V130_div_V312(Feature):
    def create_features(self):
        self.train[self.__class__.__name__] = train_transaction['V130'] / (train_transaction['V312'] + 1e-3)
        self.test[self.__class__.__name__] = test_transaction['V130'] / (test_transaction['V312'] + 1e-3)


class V130_div_V308(Feature):
    def create_features(self):
        self.train[self.__class__.__name__] = train_transaction['V130'] / (train_transaction['V308'] + 1e-3)
        self.test[self.__class__.__name__] = test_transaction['V130'] / (test_transaction['V308'] + 1e-3)


class V130_div_V127(Feature):
    def create_features(self):
        self.train[self.__class__.__name__] = train_transaction['V130'] / (train_transaction['V127'] + 1e-3)
        self.test[self.__class__.__name__] = test_transaction['V130'] / (test_transaction['V127'] + 1e-3)


class V130_div_V282(Feature):
    def create_features(self):
        self.train[self.__class__.__name__] = train_transaction['V130'] / (train_transaction['V282'] + 1e-3)
        self.test[self.__class__.__name__] = test_transaction['V130'] / (test_transaction['V282'] + 1e-3)


class V130_div_V45(Feature):
    def create_features(self):
        self.train[self.__class__.__name__] = train_transaction['V130'] / (train_transaction['V45'] + 1e-3)
        self.test[self.__class__.__name__] = test_transaction['V130'] / (test_transaction['V45'] + 1e-3)


class V130_div_V131(Feature):
    def create_features(self):
        self.train[self.__class__.__name__] = train_transaction['V130'] / (train_transaction['V131'] + 1e-3)
        self.test[self.__class__.__name__] = test_transaction['V130'] / (test_transaction['V131'] + 1e-3)


class V314_div_V315(Feature):
    def create_features(self):
        self.train[self.__class__.__name__] = train_transaction['V314'] / (train_transaction['V315'] + 1e-3)
        self.test[self.__class__.__name__] = test_transaction['V314'] / (test_transaction['V315'] + 1e-3)


class V314_div_V317(Feature):
    def create_features(self):
        self.train[self.__class__.__name__] = train_transaction['V314'] / (train_transaction['V317'] + 1e-3)
        self.test[self.__class__.__name__] = test_transaction['V314'] / (test_transaction['V317'] + 1e-3)


class V314_div_V285(Feature):
    def create_features(self):
        self.train[self.__class__.__name__] = train_transaction['V314'] / (train_transaction['V285'] + 1e-3)
        self.test[self.__class__.__name__] = test_transaction['V314'] / (test_transaction['V285'] + 1e-3)


class V314_div_V283(Feature):
    def create_features(self):
        self.train[self.__class__.__name__] = train_transaction['V314'] / (train_transaction['V283'] + 1e-3)
        self.test[self.__class__.__name__] = test_transaction['V314'] / (test_transaction['V283'] + 1e-3)


class V314_div_V312(Feature):
    def create_features(self):
        self.train[self.__class__.__name__] = train_transaction['V314'] / (train_transaction['V312'] + 1e-3)
        self.test[self.__class__.__name__] = test_transaction['V314'] / (test_transaction['V312'] + 1e-3)


class V314_div_V308(Feature):
    def create_features(self):
        self.train[self.__class__.__name__] = train_transaction['V314'] / (train_transaction['V308'] + 1e-3)
        self.test[self.__class__.__name__] = test_transaction['V314'] / (test_transaction['V308'] + 1e-3)


class V314_div_V127(Feature):
    def create_features(self):
        self.train[self.__class__.__name__] = train_transaction['V314'] / (train_transaction['V127'] + 1e-3)
        self.test[self.__class__.__name__] = test_transaction['V314'] / (test_transaction['V127'] + 1e-3)


class V314_div_V282(Feature):
    def create_features(self):
        self.train[self.__class__.__name__] = train_transaction['V314'] / (train_transaction['V282'] + 1e-3)
        self.test[self.__class__.__name__] = test_transaction['V314'] / (test_transaction['V282'] + 1e-3)


class V314_div_V45(Feature):
    def create_features(self):
        self.train[self.__class__.__name__] = train_transaction['V314'] / (train_transaction['V45'] + 1e-3)
        self.test[self.__class__.__name__] = test_transaction['V314'] / (test_transaction['V45'] + 1e-3)


class V314_div_V131(Feature):
    def create_features(self):
        self.train[self.__class__.__name__] = train_transaction['V314'] / (train_transaction['V131'] + 1e-3)
        self.test[self.__class__.__name__] = test_transaction['V314'] / (test_transaction['V131'] + 1e-3)


class V315_div_V317(Feature):
    def create_features(self):
        self.train[self.__class__.__name__] = train_transaction['V315'] / (train_transaction['V317'] + 1e-3)
        self.test[self.__class__.__name__] = test_transaction['V315'] / (test_transaction['V317'] + 1e-3)


class V315_div_V285(Feature):
    def create_features(self):
        self.train[self.__class__.__name__] = train_transaction['V315'] / (train_transaction['V285'] + 1e-3)
        self.test[self.__class__.__name__] = test_transaction['V315'] / (test_transaction['V285'] + 1e-3)


class V315_div_V283(Feature):
    def create_features(self):
        self.train[self.__class__.__name__] = train_transaction['V315'] / (train_transaction['V283'] + 1e-3)
        self.test[self.__class__.__name__] = test_transaction['V315'] / (test_transaction['V283'] + 1e-3)


class V315_div_V312(Feature):
    def create_features(self):
        self.train[self.__class__.__name__] = train_transaction['V315'] / (train_transaction['V312'] + 1e-3)
        self.test[self.__class__.__name__] = test_transaction['V315'] / (test_transaction['V312'] + 1e-3)


class V315_div_V308(Feature):
    def create_features(self):
        self.train[self.__class__.__name__] = train_transaction['V315'] / (train_transaction['V308'] + 1e-3)
        self.test[self.__class__.__name__] = test_transaction['V315'] / (test_transaction['V308'] + 1e-3)


class V315_div_V127(Feature):
    def create_features(self):
        self.train[self.__class__.__name__] = train_transaction['V315'] / (train_transaction['V127'] + 1e-3)
        self.test[self.__class__.__name__] = test_transaction['V315'] / (test_transaction['V127'] + 1e-3)


class V315_div_V282(Feature):
    def create_features(self):
        self.train[self.__class__.__name__] = train_transaction['V315'] / (train_transaction['V282'] + 1e-3)
        self.test[self.__class__.__name__] = test_transaction['V315'] / (test_transaction['V282'] + 1e-3)


class V315_div_V45(Feature):
    def create_features(self):
        self.train[self.__class__.__name__] = train_transaction['V315'] / (train_transaction['V45'] + 1e-3)
        self.test[self.__class__.__name__] = test_transaction['V315'] / (test_transaction['V45'] + 1e-3)


class V315_div_V131(Feature):
    def create_features(self):
        self.train[self.__class__.__name__] = train_transaction['V315'] / (train_transaction['V131'] + 1e-3)
        self.test[self.__class__.__name__] = test_transaction['V315'] / (test_transaction['V131'] + 1e-3)


class V317_div_V285(Feature):
    def create_features(self):
        self.train[self.__class__.__name__] = train_transaction['V317'] / (train_transaction['V285'] + 1e-3)
        self.test[self.__class__.__name__] = test_transaction['V317'] / (test_transaction['V285'] + 1e-3)


class V317_div_V283(Feature):
    def create_features(self):
        self.train[self.__class__.__name__] = train_transaction['V317'] / (train_transaction['V283'] + 1e-3)
        self.test[self.__class__.__name__] = test_transaction['V317'] / (test_transaction['V283'] + 1e-3)


class V317_div_V312(Feature):
    def create_features(self):
        self.train[self.__class__.__name__] = train_transaction['V317'] / (train_transaction['V312'] + 1e-3)
        self.test[self.__class__.__name__] = test_transaction['V317'] / (test_transaction['V312'] + 1e-3)


class V317_div_V308(Feature):
    def create_features(self):
        self.train[self.__class__.__name__] = train_transaction['V317'] / (train_transaction['V308'] + 1e-3)
        self.test[self.__class__.__name__] = test_transaction['V317'] / (test_transaction['V308'] + 1e-3)


class V317_div_V127(Feature):
    def create_features(self):
        self.train[self.__class__.__name__] = train_transaction['V317'] / (train_transaction['V127'] + 1e-3)
        self.test[self.__class__.__name__] = test_transaction['V317'] / (test_transaction['V127'] + 1e-3)


class V317_div_V282(Feature):
    def create_features(self):
        self.train[self.__class__.__name__] = train_transaction['V317'] / (train_transaction['V282'] + 1e-3)
        self.test[self.__class__.__name__] = test_transaction['V317'] / (test_transaction['V282'] + 1e-3)


class V317_div_V45(Feature):
    def create_features(self):
        self.train[self.__class__.__name__] = train_transaction['V317'] / (train_transaction['V45'] + 1e-3)
        self.test[self.__class__.__name__] = test_transaction['V317'] / (test_transaction['V45'] + 1e-3)


class V317_div_V131(Feature):
    def create_features(self):
        self.train[self.__class__.__name__] = train_transaction['V317'] / (train_transaction['V131'] + 1e-3)
        self.test[self.__class__.__name__] = test_transaction['V317'] / (test_transaction['V131'] + 1e-3)


class V285_div_V283(Feature):
    def create_features(self):
        self.train[self.__class__.__name__] = train_transaction['V285'] / (train_transaction['V283'] + 1e-3)
        self.test[self.__class__.__name__] = test_transaction['V285'] / (test_transaction['V283'] + 1e-3)


class V285_div_V312(Feature):
    def create_features(self):
        self.train[self.__class__.__name__] = train_transaction['V285'] / (train_transaction['V312'] + 1e-3)
        self.test[self.__class__.__name__] = test_transaction['V285'] / (test_transaction['V312'] + 1e-3)


class V285_div_V308(Feature):
    def create_features(self):
        self.train[self.__class__.__name__] = train_transaction['V285'] / (train_transaction['V308'] + 1e-3)
        self.test[self.__class__.__name__] = test_transaction['V285'] / (test_transaction['V308'] + 1e-3)


class V285_div_V127(Feature):
    def create_features(self):
        self.train[self.__class__.__name__] = train_transaction['V285'] / (train_transaction['V127'] + 1e-3)
        self.test[self.__class__.__name__] = test_transaction['V285'] / (test_transaction['V127'] + 1e-3)


class V285_div_V282(Feature):
    def create_features(self):
        self.train[self.__class__.__name__] = train_transaction['V285'] / (train_transaction['V282'] + 1e-3)
        self.test[self.__class__.__name__] = test_transaction['V285'] / (test_transaction['V282'] + 1e-3)


class V285_div_V45(Feature):
    def create_features(self):
        self.train[self.__class__.__name__] = train_transaction['V285'] / (train_transaction['V45'] + 1e-3)
        self.test[self.__class__.__name__] = test_transaction['V285'] / (test_transaction['V45'] + 1e-3)


class V285_div_V131(Feature):
    def create_features(self):
        self.train[self.__class__.__name__] = train_transaction['V285'] / (train_transaction['V131'] + 1e-3)
        self.test[self.__class__.__name__] = test_transaction['V285'] / (test_transaction['V131'] + 1e-3)


class V283_div_V312(Feature):
    def create_features(self):
        self.train[self.__class__.__name__] = train_transaction['V283'] / (train_transaction['V312'] + 1e-3)
        self.test[self.__class__.__name__] = test_transaction['V283'] / (test_transaction['V312'] + 1e-3)


class V283_div_V308(Feature):
    def create_features(self):
        self.train[self.__class__.__name__] = train_transaction['V283'] / (train_transaction['V308'] + 1e-3)
        self.test[self.__class__.__name__] = test_transaction['V283'] / (test_transaction['V308'] + 1e-3)


class V283_div_V127(Feature):
    def create_features(self):
        self.train[self.__class__.__name__] = train_transaction['V283'] / (train_transaction['V127'] + 1e-3)
        self.test[self.__class__.__name__] = test_transaction['V283'] / (test_transaction['V127'] + 1e-3)


class V283_div_V282(Feature):
    def create_features(self):
        self.train[self.__class__.__name__] = train_transaction['V283'] / (train_transaction['V282'] + 1e-3)
        self.test[self.__class__.__name__] = test_transaction['V283'] / (test_transaction['V282'] + 1e-3)


class V283_div_V45(Feature):
    def create_features(self):
        self.train[self.__class__.__name__] = train_transaction['V283'] / (train_transaction['V45'] + 1e-3)
        self.test[self.__class__.__name__] = test_transaction['V283'] / (test_transaction['V45'] + 1e-3)


class V283_div_V131(Feature):
    def create_features(self):
        self.train[self.__class__.__name__] = train_transaction['V283'] / (train_transaction['V131'] + 1e-3)
        self.test[self.__class__.__name__] = test_transaction['V283'] / (test_transaction['V131'] + 1e-3)


class V312_div_V308(Feature):
    def create_features(self):
        self.train[self.__class__.__name__] = train_transaction['V312'] / (train_transaction['V308'] + 1e-3)
        self.test[self.__class__.__name__] = test_transaction['V312'] / (test_transaction['V308'] + 1e-3)


class V312_div_V127(Feature):
    def create_features(self):
        self.train[self.__class__.__name__] = train_transaction['V312'] / (train_transaction['V127'] + 1e-3)
        self.test[self.__class__.__name__] = test_transaction['V312'] / (test_transaction['V127'] + 1e-3)


class V312_div_V282(Feature):
    def create_features(self):
        self.train[self.__class__.__name__] = train_transaction['V312'] / (train_transaction['V282'] + 1e-3)
        self.test[self.__class__.__name__] = test_transaction['V312'] / (test_transaction['V282'] + 1e-3)


class V312_div_V45(Feature):
    def create_features(self):
        self.train[self.__class__.__name__] = train_transaction['V312'] / (train_transaction['V45'] + 1e-3)
        self.test[self.__class__.__name__] = test_transaction['V312'] / (test_transaction['V45'] + 1e-3)


class V312_div_V131(Feature):
    def create_features(self):
        self.train[self.__class__.__name__] = train_transaction['V312'] / (train_transaction['V131'] + 1e-3)
        self.test[self.__class__.__name__] = test_transaction['V312'] / (test_transaction['V131'] + 1e-3)


class V308_div_V127(Feature):
    def create_features(self):
        self.train[self.__class__.__name__] = train_transaction['V308'] / (train_transaction['V127'] + 1e-3)
        self.test[self.__class__.__name__] = test_transaction['V308'] / (test_transaction['V127'] + 1e-3)


class V308_div_V282(Feature):
    def create_features(self):
        self.train[self.__class__.__name__] = train_transaction['V308'] / (train_transaction['V282'] + 1e-3)
        self.test[self.__class__.__name__] = test_transaction['V308'] / (test_transaction['V282'] + 1e-3)


class V308_div_V45(Feature):
    def create_features(self):
        self.train[self.__class__.__name__] = train_transaction['V308'] / (train_transaction['V45'] + 1e-3)
        self.test[self.__class__.__name__] = test_transaction['V308'] / (test_transaction['V45'] + 1e-3)


class V308_div_V131(Feature):
    def create_features(self):
        self.train[self.__class__.__name__] = train_transaction['V308'] / (train_transaction['V131'] + 1e-3)
        self.test[self.__class__.__name__] = test_transaction['V308'] / (test_transaction['V131'] + 1e-3)


class V127_div_V282(Feature):
    def create_features(self):
        self.train[self.__class__.__name__] = train_transaction['V127'] / (train_transaction['V282'] + 1e-3)
        self.test[self.__class__.__name__] = test_transaction['V127'] / (test_transaction['V282'] + 1e-3)


class V127_div_V45(Feature):
    def create_features(self):
        self.train[self.__class__.__name__] = train_transaction['V127'] / (train_transaction['V45'] + 1e-3)
        self.test[self.__class__.__name__] = test_transaction['V127'] / (test_transaction['V45'] + 1e-3)


class V127_div_V131(Feature):
    def create_features(self):
        self.train[self.__class__.__name__] = train_transaction['V127'] / (train_transaction['V131'] + 1e-3)
        self.test[self.__class__.__name__] = test_transaction['V127'] / (test_transaction['V131'] + 1e-3)


class V282_div_V45(Feature):
    def create_features(self):
        self.train[self.__class__.__name__] = train_transaction['V282'] / (train_transaction['V45'] + 1e-3)
        self.test[self.__class__.__name__] = test_transaction['V282'] / (test_transaction['V45'] + 1e-3)


class V282_div_V131(Feature):
    def create_features(self):
        self.train[self.__class__.__name__] = train_transaction['V282'] / (train_transaction['V131'] + 1e-3)
        self.test[self.__class__.__name__] = test_transaction['V282'] / (test_transaction['V131'] + 1e-3)


class V45_div_V131(Feature):
    def create_features(self):
        self.train[self.__class__.__name__] = train_transaction['V45'] / (train_transaction['V131'] + 1e-3)
        self.test[self.__class__.__name__] = test_transaction['V45'] / (test_transaction['V131'] + 1e-3)


class ca1_ca5_amt_p_r_ad1_ad2_di1_di2(Feature):
    def create_features(self):
        len_ = len(train_transaction)
        feats = ['card1', 'card5', 'TransactionAmt', 'P_emaildomain', 'R_emaildomain', 'addr1', 'addr2', 'dist1', 'dist2']
        whole = pd.concat([train_transaction[feats], test_transaction[feats]], axis=0)
        whole['id_'] = whole['card1'].astype(str) + '_' + \
            whole['card5'].astype(str) + '_' + \
            whole['TransactionAmt'].astype(str) + '_' + \
            whole['P_emaildomain'].astype(str) + '_' + \
            whole['R_emaildomain'].astype(str) + '_' + \
            whole['addr1'].astype(str) + '_' + \
            whole['addr2'].astype(str) + '_' + \
            whole['dist1'].astype(str) + '_' + \
            whole['dist2'].astype(str)

        le = whole['id_'].value_counts().to_dict()
        whole[self.__class__.__name__] = whole['id_'].map(le)
        self.train[self.__class__.__name__] = whole[self.__class__.__name__].values[:len_]
        self.test[self.__class__.__name__] = whole[self.__class__.__name__].values[len_:]


class ca1_ca5_amt_c2_d9(Feature):
    def create_features(self):
        len_ = len(train_transaction)
        feats = ['card1', 'card5', 'TransactionAmt', 'C2', 'D9']
        whole = pd.concat([train_transaction[feats], test_transaction[feats]], axis=0)
        whole['id_'] = whole['card1'].astype(str) + '_' + \
            whole['card5'].astype(str) + '_' + \
            whole['TransactionAmt'].astype(str) + '_' + \
            whole['C2'].astype(str) + '_' + \
            whole['D9'].astype(str)

        le = whole['id_'].value_counts().to_dict()
        whole[self.__class__.__name__] = whole['id_'].map(le)
        self.train[self.__class__.__name__] = whole[self.__class__.__name__].values[:len_]
        self.test[self.__class__.__name__] = whole[self.__class__.__name__].values[len_:]


class diff_transaction_dt_from_min_ca1_ca5_amt_dt(Feature):
    def create_features(self):
        len_ = len(train_transaction)
        feats = ['card1', 'card5', 'TransactionAmt', 'TransactionDT']
        whole = pd.concat([train_transaction[feats], test_transaction[feats]], axis=0)
        whole['id_'] = whole['card1'].astype(str) + '_' + \
            whole['card5'].astype(str) + '_' + \
            whole['TransactionAmt'].astype(str)
        whole_min = whole.groupby('id_')['TransactionDT'].min().to_frame('min_TransactionDT')
        whole_ = pd.merge(whole, whole_min, on='id_', how='left')
        whole_[self.__class__.__name__] = whole_['TransactionDT'] - whole_['min_TransactionDT']
        self.train[self.__class__.__name__] = whole_[self.__class__.__name__].values[:len_]
        self.test[self.__class__.__name__] = whole_[self.__class__.__name__].values[len_:]


class transaction_num_3days_each_card1_card5_amt(Feature):
    def create_features(self):
        len_ = len(train_transaction)
        feats = ['card1', 'card5', 'TransactionDT', 'TransactionAmt']
        whole = pd.concat([train_transaction[feats], test_transaction[feats]], axis=0).reset_index(drop=True)
        whole['date_'] = whole['TransactionDT'].apply(lambda x: (START_DATE + datetime.timedelta(seconds=x)))
        whole['id_'] = whole['card1'].astype('str') + '_' + whole['card5'].astype(str) + '_' + whole['TransactionAmt'].apply(lambda x: int(x * 1000)).astype(str)
        df_count = whole.pivot_table(index='id_', columns='date_', values='card1', aggfunc='count', fill_value=0)
        id2index = {id_: idx for idx, id_ in enumerate(df_count.index)}
        date2col = {col: idx for idx, col in enumerate(df_count.columns)}

        count_matrix = df_count.values

        values = np.zeros(len(whole))

        for idx in whole['id_'].index:
            id_ = whole.loc[idx, 'id_']
            date_ = whole.loc[idx, 'date_']
            
            i, j = id2index[id_], date2col[date_]
            if j == 0:
                s, e = j, j + 2
            else:
                s, e = j - 1, j + 2

            values[idx] += np.sum(count_matrix[i, s:e])

        self.train[self.__class__.__name__] = values[:len_]
        self.test[self.__class__.__name__] = values[len_:]


class transaction_num_5days_each_card1_card5_amt(Feature):
    def create_features(self):
        len_ = len(train_transaction)
        feats = ['card1', 'card5', 'TransactionDT', 'TransactionAmt']
        whole = pd.concat([train_transaction[feats], test_transaction[feats]], axis=0).reset_index(drop=True)
        whole['date_'] = whole['TransactionDT'].apply(lambda x: (START_DATE + datetime.timedelta(seconds=x)))
        whole['id_'] = whole['card1'].astype('str') + '_' + whole['card5'].astype(str) + '_' + whole['TransactionAmt'].apply(lambda x: int(x * 1000)).astype(str)
        df_count = whole.pivot_table(index='id_', columns='date_', values='card1', aggfunc='count', fill_value=0)
        id2index = {id_: idx for idx, id_ in enumerate(df_count.index)}
        date2col = {col: idx for idx, col in enumerate(df_count.columns)}

        count_matrix = df_count.values

        values = np.zeros(len(whole))

        for idx in whole['id_'].index:
            id_ = whole.loc[idx, 'id_']
            date_ = whole.loc[idx, 'date_']
            
            i, j = id2index[id_], date2col[date_]
            if j == 0:
                s, e = j, j + 3
            else:
                s, e = j - 2, j + 3

            values[idx] += np.sum(count_matrix[i, s:e])

        self.train[self.__class__.__name__] = values[:len_]
        self.test[self.__class__.__name__] = values[len_:]


class transaction_num_36hours_each_card1_card5_amt(Feature):
    def create_features(self):
        len_ = len(train_transaction)
        feats = ['card1', 'card5', 'TransactionDT', 'TransactionAmt']
        whole = pd.concat([train_transaction[feats], test_transaction[feats]], axis=0).reset_index(drop=True)
        whole['dt_'] = (whole['TransactionDT'] - 86400) // (3600 * 12)
        whole['id_'] = whole['card1'].astype('str') + '_' + whole['card5'].astype(str) + '_' + whole['TransactionAmt'].apply(lambda x: int(x * 1000)).astype(str)
        df_count = whole.pivot_table(index='id_', columns='dt_', values='card1', aggfunc='count', fill_value=0)
        id2index = {id_: idx for idx, id_ in enumerate(df_count.index)}
        date2col = {col: idx for idx, col in enumerate(df_count.columns)}

        count_matrix = df_count.values

        values = np.zeros(len(whole))

        for idx in whole['id_'].index:
            id_ = whole.loc[idx, 'id_']
            dt_ = whole.loc[idx, 'dt_']

            i, j = id2index[id_], date2col[dt_]
            if j == 0:
                s, e = j, j + 2
            else:
                s, e = j - 1, j + 2

            values[idx] += np.sum(count_matrix[i, s:e])

        self.train[self.__class__.__name__] = values[:len_]
        self.test[self.__class__.__name__] = values[len_:]


class transaction_num_12hours_each_card1_card5_amt(Feature):
    def create_features(self):
        len_ = len(train_transaction)
        feats = ['card1', 'card5', 'TransactionDT', 'TransactionAmt']
        whole = pd.concat([train_transaction[feats], test_transaction[feats]], axis=0).reset_index(drop=True)
        whole['dt_'] = (whole['TransactionDT'] - 86400) // (3600 * 4)
        whole['id_'] = whole['card1'].astype('str') + '_' + whole['card5'].astype(str) + '_' + whole['TransactionAmt'].apply(lambda x: int(x * 1000)).astype(str)
        df_count = whole.pivot_table(index='id_', columns='dt_', values='card1', aggfunc='count', fill_value=0)
        id2index = {id_: idx for idx, id_ in enumerate(df_count.index)}
        date2col = {col: idx for idx, col in enumerate(df_count.columns)}

        count_matrix = df_count.values

        values = np.zeros(len(whole))

        for idx in whole['id_'].index:
            id_ = whole.loc[idx, 'id_']
            dt_ = whole.loc[idx, 'dt_']

            i, j = id2index[id_], date2col[dt_]
            if j == 0:
                s, e = j, j + 2
            else:
                s, e = j - 1, j + 2

            values[idx] += np.sum(count_matrix[i, s:e])

        self.train[self.__class__.__name__] = values[:len_]
        self.test[self.__class__.__name__] = values[len_:]


class elapsed_from_last_card1_transaction(Feature):
    def create_features(self):
        len_ = len(train_transaction)
        feats = ['card1', 'TransactionDT']
        whole = pd.concat([train_transaction[feats], test_transaction[feats]], axis=0).reset_index(drop=True)

        dict_dt = {}
        last_dt = []
        for idx in whole.index:
            card1 = whole.loc[idx, 'card1']
            dt = whole.loc[idx, 'TransactionDT']
            if card1 not in dict_dt.keys():
                last_dt.append(dt)
            else:
                last_dt.append(dict_dt[card1])

            dict_dt[card1] = dt

        whole['last_dt'] = last_dt
        whole[self.__class__.__name__] = whole['TransactionDT'] - whole['last_dt']

        self.train[self.__class__.__name__] = whole[self.__class__.__name__].values[:len_]
        self.test[self.__class__.__name__] = whole[self.__class__.__name__].values[len_:]


class elapsed_from_last_card1_card5_transaction(Feature):
    def create_features(self):
        len_ = len(train_transaction)
        feats = ['card1', 'card5', 'TransactionDT']
        whole = pd.concat([train_transaction[feats], test_transaction[feats]], axis=0).reset_index(drop=True)
        whole['card1_card5'] = whole['card1'].astype(str) + '_' + whole['card5'].astype(str)

        dict_dt = {}
        last_dt = []
        for idx in whole.index:
            card1 = whole.loc[idx, 'card1_card5']
            dt = whole.loc[idx, 'TransactionDT']
            if card1 not in dict_dt.keys():
                last_dt.append(dt)
            else:
                last_dt.append(dict_dt[card1])

            dict_dt[card1] = dt

        whole['last_dt'] = last_dt
        whole[self.__class__.__name__] = whole['TransactionDT'] - whole['last_dt']

        self.train[self.__class__.__name__] = whole[self.__class__.__name__].values[:len_]
        self.test[self.__class__.__name__] = whole[self.__class__.__name__].values[len_:]


class elapsed_from_last_card1_card5_amt_transaction(Feature):
    def create_features(self):
        len_ = len(train_transaction)
        feats = ['card1', 'card5', 'TransactionAmt', 'TransactionDT']
        whole = pd.concat([train_transaction[feats], test_transaction[feats]], axis=0).reset_index(drop=True)
        whole['card1_card5_amt'] = whole['card1'].astype(str) + '_' + whole['card5'].astype(str) + '_' + whole['TransactionAmt'].apply(lambda x: int(x * 1000)).astype(str)

        dict_dt = {}
        last_dt = []
        for idx in whole.index:
            card1 = whole.loc[idx, 'card1_card5_amt']
            dt = whole.loc[idx, 'TransactionDT']
            if card1 not in dict_dt.keys():
                last_dt.append(dt)
            else:
                last_dt.append(dict_dt[card1])

            dict_dt[card1] = dt

        whole['last_dt'] = last_dt
        whole[self.__class__.__name__] = whole['TransactionDT'] - whole['last_dt']

        self.train[self.__class__.__name__] = whole[self.__class__.__name__].values[:len_]
        self.test[self.__class__.__name__] = whole[self.__class__.__name__].values[len_:]


class elapsed_from_last_card1_card5_amt_c2_transaction(Feature):
    def create_features(self):
        len_ = len(train_transaction)
        feats = ['card1', 'card5', 'TransactionAmt', 'TransactionDT', 'C2']
        whole = pd.concat([train_transaction[feats], test_transaction[feats]], axis=0).reset_index(drop=True)
        whole['card1_card5_amt_c2'] = whole['card1'].astype(str) + '_' + whole['card5'].astype(str) + '_' + whole['TransactionAmt'].apply(lambda x: int(x * 1000)).astype(str) + '_' + whole['C2'].astype(str)

        dict_dt = {}
        last_dt = []
        for idx in whole.index:
            card1 = whole.loc[idx, 'card1_card5_amt_c2']
            dt = whole.loc[idx, 'TransactionDT']
            if card1 not in dict_dt.keys():
                last_dt.append(dt)
            else:
                last_dt.append(dict_dt[card1])

            dict_dt[card1] = dt

        whole['last_dt'] = last_dt
        whole[self.__class__.__name__] = whole['TransactionDT'] - whole['last_dt']

        self.train[self.__class__.__name__] = whole[self.__class__.__name__].values[:len_]
        self.test[self.__class__.__name__] = whole[self.__class__.__name__].values[len_:]


class transaction_amt_div_v283(Feature):
    def create_features(self):
        s = 1e-3
        self.train[self.__class__.__name__] = train_transaction['TransactionAmt'] / (train_transaction['V283'] + s)
        self.test[self.__class__.__name__] = test_transaction['TransactionAmt'] / (test_transaction['V283'] + s)


class c2_div_v283(Feature):
    def create_features(self):
        s = 1e-3
        self.train[self.__class__.__name__] = train_transaction['C2'] / (train_transaction['V283'] + s)
        self.test[self.__class__.__name__] = test_transaction['C2'] / (test_transaction['V283'] + s)


class V283_div_mean_each_card1(Feature):
    def create_features(self):
        s = 1e-3
        feats = ['card1', 'V283']
        len_ = len(train_transaction)

        whole = pd.concat([train_transaction[feats], test_transaction[feats]], axis=0)
        df_mean = whole.groupby('card1')['V283'].mean().to_frame('mean_V283')
        whole = pd.merge(whole, df_mean, on='card1', how='left')
        whole[self.__class__.__name__] = whole['V283'] / (whole['mean_V283'] + s)

        self.train[self.__class__.__name__] = whole[self.__class__.__name__].values[:len_]
        self.test[self.__class__.__name__] = whole[self.__class__.__name__].values[len_:]


class V283_div_C2(Feature):
    def create_features(self):
        s = 1e-3
        self.train[self.__class__.__name__] = train_transaction['V283'] / (train_transaction['C2'] + s)
        self.test[self.__class__.__name__] = test_transaction['V283'] / (test_transaction['C2'] + s)


class transaction_amt_div_v283_mean(Feature):
    def create_features(self):
        feats = ['V283', 'TransactionAmt']
        whole = pd.concat([train_transaction[feats], test_transaction[feats]], axis=0)
        df_mean = whole.groupby(feats[0])[feats[1]].mean().reset_index()
        df_mean.columns = ['V283', 'mean_']
        train = pd.merge(train_transaction[feats], df_mean, on=feats[0], how='left')
        test = pd.merge(test_transaction[feats], df_mean, on=feats[0], how='left')
        self.train[self.__class__.__name__] = train['TransactionAmt'] / train['mean_']
        self.test[self.__class__.__name__] = test['TransactionAmt'] / test['mean_']


class v283_div_mean_each_date(Feature):
    def create_features(self):
        len_ = len(train_transaction)
        feats = ['V283', 'TransactionDT']
        whole = pd.concat([train_transaction[feats], test_transaction[feats]], axis=0).reset_index(drop=True)
        whole['date_'] = (whole['TransactionDT'] - 86400) // (3600 * 24)
        df_mean = whole.groupby('date_')['V283'].mean().reset_index()
        df_mean.columns = ['date_', 'mean_']
        whole = pd.merge(whole, df_mean, on='date_', how='left')
        whole[self.__class__.__name__] = whole['V283'] / whole['mean_']

        self.train[self.__class__.__name__] = whole[self.__class__.__name__].values[:len_]
        self.test[self.__class__.__name__] = whole[self.__class__.__name__].values[len_:]


class card1__V283(Feature):
    def create_features(self):
        len_ = len(train_transaction)
        feats = self.__class__.__name__
        col1, col2 = feats.split('__')
        whole = pd.concat([train_transaction[[col1, col2]], test_transaction[[col1, col2]]], axis=0)
        whole[feats] = whole[col1].astype(str) + '_' + whole[col2].astype(str)
        le = whole[feats].value_counts().to_dict()
        whole[self.__class__.__name__] = whole[feats].map(le)
        self.train[self.__class__.__name__] = whole[self.__class__.__name__].values[:len_]
        self.test[self.__class__.__name__] = whole[self.__class__.__name__].values[len_:]


class card1_regis_day(Feature):
    def create_features(self):
        len_ = len(train_transaction)
        feats = ['card1', 'TransactionDT', 'D1']
        whole = pd.concat([train_transaction[feats], test_transaction[feats]], axis=0)
        whole['day'] = (whole['TransactionDT'] - 86400) // (3600 * 24)
        whole['regis_day'] = whole['day'] - whole['D1']
        whole['uid'] = whole['card1'].astype(str) + '_' + whole['D1'].astype(str)
        le = {id_: i for i, id_ in enumerate(whole['uid'].unique())}
        whole[self.__class__.__name__] = whole['uid'].map(le)
        self.train[self.__class__.__name__] = whole[self.__class__.__name__].values[:len_]
        self.test[self.__class__.__name__] = whole[self.__class__.__name__].values[len_:]


# ===========================================================================================================================
# identity
# ===========================================================================================================================
class id_01(Feature):
    def create_features(self):
        feats = ['TransactionID', 'id_01']
        whole = pd.concat([train_identity[feats], test_identity[feats]], axis=0).reset_index(drop=True)
        self.train['id_01'] = pd.merge(train_transaction['TransactionID'], whole, on='TransactionID', how='left')['id_01']
        self.test['id_01'] = pd.merge(test_transaction['TransactionID'], whole, on='TransactionID', how='left')['id_01']


class id_02(Feature):
    def create_features(self):
        feats = ['TransactionID', 'id_02']
        whole = pd.concat([train_identity[feats], test_identity[feats]], axis=0).reset_index(drop=True)
        self.train['id_02'] = pd.merge(train_transaction['TransactionID'], whole, on='TransactionID', how='left')['id_02']
        self.test['id_02'] = pd.merge(test_transaction['TransactionID'], whole, on='TransactionID', how='left')['id_02']


class mean_id_02_each_card1(Feature):
    def create_features(self):
        identity_feats = ['TransactionID', 'id_02']
        transaction_feats = ['TransactionID', 'card1']

        identity_whole = pd.concat([train_identity[identity_feats], test_identity[identity_feats]], axis=0)
        transaction_whole = pd.concat([train_transaction[transaction_feats], test_transaction[transaction_feats]], axis=0)

        whole = pd.merge(transaction_whole, identity_whole, on='TransactionID', how='left')

        df_mean = whole.groupby('card1')['id_02'].mean().to_frame(self.__class__.__name__)

        self.train[self.__class__.__name__] = pd.merge(train_transaction, df_mean, on='card1', how='left')[self.__class__.__name__]
        self.test[self.__class__.__name__] = pd.merge(test_transaction, df_mean, on='card1', how='left')[self.__class__.__name__]


class id_02_div_mean_each_card1(Feature):
    def create_features(self):
        identity_feats = ['TransactionID', 'id_02']
        transaction_feats = ['TransactionID', 'card1']
        len_ = len(train_transaction)

        identity_whole = pd.concat([train_identity[identity_feats], test_identity[identity_feats]], axis=0)
        transaction_whole = pd.concat([train_transaction[transaction_feats], test_transaction[transaction_feats]], axis=0)
        whole = pd.merge(transaction_whole, identity_whole, on='TransactionID', how='left')

        df_mean = whole.groupby('card1')['id_02'].mean().to_frame('mean_id_02_each_card1')
        whole = pd.merge(whole, df_mean, on='card1', how='left')
        whole[self.__class__.__name__] = whole['id_02'] / whole['mean_id_02_each_card1']

        self.train[self.__class__.__name__] = whole[self.__class__.__name__].values[:len_]
        self.test[self.__class__.__name__] = whole[self.__class__.__name__].values[len_:]


class id_02_div_transaction_amt(Feature):
    def create_features(self):
        identity_feats = ['TransactionID', 'id_02']
        transaction_feats = ['TransactionID', 'TransactionAmt']
        len_ = len(train_transaction)
        whole_identity = pd.concat([train_identity[identity_feats], test_identity[identity_feats]], axis=0)
        whole_transaction = pd.concat([train_transaction[transaction_feats], test_transaction[transaction_feats]], axis=0)
        whole = pd.merge(whole_transaction, whole_identity, on='TransactionID', how='left')
        whole[self.__class__.__name__] = whole['id_02'] / whole['TransactionAmt']
        self.train[self.__class__.__name__] = whole[self.__class__.__name__].values[:len_]
        self.test[self.__class__.__name__] = whole[self.__class__.__name__].values[len_:]


class id_03(Feature):
    def create_features(self):
        feats = ['TransactionID', 'id_03']
        whole = pd.concat([train_identity[feats], test_identity[feats]], axis=0).reset_index(drop=True)
        self.train['id_03'] = pd.merge(train_transaction['TransactionID'], whole, on='TransactionID', how='left')['id_03']
        self.test['id_03'] = pd.merge(test_transaction['TransactionID'], whole, on='TransactionID', how='left')['id_03']


class id_04(Feature):
    def create_features(self):
        feats = ['TransactionID', 'id_04']
        whole = pd.concat([train_identity[feats], test_identity[feats]], axis=0).reset_index(drop=True)
        self.train['id_04'] = pd.merge(train_transaction['TransactionID'], whole, on='TransactionID', how='left')['id_04']
        self.test['id_04'] = pd.merge(test_transaction['TransactionID'], whole, on='TransactionID', how='left')['id_04']


class id_05(Feature):
    def create_features(self):
        feats = ['TransactionID', 'id_05']
        whole = pd.concat([train_identity[feats], test_identity[feats]], axis=0).reset_index(drop=True)
        self.train['id_05'] = pd.merge(train_transaction['TransactionID'], whole, on='TransactionID', how='left')['id_05']
        self.test['id_05'] = pd.merge(test_transaction['TransactionID'], whole, on='TransactionID', how='left')['id_05']


class id_06(Feature):
    def create_features(self):
        feats = ['TransactionID', 'id_06']
        whole = pd.concat([train_identity[feats], test_identity[feats]], axis=0).reset_index(drop=True)
        self.train['id_06'] = pd.merge(train_transaction['TransactionID'], whole, on='TransactionID', how='left')['id_06']
        self.test['id_06'] = pd.merge(test_transaction['TransactionID'], whole, on='TransactionID', how='left')['id_06']


class id_07(Feature):
    def create_features(self):
        feats = ['TransactionID', 'id_07']
        whole = pd.concat([train_identity[feats], test_identity[feats]], axis=0).reset_index(drop=True)
        self.train['id_07'] = pd.merge(train_transaction['TransactionID'], whole, on='TransactionID', how='left')['id_07']
        self.test['id_07'] = pd.merge(test_transaction['TransactionID'], whole, on='TransactionID', how='left')['id_07']


class id_08(Feature):
    def create_features(self):
        feats = ['TransactionID', 'id_08']
        whole = pd.concat([train_identity[feats], test_identity[feats]], axis=0).reset_index(drop=True)
        self.train['id_08'] = pd.merge(train_transaction['TransactionID'], whole, on='TransactionID', how='left')['id_08']
        self.test['id_08'] = pd.merge(test_transaction['TransactionID'], whole, on='TransactionID', how='left')['id_08']


class id_09(Feature):
    def create_features(self):
        feats = ['TransactionID', 'id_09']
        whole = pd.concat([train_identity[feats], test_identity[feats]], axis=0).reset_index(drop=True)
        self.train['id_09'] = pd.merge(train_transaction['TransactionID'], whole, on='TransactionID', how='left')['id_09']
        self.test['id_09'] = pd.merge(test_transaction['TransactionID'], whole, on='TransactionID', how='left')['id_09']


class id_10(Feature):
    def create_features(self):
        feats = ['TransactionID', 'id_10']
        whole = pd.concat([train_identity[feats], test_identity[feats]], axis=0).reset_index(drop=True)
        self.train['id_10'] = pd.merge(train_transaction['TransactionID'], whole, on='TransactionID', how='left')['id_10']
        self.test['id_10'] = pd.merge(test_transaction['TransactionID'], whole, on='TransactionID', how='left')['id_10']


class id_11(Feature):
    def create_features(self):
        feats = ['TransactionID', 'id_11']
        whole = pd.concat([train_identity[feats], test_identity[feats]], axis=0).reset_index(drop=True)
        self.train['id_11'] = pd.merge(train_transaction['TransactionID'], whole, on='TransactionID', how='left')['id_11']
        self.test['id_11'] = pd.merge(test_transaction['TransactionID'], whole, on='TransactionID', how='left')['id_11']


class id_12(Feature):
    def create_features(self):
        feats = ['TransactionID', 'id_12']
        whole = pd.concat([train_identity[feats], test_identity[feats]], axis=0).reset_index(drop=True)
        le = {'NotFound': 0, 'Found': 1}
        whole['id_12'] = whole['id_12'].apply(lambda x: le[x] if type(x) == str else x)
        self.train['id_12'] = pd.merge(train_transaction['TransactionID'], whole, on='TransactionID', how='left')['id_12']
        self.test['id_12'] = pd.merge(test_transaction['TransactionID'], whole, on='TransactionID', how='left')['id_12']


class id_13(Feature):
    def create_features(self):
        feats = ['TransactionID', 'id_13']
        whole = pd.concat([train_identity[feats], test_identity[feats]], axis=0).reset_index(drop=True)
        self.train['id_13'] = pd.merge(train_transaction['TransactionID'], whole, on='TransactionID', how='left')['id_13']
        self.test['id_13'] = pd.merge(test_transaction['TransactionID'], whole, on='TransactionID', how='left')['id_13']


class id_14(Feature):
    def create_features(self):
        feats = ['TransactionID', 'id_14']
        whole = pd.concat([train_identity[feats], test_identity[feats]], axis=0).reset_index(drop=True)
        self.train['id_14'] = pd.merge(train_transaction['TransactionID'], whole, on='TransactionID', how='left')['id_14']
        self.test['id_14'] = pd.merge(test_transaction['TransactionID'], whole, on='TransactionID', how='left')['id_14']


class id_15(Feature):
    def create_features(self):
        feats = ['TransactionID', 'id_15']
        whole = pd.concat([train_identity[feats], test_identity[feats]], axis=0).reset_index(drop=True)
        le = {'Found': 0, 'New': 1, "Unknown": 2}
        whole['id_15'] = whole['id_15'].apply(lambda x: le[x] if type(x) == str else x)
        self.train['id_15'] = pd.merge(train_transaction['TransactionID'], whole, on='TransactionID', how='left')['id_15']
        self.test['id_15'] = pd.merge(test_transaction['TransactionID'], whole, on='TransactionID', how='left')['id_15']


class id_16(Feature):
    def create_features(self):
        feats = ['TransactionID', 'id_16']
        whole = pd.concat([train_identity[feats], test_identity[feats]], axis=0).reset_index(drop=True)
        le = {'NotFound': 0, 'Found': 1}
        whole['id_16'] = whole['id_16'].apply(lambda x: le[x] if type(x) == str else x)
        self.train['id_16'] = pd.merge(train_transaction['TransactionID'], whole, on='TransactionID', how='left')['id_16']
        self.test['id_16'] = pd.merge(test_transaction['TransactionID'], whole, on='TransactionID', how='left')['id_16']


class id_17(Feature):
    def create_features(self):
        feats = ['TransactionID', 'id_17']
        whole = pd.concat([train_identity[feats], test_identity[feats]], axis=0).reset_index(drop=True)
        self.train['id_17'] = pd.merge(train_transaction['TransactionID'], whole, on='TransactionID', how='left')['id_17']
        self.test['id_17'] = pd.merge(test_transaction['TransactionID'], whole, on='TransactionID', how='left')['id_17']


class id_18(Feature):
    def create_features(self):
        feats = ['TransactionID', 'id_18']
        whole = pd.concat([train_identity[feats], test_identity[feats]], axis=0).reset_index(drop=True)
        self.train['id_18'] = pd.merge(train_transaction['TransactionID'], whole, on='TransactionID', how='left')['id_18']
        self.test['id_18'] = pd.merge(test_transaction['TransactionID'], whole, on='TransactionID', how='left')['id_18']


class id_19(Feature):
    def create_features(self):
        feats = ['TransactionID', 'id_19']
        whole = pd.concat([train_identity[feats], test_identity[feats]], axis=0).reset_index(drop=True)
        self.train['id_19'] = pd.merge(train_transaction['TransactionID'], whole, on='TransactionID', how='left')['id_19']
        self.test['id_19'] = pd.merge(test_transaction['TransactionID'], whole, on='TransactionID', how='left')['id_19']


class id_20(Feature):
    def create_features(self):
        feats = ['TransactionID', 'id_20']
        whole = pd.concat([train_identity[feats], test_identity[feats]], axis=0).reset_index(drop=True)
        self.train['id_20'] = pd.merge(train_transaction['TransactionID'], whole, on='TransactionID', how='left')['id_20']
        self.test['id_20'] = pd.merge(test_transaction['TransactionID'], whole, on='TransactionID', how='left')['id_20']


class mean_id_20_each_card1(Feature):
    def create_features(self):
        identity_feats = ['TransactionID', 'id_20']
        transaction_feats = ['TransactionID', 'card1']

        identity_whole = pd.concat([train_identity[identity_feats], test_identity[identity_feats]], axis=0)
        transaction_whole = pd.concat([train_transaction[transaction_feats], test_transaction[transaction_feats]], axis=0)

        whole = pd.merge(transaction_whole, identity_whole, on='TransactionID', how='left')

        df_mean = whole.groupby('card1')['id_20'].mean().to_frame(self.__class__.__name__)

        self.train[self.__class__.__name__] = pd.merge(train_transaction, df_mean, on='card1', how='left')[self.__class__.__name__]
        self.test[self.__class__.__name__] = pd.merge(test_transaction, df_mean, on='card1', how='left')[self.__class__.__name__]


class id_20_div_mean_each_card1(Feature):
    def create_features(self):
        identity_feats = ['TransactionID', 'id_20']
        transaction_feats = ['TransactionID', 'card1']
        len_ = len(train_transaction)

        identity_whole = pd.concat([train_identity[identity_feats], test_identity[identity_feats]], axis=0)
        transaction_whole = pd.concat([train_transaction[transaction_feats], test_transaction[transaction_feats]], axis=0)
        whole = pd.merge(transaction_whole, identity_whole, on='TransactionID', how='left')

        df_mean = whole.groupby('card1')['id_20'].mean().to_frame('mean_id_20_each_card1')
        whole = pd.merge(whole, df_mean, on='card1', how='left')
        whole[self.__class__.__name__] = whole['id_20'] / whole['mean_id_20_each_card1']

        self.train[self.__class__.__name__] = whole[self.__class__.__name__].values[:len_]
        self.test[self.__class__.__name__] = whole[self.__class__.__name__].values[len_:]


class id_20_div_transaction_amt(Feature):
    def create_features(self):
        identity_feats = ['TransactionID', 'id_20']
        transaction_feats = ['TransactionID', 'TransactionAmt']
        len_ = len(train_transaction)
        whole_identity = pd.concat([train_identity[identity_feats], test_identity[identity_feats]], axis=0)
        whole_transaction = pd.concat([train_transaction[transaction_feats], test_transaction[transaction_feats]], axis=0)
        whole = pd.merge(whole_transaction, whole_identity, on='TransactionID', how='left')
        whole[self.__class__.__name__] = whole['id_20'] / whole['TransactionAmt']
        self.train[self.__class__.__name__] = whole[self.__class__.__name__].values[:len_]
        self.test[self.__class__.__name__] = whole[self.__class__.__name__].values[len_:]


class id_21(Feature):
    def create_features(self):
        feats = ['TransactionID', 'id_21']
        whole = pd.concat([train_identity[feats], test_identity[feats]], axis=0).reset_index(drop=True)
        self.train['id_21'] = pd.merge(train_transaction['TransactionID'], whole, on='TransactionID', how='left')['id_21']
        self.test['id_21'] = pd.merge(test_transaction['TransactionID'], whole, on='TransactionID', how='left')['id_21']


class id_22(Feature):
    def create_features(self):
        feats = ['TransactionID', 'id_22']
        whole = pd.concat([train_identity[feats], test_identity[feats]], axis=0).reset_index(drop=True)
        self.train['id_22'] = pd.merge(train_transaction['TransactionID'], whole, on='TransactionID', how='left')['id_22']
        self.test['id_22'] = pd.merge(test_transaction['TransactionID'], whole, on='TransactionID', how='left')['id_22']


class id_23(Feature):
    def create_features(self):
        feats = ['TransactionID', 'id_23']
        whole = pd.concat([train_identity[feats], test_identity[feats]], axis=0).reset_index(drop=True)
        le = {'IP_PROXY:TRANSPARENT': 0, 'IP_PROXY:ANONYMOUS': 1, 'IP_PROXY:HIDDEN': 2}
        whole['id_23'] = whole['id_23'].apply(lambda x: le[x] if type(x) == str else x)
        self.train['id_23'] = pd.merge(train_transaction['TransactionID'], whole, on='TransactionID', how='left')['id_23']
        self.test['id_23'] = pd.merge(test_transaction['TransactionID'], whole, on='TransactionID', how='left')['id_23']


class id_24(Feature):
    def create_features(self):
        feats = ['TransactionID', 'id_24']
        whole = pd.concat([train_identity[feats], test_identity[feats]], axis=0).reset_index(drop=True)
        self.train['id_24'] = pd.merge(train_transaction['TransactionID'], whole, on='TransactionID', how='left')['id_24']
        self.test['id_24'] = pd.merge(test_transaction['TransactionID'], whole, on='TransactionID', how='left')['id_24']


class id_25(Feature):
    def create_features(self):
        feats = ['TransactionID', 'id_25']
        whole = pd.concat([train_identity[feats], test_identity[feats]], axis=0).reset_index(drop=True)
        self.train['id_25'] = pd.merge(train_transaction['TransactionID'], whole, on='TransactionID', how='left')['id_25']
        self.test['id_25'] = pd.merge(test_transaction['TransactionID'], whole, on='TransactionID', how='left')['id_25']


class id_26(Feature):
    def create_features(self):
        feats = ['TransactionID', 'id_26']
        whole = pd.concat([train_identity[feats], test_identity[feats]], axis=0).reset_index(drop=True)
        self.train['id_26'] = pd.merge(train_transaction['TransactionID'], whole, on='TransactionID', how='left')['id_26']
        self.test['id_26'] = pd.merge(test_transaction['TransactionID'], whole, on='TransactionID', how='left')['id_26']


class id_27(Feature):
    def create_features(self):
        feats = ['TransactionID', 'id_27']
        whole = pd.concat([train_identity[feats], test_identity[feats]], axis=0).reset_index(drop=True)
        le = {'Found': 0, 'NotFound': 1}
        whole['id_27'] = whole['id_27'].apply(lambda x: le[x] if type(x) == str else x)
        self.train['id_27'] = pd.merge(train_transaction['TransactionID'], whole, on='TransactionID', how='left')['id_27']
        self.test['id_27'] = pd.merge(test_transaction['TransactionID'], whole, on='TransactionID', how='left')['id_27']


class id_28(Feature):
    def create_features(self):
        feats = ['TransactionID', 'id_28']
        whole = pd.concat([train_identity[feats], test_identity[feats]], axis=0).reset_index(drop=True)
        le = {'Found': 0, 'New': 1}
        whole['id_28'] = whole['id_28'].apply(lambda x: le[x] if type(x) == str else x)
        self.train['id_28'] = pd.merge(train_transaction['TransactionID'], whole, on='TransactionID', how='left')['id_28']
        self.test['id_28'] = pd.merge(test_transaction['TransactionID'], whole, on='TransactionID', how='left')['id_28']


class id_29(Feature):
    def create_features(self):
        feats = ['TransactionID', 'id_29']
        whole = pd.concat([train_identity[feats], test_identity[feats]], axis=0).reset_index(drop=True)
        le = {'Found': 0, 'NotFound': 1}
        whole['id_29'] = whole['id_29'].apply(lambda x: le[x] if type(x) == str else x)
        self.train['id_29'] = pd.merge(train_transaction['TransactionID'], whole, on='TransactionID', how='left')['id_29']
        self.test['id_29'] = pd.merge(test_transaction['TransactionID'], whole, on='TransactionID', how='left')['id_29']


class id_30(Feature):
    def create_features(self):
        feats = ['TransactionID', 'id_30']
        whole = pd.concat([train_identity[feats], test_identity[feats]], axis=0).reset_index(drop=True)
        le = {v: i for i, v in enumerate(whole['id_30'].unique())}
        whole['id_30'] = whole['id_30'].apply(lambda x: le[x] if type(x) == str else x)
        self.train['id_30'] = pd.merge(train_transaction['TransactionID'], whole, on='TransactionID', how='left')['id_30']
        self.test['id_30'] = pd.merge(test_transaction['TransactionID'], whole, on='TransactionID', how='left')['id_30']


class id_30_os(Feature):
    def create_features(self):
        p = r'^\w+'
        feats = ['TransactionID', 'id_30']
        whole = pd.concat([train_identity[feats], test_identity[feats]], axis=0).reset_index(drop=True)
        whole['id_30_os'] = whole['id_30'].apply(lambda x: re.search(p, x).group(0) if x is not None else x)
        le = {os_: i for i, os_ in enumerate(whole['id_30_os'].unique())}
        whole['id_30_os'] = whole['id_30_os'].apply(lambda x: le[x] if type(x) == str else x)
        self.train['id_30_os'] = pd.merge(train_transaction['TransactionID'], whole, on='TransactionID', how='left')['id_30_os']
        self.test['id_30_os'] = pd.merge(test_transaction['TransactionID'], whole, on='TransactionID', how='left')['id_30_os']


class elapsed_from_os_release(Feature):
    def create_features(self):
        identity_col = ['TransactionID', 'id_30']
        whole_identity = pd.concat([train_identity[identity_col], test_identity[identity_col]], axis=0)
        whole_identity['release_date'] = whole_identity['id_30'].apply(lambda x: os2release_date[x] if type(x) == str else x)

        transaction_col = ['TransactionID', 'TransactionDT']
        whole_transaction = pd.concat([train_transaction[transaction_col], test_transaction[transaction_col]], axis=0)
        whole_transaction['date_'] = whole_transaction['TransactionDT'].apply(lambda x: (START_DATE + datetime.timedelta(seconds=x)))

        whole = pd.merge(whole_transaction, whole_identity, on='TransactionID', how='left')
        whole[self.__class__.__name__] = (whole['date_'] - whole['release_date'].fillna(datetime.date(1990, 1, 1))).dt.days
        whole[self.__class__.__name__] = np.where(whole[self.__class__.__name__].values < 10000, whole[self.__class__.__name__], np.nan)

        self.train[self.__class__.__name__] = whole[self.__class__.__name__].values[:len(train_transaction)]
        self.test[self.__class__.__name__] = whole[self.__class__.__name__].values[len(train_transaction):]


class id_31(Feature):
    def create_features(self):
        feats = ['TransactionID', 'id_31']
        whole = pd.concat([train_identity[feats], test_identity[feats]], axis=0).reset_index(drop=True)
        le = {v: i for i, v in enumerate(whole['id_31'].unique())}
        whole['id_31'] = whole['id_31'].apply(lambda x: le[x] if type(x) == str else x)
        self.train['id_31'] = pd.merge(train_transaction['TransactionID'], whole, on='TransactionID', how='left')['id_31']
        self.test['id_31'] = pd.merge(test_transaction['TransactionID'], whole, on='TransactionID', how='left')['id_31']


class id_31_top(Feature):
    def create_features(self):
        feats = ['TransactionID', 'id_31']
        whole = pd.concat([train_identity[feats], test_identity[feats]], axis=0).reset_index(drop=True)
        whole['id_31_top'] = whole['id_31'].apply(lambda x: x.split(' ')[0] if x is not None else x)
        le = {v: i for i, v in enumerate(whole['id_31_top'].unique())}
        whole['id_31_top'] = whole['id_31_top'].apply(lambda x: le[x] if type(x) == str else x)
        self.train['id_31_top'] = pd.merge(train_transaction['TransactionID'], whole, on='TransactionID', how='left')['id_31_top']
        self.test['id_31_top'] = pd.merge(test_transaction['TransactionID'], whole, on='TransactionID', how='left')['id_31_top']


class elapsed_from_brawser_release(Feature):
    def create_features(self):
        identity_col = ['TransactionID', 'id_31']
        whole_identity = pd.concat([train_identity[identity_col], test_identity[identity_col]], axis=0)
        whole_identity['release_date'] = whole_identity['id_31'].apply(lambda x: brawser2release_date[x] if type(x) == str else x)

        transaction_col = ['TransactionID', 'TransactionDT']
        whole_transaction = pd.concat([train_transaction[transaction_col], test_transaction[transaction_col]], axis=0)
        whole_transaction['date_'] = whole_transaction['TransactionDT'].apply(lambda x: (START_DATE + datetime.timedelta(seconds=x)))

        whole = pd.merge(whole_transaction, whole_identity, on='TransactionID', how='left')
        whole[self.__class__.__name__] = (whole['date_'] - whole['release_date'].fillna(datetime.date(1990, 1, 1))).dt.days
        whole[self.__class__.__name__] = np.where(whole[self.__class__.__name__].values < 10000, whole[self.__class__.__name__], np.nan)

        self.train[self.__class__.__name__] = whole[self.__class__.__name__].values[:len(train_transaction)]
        self.test[self.__class__.__name__] = whole[self.__class__.__name__].values[len(train_transaction):]


class id_32(Feature):
    def create_features(self):
        feats = ['TransactionID', 'id_32']
        whole = pd.concat([train_identity[feats], test_identity[feats]], axis=0).reset_index(drop=True)
        self.train['id_32'] = pd.merge(train_transaction['TransactionID'], whole, on='TransactionID', how='left')['id_32']
        self.test['id_32'] = pd.merge(test_transaction['TransactionID'], whole, on='TransactionID', how='left')['id_32']


class id_33(Feature):
    def create_features(self):
        feats = ['TransactionID', 'id_33']
        whole = pd.concat([train_identity[feats], test_identity[feats]], axis=0).reset_index(drop=True)
        le = {v: i for i, v in enumerate(whole['id_33'].unique())}
        whole['id_33'] = whole['id_33'].apply(lambda x: le[x] if type(x) == str else x)
        self.train['id_33'] = pd.merge(train_transaction['TransactionID'], whole, on='TransactionID', how='left')['id_33']
        self.test['id_33'] = pd.merge(test_transaction['TransactionID'], whole, on='TransactionID', how='left')['id_33']


class id_34(Feature):
    def create_features(self):
        feats = ['TransactionID', 'id_34']
        whole = pd.concat([train_identity[feats], test_identity[feats]], axis=0).reset_index(drop=True)
        le = {'match_status:-1': -1, 'match_status:0': 0, 'match_status:1': 1, 'match_status:2': 2}
        whole['id_34'] = whole['id_34'].apply(lambda x: le[x] if type(x) == str else x)
        self.train['id_34'] = pd.merge(train_transaction['TransactionID'], whole, on='TransactionID', how='left')['id_34']
        self.test['id_34'] = pd.merge(test_transaction['TransactionID'], whole, on='TransactionID', how='left')['id_34']


class id_35(Feature):
    def create_features(self):
        feats = ['TransactionID', 'id_35']
        whole = pd.concat([train_identity[feats], test_identity[feats]], axis=0).reset_index(drop=True)
        le = {'F': 0, 'T': 1}
        whole['id_35'] = whole['id_35'].apply(lambda x: le[x] if type(x) == str else x)
        self.train['id_35'] = pd.merge(train_transaction['TransactionID'], whole, on='TransactionID', how='left')['id_35']
        self.test['id_35'] = pd.merge(test_transaction['TransactionID'], whole, on='TransactionID', how='left')['id_35']


class id_36(Feature):
    def create_features(self):
        feats = ['TransactionID', 'id_36']
        whole = pd.concat([train_identity[feats], test_identity[feats]], axis=0).reset_index(drop=True)
        le = {'F': 0, 'T': 1}
        whole['id_36'] = whole['id_36'].apply(lambda x: le[x] if type(x) == str else x)
        self.train['id_36'] = pd.merge(train_transaction['TransactionID'], whole, on='TransactionID', how='left')['id_36']
        self.test['id_36'] = pd.merge(test_transaction['TransactionID'], whole, on='TransactionID', how='left')['id_36']


class id_37(Feature):
    def create_features(self):
        feats = ['TransactionID', 'id_37']
        whole = pd.concat([train_identity[feats], test_identity[feats]], axis=0).reset_index(drop=True)
        le = {'F': 0, 'T': 1}
        whole['id_37'] = whole['id_37'].apply(lambda x: le[x] if type(x) == str else x)
        self.train['id_37'] = pd.merge(train_transaction['TransactionID'], whole, on='TransactionID', how='left')['id_37']
        self.test['id_37'] = pd.merge(test_transaction['TransactionID'], whole, on='TransactionID', how='left')['id_37']


class id_38(Feature):
    def create_features(self):
        feats = ['TransactionID', 'id_38']
        whole = pd.concat([train_identity[feats], test_identity[feats]], axis=0).reset_index(drop=True)
        le = {'F': 0, 'T': 1}
        whole['id_38'] = whole['id_38'].apply(lambda x: le[x] if type(x) == str else x)
        self.train['id_38'] = pd.merge(train_transaction['TransactionID'], whole, on='TransactionID', how='left')['id_38']
        self.test['id_38'] = pd.merge(test_transaction['TransactionID'], whole, on='TransactionID', how='left')['id_38']


class device_type(Feature):
    def create_features(self):
        feats = ['TransactionID', 'DeviceType']
        whole = pd.concat([train_identity[feats], test_identity[feats]], axis=0).reset_index(drop=True)
        le = {'desktop': 0, 'mobile': 1}
        whole['DeviceType'] = whole['DeviceType'].apply(lambda x: le[x] if type(x) == str else x)
        self.train['DeviceType'] = pd.merge(train_transaction['TransactionID'], whole, on='TransactionID', how='left')['DeviceType']
        self.test['DeviceType'] = pd.merge(test_transaction['TransactionID'], whole, on='TransactionID', how='left')['DeviceType']


class device_info(Feature):
    def create_features(self):
        feats = ['TransactionID', 'DeviceInfo']
        whole = pd.concat([train_identity[feats], test_identity[feats]], axis=0).reset_index(drop=True)
        le = {v: i for i, v in enumerate(whole['DeviceInfo'].unique())}
        whole['DeviceInfo'] = whole['DeviceInfo'].apply(lambda x: le[x] if type(x) == str else x)
        self.train['DeviceInfo'] = pd.merge(train_transaction['TransactionID'], whole, on='TransactionID', how='left')['DeviceInfo']
        self.test['DeviceInfo'] = pd.merge(test_transaction['TransactionID'], whole, on='TransactionID', how='left')['DeviceInfo']


class nunique_id_30_each_card1(Feature):
    def create_features(self):
        transaction_col = ['TransactionID', 'card1']
        identity_col = ['TransactionID', 'id_30']
        whole_transaction = pd.concat([train_transaction[transaction_col], test_transaction[transaction_col]], axis=0)
        whole_identity = pd.concat([train_identity[identity_col], test_identity[identity_col]], axis=0)
        whole = pd.merge(whole_transaction, whole_identity, on='TransactionID', how='left')
        le = whole.groupby('card1')['id_30'].nunique().to_dict()
        self.train[self.__class__.__name__] = train_transaction['card1'].map(le)
        self.test[self.__class__.__name__] = test_transaction['card1'].map(le)


class nunique_id_31_each_card1(Feature):
    def create_features(self):
        transaction_col = ['TransactionID', 'card1']
        identity_col = ['TransactionID', 'id_31']
        whole_transaction = pd.concat([train_transaction[transaction_col], test_transaction[transaction_col]], axis=0)
        whole_identity = pd.concat([train_identity[identity_col], test_identity[identity_col]], axis=0)
        whole = pd.merge(whole_transaction, whole_identity, on='TransactionID', how='left')
        le = whole.groupby('card1')['id_31'].nunique().to_dict()
        self.train[self.__class__.__name__] = train_transaction['card1'].map(le)
        self.test[self.__class__.__name__] = test_transaction['card1'].map(le)


class nunique_DeviceType_each_card1(Feature):
    def create_features(self):
        transaction_col = ['TransactionID', 'card1']
        identity_col = ['TransactionID', 'DeviceType']
        whole_transaction = pd.concat([train_transaction[transaction_col], test_transaction[transaction_col]], axis=0)
        whole_identity = pd.concat([train_identity[identity_col], test_identity[identity_col]], axis=0)
        whole = pd.merge(whole_transaction, whole_identity, on='TransactionID', how='left')
        le = whole.groupby('card1')['DeviceType'].nunique().to_dict()
        self.train[self.__class__.__name__] = train_transaction['card1'].map(le)
        self.test[self.__class__.__name__] = test_transaction['card1'].map(le)


class nunique_DeviceInfo_each_card1(Feature):
    def create_features(self):
        transaction_col = ['TransactionID', 'card1']
        identity_col = ['TransactionID', 'DeviceInfo']
        whole_transaction = pd.concat([train_transaction[transaction_col], test_transaction[transaction_col]], axis=0)
        whole_identity = pd.concat([train_identity[identity_col], test_identity[identity_col]], axis=0)
        whole = pd.merge(whole_transaction, whole_identity, on='TransactionID', how='left')
        le = whole.groupby('card1')['DeviceInfo'].nunique().to_dict()
        self.train[self.__class__.__name__] = train_transaction['card1'].map(le)
        self.test[self.__class__.__name__] = test_transaction['card1'].map(le)


class identity_flg(Feature):
    def create_features(self):
        len_ = len(train_transaction)
        transaction_whole = pd.concat([train_transaction[['TransactionID']], train_transaction[['TransactionID']]], axis=0)
        identity_whole = pd.concat([train_identity[['TransactionID']], test_identity[['TransactionID']]], axis=0)
        identity_whole['identity_flg'] = 1

        whole = pd.merge(transaction_whole, identity_whole, on='TransactionID', how='left').fillna(0).astype(int)
        self.train[self.__class__.__name__] = whole['identity_flg'].values[:len_]
        self.test[self.__class__.__name__] = whole['identity_flg'].values[len_:]


if __name__ == '__main__':
    args = get_arguments()

    train_identity = pd.read_feather('./data/input/train_identity.feather')
    train_transaction = pd.read_feather('./data/input/train_transaction.feather')
    test_identity = pd.read_feather('./data/input/test_identity.feather')
    test_transaction = pd.read_feather('./data/input/test_transaction.feather')

    generate_features(globals(), args.force)
