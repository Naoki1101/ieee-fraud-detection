import argparse
import inspect
import re
import pickle
from abc import ABCMeta, abstractmethod
from pathlib import Path

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold


SEED = 2019


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--force', '-f', action='store_true', help='Overwrite existing files'
    )
    return parser.parse_args()


def get_features(namespace):
    for k, v in namespace.items():
        if inspect.isclass(v) and issubclass(v, Feature) \
                and not inspect.isabstract(v):
            yield v()


def generate_features(namespace, overwrite):
    for f in get_features(namespace):
        if f.train_path.exists() and f.test_path.exists() and not overwrite:
            print(f.name, 'was skipped')
        else:
            f.run().save()
            f.add_columns()


class Feature(metaclass=ABCMeta):
    prefix = ''
    suffix = ''
    dir = './features/'

    def __init__(self):
        if self.__class__.__name__.isupper():
            self.name = self.__class__.__name__.lower()
        else:
            self.name = re.sub(
                "([A-Z])",
                lambda x: "_" + x.group(1).lower(), self.__class__.__name__
            ).lstrip('_')

        self.train = pd.DataFrame()
        self.test = pd.DataFrame()
        self.train_path = Path(self.dir) / f'fe_{self.name}_train.feather'
        self.test_path = Path(self.dir) / f'fe_{self.name}_test.feather'

        self.col2path_path = './data/else/col2path.pkl'
        with open(self.col2path_path, 'rb') as f:
            self.col2path = pickle.load(f)

    def run(self):
        # with timer(self.name):
        self.create_features()
        prefix = self.prefix + '_' if self.prefix else ''
        suffix = '_' + self.suffix if self.suffix else ''
        self.train.columns = prefix + self.train.columns + suffix
        self.test.columns = prefix + self.test.columns + suffix
        return self

    @abstractmethod
    def create_features(self):
        raise NotImplementedError

    def save(self):
        self.train.to_feather(str(self.train_path))
        self.test.to_feather(str(self.test_path))

    def load(self):
        self.train = pd.read_feather(str(self.train_path))
        self.test = pd.read_feather(str(self.test_path))

    def add_columns(self):
        for col in self.train.columns:
            self.col2path[col] = str(self.train_path)

        with open(self.col2path_path, 'wb') as f:
            pickle.dump(self.col2path, f)


# ===============
# Feature Module
# ===============
def sigmoid(values):
    return 1 / (1 + np.exp(-1 * values))


def minmaxscale(values):
    min_ = np.min(values)
    max_ = np.max(values)
    return (values - min_) / (max_ - min_) + 0.01


def count_encoding(tr, te, col):
    tr_len = len(tr)
    whole = pd.concat([tr[[col]], te[[col]]], axis=0)
    df_count = whole[col].value_counts().to_frame('count_').reset_index()
    df_count.rename(columns={'index': col}, inplace=True)
    whole_encoded = pd.merge(whole, df_count, on=col, how='left')
    tr_ce = whole_encoded.iloc[:tr_len, 1].values
    te_ce = whole_encoded.iloc[tr_len:, 1].values
    return tr_ce, te_ce


def target_encoding(tr, te, target, feat, folds):

    target_tr = np.zeros(len(tr))
    target_te = np.zeros(len(te))
    n_splits = len(folds['fold_id'].unique())

    mean_all = tr[target].mean()
    le_all = dict(tr.groupby(feat)[target].mean())

    target_te = te[feat].apply(lambda x: le_all[x] if x in le_all.keys() else mean_all).values

    for fold_ in range(n_splits):
        X_train, X_val = tr[folds['fold_id'] != fold_], tr[folds['fold_id'] == fold_]
        mean_ = np.mean(X_train[target])

        le = dict(X_train.groupby(feat)[target].mean())

        target_tr[X_val.index] = X_val[feat].apply(lambda x: le[x] if x in le.keys() else mean_)

    return target_tr, target_te


class SinCos():
    def __init__(self, feature_name, period):
        '''
        input
        ---
        feature_name(str): name of feature
        period(int): period of feature
        '''
        self.feature_name = feature_name
        self.period = period

    def create_features(self, df):
        df['{}_sin'.format(self.feature_name)] = np.round(np.sin(2 * np.pi * df[self.feature_name]/self.period), 3)
        df['{}_cos'.format(self.feature_name)] = np.round(np.cos(2 * np.pi * df[self.feature_name] / self.period), 3)
        new_cols = ["{}_{}".format(self.feature_name, key) for key in ["sin", "cos"]]

        return df, new_cols
