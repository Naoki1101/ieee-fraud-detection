import os
import time
import glob
import pandas as pd
from contextlib import contextmanager


@contextmanager
def timer(name, logging):
    t0 = time.time()
    yield
    print('[{name}] done in {end:.0f} s'.format(name=name, end=time.time() - t0))
    logging.debug('[{name}] done in {end:.0f} s'.format(name=name, end=time.time() - t0))


def load_datasets(feats, dict_path, fake=False):
    dfs = [pd.read_feather(dict_path[f])[f] for f in feats]
    X_train = pd.concat(dfs, axis=1)
    dfs = [pd.read_feather(dict_path[f].replace('train', 'test'))[f] for f in feats]
    X_test = pd.concat(dfs, axis=1)
    return X_train, X_test


def load_target(target_name):
    train = pd.read_csv('./data/input/train_transaction.csv')
    y_train = train[target_name]
    return y_train
