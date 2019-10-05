import numpy as np
import pandas as pd
import re
import unicodedata
import datetime

# from sklearn.manifold import TSNE
# from bhtsne import tsne
import umap
from scipy.sparse.csgraph import connected_components
import matplotlib.cm as cm
from sklearn.decomposition import PCA

from base import Feature, get_arguments, generate_features, sigmoid, minmaxscale, count_encoding, target_encoding

# import MeCab
import lda
from sklearn.feature_extraction.text import CountVectorizer

import warnings
warnings.filterwarnings('ignore')


SEED = 2019
START_DATE = '2017-11-30'

Feature.dir = 'features'

folds4 = pd.read_feather('./folds/folds4.feather')
folds1 = pd.read_feather('./folds/folds1.feather')


"""
encoding系
"""


class card1_ce(Feature):
    def create_features(self):
        col = 'card1'
        train_card1_ce, test_card1_ce = count_encoding(train_transaction, test_transaction, col)
        self.train[self.__class__.__name__] = train_card1_ce
        self.test[self.__class__.__name__] = test_card1_ce


class card1_te(Feature):
    def create_features(self):
        target = 'isFraud'
        feat = 'card1'
        train_card1_te, test_card1_te = target_encoding(train_transaction, test_transaction, target, feat)
        self.train[self.__class__.__name__] = train_card1_te
        self.test[self.__class__.__name__] = test_card1_te


class card1_te_folds4(Feature):
    def create_features(self):
        target = 'isFraud'
        feat = 'card1'
        train_card1_te, test_card1_te = target_encoding(train_transaction, test_transaction, target, feat, folds4)
        self.train[self.__class__.__name__] = train_card1_te
        self.test[self.__class__.__name__] = test_card1_te


class card2_te(Feature):
    def create_features(self):
        target = 'isFraud'
        feat = 'card2'
        train_card2_te, test_card2_te = target_encoding(train_transaction, test_transaction, target, feat)
        self.train['card2_te'] = train_card2_te
        self.test['card2_te'] = test_card2_te


class card2_te_folds4(Feature):
    def create_features(self):
        target = 'isFraud'
        feat = 'card2'
        train_card1_te, test_card1_te = target_encoding(train_transaction, test_transaction, target, feat, folds4)
        self.train[self.__class__.__name__] = train_card1_te
        self.test[self.__class__.__name__] = test_card1_te


class card3_ce(Feature):
    def create_features(self):
        col = 'card3'
        train_card3_ce, test_card3_ce = count_encoding(train_transaction, test_transaction, col)
        self.train['card3_ce'] = train_card3_ce
        self.test['card3_ce'] = test_card3_ce


class card5_ce(Feature):
    def create_features(self):
        col = 'card5'
        train_card5_ce, test_card5_ce = count_encoding(train_transaction, test_transaction, col)
        self.train[self.__class__.__name__] = train_card5_ce
        self.test[self.__class__.__name__] = test_card5_ce


class card5_te(Feature):
    def create_features(self):
        target = 'isFraud'
        feat = 'card5'
        train_card5_te, test_card5_te = target_encoding(train_transaction, test_transaction, target, feat)
        self.train[self.__class__.__name__] = train_card5_te
        self.test[self.__class__.__name__] = test_card5_te


class card5_te_folds4(Feature):
    def create_features(self):
        target = 'isFraud'
        feat = 'card5'
        train_card1_te, test_card1_te = target_encoding(train_transaction, test_transaction, target, feat, folds4)
        self.train[self.__class__.__name__] = train_card1_te
        self.test[self.__class__.__name__] = test_card1_te


class addr1_ce(Feature):
    def create_features(self):
        col = 'addr1'
        train_addr1_ce, test_addr1_ce = count_encoding(train_transaction, test_transaction, col)
        self.train['addr1_ce'] = train_addr1_ce
        self.test['addr1_ce'] = test_addr1_ce


class id_13_ce(Feature):
    def create_features(self):
        feats = ['TransactionID', 'id_13']
        whole = pd.concat([train_identity[feats], test_identity[feats]], axis=0)
        le = whole['id_13'].value_counts().to_dict()
        self.train[self.__class__.__name__] = pd.merge(train_transaction['TransactionID'], whole, on='TransactionID', how='left')['id_13'].map(le)
        self.test[self.__class__.__name__] = pd.merge(test_transaction['TransactionID'], whole, on='TransactionID', how='left')['id_13'].map(le)


class card1_card5_te_folds4(Feature):
    def create_features(self):
        target = 'isFraud'
        feat = 'card1_card5'
        train_, test_ = train_transaction[[target] + feat.split('_')], test_transaction[feat.split('_')]
        train_[feat] = train_['card1'].astype('str') + '_' + train_['card5'].astype(str)
        test_[feat] = test_['card1'].astype('str') + '_' + test_['card5'].astype(str)
        train_card1_card5_te, test_card1_card5_te = target_encoding(train_, test_, target, feat, folds4)
        self.train[self.__class__.__name__] = train_card1_card5_te
        self.test[self.__class__.__name__] = test_card1_card5_te


class card1_card5_amt_te_folds4(Feature):
    def create_features(self):
        target = 'isFraud'
        feat = 'card1_card5_TransactionAmt'
        train_, test_ = train_transaction[[target] + feat.split('_')], test_transaction[feat.split('_')]
        train_[feat] = train_['card1'].astype('str') + '_' + train_['card5'].astype(str) + '_' + train_['TransactionAmt'].apply(lambda x: int(x * 1000)).astype(str)
        test_[feat] = test_['card1'].astype('str') + '_' + test_['card5'].astype(str) + '_' + test_['TransactionAmt'].apply(lambda x: int(x * 1000)).astype(str)
        train_card1_card5_amt_te, test_card1_card5_amt_te = target_encoding(train_, test_, target, feat, folds4)
        self.train[self.__class__.__name__] = train_card1_card5_amt_te
        self.test[self.__class__.__name__] = test_card1_card5_amt_te


if __name__ == '__main__':
    args = get_arguments()

    train_identity = pd.read_feather('./data/input/train_identity.feather')
    train_transaction = pd.read_feather('./data/input/train_transaction.feather')
    test_identity = pd.read_feather('./data/input/test_identity.feather')
    test_transaction = pd.read_feather('./data/input/test_transaction.feather')

    generate_features(globals(), args.force)
