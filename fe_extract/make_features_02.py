import yaml
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

from base import Feature, get_arguments, generate_features, sigmoid, minmaxscale, target_encoding

# import MeCab
import lda
from sklearn.feature_extraction.text import CountVectorizer


with open('./configs/default.yaml', 'r') as yf:
    config = yaml.load(yf)
    
with open('./configs/email.yaml', 'r') as yf:
    email = yaml.load(yf)

SEED = config['seed']
START_DATE = '2017-12-01'

Feature.dir = 'features'


"""
groupby系の特徴量を作りまくる
"""


class card1_aggs_with_TransactionAmt(Feature):
    def create_features(self):
        feats = ['card1', 'TransactionAmt']
        aggs = {'sum', 'max', 'min', 'mean', 'median', 'nunique'}
        whole = pd.concat([train_transaction[feats], test_transaction[feats]], axis=0)
        df_aggs = whole.groupby(feats[0])[feats[1]].agg(aggs).reset_index()
        df_aggs.columns = feats[:1] + [f'{col}_{feats[1]}_each_{feats[0]}' for col in df_aggs.columns[1:]]
        for col in df_aggs.columns[1:]:
            self.train[col] = pd.merge(train_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]
            self.test[col] = pd.merge(test_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]


class card1_aggs_with_dist1(Feature):
    def create_features(self):
        feats = ['card1', 'dist1']
        aggs = {'sum', 'max', 'min', 'mean', 'median', 'nunique'}
        whole = pd.concat([train_transaction[feats], test_transaction[feats]], axis=0)
        df_aggs = whole.groupby(feats[0])[feats[1]].agg(aggs).reset_index()
        df_aggs.columns = feats[:1] + [f'{col}_{feats[1]}_each_{feats[0]}' for col in df_aggs.columns[1:]]
        for col in df_aggs.columns[1:]:
            self.train[col] = pd.merge(train_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]
            self.test[col] = pd.merge(test_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]


class card1_aggs_with_d1(Feature):
    def create_features(self):
        feats = ['card1', 'D1']
        aggs = {'sum', 'max', 'min', 'mean', 'median', 'nunique'}
        whole = pd.concat([train_transaction[feats], test_transaction[feats]], axis=0)
        df_aggs = whole.groupby(feats[0])[feats[1]].agg(aggs).reset_index()
        df_aggs.columns = feats[:1] + [f'{col}_{feats[1]}_each_{feats[0]}' for col in df_aggs.columns[1:]]
        for col in df_aggs.columns[1:]:
            self.train[col] = pd.merge(train_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]
            self.test[col] = pd.merge(test_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]


class card1_aggs_with_d2(Feature):
    def create_features(self):
        feats = ['card1', 'D2']
        aggs = {'sum', 'max', 'min', 'mean', 'median', 'nunique'}
        whole = pd.concat([train_transaction[feats], test_transaction[feats]], axis=0)
        df_aggs = whole.groupby(feats[0])[feats[1]].agg(aggs).reset_index()
        df_aggs.columns = feats[:1] + [f'{col}_{feats[1]}_each_{feats[0]}' for col in df_aggs.columns[1:]]
        for col in df_aggs.columns[1:]:
            self.train[col] = pd.merge(train_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]
            self.test[col] = pd.merge(test_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]


class card1_aggs_with_d3(Feature):
    def create_features(self):
        feats = ['card1', 'D3']
        aggs = {'sum', 'max', 'min', 'mean', 'median', 'nunique'}
        whole = pd.concat([train_transaction[feats], test_transaction[feats]], axis=0)
        df_aggs = whole.groupby(feats[0])[feats[1]].agg(aggs).reset_index()
        df_aggs.columns = feats[:1] + [f'{col}_{feats[1]}_each_{feats[0]}' for col in df_aggs.columns[1:]]
        for col in df_aggs.columns[1:]:
            self.train[col] = pd.merge(train_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]
            self.test[col] = pd.merge(test_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]


class card1_aggs_with_d4(Feature):
    def create_features(self):
        feats = ['card1', 'D4']
        aggs = {'sum', 'max', 'min', 'mean', 'median', 'nunique'}
        whole = pd.concat([train_transaction[feats], test_transaction[feats]], axis=0)
        df_aggs = whole.groupby(feats[0])[feats[1]].agg(aggs).reset_index()
        df_aggs.columns = feats[:1] + [f'{col}_{feats[1]}_each_{feats[0]}' for col in df_aggs.columns[1:]]
        for col in df_aggs.columns[1:]:
            self.train[col] = pd.merge(train_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]
            self.test[col] = pd.merge(test_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]


class card1_aggs_with_d5(Feature):
    def create_features(self):
        feats = ['card1', 'D5']
        aggs = {'sum', 'max', 'min', 'mean', 'median', 'nunique'}
        whole = pd.concat([train_transaction[feats], test_transaction[feats]], axis=0)
        df_aggs = whole.groupby(feats[0])[feats[1]].agg(aggs).reset_index()
        df_aggs.columns = feats[:1] + [f'{col}_{feats[1]}_each_{feats[0]}' for col in df_aggs.columns[1:]]
        for col in df_aggs.columns[1:]:
            self.train[col] = pd.merge(train_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]
            self.test[col] = pd.merge(test_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]


class card1_aggs_with_d6(Feature):
    def create_features(self):
        feats = ['card1', 'D6']
        aggs = {'sum', 'max', 'min', 'mean', 'median', 'nunique'}
        whole = pd.concat([train_transaction[feats], test_transaction[feats]], axis=0)
        df_aggs = whole.groupby(feats[0])[feats[1]].agg(aggs).reset_index()
        df_aggs.columns = feats[:1] + [f'{col}_{feats[1]}_each_{feats[0]}' for col in df_aggs.columns[1:]]
        for col in df_aggs.columns[1:]:
            self.train[col] = pd.merge(train_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]
            self.test[col] = pd.merge(test_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]


class card1_aggs_with_d7(Feature):
    def create_features(self):
        feats = ['card1', 'D7']
        aggs = {'sum', 'max', 'min', 'mean', 'median', 'nunique'}
        whole = pd.concat([train_transaction[feats], test_transaction[feats]], axis=0)
        df_aggs = whole.groupby(feats[0])[feats[1]].agg(aggs).reset_index()
        df_aggs.columns = feats[:1] + [f'{col}_{feats[1]}_each_{feats[0]}' for col in df_aggs.columns[1:]]
        for col in df_aggs.columns[1:]:
            self.train[col] = pd.merge(train_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]
            self.test[col] = pd.merge(test_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]


class card1_aggs_with_d8(Feature):
    def create_features(self):
        feats = ['card1', 'D8']
        aggs = {'sum', 'max', 'min', 'mean', 'median', 'nunique'}
        whole = pd.concat([train_transaction[feats], test_transaction[feats]], axis=0)
        df_aggs = whole.groupby(feats[0])[feats[1]].agg(aggs).reset_index()
        df_aggs.columns = feats[:1] + [f'{col}_{feats[1]}_each_{feats[0]}' for col in df_aggs.columns[1:]]
        for col in df_aggs.columns[1:]:
            self.train[col] = pd.merge(train_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]
            self.test[col] = pd.merge(test_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]


class card1_aggs_with_d9(Feature):
    def create_features(self):
        feats = ['card1', 'D9']
        aggs = {'sum', 'max', 'min', 'mean', 'median', 'nunique'}
        whole = pd.concat([train_transaction[feats], test_transaction[feats]], axis=0)
        df_aggs = whole.groupby(feats[0])[feats[1]].agg(aggs).reset_index()
        df_aggs.columns = feats[:1] + [f'{col}_{feats[1]}_each_{feats[0]}' for col in df_aggs.columns[1:]]
        for col in df_aggs.columns[1:]:
            self.train[col] = pd.merge(train_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]
            self.test[col] = pd.merge(test_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]


class card1_aggs_with_d10(Feature):
    def create_features(self):
        feats = ['card1', 'D10']
        aggs = {'sum', 'max', 'min', 'mean', 'median', 'nunique'}
        whole = pd.concat([train_transaction[feats], test_transaction[feats]], axis=0)
        df_aggs = whole.groupby(feats[0])[feats[1]].agg(aggs).reset_index()
        df_aggs.columns = feats[:1] + [f'{col}_{feats[1]}_each_{feats[0]}' for col in df_aggs.columns[1:]]
        for col in df_aggs.columns[1:]:
            self.train[col] = pd.merge(train_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]
            self.test[col] = pd.merge(test_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]


class card1_aggs_with_d11(Feature):
    def create_features(self):
        feats = ['card1', 'D11']
        aggs = {'sum', 'max', 'min', 'mean', 'median', 'nunique'}
        whole = pd.concat([train_transaction[feats], test_transaction[feats]], axis=0)
        df_aggs = whole.groupby(feats[0])[feats[1]].agg(aggs).reset_index()
        df_aggs.columns = feats[:1] + [f'{col}_{feats[1]}_each_{feats[0]}' for col in df_aggs.columns[1:]]
        for col in df_aggs.columns[1:]:
            self.train[col] = pd.merge(train_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]
            self.test[col] = pd.merge(test_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]


class card1_aggs_with_d12(Feature):
    def create_features(self):
        feats = ['card1', 'D12']
        aggs = {'sum', 'max', 'min', 'mean', 'median', 'nunique'}
        whole = pd.concat([train_transaction[feats], test_transaction[feats]], axis=0)
        df_aggs = whole.groupby(feats[0])[feats[1]].agg(aggs).reset_index()
        df_aggs.columns = feats[:1] + [f'{col}_{feats[1]}_each_{feats[0]}' for col in df_aggs.columns[1:]]
        for col in df_aggs.columns[1:]:
            self.train[col] = pd.merge(train_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]
            self.test[col] = pd.merge(test_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]


class card1_aggs_with_d13(Feature):
    def create_features(self):
        feats = ['card1', 'D13']
        aggs = {'sum', 'max', 'min', 'mean', 'median', 'nunique'}
        whole = pd.concat([train_transaction[feats], test_transaction[feats]], axis=0)
        df_aggs = whole.groupby(feats[0])[feats[1]].agg(aggs).reset_index()
        df_aggs.columns = feats[:1] + [f'{col}_{feats[1]}_each_{feats[0]}' for col in df_aggs.columns[1:]]
        for col in df_aggs.columns[1:]:
            self.train[col] = pd.merge(train_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]
            self.test[col] = pd.merge(test_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]


class card1_aggs_with_d14(Feature):
    def create_features(self):
        feats = ['card1', 'D14']
        aggs = {'sum', 'max', 'min', 'mean', 'median', 'nunique'}
        whole = pd.concat([train_transaction[feats], test_transaction[feats]], axis=0)
        df_aggs = whole.groupby(feats[0])[feats[1]].agg(aggs).reset_index()
        df_aggs.columns = feats[:1] + [f'{col}_{feats[1]}_each_{feats[0]}' for col in df_aggs.columns[1:]]
        for col in df_aggs.columns[1:]:
            self.train[col] = pd.merge(train_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]
            self.test[col] = pd.merge(test_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]


class card1_aggs_with_d15(Feature):
    def create_features(self):
        feats = ['card1', 'D15']
        aggs = {'sum', 'max', 'min', 'mean', 'median', 'nunique'}
        whole = pd.concat([train_transaction[feats], test_transaction[feats]], axis=0)
        df_aggs = whole.groupby(feats[0])[feats[1]].agg(aggs).reset_index()
        df_aggs.columns = feats[:1] + [f'{col}_{feats[1]}_each_{feats[0]}' for col in df_aggs.columns[1:]]
        for col in df_aggs.columns[1:]:
            self.train[col] = pd.merge(train_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]
            self.test[col] = pd.merge(test_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]


class card1_aggs_with_v96(Feature):
    def create_features(self):
        feats = ['card1', 'V96']
        aggs = {'sum', 'max', 'min', 'mean', 'median', 'nunique'}
        whole = pd.concat([train_transaction[feats], test_transaction[feats]], axis=0)
        df_aggs = whole.groupby(feats[0])[feats[1]].agg(aggs).reset_index()
        df_aggs.columns = feats[:1] + [f'{col}_{feats[1]}_each_{feats[0]}' for col in df_aggs.columns[1:]]
        for col in df_aggs.columns[1:]:
            self.train[col] = pd.merge(train_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]
            self.test[col] = pd.merge(test_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]


class card1_aggs_with_v126(Feature):
    def create_features(self):
        feats = ['card1', 'V126']
        aggs = {'sum', 'max', 'min', 'mean', 'median', 'nunique'}
        whole = pd.concat([train_transaction[feats], test_transaction[feats]], axis=0)
        df_aggs = whole.groupby(feats[0])[feats[1]].agg(aggs).reset_index()
        df_aggs.columns = feats[:1] + [f'{col}_{feats[1]}_each_{feats[0]}' for col in df_aggs.columns[1:]]
        for col in df_aggs.columns[1:]:
            self.train[col] = pd.merge(train_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]
            self.test[col] = pd.merge(test_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]


class card1_aggs_with_v127(Feature):
    def create_features(self):
        feats = ['card1', 'V127']
        aggs = {'sum', 'max', 'min', 'mean', 'median', 'nunique'}
        whole = pd.concat([train_transaction[feats], test_transaction[feats]], axis=0)
        df_aggs = whole.groupby(feats[0])[feats[1]].agg(aggs).reset_index()
        df_aggs.columns = feats[:1] + [f'{col}_{feats[1]}_each_{feats[0]}' for col in df_aggs.columns[1:]]
        for col in df_aggs.columns[1:]:
            self.train[col] = pd.merge(train_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]
            self.test[col] = pd.merge(test_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]


class card1_aggs_with_v128(Feature):
    def create_features(self):
        feats = ['card1', 'V128']
        aggs = {'sum', 'max', 'min', 'mean', 'median', 'nunique'}
        whole = pd.concat([train_transaction[feats], test_transaction[feats]], axis=0)
        df_aggs = whole.groupby(feats[0])[feats[1]].agg(aggs).reset_index()
        df_aggs.columns = feats[:1] + [f'{col}_{feats[1]}_each_{feats[0]}' for col in df_aggs.columns[1:]]
        for col in df_aggs.columns[1:]:
            self.train[col] = pd.merge(train_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]
            self.test[col] = pd.merge(test_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]



class card1_aggs_with_v129(Feature):
    def create_features(self):
        feats = ['card1', 'V129']
        aggs = {'sum', 'max', 'min', 'mean', 'median', 'nunique'}
        whole = pd.concat([train_transaction[feats], test_transaction[feats]], axis=0)
        df_aggs = whole.groupby(feats[0])[feats[1]].agg(aggs).reset_index()
        df_aggs.columns = feats[:1] + [f'{col}_{feats[1]}_each_{feats[0]}' for col in df_aggs.columns[1:]]
        for col in df_aggs.columns[1:]:
            self.train[col] = pd.merge(train_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]
            self.test[col] = pd.merge(test_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]


class card1_aggs_with_v130(Feature):
    def create_features(self):
        feats = ['card1', 'V130']
        aggs = {'sum', 'max', 'min', 'mean', 'median', 'nunique'}
        whole = pd.concat([train_transaction[feats], test_transaction[feats]], axis=0)
        df_aggs = whole.groupby(feats[0])[feats[1]].agg(aggs).reset_index()
        df_aggs.columns = feats[:1] + [f'{col}_{feats[1]}_each_{feats[0]}' for col in df_aggs.columns[1:]]
        for col in df_aggs.columns[1:]:
            self.train[col] = pd.merge(train_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]
            self.test[col] = pd.merge(test_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]


class card1_aggs_with_v131(Feature):
    def create_features(self):
        feats = ['card1', 'V131']
        aggs = {'sum', 'max', 'min', 'mean', 'median', 'nunique'}
        whole = pd.concat([train_transaction[feats], test_transaction[feats]], axis=0)
        df_aggs = whole.groupby(feats[0])[feats[1]].agg(aggs).reset_index()
        df_aggs.columns = feats[:1] + [f'{col}_{feats[1]}_each_{feats[0]}' for col in df_aggs.columns[1:]]
        for col in df_aggs.columns[1:]:
            self.train[col] = pd.merge(train_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]
            self.test[col] = pd.merge(test_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]


class card1_aggs_with_v132(Feature):
    def create_features(self):
        feats = ['card1', 'V132']
        aggs = {'sum', 'max', 'min', 'mean', 'median', 'nunique'}
        whole = pd.concat([train_transaction[feats], test_transaction[feats]], axis=0)
        df_aggs = whole.groupby(feats[0])[feats[1]].agg(aggs).reset_index()
        df_aggs.columns = feats[:1] + [f'{col}_{feats[1]}_each_{feats[0]}' for col in df_aggs.columns[1:]]
        for col in df_aggs.columns[1:]:
            self.train[col] = pd.merge(train_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]
            self.test[col] = pd.merge(test_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]


class card1_aggs_with_v133(Feature):
    def create_features(self):
        feats = ['card1', 'V133']
        aggs = {'sum', 'max', 'min', 'mean', 'median', 'nunique'}
        whole = pd.concat([train_transaction[feats], test_transaction[feats]], axis=0)
        df_aggs = whole.groupby(feats[0])[feats[1]].agg(aggs).reset_index()
        df_aggs.columns = feats[:1] + [f'{col}_{feats[1]}_each_{feats[0]}' for col in df_aggs.columns[1:]]
        for col in df_aggs.columns[1:]:
            self.train[col] = pd.merge(train_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]
            self.test[col] = pd.merge(test_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]


class card1_aggs_with_v134(Feature):
    def create_features(self):
        feats = ['card1', 'V134']
        aggs = {'sum', 'max', 'min', 'mean', 'median', 'nunique'}
        whole = pd.concat([train_transaction[feats], test_transaction[feats]], axis=0)
        df_aggs = whole.groupby(feats[0])[feats[1]].agg(aggs).reset_index()
        df_aggs.columns = feats[:1] + [f'{col}_{feats[1]}_each_{feats[0]}' for col in df_aggs.columns[1:]]
        for col in df_aggs.columns[1:]:
            self.train[col] = pd.merge(train_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]
            self.test[col] = pd.merge(test_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]


class card1_aggs_with_v135(Feature):
    def create_features(self):
        feats = ['card1', 'V135']
        aggs = {'sum', 'max', 'min', 'mean', 'median', 'nunique'}
        whole = pd.concat([train_transaction[feats], test_transaction[feats]], axis=0)
        df_aggs = whole.groupby(feats[0])[feats[1]].agg(aggs).reset_index()
        df_aggs.columns = feats[:1] + [f'{col}_{feats[1]}_each_{feats[0]}' for col in df_aggs.columns[1:]]
        for col in df_aggs.columns[1:]:
            self.train[col] = pd.merge(train_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]
            self.test[col] = pd.merge(test_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]


class card1_aggs_with_v136(Feature):
    def create_features(self):
        feats = ['card1', 'V136']
        aggs = {'sum', 'max', 'min', 'mean', 'median', 'nunique'}
        whole = pd.concat([train_transaction[feats], test_transaction[feats]], axis=0)
        df_aggs = whole.groupby(feats[0])[feats[1]].agg(aggs).reset_index()
        df_aggs.columns = feats[:1] + [f'{col}_{feats[1]}_each_{feats[0]}' for col in df_aggs.columns[1:]]
        for col in df_aggs.columns[1:]:
            self.train[col] = pd.merge(train_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]
            self.test[col] = pd.merge(test_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]


class card1_aggs_with_v137(Feature):
    def create_features(self):
        feats = ['card1', 'V137']
        aggs = {'sum', 'max', 'min', 'mean', 'median', 'nunique'}
        whole = pd.concat([train_transaction[feats], test_transaction[feats]], axis=0)
        df_aggs = whole.groupby(feats[0])[feats[1]].agg(aggs).reset_index()
        df_aggs.columns = feats[:1] + [f'{col}_{feats[1]}_each_{feats[0]}' for col in df_aggs.columns[1:]]
        for col in df_aggs.columns[1:]:
            self.train[col] = pd.merge(train_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]
            self.test[col] = pd.merge(test_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]


class card1_aggs_with_v150(Feature):
    def create_features(self):
        feats = ['card1', 'V150']
        aggs = {'sum', 'max', 'min', 'mean', 'median', 'nunique'}
        whole = pd.concat([train_transaction[feats], test_transaction[feats]], axis=0)
        df_aggs = whole.groupby(feats[0])[feats[1]].agg(aggs).reset_index()
        df_aggs.columns = feats[:1] + [f'{col}_{feats[1]}_each_{feats[0]}' for col in df_aggs.columns[1:]]
        for col in df_aggs.columns[1:]:
            self.train[col] = pd.merge(train_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]
            self.test[col] = pd.merge(test_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]


class card1_aggs_with_v159(Feature):
    def create_features(self):
        feats = ['card1', 'V159']
        aggs = {'sum', 'max', 'min', 'mean', 'median', 'nunique'}
        whole = pd.concat([train_transaction[feats], test_transaction[feats]], axis=0)
        df_aggs = whole.groupby(feats[0])[feats[1]].agg(aggs).reset_index()
        df_aggs.columns = feats[:1] + [f'{col}_{feats[1]}_each_{feats[0]}' for col in df_aggs.columns[1:]]
        for col in df_aggs.columns[1:]:
            self.train[col] = pd.merge(train_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]
            self.test[col] = pd.merge(test_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]


class card1_aggs_with_v160(Feature):
    def create_features(self):
        feats = ['card1', 'V160']
        aggs = {'sum', 'max', 'min', 'mean', 'median', 'nunique'}
        whole = pd.concat([train_transaction[feats], test_transaction[feats]], axis=0)
        df_aggs = whole.groupby(feats[0])[feats[1]].agg(aggs).reset_index()
        df_aggs.columns = feats[:1] + [f'{col}_{feats[1]}_each_{feats[0]}' for col in df_aggs.columns[1:]]
        for col in df_aggs.columns[1:]:
            self.train[col] = pd.merge(train_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]
            self.test[col] = pd.merge(test_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]


class card1_aggs_with_v166(Feature):
    def create_features(self):
        feats = ['card1', 'V166']
        aggs = {'sum', 'max', 'min', 'mean', 'median', 'nunique'}
        whole = pd.concat([train_transaction[feats], test_transaction[feats]], axis=0)
        df_aggs = whole.groupby(feats[0])[feats[1]].agg(aggs).reset_index()
        df_aggs.columns = feats[:1] + [f'{col}_{feats[1]}_each_{feats[0]}' for col in df_aggs.columns[1:]]
        for col in df_aggs.columns[1:]:
            self.train[col] = pd.merge(train_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]
            self.test[col] = pd.merge(test_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]


class card1_aggs_with_v202(Feature):
    def create_features(self):
        feats = ['card1', 'V202']
        aggs = {'sum', 'max', 'min', 'mean', 'median', 'nunique'}
        whole = pd.concat([train_transaction[feats], test_transaction[feats]], axis=0)
        df_aggs = whole.groupby(feats[0])[feats[1]].agg(aggs).reset_index()
        df_aggs.columns = feats[:1] + [f'{col}_{feats[1]}_each_{feats[0]}' for col in df_aggs.columns[1:]]
        for col in df_aggs.columns[1:]:
            self.train[col] = pd.merge(train_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]
            self.test[col] = pd.merge(test_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]


class card1_aggs_with_v203(Feature):
    def create_features(self):
        feats = ['card1', 'V203']
        aggs = {'sum', 'max', 'min', 'mean', 'median', 'nunique'}
        whole = pd.concat([train_transaction[feats], test_transaction[feats]], axis=0)
        df_aggs = whole.groupby(feats[0])[feats[1]].agg(aggs).reset_index()
        df_aggs.columns = feats[:1] + [f'{col}_{feats[1]}_each_{feats[0]}' for col in df_aggs.columns[1:]]
        for col in df_aggs.columns[1:]:
            self.train[col] = pd.merge(train_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]
            self.test[col] = pd.merge(test_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]


class card1_aggs_with_v204(Feature):
    def create_features(self):
        feats = ['card1', 'V204']
        aggs = {'sum', 'max', 'min', 'mean', 'median', 'nunique'}
        whole = pd.concat([train_transaction[feats], test_transaction[feats]], axis=0)
        df_aggs = whole.groupby(feats[0])[feats[1]].agg(aggs).reset_index()
        df_aggs.columns = feats[:1] + [f'{col}_{feats[1]}_each_{feats[0]}' for col in df_aggs.columns[1:]]
        for col in df_aggs.columns[1:]:
            self.train[col] = pd.merge(train_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]
            self.test[col] = pd.merge(test_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]


class card1_aggs_with_v207(Feature):
    def create_features(self):
        feats = ['card1', 'V207']
        aggs = {'sum', 'max', 'min', 'mean', 'median', 'nunique'}
        whole = pd.concat([train_transaction[feats], test_transaction[feats]], axis=0)
        df_aggs = whole.groupby(feats[0])[feats[1]].agg(aggs).reset_index()
        df_aggs.columns = feats[:1] + [f'{col}_{feats[1]}_each_{feats[0]}' for col in df_aggs.columns[1:]]
        for col in df_aggs.columns[1:]:
            self.train[col] = pd.merge(train_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]
            self.test[col] = pd.merge(test_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]


class card1_aggs_with_v208(Feature):
    def create_features(self):
        feats = ['card1', 'V208']
        aggs = {'sum', 'max', 'min', 'mean', 'median', 'nunique'}
        whole = pd.concat([train_transaction[feats], test_transaction[feats]], axis=0)
        df_aggs = whole.groupby(feats[0])[feats[1]].agg(aggs).reset_index()
        df_aggs.columns = feats[:1] + [f'{col}_{feats[1]}_each_{feats[0]}' for col in df_aggs.columns[1:]]
        for col in df_aggs.columns[1:]:
            self.train[col] = pd.merge(train_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]
            self.test[col] = pd.merge(test_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]


class card1_aggs_with_v209(Feature):
    def create_features(self):
        feats = ['card1', 'V209']
        aggs = {'sum', 'max', 'min', 'mean', 'median', 'nunique'}
        whole = pd.concat([train_transaction[feats], test_transaction[feats]], axis=0)
        df_aggs = whole.groupby(feats[0])[feats[1]].agg(aggs).reset_index()
        df_aggs.columns = feats[:1] + [f'{col}_{feats[1]}_each_{feats[0]}' for col in df_aggs.columns[1:]]
        for col in df_aggs.columns[1:]:
            self.train[col] = pd.merge(train_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]
            self.test[col] = pd.merge(test_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]


class card1_aggs_with_v210(Feature):
    def create_features(self):
        feats = ['card1', 'V210']
        aggs = {'sum', 'max', 'min', 'mean', 'median', 'nunique'}
        whole = pd.concat([train_transaction[feats], test_transaction[feats]], axis=0)
        df_aggs = whole.groupby(feats[0])[feats[1]].agg(aggs).reset_index()
        df_aggs.columns = feats[:1] + [f'{col}_{feats[1]}_each_{feats[0]}' for col in df_aggs.columns[1:]]
        for col in df_aggs.columns[1:]:
            self.train[col] = pd.merge(train_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]
            self.test[col] = pd.merge(test_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]


class card1_aggs_with_v212(Feature):
    def create_features(self):
        feats = ['card1', 'V212']
        aggs = {'sum', 'max', 'min', 'mean', 'median', 'nunique'}
        whole = pd.concat([train_transaction[feats], test_transaction[feats]], axis=0)
        df_aggs = whole.groupby(feats[0])[feats[1]].agg(aggs).reset_index()
        df_aggs.columns = feats[:1] + [f'{col}_{feats[1]}_each_{feats[0]}' for col in df_aggs.columns[1:]]
        for col in df_aggs.columns[1:]:
            self.train[col] = pd.merge(train_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]
            self.test[col] = pd.merge(test_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]

class card1_aggs_with_v214(Feature):
    def create_features(self):
        feats = ['card1', 'V214']
        aggs = {'sum', 'max', 'min', 'mean', 'median', 'nunique'}
        whole = pd.concat([train_transaction[feats], test_transaction[feats]], axis=0)
        df_aggs = whole.groupby(feats[0])[feats[1]].agg(aggs).reset_index()
        df_aggs.columns = feats[:1] + [f'{col}_{feats[1]}_each_{feats[0]}' for col in df_aggs.columns[1:]]
        for col in df_aggs.columns[1:]:
            self.train[col] = pd.merge(train_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]
            self.test[col] = pd.merge(test_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]


class card1_aggs_with_v215(Feature):
    def create_features(self):
        feats = ['card1', 'V215']
        aggs = {'sum', 'max', 'min', 'mean', 'median', 'nunique'}
        whole = pd.concat([train_transaction[feats], test_transaction[feats]], axis=0)
        df_aggs = whole.groupby(feats[0])[feats[1]].agg(aggs).reset_index()
        df_aggs.columns = feats[:1] + [f'{col}_{feats[1]}_each_{feats[0]}' for col in df_aggs.columns[1:]]
        for col in df_aggs.columns[1:]:
            self.train[col] = pd.merge(train_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]
            self.test[col] = pd.merge(test_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]


class card1_aggs_with_v216(Feature):
    def create_features(self):
        feats = ['card1', 'V216']
        aggs = {'sum', 'max', 'min', 'mean', 'median', 'nunique'}
        whole = pd.concat([train_transaction[feats], test_transaction[feats]], axis=0)
        df_aggs = whole.groupby(feats[0])[feats[1]].agg(aggs).reset_index()
        df_aggs.columns = feats[:1] + [f'{col}_{feats[1]}_each_{feats[0]}' for col in df_aggs.columns[1:]]
        for col in df_aggs.columns[1:]:
            self.train[col] = pd.merge(train_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]
            self.test[col] = pd.merge(test_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]


class card1_aggs_with_v263(Feature):
    def create_features(self):
        feats = ['card1', 'V263']
        aggs = {'sum', 'max', 'min', 'mean', 'median', 'nunique'}
        whole = pd.concat([train_transaction[feats], test_transaction[feats]], axis=0)
        df_aggs = whole.groupby(feats[0])[feats[1]].agg(aggs).reset_index()
        df_aggs.columns = feats[:1] + [f'{col}_{feats[1]}_each_{feats[0]}' for col in df_aggs.columns[1:]]
        for col in df_aggs.columns[1:]:
            self.train[col] = pd.merge(train_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]
            self.test[col] = pd.merge(test_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]


class card1_aggs_with_v264(Feature):
    def create_features(self):
        feats = ['card1', 'V264']
        aggs = {'sum', 'max', 'min', 'mean', 'median', 'nunique'}
        whole = pd.concat([train_transaction[feats], test_transaction[feats]], axis=0)
        df_aggs = whole.groupby(feats[0])[feats[1]].agg(aggs).reset_index()
        df_aggs.columns = feats[:1] + [f'{col}_{feats[1]}_each_{feats[0]}' for col in df_aggs.columns[1:]]
        for col in df_aggs.columns[1:]:
            self.train[col] = pd.merge(train_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]
            self.test[col] = pd.merge(test_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]


class card1_aggs_with_v265(Feature):
    def create_features(self):
        feats = ['card1', 'V265']
        aggs = {'sum', 'max', 'min', 'mean', 'median', 'nunique'}
        whole = pd.concat([train_transaction[feats], test_transaction[feats]], axis=0)
        df_aggs = whole.groupby(feats[0])[feats[1]].agg(aggs).reset_index()
        df_aggs.columns = feats[:1] + [f'{col}_{feats[1]}_each_{feats[0]}' for col in df_aggs.columns[1:]]
        for col in df_aggs.columns[1:]:
            self.train[col] = pd.merge(train_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]
            self.test[col] = pd.merge(test_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]


class card1_aggs_with_v267(Feature):
    def create_features(self):
        feats = ['card1', 'V267']
        aggs = {'sum', 'max', 'min', 'mean', 'median', 'nunique'}
        whole = pd.concat([train_transaction[feats], test_transaction[feats]], axis=0)
        df_aggs = whole.groupby(feats[0])[feats[1]].agg(aggs).reset_index()
        df_aggs.columns = feats[:1] + [f'{col}_{feats[1]}_each_{feats[0]}' for col in df_aggs.columns[1:]]
        for col in df_aggs.columns[1:]:
            self.train[col] = pd.merge(train_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]
            self.test[col] = pd.merge(test_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]


class card1_aggs_with_v274(Feature):
    def create_features(self):
        feats = ['card1', 'V274']
        aggs = {'sum', 'max', 'min', 'mean', 'median', 'nunique'}
        whole = pd.concat([train_transaction[feats], test_transaction[feats]], axis=0)
        df_aggs = whole.groupby(feats[0])[feats[1]].agg(aggs).reset_index()
        df_aggs.columns = feats[:1] + [f'{col}_{feats[1]}_each_{feats[0]}' for col in df_aggs.columns[1:]]
        for col in df_aggs.columns[1:]:
            self.train[col] = pd.merge(train_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]
            self.test[col] = pd.merge(test_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]


class card1_aggs_with_v275(Feature):
    def create_features(self):
        feats = ['card1', 'V275']
        aggs = {'sum', 'max', 'min', 'mean', 'median', 'nunique'}
        whole = pd.concat([train_transaction[feats], test_transaction[feats]], axis=0)
        df_aggs = whole.groupby(feats[0])[feats[1]].agg(aggs).reset_index()
        df_aggs.columns = feats[:1] + [f'{col}_{feats[1]}_each_{feats[0]}' for col in df_aggs.columns[1:]]
        for col in df_aggs.columns[1:]:
            self.train[col] = pd.merge(train_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]
            self.test[col] = pd.merge(test_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]


class card1_aggs_with_v276(Feature):
    def create_features(self):
        feats = ['card1', 'V276']
        aggs = {'sum', 'max', 'min', 'mean', 'median', 'nunique'}
        whole = pd.concat([train_transaction[feats], test_transaction[feats]], axis=0)
        df_aggs = whole.groupby(feats[0])[feats[1]].agg(aggs).reset_index()
        df_aggs.columns = feats[:1] + [f'{col}_{feats[1]}_each_{feats[0]}' for col in df_aggs.columns[1:]]
        for col in df_aggs.columns[1:]:
            self.train[col] = pd.merge(train_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]
            self.test[col] = pd.merge(test_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]


class card1_aggs_with_v277(Feature):
    def create_features(self):
        feats = ['card1', 'V277']
        aggs = {'sum', 'max', 'min', 'mean', 'median', 'nunique'}
        whole = pd.concat([train_transaction[feats], test_transaction[feats]], axis=0)
        df_aggs = whole.groupby(feats[0])[feats[1]].agg(aggs).reset_index()
        df_aggs.columns = feats[:1] + [f'{col}_{feats[1]}_each_{feats[0]}' for col in df_aggs.columns[1:]]
        for col in df_aggs.columns[1:]:
            self.train[col] = pd.merge(train_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]
            self.test[col] = pd.merge(test_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]


class card1_aggs_with_v278(Feature):
    def create_features(self):
        feats = ['card1', 'V278']
        aggs = {'sum', 'max', 'min', 'mean', 'median', 'nunique'}
        whole = pd.concat([train_transaction[feats], test_transaction[feats]], axis=0)
        df_aggs = whole.groupby(feats[0])[feats[1]].agg(aggs).reset_index()
        df_aggs.columns = feats[:1] + [f'{col}_{feats[1]}_each_{feats[0]}' for col in df_aggs.columns[1:]]
        for col in df_aggs.columns[1:]:
            self.train[col] = pd.merge(train_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]
            self.test[col] = pd.merge(test_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]


class card1_aggs_with_v280(Feature):
    def create_features(self):
        feats = ['card1', 'V280']
        aggs = {'sum', 'max', 'min', 'mean', 'median', 'nunique'}
        whole = pd.concat([train_transaction[feats], test_transaction[feats]], axis=0)
        df_aggs = whole.groupby(feats[0])[feats[1]].agg(aggs).reset_index()
        df_aggs.columns = feats[:1] + [f'{col}_{feats[1]}_each_{feats[0]}' for col in df_aggs.columns[1:]]
        for col in df_aggs.columns[1:]:
            self.train[col] = pd.merge(train_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]
            self.test[col] = pd.merge(test_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]


class card1_aggs_with_v294(Feature):
    def create_features(self):
        feats = ['card1', 'V294']
        aggs = {'sum', 'max', 'min', 'mean', 'median', 'nunique'}
        whole = pd.concat([train_transaction[feats], test_transaction[feats]], axis=0)
        df_aggs = whole.groupby(feats[0])[feats[1]].agg(aggs).reset_index()
        df_aggs.columns = feats[:1] + [f'{col}_{feats[1]}_each_{feats[0]}' for col in df_aggs.columns[1:]]
        for col in df_aggs.columns[1:]:
            self.train[col] = pd.merge(train_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]
            self.test[col] = pd.merge(test_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]


class card1_aggs_with_v306(Feature):
    def create_features(self):
        feats = ['card1', 'V306']
        aggs = {'sum', 'max', 'min', 'mean', 'median', 'nunique'}
        whole = pd.concat([train_transaction[feats], test_transaction[feats]], axis=0)
        df_aggs = whole.groupby(feats[0])[feats[1]].agg(aggs).reset_index()
        df_aggs.columns = feats[:1] + [f'{col}_{feats[1]}_each_{feats[0]}' for col in df_aggs.columns[1:]]
        for col in df_aggs.columns[1:]:
            self.train[col] = pd.merge(train_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]
            self.test[col] = pd.merge(test_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]


class card1_aggs_with_v307(Feature):
    def create_features(self):
        feats = ['card1', 'V307']
        aggs = {'sum', 'max', 'min', 'mean', 'median', 'nunique'}
        whole = pd.concat([train_transaction[feats], test_transaction[feats]], axis=0)
        df_aggs = whole.groupby(feats[0])[feats[1]].agg(aggs).reset_index()
        df_aggs.columns = feats[:1] + [f'{col}_{feats[1]}_each_{feats[0]}' for col in df_aggs.columns[1:]]
        for col in df_aggs.columns[1:]:
            self.train[col] = pd.merge(train_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]
            self.test[col] = pd.merge(test_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]


class card1_aggs_with_v308(Feature):
    def create_features(self):
        feats = ['card1', 'V308']
        aggs = {'sum', 'max', 'min', 'mean', 'median', 'nunique'}
        whole = pd.concat([train_transaction[feats], test_transaction[feats]], axis=0)
        df_aggs = whole.groupby(feats[0])[feats[1]].agg(aggs).reset_index()
        df_aggs.columns = feats[:1] + [f'{col}_{feats[1]}_each_{feats[0]}' for col in df_aggs.columns[1:]]
        for col in df_aggs.columns[1:]:
            self.train[col] = pd.merge(train_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]
            self.test[col] = pd.merge(test_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]


class card1_aggs_with_v309(Feature):
    def create_features(self):
        feats = ['card1', 'V309']
        aggs = {'sum', 'max', 'min', 'mean', 'median', 'nunique'}
        whole = pd.concat([train_transaction[feats], test_transaction[feats]], axis=0)
        df_aggs = whole.groupby(feats[0])[feats[1]].agg(aggs).reset_index()
        df_aggs.columns = feats[:1] + [f'{col}_{feats[1]}_each_{feats[0]}' for col in df_aggs.columns[1:]]
        for col in df_aggs.columns[1:]:
            self.train[col] = pd.merge(train_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]
            self.test[col] = pd.merge(test_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]

class card1_aggs_with_v310(Feature):
    def create_features(self):
        feats = ['card1', 'V310']
        aggs = {'sum', 'max', 'min', 'mean', 'median', 'nunique'}
        whole = pd.concat([train_transaction[feats], test_transaction[feats]], axis=0)
        df_aggs = whole.groupby(feats[0])[feats[1]].agg(aggs).reset_index()
        df_aggs.columns = feats[:1] + [f'{col}_{feats[1]}_each_{feats[0]}' for col in df_aggs.columns[1:]]
        for col in df_aggs.columns[1:]:
            self.train[col] = pd.merge(train_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]
            self.test[col] = pd.merge(test_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]


class card1_aggs_with_v311(Feature):
    def create_features(self):
        feats = ['card1', 'V311']
        aggs = {'sum', 'max', 'min', 'mean', 'median', 'nunique'}
        whole = pd.concat([train_transaction[feats], test_transaction[feats]], axis=0)
        df_aggs = whole.groupby(feats[0])[feats[1]].agg(aggs).reset_index()
        df_aggs.columns = feats[:1] + [f'{col}_{feats[1]}_each_{feats[0]}' for col in df_aggs.columns[1:]]
        for col in df_aggs.columns[1:]:
            self.train[col] = pd.merge(train_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]
            self.test[col] = pd.merge(test_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]


class card1_aggs_with_v312(Feature):
    def create_features(self):
        feats = ['card1', 'V312']
        aggs = {'sum', 'max', 'min', 'mean', 'median', 'nunique'}
        whole = pd.concat([train_transaction[feats], test_transaction[feats]], axis=0)
        df_aggs = whole.groupby(feats[0])[feats[1]].agg(aggs).reset_index()
        df_aggs.columns = feats[:1] + [f'{col}_{feats[1]}_each_{feats[0]}' for col in df_aggs.columns[1:]]
        for col in df_aggs.columns[1:]:
            self.train[col] = pd.merge(train_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]
            self.test[col] = pd.merge(test_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]


class card1_aggs_with_v313(Feature):
    def create_features(self):
        feats = ['card1', 'V313']
        aggs = {'sum', 'max', 'min', 'mean', 'median', 'nunique'}
        whole = pd.concat([train_transaction[feats], test_transaction[feats]], axis=0)
        df_aggs = whole.groupby(feats[0])[feats[1]].agg(aggs).reset_index()
        df_aggs.columns = feats[:1] + [f'{col}_{feats[1]}_each_{feats[0]}' for col in df_aggs.columns[1:]]
        for col in df_aggs.columns[1:]:
            self.train[col] = pd.merge(train_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]
            self.test[col] = pd.merge(test_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]


class card1_aggs_with_v314(Feature):
    def create_features(self):
        feats = ['card1', 'V314']
        aggs = {'sum', 'max', 'min', 'mean', 'median', 'nunique'}
        whole = pd.concat([train_transaction[feats], test_transaction[feats]], axis=0)
        df_aggs = whole.groupby(feats[0])[feats[1]].agg(aggs).reset_index()
        df_aggs.columns = feats[:1] + [f'{col}_{feats[1]}_each_{feats[0]}' for col in df_aggs.columns[1:]]
        for col in df_aggs.columns[1:]:
            self.train[col] = pd.merge(train_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]
            self.test[col] = pd.merge(test_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]


class card1_aggs_with_v315(Feature):
    def create_features(self):
        feats = ['card1', 'V315']
        aggs = {'sum', 'max', 'min', 'mean', 'median', 'nunique'}
        whole = pd.concat([train_transaction[feats], test_transaction[feats]], axis=0)
        df_aggs = whole.groupby(feats[0])[feats[1]].agg(aggs).reset_index()
        df_aggs.columns = feats[:1] + [f'{col}_{feats[1]}_each_{feats[0]}' for col in df_aggs.columns[1:]]
        for col in df_aggs.columns[1:]:
            self.train[col] = pd.merge(train_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]
            self.test[col] = pd.merge(test_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]


class card1_aggs_with_v316(Feature):
    def create_features(self):
        feats = ['card1', 'V316']
        aggs = {'sum', 'max', 'min', 'mean', 'median', 'nunique'}
        whole = pd.concat([train_transaction[feats], test_transaction[feats]], axis=0)
        df_aggs = whole.groupby(feats[0])[feats[1]].agg(aggs).reset_index()
        df_aggs.columns = feats[:1] + [f'{col}_{feats[1]}_each_{feats[0]}' for col in df_aggs.columns[1:]]
        for col in df_aggs.columns[1:]:
            self.train[col] = pd.merge(train_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]
            self.test[col] = pd.merge(test_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]


class card1_aggs_with_v317(Feature):
    def create_features(self):
        feats = ['card1', 'V317']
        aggs = {'sum', 'max', 'min', 'mean', 'median', 'nunique'}
        whole = pd.concat([train_transaction[feats], test_transaction[feats]], axis=0)
        df_aggs = whole.groupby(feats[0])[feats[1]].agg(aggs).reset_index()
        df_aggs.columns = feats[:1] + [f'{col}_{feats[1]}_each_{feats[0]}' for col in df_aggs.columns[1:]]
        for col in df_aggs.columns[1:]:
            self.train[col] = pd.merge(train_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]
            self.test[col] = pd.merge(test_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]


class card1_aggs_with_v318(Feature):
    def create_features(self):
        feats = ['card1', 'V318']
        aggs = {'sum', 'max', 'min', 'mean', 'median', 'nunique'}
        whole = pd.concat([train_transaction[feats], test_transaction[feats]], axis=0)
        df_aggs = whole.groupby(feats[0])[feats[1]].agg(aggs).reset_index()
        df_aggs.columns = feats[:1] + [f'{col}_{feats[1]}_each_{feats[0]}' for col in df_aggs.columns[1:]]
        for col in df_aggs.columns[1:]:
            self.train[col] = pd.merge(train_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]
            self.test[col] = pd.merge(test_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]


class card1_aggs_with_v319(Feature):
    def create_features(self):
        feats = ['card1', 'V319']
        aggs = {'sum', 'max', 'min', 'mean', 'median', 'nunique'}
        whole = pd.concat([train_transaction[feats], test_transaction[feats]], axis=0)
        df_aggs = whole.groupby(feats[0])[feats[1]].agg(aggs).reset_index()
        df_aggs.columns = feats[:1] + [f'{col}_{feats[1]}_each_{feats[0]}' for col in df_aggs.columns[1:]]
        for col in df_aggs.columns[1:]:
            self.train[col] = pd.merge(train_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]
            self.test[col] = pd.merge(test_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]


class card1_aggs_with_v320(Feature):
    def create_features(self):
        feats = ['card1', 'V320']
        aggs = {'sum', 'max', 'min', 'mean', 'median', 'nunique'}
        whole = pd.concat([train_transaction[feats], test_transaction[feats]], axis=0)
        df_aggs = whole.groupby(feats[0])[feats[1]].agg(aggs).reset_index()
        df_aggs.columns = feats[:1] + [f'{col}_{feats[1]}_each_{feats[0]}' for col in df_aggs.columns[1:]]
        for col in df_aggs.columns[1:]:
            self.train[col] = pd.merge(train_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]
            self.test[col] = pd.merge(test_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]


class card1_aggs_with_v321(Feature):
    def create_features(self):
        feats = ['card1', 'V321']
        aggs = {'sum', 'max', 'min', 'mean', 'median', 'nunique'}
        whole = pd.concat([train_transaction[feats], test_transaction[feats]], axis=0)
        df_aggs = whole.groupby(feats[0])[feats[1]].agg(aggs).reset_index()
        df_aggs.columns = feats[:1] + [f'{col}_{feats[1]}_each_{feats[0]}' for col in df_aggs.columns[1:]]
        for col in df_aggs.columns[1:]:
            self.train[col] = pd.merge(train_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]
            self.test[col] = pd.merge(test_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]


class card1_aggs_with_v332(Feature):
    def create_features(self):
        feats = ['card1', 'V332']
        aggs = {'sum', 'max', 'min', 'mean', 'median', 'nunique'}
        whole = pd.concat([train_transaction[feats], test_transaction[feats]], axis=0)
        df_aggs = whole.groupby(feats[0])[feats[1]].agg(aggs).reset_index()
        df_aggs.columns = feats[:1] + [f'{col}_{feats[1]}_each_{feats[0]}' for col in df_aggs.columns[1:]]
        for col in df_aggs.columns[1:]:
            self.train[col] = pd.merge(train_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]
            self.test[col] = pd.merge(test_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]


class card1_aggs_with_v333(Feature):
    def create_features(self):
        feats = ['card1', 'V333']
        aggs = {'sum', 'max', 'min', 'mean', 'median', 'nunique'}
        whole = pd.concat([train_transaction[feats], test_transaction[feats]], axis=0)
        df_aggs = whole.groupby(feats[0])[feats[1]].agg(aggs).reset_index()
        df_aggs.columns = feats[:1] + [f'{col}_{feats[1]}_each_{feats[0]}' for col in df_aggs.columns[1:]]
        for col in df_aggs.columns[1:]:
            self.train[col] = pd.merge(train_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]
            self.test[col] = pd.merge(test_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]


class card2_aggs_with_transactionamt(Feature):
    def create_features(self):
        feats = ['card2', 'TransactionAmt']
        aggs = {'sum', 'max', 'min', 'mean', 'median', 'nunique'}
        whole = pd.concat([train_transaction[feats], test_transaction[feats]], axis=0)
        df_aggs = whole.groupby(feats[0])[feats[1]].agg(aggs).reset_index()
        df_aggs.columns = feats[:1] + [f'{col}_{feats[1]}_each_{feats[0]}' for col in df_aggs.columns[1:]]
        for col in df_aggs.columns[1:]:
            self.train[col] = pd.merge(train_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]
            self.test[col] = pd.merge(test_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]


class card2_aggs_with_dist1(Feature):
    def create_features(self):
        feats = ['card2', 'dist1']
        aggs = {'sum', 'max', 'min', 'mean', 'median', 'nunique'}
        whole = pd.concat([train_transaction[feats], test_transaction[feats]], axis=0)
        df_aggs = whole.groupby(feats[0])[feats[1]].agg(aggs).reset_index()
        df_aggs.columns = feats[:1] + [f'{col}_{feats[1]}_each_{feats[0]}' for col in df_aggs.columns[1:]]
        for col in df_aggs.columns[1:]:
            self.train[col] = pd.merge(train_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]
            self.test[col] = pd.merge(test_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]


class card2_aggs_with_d1(Feature):
    def create_features(self):
        feats = ['card2', 'D1']
        aggs = {'sum', 'max', 'min', 'mean', 'median', 'nunique'}
        whole = pd.concat([train_transaction[feats], test_transaction[feats]], axis=0)
        df_aggs = whole.groupby(feats[0])[feats[1]].agg(aggs).reset_index()
        df_aggs.columns = feats[:1] + [f'{col}_{feats[1]}_each_{feats[0]}' for col in df_aggs.columns[1:]]
        for col in df_aggs.columns[1:]:
            self.train[col] = pd.merge(train_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]
            self.test[col] = pd.merge(test_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]


class card2_aggs_with_d2(Feature):
    def create_features(self):
        feats = ['card2', 'D2']
        aggs = {'sum', 'max', 'min', 'mean', 'median', 'nunique'}
        whole = pd.concat([train_transaction[feats], test_transaction[feats]], axis=0)
        df_aggs = whole.groupby(feats[0])[feats[1]].agg(aggs).reset_index()
        df_aggs.columns = feats[:1] + [f'{col}_{feats[1]}_each_{feats[0]}' for col in df_aggs.columns[1:]]
        for col in df_aggs.columns[1:]:
            self.train[col] = pd.merge(train_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]
            self.test[col] = pd.merge(test_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]


class card2_aggs_with_d3(Feature):
    def create_features(self):
        feats = ['card2', 'D3']
        aggs = {'sum', 'max', 'min', 'mean', 'median', 'nunique'}
        whole = pd.concat([train_transaction[feats], test_transaction[feats]], axis=0)
        df_aggs = whole.groupby(feats[0])[feats[1]].agg(aggs).reset_index()
        df_aggs.columns = feats[:1] + [f'{col}_{feats[1]}_each_{feats[0]}' for col in df_aggs.columns[1:]]
        for col in df_aggs.columns[1:]:
            self.train[col] = pd.merge(train_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]
            self.test[col] = pd.merge(test_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]


class card2_aggs_with_d4(Feature):
    def create_features(self):
        feats = ['card2', 'D4']
        aggs = {'sum', 'max', 'min', 'mean', 'median', 'nunique'}
        whole = pd.concat([train_transaction[feats], test_transaction[feats]], axis=0)
        df_aggs = whole.groupby(feats[0])[feats[1]].agg(aggs).reset_index()
        df_aggs.columns = feats[:1] + [f'{col}_{feats[1]}_each_{feats[0]}' for col in df_aggs.columns[1:]]
        for col in df_aggs.columns[1:]:
            self.train[col] = pd.merge(train_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]
            self.test[col] = pd.merge(test_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]


class card2_aggs_with_d5(Feature):
    def create_features(self):
        feats = ['card2', 'D5']
        aggs = {'sum', 'max', 'min', 'mean', 'median', 'nunique'}
        whole = pd.concat([train_transaction[feats], test_transaction[feats]], axis=0)
        df_aggs = whole.groupby(feats[0])[feats[1]].agg(aggs).reset_index()
        df_aggs.columns = feats[:1] + [f'{col}_{feats[1]}_each_{feats[0]}' for col in df_aggs.columns[1:]]
        for col in df_aggs.columns[1:]:
            self.train[col] = pd.merge(train_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]
            self.test[col] = pd.merge(test_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]


class card2_aggs_with_d6(Feature):
    def create_features(self):
        feats = ['card2', 'D6']
        aggs = {'sum', 'max', 'min', 'mean', 'median', 'nunique'}
        whole = pd.concat([train_transaction[feats], test_transaction[feats]], axis=0)
        df_aggs = whole.groupby(feats[0])[feats[1]].agg(aggs).reset_index()
        df_aggs.columns = feats[:1] + [f'{col}_{feats[1]}_each_{feats[0]}' for col in df_aggs.columns[1:]]
        for col in df_aggs.columns[1:]:
            self.train[col] = pd.merge(train_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]
            self.test[col] = pd.merge(test_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]


class card2_aggs_with_d7(Feature):
    def create_features(self):
        feats = ['card2', 'D7']
        aggs = {'sum', 'max', 'min', 'mean', 'median', 'nunique'}
        whole = pd.concat([train_transaction[feats], test_transaction[feats]], axis=0)
        df_aggs = whole.groupby(feats[0])[feats[1]].agg(aggs).reset_index()
        df_aggs.columns = feats[:1] + [f'{col}_{feats[1]}_each_{feats[0]}' for col in df_aggs.columns[1:]]
        for col in df_aggs.columns[1:]:
            self.train[col] = pd.merge(train_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]
            self.test[col] = pd.merge(test_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]


class card2_aggs_with_d8(Feature):
    def create_features(self):
        feats = ['card2', 'D8']
        aggs = {'sum', 'max', 'min', 'mean', 'median', 'nunique'}
        whole = pd.concat([train_transaction[feats], test_transaction[feats]], axis=0)
        df_aggs = whole.groupby(feats[0])[feats[1]].agg(aggs).reset_index()
        df_aggs.columns = feats[:1] + [f'{col}_{feats[1]}_each_{feats[0]}' for col in df_aggs.columns[1:]]
        for col in df_aggs.columns[1:]:
            self.train[col] = pd.merge(train_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]
            self.test[col] = pd.merge(test_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]


class card2_aggs_with_d9(Feature):
    def create_features(self):
        feats =['card2', 'D9']
        aggs = {'sum', 'max', 'min', 'mean', 'median', 'nunique'}
        whole = pd.concat([train_transaction[feats], test_transaction[feats]], axis=0)
        df_aggs = whole.groupby(feats[0])[feats[1]].agg(aggs).reset_index()
        df_aggs.columns = feats[:1] + [f'{col}_{feats[1]}_each_{feats[0]}' for col in df_aggs.columns[1:]]
        for col in df_aggs.columns[1:]:
            self.train[col] = pd.merge(train_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]
            self.test[col] = pd.merge(test_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]


class card2_aggs_with_d10(Feature):
    def create_features(self):
        feats = ['card2', 'D10']
        aggs = {'sum', 'max', 'min', 'mean', 'median', 'nunique'}
        whole = pd.concat([train_transaction[feats], test_transaction[feats]], axis=0)
        df_aggs = whole.groupby(feats[0])[feats[1]].agg(aggs).reset_index()
        df_aggs.columns = feats[:1] + [f'{col}_{feats[1]}_each_{feats[0]}' for col in df_aggs.columns[1:]]
        for col in df_aggs.columns[1:]:
            self.train[col] = pd.merge(train_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]
            self.test[col] = pd.merge(test_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]


class card2_aggs_with_d11(Feature):
    def create_features(self):
        feats =['card2', 'D11']
        aggs = {'sum', 'max', 'min', 'mean', 'median', 'nunique'}
        whole = pd.concat([train_transaction[feats], test_transaction[feats]], axis=0)
        df_aggs = whole.groupby(feats[0])[feats[1]].agg(aggs).reset_index()
        df_aggs.columns = feats[:1] + [f'{col}_{feats[1]}_each_{feats[0]}' for col in df_aggs.columns[1:]]
        for col in df_aggs.columns[1:]:
            self.train[col] = pd.merge(train_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]
            self.test[col] = pd.merge(test_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]


class card2_aggs_with_d12(Feature):
    def create_features(self):
        feats = ['card2', 'D12']
        aggs = {'sum', 'max', 'min', 'mean', 'median', 'nunique'}
        whole = pd.concat([train_transaction[feats], test_transaction[feats]], axis=0)
        df_aggs = whole.groupby(feats[0])[feats[1]].agg(aggs).reset_index()
        df_aggs.columns = feats[:1] + [f'{col}_{feats[1]}_each_{feats[0]}' for col in df_aggs.columns[1:]]
        for col in df_aggs.columns[1:]:
            self.train[col] = pd.merge(train_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]
            self.test[col] = pd.merge(test_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]


class card2_aggs_with_d13(Feature):
    def create_features(self):
        feats = ['card2', 'D13']
        aggs = {'sum', 'max', 'min', 'mean', 'median', 'nunique'}
        whole = pd.concat([train_transaction[feats], test_transaction[feats]], axis=0)
        df_aggs = whole.groupby(feats[0])[feats[1]].agg(aggs).reset_index()
        df_aggs.columns = feats[:1] + [f'{col}_{feats[1]}_each_{feats[0]}' for col in df_aggs.columns[1:]]
        for col in df_aggs.columns[1:]:
            self.train[col] = pd.merge(train_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]
            self.test[col] = pd.merge(test_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]


class card2_aggs_with_d14(Feature):
    def create_features(self):
        feats = ['card2', 'D14']
        aggs = {'sum', 'max', 'min', 'mean', 'median', 'nunique'}
        whole = pd.concat([train_transaction[feats], test_transaction[feats]], axis=0)
        df_aggs = whole.groupby(feats[0])[feats[1]].agg(aggs).reset_index()
        df_aggs.columns = feats[:1] + [f'{col}_{feats[1]}_each_{feats[0]}' for col in df_aggs.columns[1:]]
        for col in df_aggs.columns[1:]:
            self.train[col] = pd.merge(train_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]
            self.test[col] = pd.merge(test_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]


class card2_aggs_with_d15(Feature):
    def create_features(self):
        feats = ['card2', 'D15']
        aggs = {'sum', 'max', 'min', 'mean', 'median', 'nunique'}
        whole = pd.concat([train_transaction[feats], test_transaction[feats]], axis=0)
        df_aggs = whole.groupby(feats[0])[feats[1]].agg(aggs).reset_index()
        df_aggs.columns = feats[:1] + [f'{col}_{feats[1]}_each_{feats[0]}' for col in df_aggs.columns[1:]]
        for col in df_aggs.columns[1:]:
            self.train[col] = pd.merge(train_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]
            self.test[col] = pd.merge(test_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]


class card2_aggs_with_v96(Feature):
    def create_features(self):
        feats = ['card2', 'V96']
        aggs = {'sum', 'max', 'min', 'mean', 'median', 'nunique'}
        whole = pd.concat([train_transaction[feats], test_transaction[feats]], axis=0)
        df_aggs = whole.groupby(feats[0])[feats[1]].agg(aggs).reset_index()
        df_aggs.columns = feats[:1] + [f'{col}_{feats[1]}_each_{feats[0]}' for col in df_aggs.columns[1:]]
        for col in df_aggs.columns[1:]:
            self.train[col] = pd.merge(train_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]
            self.test[col] = pd.merge(test_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]


class card2_aggs_with_v126(Feature):
    def create_features(self):
        feats = ['card2', 'V126']
        aggs = {'sum', 'max', 'min', 'mean', 'median', 'nunique'}
        whole = pd.concat([train_transaction[feats], test_transaction[feats]], axis=0)
        df_aggs = whole.groupby(feats[0])[feats[1]].agg(aggs).reset_index()
        df_aggs.columns = feats[:1] + [f'{col}_{feats[1]}_each_{feats[0]}' for col in df_aggs.columns[1:]]
        for col in df_aggs.columns[1:]:
            self.train[col] = pd.merge(train_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]
            self.test[col] = pd.merge(test_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]


class card2_aggs_with_v127(Feature):
    def create_features(self):
        feats = ['card2', 'V127']
        aggs = {'sum', 'max', 'min', 'mean', 'median', 'nunique'}
        whole = pd.concat([train_transaction[feats], test_transaction[feats]], axis=0)
        df_aggs = whole.groupby(feats[0])[feats[1]].agg(aggs).reset_index()
        df_aggs.columns = feats[:1] + [f'{col}_{feats[1]}_each_{feats[0]}' for col in df_aggs.columns[1:]]
        for col in df_aggs.columns[1:]:
            self.train[col] = pd.merge(train_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]
            self.test[col] = pd.merge(test_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]


class card2_aggs_with_v128(Feature):
    def create_features(self):
        feats = ['card2', 'V128']
        aggs = {'sum', 'max', 'min', 'mean', 'median', 'nunique'}
        whole = pd.concat([train_transaction[feats], test_transaction[feats]], axis=0)
        df_aggs = whole.groupby(feats[0])[feats[1]].agg(aggs).reset_index()
        df_aggs.columns = feats[:1] + [f'{col}_{feats[1]}_each_{feats[0]}' for col in df_aggs.columns[1:]]
        for col in df_aggs.columns[1:]:
            self.train[col] = pd.merge(train_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]
            self.test[col] = pd.merge(test_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]



class card2_aggs_with_v129(Feature):
    def create_features(self):
        feats = ['card2', 'V129']
        aggs = {'sum', 'max', 'min', 'mean', 'median', 'nunique'}
        whole = pd.concat([train_transaction[feats], test_transaction[feats]], axis=0)
        df_aggs = whole.groupby(feats[0])[feats[1]].agg(aggs).reset_index()
        df_aggs.columns = feats[:1] + [f'{col}_{feats[1]}_each_{feats[0]}' for col in df_aggs.columns[1:]]
        for col in df_aggs.columns[1:]:
            self.train[col] = pd.merge(train_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]
            self.test[col] = pd.merge(test_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]


class card2_aggs_with_v130(Feature):
    def create_features(self):
        feats = ['card2', 'V130']
        aggs = {'sum', 'max', 'min', 'mean', 'median', 'nunique'}
        whole = pd.concat([train_transaction[feats], test_transaction[feats]], axis=0)
        df_aggs = whole.groupby(feats[0])[feats[1]].agg(aggs).reset_index()
        df_aggs.columns = feats[:1] + [f'{col}_{feats[1]}_each_{feats[0]}' for col in df_aggs.columns[1:]]
        for col in df_aggs.columns[1:]:
            self.train[col] = pd.merge(train_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]
            self.test[col] = pd.merge(test_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]


class card2_aggs_with_v131(Feature):
    def create_features(self):
        feats = ['card2', 'V131']
        aggs = {'sum', 'max', 'min', 'mean', 'median', 'nunique'}
        whole = pd.concat([train_transaction[feats], test_transaction[feats]], axis=0)
        df_aggs = whole.groupby(feats[0])[feats[1]].agg(aggs).reset_index()
        df_aggs.columns = feats[:1] + [f'{col}_{feats[1]}_each_{feats[0]}' for col in df_aggs.columns[1:]]
        for col in df_aggs.columns[1:]:
            self.train[col] = pd.merge(train_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]
            self.test[col] = pd.merge(test_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]


class card2_aggs_with_v132(Feature):
    def create_features(self):
        feats = ['card2', 'V132']
        aggs = {'sum', 'max', 'min', 'mean', 'median', 'nunique'}
        whole = pd.concat([train_transaction[feats], test_transaction[feats]], axis=0)
        df_aggs = whole.groupby(feats[0])[feats[1]].agg(aggs).reset_index()
        df_aggs.columns = feats[:1] + [f'{col}_{feats[1]}_each_{feats[0]}' for col in df_aggs.columns[1:]]
        for col in df_aggs.columns[1:]:
            self.train[col] = pd.merge(train_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]
            self.test[col] = pd.merge(test_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]


class card2_aggs_with_v133(Feature):
    def create_features(self):
        feats = ['card2', 'V133']
        aggs = {'sum', 'max', 'min', 'mean', 'median', 'nunique'}
        whole = pd.concat([train_transaction[feats], test_transaction[feats]], axis=0)
        df_aggs = whole.groupby(feats[0])[feats[1]].agg(aggs).reset_index()
        df_aggs.columns = feats[:1] + [f'{col}_{feats[1]}_each_{feats[0]}' for col in df_aggs.columns[1:]]
        for col in df_aggs.columns[1:]:
            self.train[col] = pd.merge(train_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]
            self.test[col] = pd.merge(test_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]


class card2_aggs_with_v134(Feature):
    def create_features(self):
        feats = ['card2', 'V134']
        aggs = {'sum', 'max', 'min', 'mean', 'median', 'nunique'}
        whole = pd.concat([train_transaction[feats], test_transaction[feats]], axis=0)
        df_aggs = whole.groupby(feats[0])[feats[1]].agg(aggs).reset_index()
        df_aggs.columns = feats[:1] + [f'{col}_{feats[1]}_each_{feats[0]}' for col in df_aggs.columns[1:]]
        for col in df_aggs.columns[1:]:
            self.train[col] = pd.merge(train_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]
            self.test[col] = pd.merge(test_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]


class card2_aggs_with_v135(Feature):
    def create_features(self):
        feats = ['card2', 'V135']
        aggs = {'sum', 'max', 'min', 'mean', 'median', 'nunique'}
        whole = pd.concat([train_transaction[feats], test_transaction[feats]], axis=0)
        df_aggs = whole.groupby(feats[0])[feats[1]].agg(aggs).reset_index()
        df_aggs.columns = feats[:1] + [f'{col}_{feats[1]}_each_{feats[0]}' for col in df_aggs.columns[1:]]
        for col in df_aggs.columns[1:]:
            self.train[col] = pd.merge(train_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]
            self.test[col] = pd.merge(test_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]


class card2_aggs_with_v136(Feature):
    def create_features(self):
        feats = ['card2', 'V136']
        aggs = {'sum', 'max', 'min', 'mean', 'median', 'nunique'}
        whole = pd.concat([train_transaction[feats], test_transaction[feats]], axis=0)
        df_aggs = whole.groupby(feats[0])[feats[1]].agg(aggs).reset_index()
        df_aggs.columns = feats[:1] + [f'{col}_{feats[1]}_each_{feats[0]}' for col in df_aggs.columns[1:]]
        for col in df_aggs.columns[1:]:
            self.train[col] = pd.merge(train_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]
            self.test[col] = pd.merge(test_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]


class card2_aggs_with_v137(Feature):
    def create_features(self):
        feats = ['card2', 'V137']
        aggs = {'sum', 'max', 'min', 'mean', 'median', 'nunique'}
        whole = pd.concat([train_transaction[feats], test_transaction[feats]], axis=0)
        df_aggs = whole.groupby(feats[0])[feats[1]].agg(aggs).reset_index()
        df_aggs.columns = feats[:1] + [f'{col}_{feats[1]}_each_{feats[0]}' for col in df_aggs.columns[1:]]
        for col in df_aggs.columns[1:]:
            self.train[col] = pd.merge(train_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]
            self.test[col] = pd.merge(test_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]


class card2_aggs_with_v150(Feature):
    def create_features(self):
        feats = ['card2', 'V150']
        aggs = {'sum', 'max', 'min', 'mean', 'median', 'nunique'}
        whole = pd.concat([train_transaction[feats], test_transaction[feats]], axis=0)
        df_aggs = whole.groupby(feats[0])[feats[1]].agg(aggs).reset_index()
        df_aggs.columns = feats[:1] + [f'{col}_{feats[1]}_each_{feats[0]}' for col in df_aggs.columns[1:]]
        for col in df_aggs.columns[1:]:
            self.train[col] = pd.merge(train_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]
            self.test[col] = pd.merge(test_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]


class card2_aggs_with_v159(Feature):
    def create_features(self):
        feats = ['card2', 'V159']
        aggs = {'sum', 'max', 'min', 'mean', 'median', 'nunique'}
        whole = pd.concat([train_transaction[feats], test_transaction[feats]], axis=0)
        df_aggs = whole.groupby(feats[0])[feats[1]].agg(aggs).reset_index()
        df_aggs.columns = feats[:1] + [f'{col}_{feats[1]}_each_{feats[0]}' for col in df_aggs.columns[1:]]
        for col in df_aggs.columns[1:]:
            self.train[col] = pd.merge(train_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]
            self.test[col] = pd.merge(test_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]


class card2_aggs_with_v160(Feature):
    def create_features(self):
        feats = ['card2', 'V160']
        aggs = {'sum', 'max', 'min', 'mean', 'median', 'nunique'}
        whole = pd.concat([train_transaction[feats], test_transaction[feats]], axis=0)
        df_aggs = whole.groupby(feats[0])[feats[1]].agg(aggs).reset_index()
        df_aggs.columns = feats[:1] + [f'{col}_{feats[1]}_each_{feats[0]}' for col in df_aggs.columns[1:]]
        for col in df_aggs.columns[1:]:
            self.train[col] = pd.merge(train_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]
            self.test[col] = pd.merge(test_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]


class card2_aggs_with_v166(Feature):
    def create_features(self):
        feats = ['card2', 'V166']
        aggs = {'sum', 'max', 'min', 'mean', 'median', 'nunique'}
        whole = pd.concat([train_transaction[feats], test_transaction[feats]], axis=0)
        df_aggs = whole.groupby(feats[0])[feats[1]].agg(aggs).reset_index()
        df_aggs.columns = feats[:1] + [f'{col}_{feats[1]}_each_{feats[0]}' for col in df_aggs.columns[1:]]
        for col in df_aggs.columns[1:]:
            self.train[col] = pd.merge(train_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]
            self.test[col] = pd.merge(test_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]


class card2_aggs_with_v202(Feature):
    def create_features(self):
        feats = ['card2', 'V202']
        aggs = {'sum', 'max', 'min', 'mean', 'median', 'nunique'}
        whole = pd.concat([train_transaction[feats], test_transaction[feats]], axis=0)
        df_aggs = whole.groupby(feats[0])[feats[1]].agg(aggs).reset_index()
        df_aggs.columns = feats[:1] + [f'{col}_{feats[1]}_each_{feats[0]}' for col in df_aggs.columns[1:]]
        for col in df_aggs.columns[1:]:
            self.train[col] = pd.merge(train_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]
            self.test[col] = pd.merge(test_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]


class card2_aggs_with_v203(Feature):
    def create_features(self):
        feats = ['card2', 'V203']
        aggs = {'sum', 'max', 'min', 'mean', 'median', 'nunique'}
        whole = pd.concat([train_transaction[feats], test_transaction[feats]], axis=0)
        df_aggs = whole.groupby(feats[0])[feats[1]].agg(aggs).reset_index()
        df_aggs.columns = feats[:1] + [f'{col}_{feats[1]}_each_{feats[0]}' for col in df_aggs.columns[1:]]
        for col in df_aggs.columns[1:]:
            self.train[col] = pd.merge(train_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]
            self.test[col] = pd.merge(test_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]


class card2_aggs_with_v204(Feature):
    def create_features(self):
        feats = ['card2', 'V204']
        aggs = {'sum', 'max', 'min', 'mean', 'median', 'nunique'}
        whole = pd.concat([train_transaction[feats], test_transaction[feats]], axis=0)
        df_aggs = whole.groupby(feats[0])[feats[1]].agg(aggs).reset_index()
        df_aggs.columns = feats[:1] + [f'{col}_{feats[1]}_each_{feats[0]}' for col in df_aggs.columns[1:]]
        for col in df_aggs.columns[1:]:
            self.train[col] = pd.merge(train_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]
            self.test[col] = pd.merge(test_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]


class card2_aggs_with_v207(Feature):
    def create_features(self):
        feats = ['card2', 'V207']
        aggs = {'sum', 'max', 'min', 'mean', 'median', 'nunique'}
        whole = pd.concat([train_transaction[feats], test_transaction[feats]], axis=0)
        df_aggs = whole.groupby(feats[0])[feats[1]].agg(aggs).reset_index()
        df_aggs.columns = feats[:1] + [f'{col}_{feats[1]}_each_{feats[0]}' for col in df_aggs.columns[1:]]
        for col in df_aggs.columns[1:]:
            self.train[col] = pd.merge(train_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]
            self.test[col] = pd.merge(test_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]


class card2_aggs_with_v208(Feature):
    def create_features(self):
        feats = ['card2', 'V208']
        aggs = {'sum', 'max', 'min', 'mean', 'median', 'nunique'}
        whole = pd.concat([train_transaction[feats], test_transaction[feats]], axis=0)
        df_aggs = whole.groupby(feats[0])[feats[1]].agg(aggs).reset_index()
        df_aggs.columns = feats[:1] + [f'{col}_{feats[1]}_each_{feats[0]}' for col in df_aggs.columns[1:]]
        for col in df_aggs.columns[1:]:
            self.train[col] = pd.merge(train_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]
            self.test[col] = pd.merge(test_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]


class card2_aggs_with_v209(Feature):
    def create_features(self):
        feats = ['card2', 'V209']
        aggs = {'sum', 'max', 'min', 'mean', 'median', 'nunique'}
        whole = pd.concat([train_transaction[feats], test_transaction[feats]], axis=0)
        df_aggs = whole.groupby(feats[0])[feats[1]].agg(aggs).reset_index()
        df_aggs.columns = feats[:1] + [f'{col}_{feats[1]}_each_{feats[0]}' for col in df_aggs.columns[1:]]
        for col in df_aggs.columns[1:]:
            self.train[col] = pd.merge(train_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]
            self.test[col] = pd.merge(test_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]


class card2_aggs_with_v210(Feature):
    def create_features(self):
        feats = ['card2', 'V210']
        aggs = {'sum', 'max', 'min', 'mean', 'median', 'nunique'}
        whole = pd.concat([train_transaction[feats], test_transaction[feats]], axis=0)
        df_aggs = whole.groupby(feats[0])[feats[1]].agg(aggs).reset_index()
        df_aggs.columns = feats[:1] + [f'{col}_{feats[1]}_each_{feats[0]}' for col in df_aggs.columns[1:]]
        for col in df_aggs.columns[1:]:
            self.train[col] = pd.merge(train_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]
            self.test[col] = pd.merge(test_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]


class card2_aggs_with_v212(Feature):
    def create_features(self):
        feats = ['card2', 'V212']
        aggs = {'sum', 'max', 'min', 'mean', 'median', 'nunique'}
        whole = pd.concat([train_transaction[feats], test_transaction[feats]], axis=0)
        df_aggs = whole.groupby(feats[0])[feats[1]].agg(aggs).reset_index()
        df_aggs.columns = feats[:1] + [f'{col}_{feats[1]}_each_{feats[0]}' for col in df_aggs.columns[1:]]
        for col in df_aggs.columns[1:]:
            self.train[col] = pd.merge(train_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]
            self.test[col] = pd.merge(test_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]

class card2_aggs_with_v214(Feature):
    def create_features(self):
        feats = ['card2', 'V214']
        aggs = {'sum', 'max', 'min', 'mean', 'median', 'nunique'}
        whole = pd.concat([train_transaction[feats], test_transaction[feats]], axis=0)
        df_aggs = whole.groupby(feats[0])[feats[1]].agg(aggs).reset_index()
        df_aggs.columns = feats[:1] + [f'{col}_{feats[1]}_each_{feats[0]}' for col in df_aggs.columns[1:]]
        for col in df_aggs.columns[1:]:
            self.train[col] = pd.merge(train_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]
            self.test[col] = pd.merge(test_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]


class card2_aggs_with_v215(Feature):
    def create_features(self):
        feats = ['card2', 'V215']
        aggs = {'sum', 'max', 'min', 'mean', 'median', 'nunique'}
        whole = pd.concat([train_transaction[feats], test_transaction[feats]], axis=0)
        df_aggs = whole.groupby(feats[0])[feats[1]].agg(aggs).reset_index()
        df_aggs.columns = feats[:1] + [f'{col}_{feats[1]}_each_{feats[0]}' for col in df_aggs.columns[1:]]
        for col in df_aggs.columns[1:]:
            self.train[col] = pd.merge(train_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]
            self.test[col] = pd.merge(test_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]


class card2_aggs_with_v216(Feature):
    def create_features(self):
        feats = ['card2', 'V216']
        aggs = {'sum', 'max', 'min', 'mean', 'median', 'nunique'}
        whole = pd.concat([train_transaction[feats], test_transaction[feats]], axis=0)
        df_aggs = whole.groupby(feats[0])[feats[1]].agg(aggs).reset_index()
        df_aggs.columns = feats[:1] + [f'{col}_{feats[1]}_each_{feats[0]}' for col in df_aggs.columns[1:]]
        for col in df_aggs.columns[1:]:
            self.train[col] = pd.merge(train_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]
            self.test[col] = pd.merge(test_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]


class card2_aggs_with_v263(Feature):
    def create_features(self):
        feats = ['card2', 'V263']
        aggs = {'sum', 'max', 'min', 'mean', 'median', 'nunique'}
        whole = pd.concat([train_transaction[feats], test_transaction[feats]], axis=0)
        df_aggs = whole.groupby(feats[0])[feats[1]].agg(aggs).reset_index()
        df_aggs.columns = feats[:1] + [f'{col}_{feats[1]}_each_{feats[0]}' for col in df_aggs.columns[1:]]
        for col in df_aggs.columns[1:]:
            self.train[col] = pd.merge(train_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]
            self.test[col] = pd.merge(test_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]


class card2_aggs_with_v264(Feature):
    def create_features(self):
        feats = ['card2', 'V264']
        aggs = {'sum', 'max', 'min', 'mean', 'median', 'nunique'}
        whole = pd.concat([train_transaction[feats], test_transaction[feats]], axis=0)
        df_aggs = whole.groupby(feats[0])[feats[1]].agg(aggs).reset_index()
        df_aggs.columns = feats[:1] + [f'{col}_{feats[1]}_each_{feats[0]}' for col in df_aggs.columns[1:]]
        for col in df_aggs.columns[1:]:
            self.train[col] = pd.merge(train_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]
            self.test[col] = pd.merge(test_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]


class card2_aggs_with_v265(Feature):
    def create_features(self):
        feats = ['card2', 'V265']
        aggs = {'sum', 'max', 'min', 'mean', 'median', 'nunique'}
        whole = pd.concat([train_transaction[feats], test_transaction[feats]], axis=0)
        df_aggs = whole.groupby(feats[0])[feats[1]].agg(aggs).reset_index()
        df_aggs.columns = feats[:1] + [f'{col}_{feats[1]}_each_{feats[0]}' for col in df_aggs.columns[1:]]
        for col in df_aggs.columns[1:]:
            self.train[col] = pd.merge(train_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]
            self.test[col] = pd.merge(test_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]


class card2_aggs_with_v267(Feature):
    def create_features(self):
        feats = ['card2', 'V267']
        aggs = {'sum', 'max', 'min', 'mean', 'median', 'nunique'}
        whole = pd.concat([train_transaction[feats], test_transaction[feats]], axis=0)
        df_aggs = whole.groupby(feats[0])[feats[1]].agg(aggs).reset_index()
        df_aggs.columns = feats[:1] + [f'{col}_{feats[1]}_each_{feats[0]}' for col in df_aggs.columns[1:]]
        for col in df_aggs.columns[1:]:
            self.train[col] = pd.merge(train_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]
            self.test[col] = pd.merge(test_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]


class card2_aggs_with_v274(Feature):
    def create_features(self):
        feats = ['card2', 'V274']
        aggs = {'sum', 'max', 'min', 'mean', 'median', 'nunique'}
        whole = pd.concat([train_transaction[feats], test_transaction[feats]], axis=0)
        df_aggs = whole.groupby(feats[0])[feats[1]].agg(aggs).reset_index()
        df_aggs.columns = feats[:1] + [f'{col}_{feats[1]}_each_{feats[0]}' for col in df_aggs.columns[1:]]
        for col in df_aggs.columns[1:]:
            self.train[col] = pd.merge(train_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]
            self.test[col] = pd.merge(test_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]


class card2_aggs_with_v275(Feature):
    def create_features(self):
        feats = ['card2', 'V275']
        aggs = {'sum', 'max', 'min', 'mean', 'median', 'nunique'}
        whole = pd.concat([train_transaction[feats], test_transaction[feats]], axis=0)
        df_aggs = whole.groupby(feats[0])[feats[1]].agg(aggs).reset_index()
        df_aggs.columns = feats[:1] + [f'{col}_{feats[1]}_each_{feats[0]}' for col in df_aggs.columns[1:]]
        for col in df_aggs.columns[1:]:
            self.train[col] = pd.merge(train_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]
            self.test[col] = pd.merge(test_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]


class card2_aggs_with_v276(Feature):
    def create_features(self):
        feats = ['card2', 'V276']
        aggs = {'sum', 'max', 'min', 'mean', 'median', 'nunique'}
        whole = pd.concat([train_transaction[feats], test_transaction[feats]], axis=0)
        df_aggs = whole.groupby(feats[0])[feats[1]].agg(aggs).reset_index()
        df_aggs.columns = feats[:1] + [f'{col}_{feats[1]}_each_{feats[0]}' for col in df_aggs.columns[1:]]
        for col in df_aggs.columns[1:]:
            self.train[col] = pd.merge(train_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]
            self.test[col] = pd.merge(test_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]


class card2_aggs_with_v277(Feature):
    def create_features(self):
        feats = ['card2', 'V277']
        aggs = {'sum', 'max', 'min', 'mean', 'median', 'nunique'}
        whole = pd.concat([train_transaction[feats], test_transaction[feats]], axis=0)
        df_aggs = whole.groupby(feats[0])[feats[1]].agg(aggs).reset_index()
        df_aggs.columns = feats[:1] + [f'{col}_{feats[1]}_each_{feats[0]}' for col in df_aggs.columns[1:]]
        for col in df_aggs.columns[1:]:
            self.train[col] = pd.merge(train_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]
            self.test[col] = pd.merge(test_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]


class card2_aggs_with_v278(Feature):
    def create_features(self):
        feats = ['card2', 'V278']
        aggs = {'sum', 'max', 'min', 'mean', 'median', 'nunique'}
        whole = pd.concat([train_transaction[feats], test_transaction[feats]], axis=0)
        df_aggs = whole.groupby(feats[0])[feats[1]].agg(aggs).reset_index()
        df_aggs.columns = feats[:1] + [f'{col}_{feats[1]}_each_{feats[0]}' for col in df_aggs.columns[1:]]
        for col in df_aggs.columns[1:]:
            self.train[col] = pd.merge(train_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]
            self.test[col] = pd.merge(test_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]


class card2_aggs_with_v280(Feature):
    def create_features(self):
        feats = ['card2', 'V280']
        aggs = {'sum', 'max', 'min', 'mean', 'median', 'nunique'}
        whole = pd.concat([train_transaction[feats], test_transaction[feats]], axis=0)
        df_aggs = whole.groupby(feats[0])[feats[1]].agg(aggs).reset_index()
        df_aggs.columns = feats[:1] + [f'{col}_{feats[1]}_each_{feats[0]}' for col in df_aggs.columns[1:]]
        for col in df_aggs.columns[1:]:
            self.train[col] = pd.merge(train_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]
            self.test[col] = pd.merge(test_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]


class card2_aggs_with_v294(Feature):
    def create_features(self):
        feats = ['card2', 'V294']
        aggs = {'sum', 'max', 'min', 'mean', 'median', 'nunique'}
        whole = pd.concat([train_transaction[feats], test_transaction[feats]], axis=0)
        df_aggs = whole.groupby(feats[0])[feats[1]].agg(aggs).reset_index()
        df_aggs.columns = feats[:1] + [f'{col}_{feats[1]}_each_{feats[0]}' for col in df_aggs.columns[1:]]
        for col in df_aggs.columns[1:]:
            self.train[col] = pd.merge(train_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]
            self.test[col] = pd.merge(test_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]


class card2_aggs_with_v306(Feature):
    def create_features(self):
        feats = ['card2', 'V306']
        aggs = {'sum', 'max', 'min', 'mean', 'median', 'nunique'}
        whole = pd.concat([train_transaction[feats], test_transaction[feats]], axis=0)
        df_aggs = whole.groupby(feats[0])[feats[1]].agg(aggs).reset_index()
        df_aggs.columns = feats[:1] + [f'{col}_{feats[1]}_each_{feats[0]}' for col in df_aggs.columns[1:]]
        for col in df_aggs.columns[1:]:
            self.train[col] = pd.merge(train_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]
            self.test[col] = pd.merge(test_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]


class card2_aggs_with_v307(Feature):
    def create_features(self):
        feats = ['card2', 'V307']
        aggs = {'sum', 'max', 'min', 'mean', 'median', 'nunique'}
        whole = pd.concat([train_transaction[feats], test_transaction[feats]], axis=0)
        df_aggs = whole.groupby(feats[0])[feats[1]].agg(aggs).reset_index()
        df_aggs.columns = feats[:1] + [f'{col}_{feats[1]}_each_{feats[0]}' for col in df_aggs.columns[1:]]
        for col in df_aggs.columns[1:]:
            self.train[col] = pd.merge(train_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]
            self.test[col] = pd.merge(test_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]


class card2_aggs_with_v308(Feature):
    def create_features(self):
        feats = ['card2', 'V308']
        aggs = {'sum', 'max', 'min', 'mean', 'median', 'nunique'}
        whole = pd.concat([train_transaction[feats], test_transaction[feats]], axis=0)
        df_aggs = whole.groupby(feats[0])[feats[1]].agg(aggs).reset_index()
        df_aggs.columns = feats[:1] + [f'{col}_{feats[1]}_each_{feats[0]}' for col in df_aggs.columns[1:]]
        for col in df_aggs.columns[1:]:
            self.train[col] = pd.merge(train_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]
            self.test[col] = pd.merge(test_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]


class card2_aggs_with_v309(Feature):
    def create_features(self):
        feats = ['card2', 'V309']
        aggs = {'sum', 'max', 'min', 'mean', 'median', 'nunique'}
        whole = pd.concat([train_transaction[feats], test_transaction[feats]], axis=0)
        df_aggs = whole.groupby(feats[0])[feats[1]].agg(aggs).reset_index()
        df_aggs.columns = feats[:1] + [f'{col}_{feats[1]}_each_{feats[0]}' for col in df_aggs.columns[1:]]
        for col in df_aggs.columns[1:]:
            self.train[col] = pd.merge(train_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]
            self.test[col] = pd.merge(test_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]

class card2_aggs_with_v310(Feature):
    def create_features(self):
        feats = ['card2', 'V310']
        aggs = {'sum', 'max', 'min', 'mean', 'median', 'nunique'}
        whole = pd.concat([train_transaction[feats], test_transaction[feats]], axis=0)
        df_aggs = whole.groupby(feats[0])[feats[1]].agg(aggs).reset_index()
        df_aggs.columns = feats[:1] + [f'{col}_{feats[1]}_each_{feats[0]}' for col in df_aggs.columns[1:]]
        for col in df_aggs.columns[1:]:
            self.train[col] = pd.merge(train_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]
            self.test[col] = pd.merge(test_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]


class card2_aggs_with_v311(Feature):
    def create_features(self):
        feats = ['card2', 'V311']
        aggs = {'sum', 'max', 'min', 'mean', 'median', 'nunique'}
        whole = pd.concat([train_transaction[feats], test_transaction[feats]], axis=0)
        df_aggs = whole.groupby(feats[0])[feats[1]].agg(aggs).reset_index()
        df_aggs.columns = feats[:1] + [f'{col}_{feats[1]}_each_{feats[0]}' for col in df_aggs.columns[1:]]
        for col in df_aggs.columns[1:]:
            self.train[col] = pd.merge(train_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]
            self.test[col] = pd.merge(test_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]


class card2_aggs_with_v312(Feature):
    def create_features(self):
        feats = ['card2', 'V312']
        aggs = {'sum', 'max', 'min', 'mean', 'median', 'nunique'}
        whole = pd.concat([train_transaction[feats], test_transaction[feats]], axis=0)
        df_aggs = whole.groupby(feats[0])[feats[1]].agg(aggs).reset_index()
        df_aggs.columns = feats[:1] + [f'{col}_{feats[1]}_each_{feats[0]}' for col in df_aggs.columns[1:]]
        for col in df_aggs.columns[1:]:
            self.train[col] = pd.merge(train_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]
            self.test[col] = pd.merge(test_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]


class card2_aggs_with_v313(Feature):
    def create_features(self):
        feats = ['card2', 'V313']
        aggs = {'sum', 'max', 'min', 'mean', 'median', 'nunique'}
        whole = pd.concat([train_transaction[feats], test_transaction[feats]], axis=0)
        df_aggs = whole.groupby(feats[0])[feats[1]].agg(aggs).reset_index()
        df_aggs.columns = feats[:1] + [f'{col}_{feats[1]}_each_{feats[0]}' for col in df_aggs.columns[1:]]
        for col in df_aggs.columns[1:]:
            self.train[col] = pd.merge(train_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]
            self.test[col] = pd.merge(test_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]


class card2_aggs_with_v314(Feature):
    def create_features(self):
        feats = ['card2', 'V314']
        aggs = {'sum', 'max', 'min', 'mean', 'median', 'nunique'}
        whole = pd.concat([train_transaction[feats], test_transaction[feats]], axis=0)
        df_aggs = whole.groupby(feats[0])[feats[1]].agg(aggs).reset_index()
        df_aggs.columns = feats[:1] + [f'{col}_{feats[1]}_each_{feats[0]}' for col in df_aggs.columns[1:]]
        for col in df_aggs.columns[1:]:
            self.train[col] = pd.merge(train_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]
            self.test[col] = pd.merge(test_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]


class card2_aggs_with_v315(Feature):
    def create_features(self):
        feats = ['card2', 'V315']
        aggs = {'sum', 'max', 'min', 'mean', 'median', 'nunique'}
        whole = pd.concat([train_transaction[feats], test_transaction[feats]], axis=0)
        df_aggs = whole.groupby(feats[0])[feats[1]].agg(aggs).reset_index()
        df_aggs.columns = feats[:1] + [f'{col}_{feats[1]}_each_{feats[0]}' for col in df_aggs.columns[1:]]
        for col in df_aggs.columns[1:]:
            self.train[col] = pd.merge(train_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]
            self.test[col] = pd.merge(test_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]


class card2_aggs_with_v316(Feature):
    def create_features(self):
        feats = ['card2', 'V316']
        aggs = {'sum', 'max', 'min', 'mean', 'median', 'nunique'}
        whole = pd.concat([train_transaction[feats], test_transaction[feats]], axis=0)
        df_aggs = whole.groupby(feats[0])[feats[1]].agg(aggs).reset_index()
        df_aggs.columns = feats[:1] + [f'{col}_{feats[1]}_each_{feats[0]}' for col in df_aggs.columns[1:]]
        for col in df_aggs.columns[1:]:
            self.train[col] = pd.merge(train_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]
            self.test[col] = pd.merge(test_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]


class card2_aggs_with_v317(Feature):
    def create_features(self):
        feats = ['card2', 'V317']
        aggs = {'sum', 'max', 'min', 'mean', 'median', 'nunique'}
        whole = pd.concat([train_transaction[feats], test_transaction[feats]], axis=0)
        df_aggs = whole.groupby(feats[0])[feats[1]].agg(aggs).reset_index()
        df_aggs.columns = feats[:1] + [f'{col}_{feats[1]}_each_{feats[0]}' for col in df_aggs.columns[1:]]
        for col in df_aggs.columns[1:]:
            self.train[col] = pd.merge(train_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]
            self.test[col] = pd.merge(test_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]


class card2_aggs_with_v318(Feature):
    def create_features(self):
        feats = ['card2', 'V318']
        aggs = {'sum', 'max', 'min', 'mean', 'median', 'nunique'}
        whole = pd.concat([train_transaction[feats], test_transaction[feats]], axis=0)
        df_aggs = whole.groupby(feats[0])[feats[1]].agg(aggs).reset_index()
        df_aggs.columns = feats[:1] + [f'{col}_{feats[1]}_each_{feats[0]}' for col in df_aggs.columns[1:]]
        for col in df_aggs.columns[1:]:
            self.train[col] = pd.merge(train_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]
            self.test[col] = pd.merge(test_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]


class card2_aggs_with_v319(Feature):
    def create_features(self):
        feats = ['card2', 'V319']
        aggs = {'sum', 'max', 'min', 'mean', 'median', 'nunique'}
        whole = pd.concat([train_transaction[feats], test_transaction[feats]], axis=0)
        df_aggs = whole.groupby(feats[0])[feats[1]].agg(aggs).reset_index()
        df_aggs.columns = feats[:1] + [f'{col}_{feats[1]}_each_{feats[0]}' for col in df_aggs.columns[1:]]
        for col in df_aggs.columns[1:]:
            self.train[col] = pd.merge(train_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]
            self.test[col] = pd.merge(test_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]


class card2_aggs_with_v320(Feature):
    def create_features(self):
        feats = ['card2', 'V320']
        aggs = {'sum', 'max', 'min', 'mean', 'median', 'nunique'}
        whole = pd.concat([train_transaction[feats], test_transaction[feats]], axis=0)
        df_aggs = whole.groupby(feats[0])[feats[1]].agg(aggs).reset_index()
        df_aggs.columns = feats[:1] + [f'{col}_{feats[1]}_each_{feats[0]}' for col in df_aggs.columns[1:]]
        for col in df_aggs.columns[1:]:
            self.train[col] = pd.merge(train_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]
            self.test[col] = pd.merge(test_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]


class card2_aggs_with_v321(Feature):
    def create_features(self):
        feats = ['card2', 'V321']
        aggs = {'sum', 'max', 'min', 'mean', 'median', 'nunique'}
        whole = pd.concat([train_transaction[feats], test_transaction[feats]], axis=0)
        df_aggs = whole.groupby(feats[0])[feats[1]].agg(aggs).reset_index()
        df_aggs.columns = feats[:1] + [f'{col}_{feats[1]}_each_{feats[0]}' for col in df_aggs.columns[1:]]
        for col in df_aggs.columns[1:]:
            self.train[col] = pd.merge(train_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]
            self.test[col] = pd.merge(test_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]


class card2_aggs_with_v332(Feature):
    def create_features(self):
        feats = ['card2', 'V332']
        aggs = {'sum', 'max', 'min', 'mean', 'median', 'nunique'}
        whole = pd.concat([train_transaction[feats], test_transaction[feats]], axis=0)
        df_aggs = whole.groupby(feats[0])[feats[1]].agg(aggs).reset_index()
        df_aggs.columns = feats[:1] + [f'{col}_{feats[1]}_each_{feats[0]}' for col in df_aggs.columns[1:]]
        for col in df_aggs.columns[1:]:
            self.train[col] = pd.merge(train_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]
            self.test[col] = pd.merge(test_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]


class card2_aggs_with_v333(Feature):
    def create_features(self):
        feats = ['card2', 'V333']
        aggs = {'sum', 'max', 'min', 'mean', 'median', 'nunique'}
        whole = pd.concat([train_transaction[feats], test_transaction[feats]], axis=0)
        df_aggs = whole.groupby(feats[0])[feats[1]].agg(aggs).reset_index()
        df_aggs.columns = feats[:1] + [f'{col}_{feats[1]}_each_{feats[0]}' for col in df_aggs.columns[1:]]
        for col in df_aggs.columns[1:]:
            self.train[col] = pd.merge(train_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]
            self.test[col] = pd.merge(test_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]


class card3_aggs_with_transactionamt(Feature):
    def create_features(self):
        feats =['card3', 'TransactionAmt']
        aggs = {'sum', 'max', 'min', 'mean', 'median', 'nunique'}
        whole = pd.concat([train_transaction[feats], test_transaction[feats]], axis=0)
        df_aggs = whole.groupby(feats[0])[feats[1]].agg(aggs).reset_index()
        df_aggs.columns = feats[:1] + [f'{col}_{feats[1]}_each_{feats[0]}' for col in df_aggs.columns[1:]]
        for col in df_aggs.columns[1:]:
            self.train[col] = pd.merge(train_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]
            self.test[col] = pd.merge(test_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]


class card3_aggs_with_dist1(Feature):
    def create_features(self):
        feats = ['card3', 'dist1']
        aggs = {'sum', 'max', 'min', 'mean', 'median', 'nunique'}
        whole = pd.concat([train_transaction[feats], test_transaction[feats]], axis=0)
        df_aggs = whole.groupby(feats[0])[feats[1]].agg(aggs).reset_index()
        df_aggs.columns = feats[:1] + [f'{col}_{feats[1]}_each_{feats[0]}' for col in df_aggs.columns[1:]]
        for col in df_aggs.columns[1:]:
            self.train[col] = pd.merge(train_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]
            self.test[col] = pd.merge(test_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]


class card3_aggs_with_d1(Feature):
    def create_features(self):
        feats = ['card3', 'D1']
        aggs = {'sum', 'max', 'min', 'mean', 'median', 'nunique'}
        whole = pd.concat([train_transaction[feats], test_transaction[feats]], axis=0)
        df_aggs = whole.groupby(feats[0])[feats[1]].agg(aggs).reset_index()
        df_aggs.columns = feats[:1] + [f'{col}_{feats[1]}_each_{feats[0]}' for col in df_aggs.columns[1:]]
        for col in df_aggs.columns[1:]:
            self.train[col] = pd.merge(train_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]
            self.test[col] = pd.merge(test_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]


class card3_aggs_with_d2(Feature):
    def create_features(self):
        feats = ['card3', 'D2']
        aggs = {'sum', 'max', 'min', 'mean', 'median', 'nunique'}
        whole = pd.concat([train_transaction[feats], test_transaction[feats]], axis=0)
        df_aggs = whole.groupby(feats[0])[feats[1]].agg(aggs).reset_index()
        df_aggs.columns = feats[:1] + [f'{col}_{feats[1]}_each_{feats[0]}' for col in df_aggs.columns[1:]]
        for col in df_aggs.columns[1:]:
            self.train[col] = pd.merge(train_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]
            self.test[col] = pd.merge(test_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]


class card3_aggs_with_d3(Feature):
    def create_features(self):
        feats = ['card3', 'D3']
        aggs = {'sum', 'max', 'min', 'mean', 'median', 'nunique'}
        whole = pd.concat([train_transaction[feats], test_transaction[feats]], axis=0)
        df_aggs = whole.groupby(feats[0])[feats[1]].agg(aggs).reset_index()
        df_aggs.columns = feats[:1] + [f'{col}_{feats[1]}_each_{feats[0]}' for col in df_aggs.columns[1:]]
        for col in df_aggs.columns[1:]:
            self.train[col] = pd.merge(train_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]
            self.test[col] = pd.merge(test_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]


class card3_aggs_with_d4(Feature):
    def create_features(self):
        feats = ['card3', 'D4']
        aggs = {'sum', 'max', 'min', 'mean', 'median', 'nunique'}
        whole = pd.concat([train_transaction[feats], test_transaction[feats]], axis=0)
        df_aggs = whole.groupby(feats[0])[feats[1]].agg(aggs).reset_index()
        df_aggs.columns = feats[:1] + [f'{col}_{feats[1]}_each_{feats[0]}' for col in df_aggs.columns[1:]]
        for col in df_aggs.columns[1:]:
            self.train[col] = pd.merge(train_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]
            self.test[col] = pd.merge(test_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]


class card3_aggs_with_d5(Feature):
    def create_features(self):
        feats = ['card3', 'D5']
        aggs = {'sum', 'max', 'min', 'mean', 'median', 'nunique'}
        whole = pd.concat([train_transaction[feats], test_transaction[feats]], axis=0)
        df_aggs = whole.groupby(feats[0])[feats[1]].agg(aggs).reset_index()
        df_aggs.columns = feats[:1] + [f'{col}_{feats[1]}_each_{feats[0]}' for col in df_aggs.columns[1:]]
        for col in df_aggs.columns[1:]:
            self.train[col] = pd.merge(train_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]
            self.test[col] = pd.merge(test_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]


class card3_aggs_with_d6(Feature):
    def create_features(self):
        feats = ['card3', 'D6']
        aggs = {'sum', 'max', 'min', 'mean', 'median', 'nunique'}
        whole = pd.concat([train_transaction[feats], test_transaction[feats]], axis=0)
        df_aggs = whole.groupby(feats[0])[feats[1]].agg(aggs).reset_index()
        df_aggs.columns = feats[:1] + [f'{col}_{feats[1]}_each_{feats[0]}' for col in df_aggs.columns[1:]]
        for col in df_aggs.columns[1:]:
            self.train[col] = pd.merge(train_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]
            self.test[col] = pd.merge(test_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]


class card3_aggs_with_d7(Feature):
    def create_features(self):
        feats = ['card3', 'D7']
        aggs = {'sum', 'max', 'min', 'mean', 'median', 'nunique'}
        whole = pd.concat([train_transaction[feats], test_transaction[feats]], axis=0)
        df_aggs = whole.groupby(feats[0])[feats[1]].agg(aggs).reset_index()
        df_aggs.columns = feats[:1] + [f'{col}_{feats[1]}_each_{feats[0]}' for col in df_aggs.columns[1:]]
        for col in df_aggs.columns[1:]:
            self.train[col] = pd.merge(train_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]
            self.test[col] = pd.merge(test_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]


class card3_aggs_with_d8(Feature):
    def create_features(self):
        feats = ['card3', 'D8']
        aggs = {'sum', 'max', 'min', 'mean', 'median', 'nunique'}
        whole = pd.concat([train_transaction[feats], test_transaction[feats]], axis=0)
        df_aggs = whole.groupby(feats[0])[feats[1]].agg(aggs).reset_index()
        df_aggs.columns = feats[:1] + [f'{col}_{feats[1]}_each_{feats[0]}' for col in df_aggs.columns[1:]]
        for col in df_aggs.columns[1:]:
            self.train[col] = pd.merge(train_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]
            self.test[col] = pd.merge(test_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]


class card3_aggs_with_d9(Feature):
    def create_features(self):
        feats = ['card3', 'D9']
        aggs = {'sum', 'max', 'min', 'mean', 'median', 'nunique'}
        whole = pd.concat([train_transaction[feats], test_transaction[feats]], axis=0)
        df_aggs = whole.groupby(feats[0])[feats[1]].agg(aggs).reset_index()
        df_aggs.columns = feats[:1] + [f'{col}_{feats[1]}_each_{feats[0]}' for col in df_aggs.columns[1:]]
        for col in df_aggs.columns[1:]:
            self.train[col] = pd.merge(train_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]
            self.test[col] = pd.merge(test_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]


class card3_aggs_with_d10(Feature):
    def create_features(self):
        feats =['card3', 'D10']
        aggs = {'sum', 'max', 'min', 'mean', 'median', 'nunique'}
        whole = pd.concat([train_transaction[feats], test_transaction[feats]], axis=0)
        df_aggs = whole.groupby(feats[0])[feats[1]].agg(aggs).reset_index()
        df_aggs.columns = feats[:1] + [f'{col}_{feats[1]}_each_{feats[0]}' for col in df_aggs.columns[1:]]
        for col in df_aggs.columns[1:]:
            self.train[col] = pd.merge(train_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]
            self.test[col] = pd.merge(test_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]


class card3_aggs_with_d11(Feature):
    def create_features(self):
        feats = ['card3', 'D11']
        aggs = {'sum', 'max', 'min', 'mean', 'median', 'nunique'}
        whole = pd.concat([train_transaction[feats], test_transaction[feats]], axis=0)
        df_aggs = whole.groupby(feats[0])[feats[1]].agg(aggs).reset_index()
        df_aggs.columns = feats[:1] + [f'{col}_{feats[1]}_each_{feats[0]}' for col in df_aggs.columns[1:]]
        for col in df_aggs.columns[1:]:
            self.train[col] = pd.merge(train_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]
            self.test[col] = pd.merge(test_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]


class card3_aggs_with_d12(Feature):
    def create_features(self):
        feats = ['card3', 'D12']
        aggs = {'sum', 'max', 'min', 'mean', 'median', 'nunique'}
        whole = pd.concat([train_transaction[feats], test_transaction[feats]], axis=0)
        df_aggs = whole.groupby(feats[0])[feats[1]].agg(aggs).reset_index()
        df_aggs.columns = feats[:1] + [f'{col}_{feats[1]}_each_{feats[0]}' for col in df_aggs.columns[1:]]
        for col in df_aggs.columns[1:]:
            self.train[col] = pd.merge(train_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]
            self.test[col] = pd.merge(test_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]


class card3_aggs_with_d13(Feature):
    def create_features(self):
        feats = ['card3', 'D13']
        aggs = {'sum', 'max', 'min', 'mean', 'median', 'nunique'}
        whole = pd.concat([train_transaction[feats], test_transaction[feats]], axis=0)
        df_aggs = whole.groupby(feats[0])[feats[1]].agg(aggs).reset_index()
        df_aggs.columns = feats[:1] + [f'{col}_{feats[1]}_each_{feats[0]}' for col in df_aggs.columns[1:]]
        for col in df_aggs.columns[1:]:
            self.train[col] = pd.merge(train_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]
            self.test[col] = pd.merge(test_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]


class card3_aggs_with_d14(Feature):
    def create_features(self):
        feats = ['card3', 'D14']
        aggs = {'sum', 'max', 'min', 'mean', 'median', 'nunique'}
        whole = pd.concat([train_transaction[feats], test_transaction[feats]], axis=0)
        df_aggs = whole.groupby(feats[0])[feats[1]].agg(aggs).reset_index()
        df_aggs.columns = feats[:1] + [f'{col}_{feats[1]}_each_{feats[0]}' for col in df_aggs.columns[1:]]
        for col in df_aggs.columns[1:]:
            self.train[col] = pd.merge(train_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]
            self.test[col] = pd.merge(test_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]


class card3_aggs_with_d15(Feature):
    def create_features(self):
        feats = ['card3', 'D15']
        aggs = {'sum', 'max', 'min', 'mean', 'median', 'nunique'}
        whole = pd.concat([train_transaction[feats], test_transaction[feats]], axis=0)
        df_aggs = whole.groupby(feats[0])[feats[1]].agg(aggs).reset_index()
        df_aggs.columns = feats[:1] + [f'{col}_{feats[1]}_each_{feats[0]}' for col in df_aggs.columns[1:]]
        for col in df_aggs.columns[1:]:
            self.train[col] = pd.merge(train_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]
            self.test[col] = pd.merge(test_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]


class card4_aggs_with_transactionamt(Feature):
    def create_features(self):
        feats = ['card4', 'TransactionAmt']
        aggs = {'sum', 'max', 'min', 'mean', 'median', 'nunique'}
        whole = pd.concat([train_transaction[feats], test_transaction[feats]], axis=0)
        df_aggs = whole.groupby(feats[0])[feats[1]].agg(aggs).reset_index()
        df_aggs.columns = feats[:1] + [f'{col}_{feats[1]}_each_{feats[0]}' for col in df_aggs.columns[1:]]
        for col in df_aggs.columns[1:]:
            self.train[col] = pd.merge(train_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]
            self.test[col] = pd.merge(test_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]


class card4_aggs_with_dist1(Feature):
    def create_features(self):
        feats = ['card4', 'dist1']
        aggs = {'sum', 'max', 'min', 'mean', 'median', 'nunique'}
        whole = pd.concat([train_transaction[feats], test_transaction[feats]], axis=0)
        df_aggs = whole.groupby(feats[0])[feats[1]].agg(aggs).reset_index()
        df_aggs.columns = feats[:1] + [f'{col}_{feats[1]}_each_{feats[0]}' for col in df_aggs.columns[1:]]
        for col in df_aggs.columns[1:]:
            self.train[col] = pd.merge(train_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]
            self.test[col] = pd.merge(test_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]


class card4_aggs_with_d1(Feature):
    def create_features(self):
        feats = ['card4', 'D1']
        aggs = {'sum', 'max', 'min', 'mean', 'median', 'nunique'}
        whole = pd.concat([train_transaction[feats], test_transaction[feats]], axis=0)
        df_aggs = whole.groupby(feats[0])[feats[1]].agg(aggs).reset_index()
        df_aggs.columns = feats[:1] + [f'{col}_{feats[1]}_each_{feats[0]}' for col in df_aggs.columns[1:]]
        for col in df_aggs.columns[1:]:
            self.train[col] = pd.merge(train_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]
            self.test[col] = pd.merge(test_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]


class card4_aggs_with_d2(Feature):
    def create_features(self):
        feats = ['card4', 'D2']
        aggs = {'sum', 'max', 'min', 'mean', 'median', 'nunique'}
        whole = pd.concat([train_transaction[feats], test_transaction[feats]], axis=0)
        df_aggs = whole.groupby(feats[0])[feats[1]].agg(aggs).reset_index()
        df_aggs.columns = feats[:1] + [f'{col}_{feats[1]}_each_{feats[0]}' for col in df_aggs.columns[1:]]
        for col in df_aggs.columns[1:]:
            self.train[col] = pd.merge(train_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]
            self.test[col] = pd.merge(test_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]


class card4_aggs_with_d3(Feature):
    def create_features(self):
        feats = ['card4', 'D3']
        aggs = {'sum', 'max', 'min', 'mean', 'median', 'nunique'}
        whole = pd.concat([train_transaction[feats], test_transaction[feats]], axis=0)
        df_aggs = whole.groupby(feats[0])[feats[1]].agg(aggs).reset_index()
        df_aggs.columns = feats[:1] + [f'{col}_{feats[1]}_each_{feats[0]}' for col in df_aggs.columns[1:]]
        for col in df_aggs.columns[1:]:
            self.train[col] = pd.merge(train_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]
            self.test[col] = pd.merge(test_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]


class card4_aggs_with_d4(Feature):
    def create_features(self):
        feats = ['card4', 'D4']
        aggs = {'sum', 'max', 'min', 'mean', 'median', 'nunique'}
        whole = pd.concat([train_transaction[feats], test_transaction[feats]], axis=0)
        df_aggs = whole.groupby(feats[0])[feats[1]].agg(aggs).reset_index()
        df_aggs.columns = feats[:1] + [f'{col}_{feats[1]}_each_{feats[0]}' for col in df_aggs.columns[1:]]
        for col in df_aggs.columns[1:]:
            self.train[col] = pd.merge(train_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]
            self.test[col] = pd.merge(test_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]


class card4_aggs_with_d5(Feature):
    def create_features(self):
        feats = ['card4', 'D5']
        aggs = {'sum', 'max', 'min', 'mean', 'median', 'nunique'}
        whole = pd.concat([train_transaction[feats], test_transaction[feats]], axis=0)
        df_aggs = whole.groupby(feats[0])[feats[1]].agg(aggs).reset_index()
        df_aggs.columns = feats[:1] + [f'{col}_{feats[1]}_each_{feats[0]}' for col in df_aggs.columns[1:]]
        for col in df_aggs.columns[1:]:
            self.train[col] = pd.merge(train_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]
            self.test[col] = pd.merge(test_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]


class card4_aggs_with_d6(Feature):
    def create_features(self):
        feats = ['card4', 'D6']
        aggs = {'sum', 'max', 'min', 'mean', 'median', 'nunique'}
        whole = pd.concat([train_transaction[feats], test_transaction[feats]], axis=0)
        df_aggs = whole.groupby(feats[0])[feats[1]].agg(aggs).reset_index()
        df_aggs.columns = feats[:1] + [f'{col}_{feats[1]}_each_{feats[0]}' for col in df_aggs.columns[1:]]
        for col in df_aggs.columns[1:]:
            self.train[col] = pd.merge(train_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]
            self.test[col] = pd.merge(test_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]


class card4_aggs_with_d7(Feature):
    def create_features(self):
        feats = ['card4', 'D7']
        aggs = {'sum', 'max', 'min', 'mean', 'median', 'nunique'}
        whole = pd.concat([train_transaction[feats], test_transaction[feats]], axis=0)
        df_aggs = whole.groupby(feats[0])[feats[1]].agg(aggs).reset_index()
        df_aggs.columns = feats[:1] + [f'{col}_{feats[1]}_each_{feats[0]}' for col in df_aggs.columns[1:]]
        for col in df_aggs.columns[1:]:
            self.train[col] = pd.merge(train_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]
            self.test[col] = pd.merge(test_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]


class card4_aggs_with_d8(Feature):
    def create_features(self):
        feats = ['card4', 'D8']
        aggs = {'sum', 'max', 'min', 'mean', 'median', 'nunique'}
        whole = pd.concat([train_transaction[feats], test_transaction[feats]], axis=0)
        df_aggs = whole.groupby(feats[0])[feats[1]].agg(aggs).reset_index()
        df_aggs.columns = feats[:1] + [f'{col}_{feats[1]}_each_{feats[0]}' for col in df_aggs.columns[1:]]
        for col in df_aggs.columns[1:]:
            self.train[col] = pd.merge(train_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]
            self.test[col] = pd.merge(test_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]


class card4_aggs_with_d9(Feature):
    def create_features(self):
        feats = ['card4', 'D9']
        aggs = {'sum', 'max', 'min', 'mean', 'median', 'nunique'}
        whole = pd.concat([train_transaction[feats], test_transaction[feats]], axis=0)
        df_aggs = whole.groupby(feats[0])[feats[1]].agg(aggs).reset_index()
        df_aggs.columns = feats[:1] + [f'{col}_{feats[1]}_each_{feats[0]}' for col in df_aggs.columns[1:]]
        for col in df_aggs.columns[1:]:
            self.train[col] = pd.merge(train_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]
            self.test[col] = pd.merge(test_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]


class card4_aggs_with_d10(Feature):
    def create_features(self):
        feats = ['card4', 'D10']
        aggs = {'sum', 'max', 'min', 'mean', 'median', 'nunique'}
        whole = pd.concat([train_transaction[feats], test_transaction[feats]], axis=0)
        df_aggs = whole.groupby(feats[0])[feats[1]].agg(aggs).reset_index()
        df_aggs.columns = feats[:1] + [f'{col}_{feats[1]}_each_{feats[0]}' for col in df_aggs.columns[1:]]
        for col in df_aggs.columns[1:]:
            self.train[col] = pd.merge(train_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]
            self.test[col] = pd.merge(test_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]


class card4_aggs_with_d11(Feature):
    def create_features(self):
        feats = ['card4', 'D11']
        aggs = {'sum', 'max', 'min', 'mean', 'median', 'nunique'}
        whole = pd.concat([train_transaction[feats], test_transaction[feats]], axis=0)
        df_aggs = whole.groupby(feats[0])[feats[1]].agg(aggs).reset_index()
        df_aggs.columns = feats[:1] + [f'{col}_{feats[1]}_each_{feats[0]}' for col in df_aggs.columns[1:]]
        for col in df_aggs.columns[1:]:
            self.train[col] = pd.merge(train_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]
            self.test[col] = pd.merge(test_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]


class card4_aggs_with_d12(Feature):
    def create_features(self):
        feats = ['card4', 'D12']
        aggs = {'sum', 'max', 'min', 'mean', 'median', 'nunique'}
        whole = pd.concat([train_transaction[feats], test_transaction[feats]], axis=0)
        df_aggs = whole.groupby(feats[0])[feats[1]].agg(aggs).reset_index()
        df_aggs.columns = feats[:1] + [f'{col}_{feats[1]}_each_{feats[0]}' for col in df_aggs.columns[1:]]
        for col in df_aggs.columns[1:]:
            self.train[col] = pd.merge(train_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]
            self.test[col] = pd.merge(test_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]


class card4_aggs_with_d13(Feature):
    def create_features(self):
        feats = ['card4', 'D13']
        aggs = {'sum', 'max', 'min', 'mean', 'median', 'nunique'}
        whole = pd.concat([train_transaction[feats], test_transaction[feats]], axis=0)
        df_aggs = whole.groupby(feats[0])[feats[1]].agg(aggs).reset_index()
        df_aggs.columns = feats[:1] + [f'{col}_{feats[1]}_each_{feats[0]}' for col in df_aggs.columns[1:]]
        for col in df_aggs.columns[1:]:
            self.train[col] = pd.merge(train_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]
            self.test[col] = pd.merge(test_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]


class card4_aggs_with_d14(Feature):
    def create_features(self):
        feats = ['card4', 'D14']
        aggs = {'sum', 'max', 'min', 'mean', 'median', 'nunique'}
        whole = pd.concat([train_transaction[feats], test_transaction[feats]], axis=0)
        df_aggs = whole.groupby(feats[0])[feats[1]].agg(aggs).reset_index()
        df_aggs.columns = feats[:1] + [f'{col}_{feats[1]}_each_{feats[0]}' for col in df_aggs.columns[1:]]
        for col in df_aggs.columns[1:]:
            self.train[col] = pd.merge(train_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]
            self.test[col] = pd.merge(test_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]


class card4_aggs_with_d15(Feature):
    def create_features(self):
        feats = ['card4', 'D15']
        aggs = {'sum', 'max', 'min', 'mean', 'median', 'nunique'}
        whole = pd.concat([train_transaction[feats], test_transaction[feats]], axis=0)
        df_aggs = whole.groupby(feats[0])[feats[1]].agg(aggs).reset_index()
        df_aggs.columns = feats[:1] + [f'{col}_{feats[1]}_each_{feats[0]}' for col in df_aggs.columns[1:]]
        for col in df_aggs.columns[1:]:
            self.train[col] = pd.merge(train_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]
            self.test[col] = pd.merge(test_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]


class card5_aggs_with_transactionamt(Feature):
    def create_features(self):
        feats = ['card5', 'TransactionAmt']
        aggs = {'sum', 'max', 'min', 'mean', 'median', 'nunique'}
        whole = pd.concat([train_transaction[feats], test_transaction[feats]], axis=0)
        df_aggs = whole.groupby(feats[0])[feats[1]].agg(aggs).reset_index()
        df_aggs.columns = feats[:1] + [f'{col}_{feats[1]}_each_{feats[0]}' for col in df_aggs.columns[1:]]
        for col in df_aggs.columns[1:]:
            self.train[col] = pd.merge(train_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]
            self.test[col] = pd.merge(test_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]


class card5_aggs_with_dist1(Feature):
    def create_features(self):
        feats = ['card5', 'dist1']
        aggs = {'sum', 'max', 'min', 'mean', 'median', 'nunique'}
        whole = pd.concat([train_transaction[feats], test_transaction[feats]], axis=0)
        df_aggs = whole.groupby(feats[0])[feats[1]].agg(aggs).reset_index()
        df_aggs.columns = feats[:1] + [f'{col}_{feats[1]}_each_{feats[0]}' for col in df_aggs.columns[1:]]
        for col in df_aggs.columns[1:]:
            self.train[col] = pd.merge(train_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]
            self.test[col] = pd.merge(test_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]


class card5_aggs_with_d1(Feature):
    def create_features(self):
        feats = ['card5', 'D1']
        aggs = {'sum', 'max', 'min', 'mean', 'median', 'nunique'}
        whole = pd.concat([train_transaction[feats], test_transaction[feats]], axis=0)
        df_aggs = whole.groupby(feats[0])[feats[1]].agg(aggs).reset_index()
        df_aggs.columns = feats[:1] + [f'{col}_{feats[1]}_each_{feats[0]}' for col in df_aggs.columns[1:]]
        for col in df_aggs.columns[1:]:
            self.train[col] = pd.merge(train_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]
            self.test[col] = pd.merge(test_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]


class card5_aggs_with_d2(Feature):
    def create_features(self):
        feats = ['card5', 'D2']
        aggs = {'sum', 'max', 'min', 'mean', 'median', 'nunique'}
        whole = pd.concat([train_transaction[feats], test_transaction[feats]], axis=0)
        df_aggs = whole.groupby(feats[0])[feats[1]].agg(aggs).reset_index()
        df_aggs.columns = feats[:1] + [f'{col}_{feats[1]}_each_{feats[0]}' for col in df_aggs.columns[1:]]
        for col in df_aggs.columns[1:]:
            self.train[col] = pd.merge(train_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]
            self.test[col] = pd.merge(test_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]


class card5_aggs_with_d3(Feature):
    def create_features(self):
        feats = ['card5', 'D3']
        aggs = {'sum', 'max', 'min', 'mean', 'median', 'nunique'}
        whole = pd.concat([train_transaction[feats], test_transaction[feats]], axis=0)
        df_aggs = whole.groupby(feats[0])[feats[1]].agg(aggs).reset_index()
        df_aggs.columns = feats[:1] + [f'{col}_{feats[1]}_each_{feats[0]}' for col in df_aggs.columns[1:]]
        for col in df_aggs.columns[1:]:
            self.train[col] = pd.merge(train_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]
            self.test[col] = pd.merge(test_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]


class card5_aggs_with_d4(Feature):
    def create_features(self):
        feats = ['card5', 'D4']
        aggs = {'sum', 'max', 'min', 'mean', 'median', 'nunique'}
        whole = pd.concat([train_transaction[feats], test_transaction[feats]], axis=0)
        df_aggs = whole.groupby(feats[0])[feats[1]].agg(aggs).reset_index()
        df_aggs.columns = feats[:1] + [f'{col}_{feats[1]}_each_{feats[0]}' for col in df_aggs.columns[1:]]
        for col in df_aggs.columns[1:]:
            self.train[col] = pd.merge(train_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]
            self.test[col] = pd.merge(test_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]


class card5_aggs_with_d5(Feature):
    def create_features(self):
        feats = ['card5', 'D5']
        aggs = {'sum', 'max', 'min', 'mean', 'median', 'nunique'}
        whole = pd.concat([train_transaction[feats], test_transaction[feats]], axis=0)
        df_aggs = whole.groupby(feats[0])[feats[1]].agg(aggs).reset_index()
        df_aggs.columns = feats[:1] + [f'{col}_{feats[1]}_each_{feats[0]}' for col in df_aggs.columns[1:]]
        for col in df_aggs.columns[1:]:
            self.train[col] = pd.merge(train_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]
            self.test[col] = pd.merge(test_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]


class card5_aggs_with_d6(Feature):
    def create_features(self):
        feats = ['card5', 'D6']
        aggs = {'sum', 'max', 'min', 'mean', 'median', 'nunique'}
        whole = pd.concat([train_transaction[feats], test_transaction[feats]], axis=0)
        df_aggs = whole.groupby(feats[0])[feats[1]].agg(aggs).reset_index()
        df_aggs.columns = feats[:1] + [f'{col}_{feats[1]}_each_{feats[0]}' for col in df_aggs.columns[1:]]
        for col in df_aggs.columns[1:]:
            self.train[col] = pd.merge(train_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]
            self.test[col] = pd.merge(test_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]


class card5_aggs_with_d7(Feature):
    def create_features(self):
        feats = ['card5', 'D7']
        aggs = {'sum', 'max', 'min', 'mean', 'median', 'nunique'}
        whole = pd.concat([train_transaction[feats], test_transaction[feats]], axis=0)
        df_aggs = whole.groupby(feats[0])[feats[1]].agg(aggs).reset_index()
        df_aggs.columns = feats[:1] + [f'{col}_{feats[1]}_each_{feats[0]}' for col in df_aggs.columns[1:]]
        for col in df_aggs.columns[1:]:
            self.train[col] = pd.merge(train_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]
            self.test[col] = pd.merge(test_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]


class card5_aggs_with_d8(Feature):
    def create_features(self):
        feats = ['card5', 'D8']
        aggs = {'sum', 'max', 'min', 'mean', 'median', 'nunique'}
        whole = pd.concat([train_transaction[feats], test_transaction[feats]], axis=0)
        df_aggs = whole.groupby(feats[0])[feats[1]].agg(aggs).reset_index()
        df_aggs.columns = feats[:1] + [f'{col}_{feats[1]}_each_{feats[0]}' for col in df_aggs.columns[1:]]
        for col in df_aggs.columns[1:]:
            self.train[col] = pd.merge(train_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]
            self.test[col] = pd.merge(test_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]


class card5_aggs_with_d9(Feature):
    def create_features(self):
        feats = ['card5', 'D9']
        aggs = {'sum', 'max', 'min', 'mean', 'median', 'nunique'}
        whole = pd.concat([train_transaction[feats], test_transaction[feats]], axis=0)
        df_aggs = whole.groupby(feats[0])[feats[1]].agg(aggs).reset_index()
        df_aggs.columns = feats[:1] + [f'{col}_{feats[1]}_each_{feats[0]}' for col in df_aggs.columns[1:]]
        for col in df_aggs.columns[1:]:
            self.train[col] = pd.merge(train_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]
            self.test[col] = pd.merge(test_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]


class card5_aggs_with_d10(Feature):
    def create_features(self):
        feats = ['card5', 'D10']
        aggs = {'sum', 'max', 'min', 'mean', 'median', 'nunique'}
        whole = pd.concat([train_transaction[feats], test_transaction[feats]], axis=0)
        df_aggs = whole.groupby(feats[0])[feats[1]].agg(aggs).reset_index()
        df_aggs.columns = feats[:1] + [f'{col}_{feats[1]}_each_{feats[0]}' for col in df_aggs.columns[1:]]
        for col in df_aggs.columns[1:]:
            self.train[col] = pd.merge(train_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]
            self.test[col] = pd.merge(test_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]


class card5_aggs_with_d11(Feature):
    def create_features(self):
        feats = ['card5', 'D11']
        aggs = {'sum', 'max', 'min', 'mean', 'median', 'nunique'}
        whole = pd.concat([train_transaction[feats], test_transaction[feats]], axis=0)
        df_aggs = whole.groupby(feats[0])[feats[1]].agg(aggs).reset_index()
        df_aggs.columns = feats[:1] + [f'{col}_{feats[1]}_each_{feats[0]}' for col in df_aggs.columns[1:]]
        for col in df_aggs.columns[1:]:
            self.train[col] = pd.merge(train_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]
            self.test[col] = pd.merge(test_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]


class card5_aggs_with_d12(Feature):
    def create_features(self):
        feats = ['card5', 'D12']
        aggs = {'sum', 'max', 'min', 'mean', 'median', 'nunique'}
        whole = pd.concat([train_transaction[feats], test_transaction[feats]], axis=0)
        df_aggs = whole.groupby(feats[0])[feats[1]].agg(aggs).reset_index()
        df_aggs.columns = feats[:1] + [f'{col}_{feats[1]}_each_{feats[0]}' for col in df_aggs.columns[1:]]
        for col in df_aggs.columns[1:]:
            self.train[col] = pd.merge(train_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]
            self.test[col] = pd.merge(test_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]


class card5_aggs_with_d13(Feature):
    def create_features(self):
        feats = ['card5', 'D13']
        aggs = {'sum', 'max', 'min', 'mean', 'median', 'nunique'}
        whole = pd.concat([train_transaction[feats], test_transaction[feats]], axis=0)
        df_aggs = whole.groupby(feats[0])[feats[1]].agg(aggs).reset_index()
        df_aggs.columns = feats[:1] + [f'{col}_{feats[1]}_each_{feats[0]}' for col in df_aggs.columns[1:]]
        for col in df_aggs.columns[1:]:
            self.train[col] = pd.merge(train_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]
            self.test[col] = pd.merge(test_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]


class card5_aggs_with_d14(Feature):
    def create_features(self):
        feats = ['card5', 'D14']
        aggs = {'sum', 'max', 'min', 'mean', 'median', 'nunique'}
        whole = pd.concat([train_transaction[feats], test_transaction[feats]], axis=0)
        df_aggs = whole.groupby(feats[0])[feats[1]].agg(aggs).reset_index()
        df_aggs.columns = feats[:1] + [f'{col}_{feats[1]}_each_{feats[0]}' for col in df_aggs.columns[1:]]
        for col in df_aggs.columns[1:]:
            self.train[col] = pd.merge(train_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]
            self.test[col] = pd.merge(test_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]


class card5_aggs_with_d15(Feature):
    def create_features(self):
        feats = ['card5', 'D15']
        aggs = {'sum', 'max', 'min', 'mean', 'median', 'nunique'}
        whole = pd.concat([train_transaction[feats], test_transaction[feats]], axis=0)
        df_aggs = whole.groupby(feats[0])[feats[1]].agg(aggs).reset_index()
        df_aggs.columns = feats[:1] + [f'{col}_{feats[1]}_each_{feats[0]}' for col in df_aggs.columns[1:]]
        for col in df_aggs.columns[1:]:
            self.train[col] = pd.merge(train_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]
            self.test[col] = pd.merge(test_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]


class card5_aggs_with_v96(Feature):
    def create_features(self):
        feats = ['card5', 'V96']
        aggs = {'sum', 'max', 'min', 'mean', 'median', 'nunique'}
        whole = pd.concat([train_transaction[feats], test_transaction[feats]], axis=0)
        df_aggs = whole.groupby(feats[0])[feats[1]].agg(aggs).reset_index()
        df_aggs.columns = feats[:1] + [f'{col}_{feats[1]}_each_{feats[0]}' for col in df_aggs.columns[1:]]
        for col in df_aggs.columns[1:]:
            self.train[col] = pd.merge(train_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]
            self.test[col] = pd.merge(test_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]


class card5_aggs_with_v126(Feature):
    def create_features(self):
        feats = ['card5', 'V126']
        aggs = {'sum', 'max', 'min', 'mean', 'median', 'nunique'}
        whole = pd.concat([train_transaction[feats], test_transaction[feats]], axis=0)
        df_aggs = whole.groupby(feats[0])[feats[1]].agg(aggs).reset_index()
        df_aggs.columns = feats[:1] + [f'{col}_{feats[1]}_each_{feats[0]}' for col in df_aggs.columns[1:]]
        for col in df_aggs.columns[1:]:
            self.train[col] = pd.merge(train_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]
            self.test[col] = pd.merge(test_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]


class card5_aggs_with_v127(Feature):
    def create_features(self):
        feats = ['card5', 'V127']
        aggs = {'sum', 'max', 'min', 'mean', 'median', 'nunique'}
        whole = pd.concat([train_transaction[feats], test_transaction[feats]], axis=0)
        df_aggs = whole.groupby(feats[0])[feats[1]].agg(aggs).reset_index()
        df_aggs.columns = feats[:1] + [f'{col}_{feats[1]}_each_{feats[0]}' for col in df_aggs.columns[1:]]
        for col in df_aggs.columns[1:]:
            self.train[col] = pd.merge(train_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]
            self.test[col] = pd.merge(test_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]


class card5_aggs_with_v128(Feature):
    def create_features(self):
        feats = ['card5', 'V128']
        aggs = {'sum', 'max', 'min', 'mean', 'median', 'nunique'}
        whole = pd.concat([train_transaction[feats], test_transaction[feats]], axis=0)
        df_aggs = whole.groupby(feats[0])[feats[1]].agg(aggs).reset_index()
        df_aggs.columns = feats[:1] + [f'{col}_{feats[1]}_each_{feats[0]}' for col in df_aggs.columns[1:]]
        for col in df_aggs.columns[1:]:
            self.train[col] = pd.merge(train_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]
            self.test[col] = pd.merge(test_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]



class card5_aggs_with_v129(Feature):
    def create_features(self):
        feats = ['card5', 'V129']
        aggs = {'sum', 'max', 'min', 'mean', 'median', 'nunique'}
        whole = pd.concat([train_transaction[feats], test_transaction[feats]], axis=0)
        df_aggs = whole.groupby(feats[0])[feats[1]].agg(aggs).reset_index()
        df_aggs.columns = feats[:1] + [f'{col}_{feats[1]}_each_{feats[0]}' for col in df_aggs.columns[1:]]
        for col in df_aggs.columns[1:]:
            self.train[col] = pd.merge(train_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]
            self.test[col] = pd.merge(test_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]


class card5_aggs_with_v130(Feature):
    def create_features(self):
        feats = ['card5', 'V130']
        aggs = {'sum', 'max', 'min', 'mean', 'median', 'nunique'}
        whole = pd.concat([train_transaction[feats], test_transaction[feats]], axis=0)
        df_aggs = whole.groupby(feats[0])[feats[1]].agg(aggs).reset_index()
        df_aggs.columns = feats[:1] + [f'{col}_{feats[1]}_each_{feats[0]}' for col in df_aggs.columns[1:]]
        for col in df_aggs.columns[1:]:
            self.train[col] = pd.merge(train_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]
            self.test[col] = pd.merge(test_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]


class card5_aggs_with_v131(Feature):
    def create_features(self):
        feats = ['card5', 'V131']
        aggs = {'sum', 'max', 'min', 'mean', 'median', 'nunique'}
        whole = pd.concat([train_transaction[feats], test_transaction[feats]], axis=0)
        df_aggs = whole.groupby(feats[0])[feats[1]].agg(aggs).reset_index()
        df_aggs.columns = feats[:1] + [f'{col}_{feats[1]}_each_{feats[0]}' for col in df_aggs.columns[1:]]
        for col in df_aggs.columns[1:]:
            self.train[col] = pd.merge(train_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]
            self.test[col] = pd.merge(test_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]


class card5_aggs_with_v132(Feature):
    def create_features(self):
        feats = ['card5', 'V132']
        aggs = {'sum', 'max', 'min', 'mean', 'median', 'nunique'}
        whole = pd.concat([train_transaction[feats], test_transaction[feats]], axis=0)
        df_aggs = whole.groupby(feats[0])[feats[1]].agg(aggs).reset_index()
        df_aggs.columns = feats[:1] + [f'{col}_{feats[1]}_each_{feats[0]}' for col in df_aggs.columns[1:]]
        for col in df_aggs.columns[1:]:
            self.train[col] = pd.merge(train_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]
            self.test[col] = pd.merge(test_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]


class card5_aggs_with_v133(Feature):
    def create_features(self):
        feats = ['card5', 'V133']
        aggs = {'sum', 'max', 'min', 'mean', 'median', 'nunique'}
        whole = pd.concat([train_transaction[feats], test_transaction[feats]], axis=0)
        df_aggs = whole.groupby(feats[0])[feats[1]].agg(aggs).reset_index()
        df_aggs.columns = feats[:1] + [f'{col}_{feats[1]}_each_{feats[0]}' for col in df_aggs.columns[1:]]
        for col in df_aggs.columns[1:]:
            self.train[col] = pd.merge(train_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]
            self.test[col] = pd.merge(test_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]


class card5_aggs_with_v134(Feature):
    def create_features(self):
        feats = ['card5', 'V134']
        aggs = {'sum', 'max', 'min', 'mean', 'median', 'nunique'}
        whole = pd.concat([train_transaction[feats], test_transaction[feats]], axis=0)
        df_aggs = whole.groupby(feats[0])[feats[1]].agg(aggs).reset_index()
        df_aggs.columns = feats[:1] + [f'{col}_{feats[1]}_each_{feats[0]}' for col in df_aggs.columns[1:]]
        for col in df_aggs.columns[1:]:
            self.train[col] = pd.merge(train_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]
            self.test[col] = pd.merge(test_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]


class card5_aggs_with_v135(Feature):
    def create_features(self):
        feats = ['card5', 'V135']
        aggs = {'sum', 'max', 'min', 'mean', 'median', 'nunique'}
        whole = pd.concat([train_transaction[feats], test_transaction[feats]], axis=0)
        df_aggs = whole.groupby(feats[0])[feats[1]].agg(aggs).reset_index()
        df_aggs.columns = feats[:1] + [f'{col}_{feats[1]}_each_{feats[0]}' for col in df_aggs.columns[1:]]
        for col in df_aggs.columns[1:]:
            self.train[col] = pd.merge(train_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]
            self.test[col] = pd.merge(test_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]


class card5_aggs_with_v136(Feature):
    def create_features(self):
        feats = ['card5', 'V136']
        aggs = {'sum', 'max', 'min', 'mean', 'median', 'nunique'}
        whole = pd.concat([train_transaction[feats], test_transaction[feats]], axis=0)
        df_aggs = whole.groupby(feats[0])[feats[1]].agg(aggs).reset_index()
        df_aggs.columns = feats[:1] + [f'{col}_{feats[1]}_each_{feats[0]}' for col in df_aggs.columns[1:]]
        for col in df_aggs.columns[1:]:
            self.train[col] = pd.merge(train_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]
            self.test[col] = pd.merge(test_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]


class card5_aggs_with_v137(Feature):
    def create_features(self):
        feats = ['card5', 'V137']
        aggs = {'sum', 'max', 'min', 'mean', 'median', 'nunique'}
        whole = pd.concat([train_transaction[feats], test_transaction[feats]], axis=0)
        df_aggs = whole.groupby(feats[0])[feats[1]].agg(aggs).reset_index()
        df_aggs.columns = feats[:1] + [f'{col}_{feats[1]}_each_{feats[0]}' for col in df_aggs.columns[1:]]
        for col in df_aggs.columns[1:]:
            self.train[col] = pd.merge(train_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]
            self.test[col] = pd.merge(test_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]


class card5_aggs_with_v150(Feature):
    def create_features(self):
        feats = ['card5', 'V150']
        aggs = {'sum', 'max', 'min', 'mean', 'median', 'nunique'}
        whole = pd.concat([train_transaction[feats], test_transaction[feats]], axis=0)
        df_aggs = whole.groupby(feats[0])[feats[1]].agg(aggs).reset_index()
        df_aggs.columns = feats[:1] + [f'{col}_{feats[1]}_each_{feats[0]}' for col in df_aggs.columns[1:]]
        for col in df_aggs.columns[1:]:
            self.train[col] = pd.merge(train_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]
            self.test[col] = pd.merge(test_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]


class card5_aggs_with_v159(Feature):
    def create_features(self):
        feats = ['card5', 'V159']
        aggs = {'sum', 'max', 'min', 'mean', 'median', 'nunique'}
        whole = pd.concat([train_transaction[feats], test_transaction[feats]], axis=0)
        df_aggs = whole.groupby(feats[0])[feats[1]].agg(aggs).reset_index()
        df_aggs.columns = feats[:1] + [f'{col}_{feats[1]}_each_{feats[0]}' for col in df_aggs.columns[1:]]
        for col in df_aggs.columns[1:]:
            self.train[col] = pd.merge(train_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]
            self.test[col] = pd.merge(test_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]


class card5_aggs_with_v160(Feature):
    def create_features(self):
        feats = ['card5', 'V160']
        aggs = {'sum', 'max', 'min', 'mean', 'median', 'nunique'}
        whole = pd.concat([train_transaction[feats], test_transaction[feats]], axis=0)
        df_aggs = whole.groupby(feats[0])[feats[1]].agg(aggs).reset_index()
        df_aggs.columns = feats[:1] + [f'{col}_{feats[1]}_each_{feats[0]}' for col in df_aggs.columns[1:]]
        for col in df_aggs.columns[1:]:
            self.train[col] = pd.merge(train_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]
            self.test[col] = pd.merge(test_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]


class card5_aggs_with_v166(Feature):
    def create_features(self):
        feats = ['card5', 'V166']
        aggs = {'sum', 'max', 'min', 'mean', 'median', 'nunique'}
        whole = pd.concat([train_transaction[feats], test_transaction[feats]], axis=0)
        df_aggs = whole.groupby(feats[0])[feats[1]].agg(aggs).reset_index()
        df_aggs.columns = feats[:1] + [f'{col}_{feats[1]}_each_{feats[0]}' for col in df_aggs.columns[1:]]
        for col in df_aggs.columns[1:]:
            self.train[col] = pd.merge(train_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]
            self.test[col] = pd.merge(test_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]


class card5_aggs_with_v202(Feature):
    def create_features(self):
        feats = ['card5', 'V202']
        aggs = {'sum', 'max', 'min', 'mean', 'median', 'nunique'}
        whole = pd.concat([train_transaction[feats], test_transaction[feats]], axis=0)
        df_aggs = whole.groupby(feats[0])[feats[1]].agg(aggs).reset_index()
        df_aggs.columns = feats[:1] + [f'{col}_{feats[1]}_each_{feats[0]}' for col in df_aggs.columns[1:]]
        for col in df_aggs.columns[1:]:
            self.train[col] = pd.merge(train_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]
            self.test[col] = pd.merge(test_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]


class card5_aggs_with_v203(Feature):
    def create_features(self):
        feats = ['card5', 'V203']
        aggs = {'sum', 'max', 'min', 'mean', 'median', 'nunique'}
        whole = pd.concat([train_transaction[feats], test_transaction[feats]], axis=0)
        df_aggs = whole.groupby(feats[0])[feats[1]].agg(aggs).reset_index()
        df_aggs.columns = feats[:1] + [f'{col}_{feats[1]}_each_{feats[0]}' for col in df_aggs.columns[1:]]
        for col in df_aggs.columns[1:]:
            self.train[col] = pd.merge(train_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]
            self.test[col] = pd.merge(test_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]


class card5_aggs_with_v204(Feature):
    def create_features(self):
        feats = ['card5', 'V204']
        aggs = {'sum', 'max', 'min', 'mean', 'median', 'nunique'}
        whole = pd.concat([train_transaction[feats], test_transaction[feats]], axis=0)
        df_aggs = whole.groupby(feats[0])[feats[1]].agg(aggs).reset_index()
        df_aggs.columns = feats[:1] + [f'{col}_{feats[1]}_each_{feats[0]}' for col in df_aggs.columns[1:]]
        for col in df_aggs.columns[1:]:
            self.train[col] = pd.merge(train_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]
            self.test[col] = pd.merge(test_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]


class card5_aggs_with_v207(Feature):
    def create_features(self):
        feats = ['card5', 'V207']
        aggs = {'sum', 'max', 'min', 'mean', 'median', 'nunique'}
        whole = pd.concat([train_transaction[feats], test_transaction[feats]], axis=0)
        df_aggs = whole.groupby(feats[0])[feats[1]].agg(aggs).reset_index()
        df_aggs.columns = feats[:1] + [f'{col}_{feats[1]}_each_{feats[0]}' for col in df_aggs.columns[1:]]
        for col in df_aggs.columns[1:]:
            self.train[col] = pd.merge(train_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]
            self.test[col] = pd.merge(test_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]


class card5_aggs_with_v208(Feature):
    def create_features(self):
        feats = ['card5', 'V208']
        aggs = {'sum', 'max', 'min', 'mean', 'median', 'nunique'}
        whole = pd.concat([train_transaction[feats], test_transaction[feats]], axis=0)
        df_aggs = whole.groupby(feats[0])[feats[1]].agg(aggs).reset_index()
        df_aggs.columns = feats[:1] + [f'{col}_{feats[1]}_each_{feats[0]}' for col in df_aggs.columns[1:]]
        for col in df_aggs.columns[1:]:
            self.train[col] = pd.merge(train_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]
            self.test[col] = pd.merge(test_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]


class card5_aggs_with_v209(Feature):
    def create_features(self):
        feats = ['card5', 'V209']
        aggs = {'sum', 'max', 'min', 'mean', 'median', 'nunique'}
        whole = pd.concat([train_transaction[feats], test_transaction[feats]], axis=0)
        df_aggs = whole.groupby(feats[0])[feats[1]].agg(aggs).reset_index()
        df_aggs.columns = feats[:1] + [f'{col}_{feats[1]}_each_{feats[0]}' for col in df_aggs.columns[1:]]
        for col in df_aggs.columns[1:]:
            self.train[col] = pd.merge(train_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]
            self.test[col] = pd.merge(test_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]


class card5_aggs_with_v210(Feature):
    def create_features(self):
        feats = ['card5', 'V210']
        aggs = {'sum', 'max', 'min', 'mean', 'median', 'nunique'}
        whole = pd.concat([train_transaction[feats], test_transaction[feats]], axis=0)
        df_aggs = whole.groupby(feats[0])[feats[1]].agg(aggs).reset_index()
        df_aggs.columns = feats[:1] + [f'{col}_{feats[1]}_each_{feats[0]}' for col in df_aggs.columns[1:]]
        for col in df_aggs.columns[1:]:
            self.train[col] = pd.merge(train_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]
            self.test[col] = pd.merge(test_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]


class card5_aggs_with_v212(Feature):
    def create_features(self):
        feats = ['card5', 'V212']
        aggs = {'sum', 'max', 'min', 'mean', 'median', 'nunique'}
        whole = pd.concat([train_transaction[feats], test_transaction[feats]], axis=0)
        df_aggs = whole.groupby(feats[0])[feats[1]].agg(aggs).reset_index()
        df_aggs.columns = feats[:1] + [f'{col}_{feats[1]}_each_{feats[0]}' for col in df_aggs.columns[1:]]
        for col in df_aggs.columns[1:]:
            self.train[col] = pd.merge(train_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]
            self.test[col] = pd.merge(test_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]

class card5_aggs_with_v214(Feature):
    def create_features(self):
        feats = ['card5', 'V214']
        aggs = {'sum', 'max', 'min', 'mean', 'median', 'nunique'}
        whole = pd.concat([train_transaction[feats], test_transaction[feats]], axis=0)
        df_aggs = whole.groupby(feats[0])[feats[1]].agg(aggs).reset_index()
        df_aggs.columns = feats[:1] + [f'{col}_{feats[1]}_each_{feats[0]}' for col in df_aggs.columns[1:]]
        for col in df_aggs.columns[1:]:
            self.train[col] = pd.merge(train_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]
            self.test[col] = pd.merge(test_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]


class card5_aggs_with_v215(Feature):
    def create_features(self):
        feats = ['card5', 'V215']
        aggs = {'sum', 'max', 'min', 'mean', 'median', 'nunique'}
        whole = pd.concat([train_transaction[feats], test_transaction[feats]], axis=0)
        df_aggs = whole.groupby(feats[0])[feats[1]].agg(aggs).reset_index()
        df_aggs.columns = feats[:1] + [f'{col}_{feats[1]}_each_{feats[0]}' for col in df_aggs.columns[1:]]
        for col in df_aggs.columns[1:]:
            self.train[col] = pd.merge(train_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]
            self.test[col] = pd.merge(test_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]


class card5_aggs_with_v216(Feature):
    def create_features(self):
        feats = ['card5', 'V216']
        aggs = {'sum', 'max', 'min', 'mean', 'median', 'nunique'}
        whole = pd.concat([train_transaction[feats], test_transaction[feats]], axis=0)
        df_aggs = whole.groupby(feats[0])[feats[1]].agg(aggs).reset_index()
        df_aggs.columns = feats[:1] + [f'{col}_{feats[1]}_each_{feats[0]}' for col in df_aggs.columns[1:]]
        for col in df_aggs.columns[1:]:
            self.train[col] = pd.merge(train_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]
            self.test[col] = pd.merge(test_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]


class card5_aggs_with_v263(Feature):
    def create_features(self):
        feats = ['card5', 'V263']
        aggs = {'sum', 'max', 'min', 'mean', 'median', 'nunique'}
        whole = pd.concat([train_transaction[feats], test_transaction[feats]], axis=0)
        df_aggs = whole.groupby(feats[0])[feats[1]].agg(aggs).reset_index()
        df_aggs.columns = feats[:1] + [f'{col}_{feats[1]}_each_{feats[0]}' for col in df_aggs.columns[1:]]
        for col in df_aggs.columns[1:]:
            self.train[col] = pd.merge(train_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]
            self.test[col] = pd.merge(test_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]


class card5_aggs_with_v264(Feature):
    def create_features(self):
        feats = ['card5', 'V264']
        aggs = {'sum', 'max', 'min', 'mean', 'median', 'nunique'}
        whole = pd.concat([train_transaction[feats], test_transaction[feats]], axis=0)
        df_aggs = whole.groupby(feats[0])[feats[1]].agg(aggs).reset_index()
        df_aggs.columns = feats[:1] + [f'{col}_{feats[1]}_each_{feats[0]}' for col in df_aggs.columns[1:]]
        for col in df_aggs.columns[1:]:
            self.train[col] = pd.merge(train_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]
            self.test[col] = pd.merge(test_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]


class card5_aggs_with_v265(Feature):
    def create_features(self):
        feats = ['card5', 'V265']
        aggs = {'sum', 'max', 'min', 'mean', 'median', 'nunique'}
        whole = pd.concat([train_transaction[feats], test_transaction[feats]], axis=0)
        df_aggs = whole.groupby(feats[0])[feats[1]].agg(aggs).reset_index()
        df_aggs.columns = feats[:1] + [f'{col}_{feats[1]}_each_{feats[0]}' for col in df_aggs.columns[1:]]
        for col in df_aggs.columns[1:]:
            self.train[col] = pd.merge(train_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]
            self.test[col] = pd.merge(test_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]


class card5_aggs_with_v267(Feature):
    def create_features(self):
        feats = ['card5', 'V267']
        aggs = {'sum', 'max', 'min', 'mean', 'median', 'nunique'}
        whole = pd.concat([train_transaction[feats], test_transaction[feats]], axis=0)
        df_aggs = whole.groupby(feats[0])[feats[1]].agg(aggs).reset_index()
        df_aggs.columns = feats[:1] + [f'{col}_{feats[1]}_each_{feats[0]}' for col in df_aggs.columns[1:]]
        for col in df_aggs.columns[1:]:
            self.train[col] = pd.merge(train_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]
            self.test[col] = pd.merge(test_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]


class card5_aggs_with_v274(Feature):
    def create_features(self):
        feats = ['card5', 'V274']
        aggs = {'sum', 'max', 'min', 'mean', 'median', 'nunique'}
        whole = pd.concat([train_transaction[feats], test_transaction[feats]], axis=0)
        df_aggs = whole.groupby(feats[0])[feats[1]].agg(aggs).reset_index()
        df_aggs.columns = feats[:1] + [f'{col}_{feats[1]}_each_{feats[0]}' for col in df_aggs.columns[1:]]
        for col in df_aggs.columns[1:]:
            self.train[col] = pd.merge(train_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]
            self.test[col] = pd.merge(test_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]


class card5_aggs_with_v275(Feature):
    def create_features(self):
        feats = ['card5', 'V275']
        aggs = {'sum', 'max', 'min', 'mean', 'median', 'nunique'}
        whole = pd.concat([train_transaction[feats], test_transaction[feats]], axis=0)
        df_aggs = whole.groupby(feats[0])[feats[1]].agg(aggs).reset_index()
        df_aggs.columns = feats[:1] + [f'{col}_{feats[1]}_each_{feats[0]}' for col in df_aggs.columns[1:]]
        for col in df_aggs.columns[1:]:
            self.train[col] = pd.merge(train_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]
            self.test[col] = pd.merge(test_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]


class card5_aggs_with_v276(Feature):
    def create_features(self):
        feats = ['card5', 'V276']
        aggs = {'sum', 'max', 'min', 'mean', 'median', 'nunique'}
        whole = pd.concat([train_transaction[feats], test_transaction[feats]], axis=0)
        df_aggs = whole.groupby(feats[0])[feats[1]].agg(aggs).reset_index()
        df_aggs.columns = feats[:1] + [f'{col}_{feats[1]}_each_{feats[0]}' for col in df_aggs.columns[1:]]
        for col in df_aggs.columns[1:]:
            self.train[col] = pd.merge(train_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]
            self.test[col] = pd.merge(test_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]


class card5_aggs_with_v277(Feature):
    def create_features(self):
        feats = ['card5', 'V277']
        aggs = {'sum', 'max', 'min', 'mean', 'median', 'nunique'}
        whole = pd.concat([train_transaction[feats], test_transaction[feats]], axis=0)
        df_aggs = whole.groupby(feats[0])[feats[1]].agg(aggs).reset_index()
        df_aggs.columns = feats[:1] + [f'{col}_{feats[1]}_each_{feats[0]}' for col in df_aggs.columns[1:]]
        for col in df_aggs.columns[1:]:
            self.train[col] = pd.merge(train_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]
            self.test[col] = pd.merge(test_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]


class card5_aggs_with_v278(Feature):
    def create_features(self):
        feats = ['card5', 'V278']
        aggs = {'sum', 'max', 'min', 'mean', 'median', 'nunique'}
        whole = pd.concat([train_transaction[feats], test_transaction[feats]], axis=0)
        df_aggs = whole.groupby(feats[0])[feats[1]].agg(aggs).reset_index()
        df_aggs.columns = feats[:1] + [f'{col}_{feats[1]}_each_{feats[0]}' for col in df_aggs.columns[1:]]
        for col in df_aggs.columns[1:]:
            self.train[col] = pd.merge(train_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]
            self.test[col] = pd.merge(test_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]


class card5_aggs_with_v280(Feature):
    def create_features(self):
        feats = ['card5', 'V280']
        aggs = {'sum', 'max', 'min', 'mean', 'median', 'nunique'}
        whole = pd.concat([train_transaction[feats], test_transaction[feats]], axis=0)
        df_aggs = whole.groupby(feats[0])[feats[1]].agg(aggs).reset_index()
        df_aggs.columns = feats[:1] + [f'{col}_{feats[1]}_each_{feats[0]}' for col in df_aggs.columns[1:]]
        for col in df_aggs.columns[1:]:
            self.train[col] = pd.merge(train_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]
            self.test[col] = pd.merge(test_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]


class card5_aggs_with_v294(Feature):
    def create_features(self):
        feats = ['card5', 'V294']
        aggs = {'sum', 'max', 'min', 'mean', 'median', 'nunique'}
        whole = pd.concat([train_transaction[feats], test_transaction[feats]], axis=0)
        df_aggs = whole.groupby(feats[0])[feats[1]].agg(aggs).reset_index()
        df_aggs.columns = feats[:1] + [f'{col}_{feats[1]}_each_{feats[0]}' for col in df_aggs.columns[1:]]
        for col in df_aggs.columns[1:]:
            self.train[col] = pd.merge(train_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]
            self.test[col] = pd.merge(test_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]


class card5_aggs_with_v306(Feature):
    def create_features(self):
        feats = ['card5', 'V306']
        aggs = {'sum', 'max', 'min', 'mean', 'median', 'nunique'}
        whole = pd.concat([train_transaction[feats], test_transaction[feats]], axis=0)
        df_aggs = whole.groupby(feats[0])[feats[1]].agg(aggs).reset_index()
        df_aggs.columns = feats[:1] + [f'{col}_{feats[1]}_each_{feats[0]}' for col in df_aggs.columns[1:]]
        for col in df_aggs.columns[1:]:
            self.train[col] = pd.merge(train_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]
            self.test[col] = pd.merge(test_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]


class card5_aggs_with_v307(Feature):
    def create_features(self):
        feats = ['card5', 'V307']
        aggs = {'sum', 'max', 'min', 'mean', 'median', 'nunique'}
        whole = pd.concat([train_transaction[feats], test_transaction[feats]], axis=0)
        df_aggs = whole.groupby(feats[0])[feats[1]].agg(aggs).reset_index()
        df_aggs.columns = feats[:1] + [f'{col}_{feats[1]}_each_{feats[0]}' for col in df_aggs.columns[1:]]
        for col in df_aggs.columns[1:]:
            self.train[col] = pd.merge(train_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]
            self.test[col] = pd.merge(test_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]


class card5_aggs_with_v308(Feature):
    def create_features(self):
        feats = ['card5', 'V308']
        aggs = {'sum', 'max', 'min', 'mean', 'median', 'nunique'}
        whole = pd.concat([train_transaction[feats], test_transaction[feats]], axis=0)
        df_aggs = whole.groupby(feats[0])[feats[1]].agg(aggs).reset_index()
        df_aggs.columns = feats[:1] + [f'{col}_{feats[1]}_each_{feats[0]}' for col in df_aggs.columns[1:]]
        for col in df_aggs.columns[1:]:
            self.train[col] = pd.merge(train_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]
            self.test[col] = pd.merge(test_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]


class card5_aggs_with_v309(Feature):
    def create_features(self):
        feats = ['card5', 'V309']
        aggs = {'sum', 'max', 'min', 'mean', 'median', 'nunique'}
        whole = pd.concat([train_transaction[feats], test_transaction[feats]], axis=0)
        df_aggs = whole.groupby(feats[0])[feats[1]].agg(aggs).reset_index()
        df_aggs.columns = feats[:1] + [f'{col}_{feats[1]}_each_{feats[0]}' for col in df_aggs.columns[1:]]
        for col in df_aggs.columns[1:]:
            self.train[col] = pd.merge(train_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]
            self.test[col] = pd.merge(test_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]

class card5_aggs_with_v310(Feature):
    def create_features(self):
        feats = ['card5', 'V310']
        aggs = {'sum', 'max', 'min', 'mean', 'median', 'nunique'}
        whole = pd.concat([train_transaction[feats], test_transaction[feats]], axis=0)
        df_aggs = whole.groupby(feats[0])[feats[1]].agg(aggs).reset_index()
        df_aggs.columns = feats[:1] + [f'{col}_{feats[1]}_each_{feats[0]}' for col in df_aggs.columns[1:]]
        for col in df_aggs.columns[1:]:
            self.train[col] = pd.merge(train_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]
            self.test[col] = pd.merge(test_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]


class card5_aggs_with_v311(Feature):
    def create_features(self):
        feats = ['card5', 'V311']
        aggs = {'sum', 'max', 'min', 'mean', 'median', 'nunique'}
        whole = pd.concat([train_transaction[feats], test_transaction[feats]], axis=0)
        df_aggs = whole.groupby(feats[0])[feats[1]].agg(aggs).reset_index()
        df_aggs.columns = feats[:1] + [f'{col}_{feats[1]}_each_{feats[0]}' for col in df_aggs.columns[1:]]
        for col in df_aggs.columns[1:]:
            self.train[col] = pd.merge(train_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]
            self.test[col] = pd.merge(test_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]


class card5_aggs_with_v312(Feature):
    def create_features(self):
        feats = ['card5', 'V312']
        aggs = {'sum', 'max', 'min', 'mean', 'median', 'nunique'}
        whole = pd.concat([train_transaction[feats], test_transaction[feats]], axis=0)
        df_aggs = whole.groupby(feats[0])[feats[1]].agg(aggs).reset_index()
        df_aggs.columns = feats[:1] + [f'{col}_{feats[1]}_each_{feats[0]}' for col in df_aggs.columns[1:]]
        for col in df_aggs.columns[1:]:
            self.train[col] = pd.merge(train_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]
            self.test[col] = pd.merge(test_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]


class card5_aggs_with_v313(Feature):
    def create_features(self):
        feats = ['card5', 'V313']
        aggs = {'sum', 'max', 'min', 'mean', 'median', 'nunique'}
        whole = pd.concat([train_transaction[feats], test_transaction[feats]], axis=0)
        df_aggs = whole.groupby(feats[0])[feats[1]].agg(aggs).reset_index()
        df_aggs.columns = feats[:1] + [f'{col}_{feats[1]}_each_{feats[0]}' for col in df_aggs.columns[1:]]
        for col in df_aggs.columns[1:]:
            self.train[col] = pd.merge(train_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]
            self.test[col] = pd.merge(test_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]


class card5_aggs_with_v314(Feature):
    def create_features(self):
        feats = ['card5', 'V314']
        aggs = {'sum', 'max', 'min', 'mean', 'median', 'nunique'}
        whole = pd.concat([train_transaction[feats], test_transaction[feats]], axis=0)
        df_aggs = whole.groupby(feats[0])[feats[1]].agg(aggs).reset_index()
        df_aggs.columns = feats[:1] + [f'{col}_{feats[1]}_each_{feats[0]}' for col in df_aggs.columns[1:]]
        for col in df_aggs.columns[1:]:
            self.train[col] = pd.merge(train_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]
            self.test[col] = pd.merge(test_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]


class card5_aggs_with_v315(Feature):
    def create_features(self):
        feats = ['card5', 'V315']
        aggs = {'sum', 'max', 'min', 'mean', 'median', 'nunique'}
        whole = pd.concat([train_transaction[feats], test_transaction[feats]], axis=0)
        df_aggs = whole.groupby(feats[0])[feats[1]].agg(aggs).reset_index()
        df_aggs.columns = feats[:1] + [f'{col}_{feats[1]}_each_{feats[0]}' for col in df_aggs.columns[1:]]
        for col in df_aggs.columns[1:]:
            self.train[col] = pd.merge(train_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]
            self.test[col] = pd.merge(test_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]


class card5_aggs_with_v316(Feature):
    def create_features(self):
        feats = ['card5', 'V316']
        aggs = {'sum', 'max', 'min', 'mean', 'median', 'nunique'}
        whole = pd.concat([train_transaction[feats], test_transaction[feats]], axis=0)
        df_aggs = whole.groupby(feats[0])[feats[1]].agg(aggs).reset_index()
        df_aggs.columns = feats[:1] + [f'{col}_{feats[1]}_each_{feats[0]}' for col in df_aggs.columns[1:]]
        for col in df_aggs.columns[1:]:
            self.train[col] = pd.merge(train_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]
            self.test[col] = pd.merge(test_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]


class card5_aggs_with_v317(Feature):
    def create_features(self):
        feats = ['card5', 'V317']
        aggs = {'sum', 'max', 'min', 'mean', 'median', 'nunique'}
        whole = pd.concat([train_transaction[feats], test_transaction[feats]], axis=0)
        df_aggs = whole.groupby(feats[0])[feats[1]].agg(aggs).reset_index()
        df_aggs.columns = feats[:1] + [f'{col}_{feats[1]}_each_{feats[0]}' for col in df_aggs.columns[1:]]
        for col in df_aggs.columns[1:]:
            self.train[col] = pd.merge(train_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]
            self.test[col] = pd.merge(test_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]


class card5_aggs_with_v318(Feature):
    def create_features(self):
        feats = ['card5', 'V318']
        aggs = {'sum', 'max', 'min', 'mean', 'median', 'nunique'}
        whole = pd.concat([train_transaction[feats], test_transaction[feats]], axis=0)
        df_aggs = whole.groupby(feats[0])[feats[1]].agg(aggs).reset_index()
        df_aggs.columns = feats[:1] + [f'{col}_{feats[1]}_each_{feats[0]}' for col in df_aggs.columns[1:]]
        for col in df_aggs.columns[1:]:
            self.train[col] = pd.merge(train_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]
            self.test[col] = pd.merge(test_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]


class card5_aggs_with_v319(Feature):
    def create_features(self):
        feats = ['card5', 'V319']
        aggs = {'sum', 'max', 'min', 'mean', 'median', 'nunique'}
        whole = pd.concat([train_transaction[feats], test_transaction[feats]], axis=0)
        df_aggs = whole.groupby(feats[0])[feats[1]].agg(aggs).reset_index()
        df_aggs.columns = feats[:1] + [f'{col}_{feats[1]}_each_{feats[0]}' for col in df_aggs.columns[1:]]
        for col in df_aggs.columns[1:]:
            self.train[col] = pd.merge(train_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]
            self.test[col] = pd.merge(test_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]


class card5_aggs_with_v320(Feature):
    def create_features(self):
        feats = ['card5', 'V320']
        aggs = {'sum', 'max', 'min', 'mean', 'median', 'nunique'}
        whole = pd.concat([train_transaction[feats], test_transaction[feats]], axis=0)
        df_aggs = whole.groupby(feats[0])[feats[1]].agg(aggs).reset_index()
        df_aggs.columns = feats[:1] + [f'{col}_{feats[1]}_each_{feats[0]}' for col in df_aggs.columns[1:]]
        for col in df_aggs.columns[1:]:
            self.train[col] = pd.merge(train_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]
            self.test[col] = pd.merge(test_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]


class card5_aggs_with_v321(Feature):
    def create_features(self):
        feats = ['card5', 'V321']
        aggs = {'sum', 'max', 'min', 'mean', 'median', 'nunique'}
        whole = pd.concat([train_transaction[feats], test_transaction[feats]], axis=0)
        df_aggs = whole.groupby(feats[0])[feats[1]].agg(aggs).reset_index()
        df_aggs.columns = feats[:1] + [f'{col}_{feats[1]}_each_{feats[0]}' for col in df_aggs.columns[1:]]
        for col in df_aggs.columns[1:]:
            self.train[col] = pd.merge(train_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]
            self.test[col] = pd.merge(test_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]


class card5_aggs_with_v332(Feature):
    def create_features(self):
        feats = ['card5', 'V332']
        aggs = {'sum', 'max', 'min', 'mean', 'median', 'nunique'}
        whole = pd.concat([train_transaction[feats], test_transaction[feats]], axis=0)
        df_aggs = whole.groupby(feats[0])[feats[1]].agg(aggs).reset_index()
        df_aggs.columns = feats[:1] + [f'{col}_{feats[1]}_each_{feats[0]}' for col in df_aggs.columns[1:]]
        for col in df_aggs.columns[1:]:
            self.train[col] = pd.merge(train_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]
            self.test[col] = pd.merge(test_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]


class card5_aggs_with_v333(Feature):
    def create_features(self):
        feats = ['card5', 'V333']
        aggs = {'sum', 'max', 'min', 'mean', 'median', 'nunique'}
        whole = pd.concat([train_transaction[feats], test_transaction[feats]], axis=0)
        df_aggs = whole.groupby(feats[0])[feats[1]].agg(aggs).reset_index()
        df_aggs.columns = feats[:1] + [f'{col}_{feats[1]}_each_{feats[0]}' for col in df_aggs.columns[1:]]
        for col in df_aggs.columns[1:]:
            self.train[col] = pd.merge(train_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]
            self.test[col] = pd.merge(test_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]


class addr1_aggs_with_transactionamt(Feature):
    def create_features(self):
        feats = ['addr1', 'TransactionAmt']
        aggs = {'sum', 'max', 'min', 'mean', 'median', 'nunique'}
        whole = pd.concat([train_transaction[feats], test_transaction[feats]], axis=0)
        df_aggs = whole.groupby(feats[0])[feats[1]].agg(aggs).reset_index()
        df_aggs.columns = feats[:1] + [f'{col}_{feats[1]}_each_{feats[0]}' for col in df_aggs.columns[1:]]
        for col in df_aggs.columns[1:]:
            self.train[col] = pd.merge(train_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]
            self.test[col] = pd.merge(test_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]


class addr1_aggs_with_dist1(Feature):
    def create_features(self):
        feats = ['addr1', 'dist1']
        aggs = {'sum', 'max', 'min', 'mean', 'median', 'nunique'}
        whole = pd.concat([train_transaction[feats], test_transaction[feats]], axis=0)
        df_aggs = whole.groupby(feats[0])[feats[1]].agg(aggs).reset_index()
        df_aggs.columns = feats[:1] + [f'{col}_{feats[1]}_each_{feats[0]}' for col in df_aggs.columns[1:]]
        for col in df_aggs.columns[1:]:
            self.train[col] = pd.merge(train_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]
            self.test[col] = pd.merge(test_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]


class addr1_aggs_with_d1(Feature):
    def create_features(self):
        feats = ['addr1', 'D1']
        aggs = {'sum', 'max', 'min', 'mean', 'median', 'nunique'}
        whole = pd.concat([train_transaction[feats], test_transaction[feats]], axis=0)
        df_aggs = whole.groupby(feats[0])[feats[1]].agg(aggs).reset_index()
        df_aggs.columns = feats[:1] + [f'{col}_{feats[1]}_each_{feats[0]}' for col in df_aggs.columns[1:]]
        for col in df_aggs.columns[1:]:
            self.train[col] = pd.merge(train_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]
            self.test[col] = pd.merge(test_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]


class addr1_aggs_with_d2(Feature):
    def create_features(self):
        feats = ['addr1', 'D2']
        aggs = {'sum', 'max', 'min', 'mean', 'median', 'nunique'}
        whole = pd.concat([train_transaction[feats], test_transaction[feats]], axis=0)
        df_aggs = whole.groupby(feats[0])[feats[1]].agg(aggs).reset_index()
        df_aggs.columns = feats[:1] + [f'{col}_{feats[1]}_each_{feats[0]}' for col in df_aggs.columns[1:]]
        for col in df_aggs.columns[1:]:
            self.train[col] = pd.merge(train_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]
            self.test[col] = pd.merge(test_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]


class addr1_aggs_with_d3(Feature):
    def create_features(self):
        feats = ['addr1', 'D3']
        aggs = {'sum', 'max', 'min', 'mean', 'median', 'nunique'}
        whole = pd.concat([train_transaction[feats], test_transaction[feats]], axis=0)
        df_aggs = whole.groupby(feats[0])[feats[1]].agg(aggs).reset_index()
        df_aggs.columns = feats[:1] + [f'{col}_{feats[1]}_each_{feats[0]}' for col in df_aggs.columns[1:]]
        for col in df_aggs.columns[1:]:
            self.train[col] = pd.merge(train_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]
            self.test[col] = pd.merge(test_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]


class addr1_aggs_with_d4(Feature):
    def create_features(self):
        feats = ['addr1', 'D4']
        aggs = {'sum', 'max', 'min', 'mean', 'median', 'nunique'}
        whole = pd.concat([train_transaction[feats], test_transaction[feats]], axis=0)
        df_aggs = whole.groupby(feats[0])[feats[1]].agg(aggs).reset_index()
        df_aggs.columns = feats[:1] + [f'{col}_{feats[1]}_each_{feats[0]}' for col in df_aggs.columns[1:]]
        for col in df_aggs.columns[1:]:
            self.train[col] = pd.merge(train_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]
            self.test[col] = pd.merge(test_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]


class addr1_aggs_with_d5(Feature):
    def create_features(self):
        feats = ['addr1', 'D5']
        aggs = {'sum', 'max', 'min', 'mean', 'median', 'nunique'}
        whole = pd.concat([train_transaction[feats], test_transaction[feats]], axis=0)
        df_aggs = whole.groupby(feats[0])[feats[1]].agg(aggs).reset_index()
        df_aggs.columns = feats[:1] + [f'{col}_{feats[1]}_each_{feats[0]}' for col in df_aggs.columns[1:]]
        for col in df_aggs.columns[1:]:
            self.train[col] = pd.merge(train_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]
            self.test[col] = pd.merge(test_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]


class addr1_aggs_with_d6(Feature):
    def create_features(self):
        feats = ['addr1', 'D6']
        aggs = {'sum', 'max', 'min', 'mean', 'median', 'nunique'}
        whole = pd.concat([train_transaction[feats], test_transaction[feats]], axis=0)
        df_aggs = whole.groupby(feats[0])[feats[1]].agg(aggs).reset_index()
        df_aggs.columns = feats[:1] + [f'{col}_{feats[1]}_each_{feats[0]}' for col in df_aggs.columns[1:]]
        for col in df_aggs.columns[1:]:
            self.train[col] = pd.merge(train_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]
            self.test[col] = pd.merge(test_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]


class addr1_aggs_with_d7(Feature):
    def create_features(self):
        feats = ['addr1', 'D7']
        aggs = {'sum', 'max', 'min', 'mean', 'median', 'nunique'}
        whole = pd.concat([train_transaction[feats], test_transaction[feats]], axis=0)
        df_aggs = whole.groupby(feats[0])[feats[1]].agg(aggs).reset_index()
        df_aggs.columns = feats[:1] + [f'{col}_{feats[1]}_each_{feats[0]}' for col in df_aggs.columns[1:]]
        for col in df_aggs.columns[1:]:
            self.train[col] = pd.merge(train_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]
            self.test[col] = pd.merge(test_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]


class addr1_aggs_with_d8(Feature):
    def create_features(self):
        feats = ['addr1', 'D8']
        aggs = {'sum', 'max', 'min', 'mean', 'median', 'nunique'}
        whole = pd.concat([train_transaction[feats], test_transaction[feats]], axis=0)
        df_aggs = whole.groupby(feats[0])[feats[1]].agg(aggs).reset_index()
        df_aggs.columns = feats[:1] + [f'{col}_{feats[1]}_each_{feats[0]}' for col in df_aggs.columns[1:]]
        for col in df_aggs.columns[1:]:
            self.train[col] = pd.merge(train_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]
            self.test[col] = pd.merge(test_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]


class addr1_aggs_with_d9(Feature):
    def create_features(self):
        feats = ['addr1', 'D9']
        aggs = {'sum', 'max', 'min', 'mean', 'median', 'nunique'}
        whole = pd.concat([train_transaction[feats], test_transaction[feats]], axis=0)
        df_aggs = whole.groupby(feats[0])[feats[1]].agg(aggs).reset_index()
        df_aggs.columns = feats[:1] + [f'{col}_{feats[1]}_each_{feats[0]}' for col in df_aggs.columns[1:]]
        for col in df_aggs.columns[1:]:
            self.train[col] = pd.merge(train_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]
            self.test[col] = pd.merge(test_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]


class addr1_aggs_with_d10(Feature):
    def create_features(self):
        feats = ['addr1', 'D10']
        aggs = {'sum', 'max', 'min', 'mean', 'median', 'nunique'}
        whole = pd.concat([train_transaction[feats], test_transaction[feats]], axis=0)
        df_aggs = whole.groupby(feats[0])[feats[1]].agg(aggs).reset_index()
        df_aggs.columns = feats[:1] + [f'{col}_{feats[1]}_each_{feats[0]}' for col in df_aggs.columns[1:]]
        for col in df_aggs.columns[1:]:
            self.train[col] = pd.merge(train_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]
            self.test[col] = pd.merge(test_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]


class addr1_aggs_with_d11(Feature):
    def create_features(self):
        feats = ['addr1', 'D11']
        aggs = {'sum', 'max', 'min', 'mean', 'median', 'nunique'}
        whole = pd.concat([train_transaction[feats], test_transaction[feats]], axis=0)
        df_aggs = whole.groupby(feats[0])[feats[1]].agg(aggs).reset_index()
        df_aggs.columns = feats[:1] + [f'{col}_{feats[1]}_each_{feats[0]}' for col in df_aggs.columns[1:]]
        for col in df_aggs.columns[1:]:
            self.train[col] = pd.merge(train_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]
            self.test[col] = pd.merge(test_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]


class addr1_aggs_with_d12(Feature):
    def create_features(self):
        feats = ['addr1', 'D12']
        aggs = {'sum', 'max', 'min', 'mean', 'median', 'nunique'}
        whole = pd.concat([train_transaction[feats], test_transaction[feats]], axis=0)
        df_aggs = whole.groupby(feats[0])[feats[1]].agg(aggs).reset_index()
        df_aggs.columns = feats[:1] + [f'{col}_{feats[1]}_each_{feats[0]}' for col in df_aggs.columns[1:]]
        for col in df_aggs.columns[1:]:
            self.train[col] = pd.merge(train_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]
            self.test[col] = pd.merge(test_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]


class addr1_aggs_with_d13(Feature):
    def create_features(self):
        feats = ['addr1', 'D13']
        aggs = {'sum', 'max', 'min', 'mean', 'median', 'nunique'}
        whole = pd.concat([train_transaction[feats], test_transaction[feats]], axis=0)
        df_aggs = whole.groupby(feats[0])[feats[1]].agg(aggs).reset_index()
        df_aggs.columns = feats[:1] + [f'{col}_{feats[1]}_each_{feats[0]}' for col in df_aggs.columns[1:]]
        for col in df_aggs.columns[1:]:
            self.train[col] = pd.merge(train_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]
            self.test[col] = pd.merge(test_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]


class addr1_aggs_with_d14(Feature):
    def create_features(self):
        feats = ['addr1', 'D14']
        aggs = {'sum', 'max', 'min', 'mean', 'median', 'nunique'}
        whole = pd.concat([train_transaction[feats], test_transaction[feats]], axis=0)
        df_aggs = whole.groupby(feats[0])[feats[1]].agg(aggs).reset_index()
        df_aggs.columns = feats[:1] + [f'{col}_{feats[1]}_each_{feats[0]}' for col in df_aggs.columns[1:]]
        for col in df_aggs.columns[1:]:
            self.train[col] = pd.merge(train_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]
            self.test[col] = pd.merge(test_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]


class addr1_aggs_with_d15(Feature):
    def create_features(self):
        feats = ['addr1', 'D15']
        aggs = {'sum', 'max', 'min', 'mean', 'median', 'nunique'}
        whole = pd.concat([train_transaction[feats], test_transaction[feats]], axis=0)
        df_aggs = whole.groupby(feats[0])[feats[1]].agg(aggs).reset_index()
        df_aggs.columns = feats[:1] + [f'{col}_{feats[1]}_each_{feats[0]}' for col in df_aggs.columns[1:]]
        for col in df_aggs.columns[1:]:
            self.train[col] = pd.merge(train_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]
            self.test[col] = pd.merge(test_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]


class transaction_amt_aggs_with_date(Feature):
    def create_features(self):
        feats = ['TransactionDT', 'TransactionAmt']
        aggs = {'sum', 'max', 'min', 'mean', 'median', 'std'}
        whole = pd.concat([train_transaction[feats], test_transaction[feats]], axis=0)
        whole['date_'] = whole['TransactionDT'] // (3600 * 24)
        df_aggs = whole.groupby('date_')['TransactionAmt'].agg(aggs).reset_index()
        df_aggs.columns = ['date_'] + [f'{col}_{feats[1]}_each_date' for col in df_aggs.columns[1:]]
        whole = pd.merge(whole['date_'], df_aggs, on='date_', how='left')
        for col in df_aggs.columns[1:]:
            self.train[col] = whole[col].values[:(len(train_transaction))]
            self.test[col] = whole[col].values[(len(train_transaction)):]


class addr1_aggs_with_v96(Feature):
    def create_features(self):
        feats = ['addr1', 'V96']
        aggs = {'sum', 'max', 'min', 'mean', 'median', 'nunique'}
        whole = pd.concat([train_transaction[feats], test_transaction[feats]], axis=0)
        df_aggs = whole.groupby(feats[0])[feats[1]].agg(aggs).reset_index()
        df_aggs.columns = feats[:1] + [f'{col}_{feats[1]}_each_{feats[0]}' for col in df_aggs.columns[1:]]
        for col in df_aggs.columns[1:]:
            self.train[col] = pd.merge(train_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]
            self.test[col] = pd.merge(test_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]


class addr1_aggs_with_v126(Feature):
    def create_features(self):
        feats = ['addr1', 'V126']
        aggs = {'sum', 'max', 'min', 'mean', 'median', 'nunique'}
        whole = pd.concat([train_transaction[feats], test_transaction[feats]], axis=0)
        df_aggs = whole.groupby(feats[0])[feats[1]].agg(aggs).reset_index()
        df_aggs.columns = feats[:1] + [f'{col}_{feats[1]}_each_{feats[0]}' for col in df_aggs.columns[1:]]
        for col in df_aggs.columns[1:]:
            self.train[col] = pd.merge(train_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]
            self.test[col] = pd.merge(test_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]


class addr1_aggs_with_v127(Feature):
    def create_features(self):
        feats = ['addr1', 'V127']
        aggs = {'sum', 'max', 'min', 'mean', 'median', 'nunique'}
        whole = pd.concat([train_transaction[feats], test_transaction[feats]], axis=0)
        df_aggs = whole.groupby(feats[0])[feats[1]].agg(aggs).reset_index()
        df_aggs.columns = feats[:1] + [f'{col}_{feats[1]}_each_{feats[0]}' for col in df_aggs.columns[1:]]
        for col in df_aggs.columns[1:]:
            self.train[col] = pd.merge(train_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]
            self.test[col] = pd.merge(test_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]


class addr1_aggs_with_v128(Feature):
    def create_features(self):
        feats = ['addr1', 'V128']
        aggs = {'sum', 'max', 'min', 'mean', 'median', 'nunique'}
        whole = pd.concat([train_transaction[feats], test_transaction[feats]], axis=0)
        df_aggs = whole.groupby(feats[0])[feats[1]].agg(aggs).reset_index()
        df_aggs.columns = feats[:1] + [f'{col}_{feats[1]}_each_{feats[0]}' for col in df_aggs.columns[1:]]
        for col in df_aggs.columns[1:]:
            self.train[col] = pd.merge(train_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]
            self.test[col] = pd.merge(test_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]



class addr1_aggs_with_v129(Feature):
    def create_features(self):
        feats = ['addr1', 'V129']
        aggs = {'sum', 'max', 'min', 'mean', 'median', 'nunique'}
        whole = pd.concat([train_transaction[feats], test_transaction[feats]], axis=0)
        df_aggs = whole.groupby(feats[0])[feats[1]].agg(aggs).reset_index()
        df_aggs.columns = feats[:1] + [f'{col}_{feats[1]}_each_{feats[0]}' for col in df_aggs.columns[1:]]
        for col in df_aggs.columns[1:]:
            self.train[col] = pd.merge(train_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]
            self.test[col] = pd.merge(test_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]


class addr1_aggs_with_v130(Feature):
    def create_features(self):
        feats = ['addr1', 'V130']
        aggs = {'sum', 'max', 'min', 'mean', 'median', 'nunique'}
        whole = pd.concat([train_transaction[feats], test_transaction[feats]], axis=0)
        df_aggs = whole.groupby(feats[0])[feats[1]].agg(aggs).reset_index()
        df_aggs.columns = feats[:1] + [f'{col}_{feats[1]}_each_{feats[0]}' for col in df_aggs.columns[1:]]
        for col in df_aggs.columns[1:]:
            self.train[col] = pd.merge(train_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]
            self.test[col] = pd.merge(test_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]


class addr1_aggs_with_v131(Feature):
    def create_features(self):
        feats = ['addr1', 'V131']
        aggs = {'sum', 'max', 'min', 'mean', 'median', 'nunique'}
        whole = pd.concat([train_transaction[feats], test_transaction[feats]], axis=0)
        df_aggs = whole.groupby(feats[0])[feats[1]].agg(aggs).reset_index()
        df_aggs.columns = feats[:1] + [f'{col}_{feats[1]}_each_{feats[0]}' for col in df_aggs.columns[1:]]
        for col in df_aggs.columns[1:]:
            self.train[col] = pd.merge(train_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]
            self.test[col] = pd.merge(test_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]


class addr1_aggs_with_v132(Feature):
    def create_features(self):
        feats = ['addr1', 'V132']
        aggs = {'sum', 'max', 'min', 'mean', 'median', 'nunique'}
        whole = pd.concat([train_transaction[feats], test_transaction[feats]], axis=0)
        df_aggs = whole.groupby(feats[0])[feats[1]].agg(aggs).reset_index()
        df_aggs.columns = feats[:1] + [f'{col}_{feats[1]}_each_{feats[0]}' for col in df_aggs.columns[1:]]
        for col in df_aggs.columns[1:]:
            self.train[col] = pd.merge(train_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]
            self.test[col] = pd.merge(test_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]


class addr1_aggs_with_v133(Feature):
    def create_features(self):
        feats = ['addr1', 'V133']
        aggs = {'sum', 'max', 'min', 'mean', 'median', 'nunique'}
        whole = pd.concat([train_transaction[feats], test_transaction[feats]], axis=0)
        df_aggs = whole.groupby(feats[0])[feats[1]].agg(aggs).reset_index()
        df_aggs.columns = feats[:1] + [f'{col}_{feats[1]}_each_{feats[0]}' for col in df_aggs.columns[1:]]
        for col in df_aggs.columns[1:]:
            self.train[col] = pd.merge(train_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]
            self.test[col] = pd.merge(test_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]


class addr1_aggs_with_v134(Feature):
    def create_features(self):
        feats = ['addr1', 'V134']
        aggs = {'sum', 'max', 'min', 'mean', 'median', 'nunique'}
        whole = pd.concat([train_transaction[feats], test_transaction[feats]], axis=0)
        df_aggs = whole.groupby(feats[0])[feats[1]].agg(aggs).reset_index()
        df_aggs.columns = feats[:1] + [f'{col}_{feats[1]}_each_{feats[0]}' for col in df_aggs.columns[1:]]
        for col in df_aggs.columns[1:]:
            self.train[col] = pd.merge(train_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]
            self.test[col] = pd.merge(test_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]


class addr1_aggs_with_v135(Feature):
    def create_features(self):
        feats = ['addr1', 'V135']
        aggs = {'sum', 'max', 'min', 'mean', 'median', 'nunique'}
        whole = pd.concat([train_transaction[feats], test_transaction[feats]], axis=0)
        df_aggs = whole.groupby(feats[0])[feats[1]].agg(aggs).reset_index()
        df_aggs.columns = feats[:1] + [f'{col}_{feats[1]}_each_{feats[0]}' for col in df_aggs.columns[1:]]
        for col in df_aggs.columns[1:]:
            self.train[col] = pd.merge(train_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]
            self.test[col] = pd.merge(test_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]


class addr1_aggs_with_v136(Feature):
    def create_features(self):
        feats = ['addr1', 'V136']
        aggs = {'sum', 'max', 'min', 'mean', 'median', 'nunique'}
        whole = pd.concat([train_transaction[feats], test_transaction[feats]], axis=0)
        df_aggs = whole.groupby(feats[0])[feats[1]].agg(aggs).reset_index()
        df_aggs.columns = feats[:1] + [f'{col}_{feats[1]}_each_{feats[0]}' for col in df_aggs.columns[1:]]
        for col in df_aggs.columns[1:]:
            self.train[col] = pd.merge(train_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]
            self.test[col] = pd.merge(test_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]


class addr1_aggs_with_v137(Feature):
    def create_features(self):
        feats = ['addr1', 'V137']
        aggs = {'sum', 'max', 'min', 'mean', 'median', 'nunique'}
        whole = pd.concat([train_transaction[feats], test_transaction[feats]], axis=0)
        df_aggs = whole.groupby(feats[0])[feats[1]].agg(aggs).reset_index()
        df_aggs.columns = feats[:1] + [f'{col}_{feats[1]}_each_{feats[0]}' for col in df_aggs.columns[1:]]
        for col in df_aggs.columns[1:]:
            self.train[col] = pd.merge(train_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]
            self.test[col] = pd.merge(test_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]


class addr1_aggs_with_v150(Feature):
    def create_features(self):
        feats = ['addr1', 'V150']
        aggs = {'sum', 'max', 'min', 'mean', 'median', 'nunique'}
        whole = pd.concat([train_transaction[feats], test_transaction[feats]], axis=0)
        df_aggs = whole.groupby(feats[0])[feats[1]].agg(aggs).reset_index()
        df_aggs.columns = feats[:1] + [f'{col}_{feats[1]}_each_{feats[0]}' for col in df_aggs.columns[1:]]
        for col in df_aggs.columns[1:]:
            self.train[col] = pd.merge(train_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]
            self.test[col] = pd.merge(test_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]


class addr1_aggs_with_v159(Feature):
    def create_features(self):
        feats = ['addr1', 'V159']
        aggs = {'sum', 'max', 'min', 'mean', 'median', 'nunique'}
        whole = pd.concat([train_transaction[feats], test_transaction[feats]], axis=0)
        df_aggs = whole.groupby(feats[0])[feats[1]].agg(aggs).reset_index()
        df_aggs.columns = feats[:1] + [f'{col}_{feats[1]}_each_{feats[0]}' for col in df_aggs.columns[1:]]
        for col in df_aggs.columns[1:]:
            self.train[col] = pd.merge(train_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]
            self.test[col] = pd.merge(test_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]


class addr1_aggs_with_v160(Feature):
    def create_features(self):
        feats = ['addr1', 'V160']
        aggs = {'sum', 'max', 'min', 'mean', 'median', 'nunique'}
        whole = pd.concat([train_transaction[feats], test_transaction[feats]], axis=0)
        df_aggs = whole.groupby(feats[0])[feats[1]].agg(aggs).reset_index()
        df_aggs.columns = feats[:1] + [f'{col}_{feats[1]}_each_{feats[0]}' for col in df_aggs.columns[1:]]
        for col in df_aggs.columns[1:]:
            self.train[col] = pd.merge(train_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]
            self.test[col] = pd.merge(test_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]


class addr1_aggs_with_v166(Feature):
    def create_features(self):
        feats = ['addr1', 'V166']
        aggs = {'sum', 'max', 'min', 'mean', 'median', 'nunique'}
        whole = pd.concat([train_transaction[feats], test_transaction[feats]], axis=0)
        df_aggs = whole.groupby(feats[0])[feats[1]].agg(aggs).reset_index()
        df_aggs.columns = feats[:1] + [f'{col}_{feats[1]}_each_{feats[0]}' for col in df_aggs.columns[1:]]
        for col in df_aggs.columns[1:]:
            self.train[col] = pd.merge(train_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]
            self.test[col] = pd.merge(test_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]


class addr1_aggs_with_v202(Feature):
    def create_features(self):
        feats = ['addr1', 'V202']
        aggs = {'sum', 'max', 'min', 'mean', 'median', 'nunique'}
        whole = pd.concat([train_transaction[feats], test_transaction[feats]], axis=0)
        df_aggs = whole.groupby(feats[0])[feats[1]].agg(aggs).reset_index()
        df_aggs.columns = feats[:1] + [f'{col}_{feats[1]}_each_{feats[0]}' for col in df_aggs.columns[1:]]
        for col in df_aggs.columns[1:]:
            self.train[col] = pd.merge(train_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]
            self.test[col] = pd.merge(test_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]


class addr1_aggs_with_v203(Feature):
    def create_features(self):
        feats = ['addr1', 'V203']
        aggs = {'sum', 'max', 'min', 'mean', 'median', 'nunique'}
        whole = pd.concat([train_transaction[feats], test_transaction[feats]], axis=0)
        df_aggs = whole.groupby(feats[0])[feats[1]].agg(aggs).reset_index()
        df_aggs.columns = feats[:1] + [f'{col}_{feats[1]}_each_{feats[0]}' for col in df_aggs.columns[1:]]
        for col in df_aggs.columns[1:]:
            self.train[col] = pd.merge(train_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]
            self.test[col] = pd.merge(test_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]


class addr1_aggs_with_v204(Feature):
    def create_features(self):
        feats = ['addr1', 'V204']
        aggs = {'sum', 'max', 'min', 'mean', 'median', 'nunique'}
        whole = pd.concat([train_transaction[feats], test_transaction[feats]], axis=0)
        df_aggs = whole.groupby(feats[0])[feats[1]].agg(aggs).reset_index()
        df_aggs.columns = feats[:1] + [f'{col}_{feats[1]}_each_{feats[0]}' for col in df_aggs.columns[1:]]
        for col in df_aggs.columns[1:]:
            self.train[col] = pd.merge(train_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]
            self.test[col] = pd.merge(test_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]


class addr1_aggs_with_v207(Feature):
    def create_features(self):
        feats = ['addr1', 'V207']
        aggs = {'sum', 'max', 'min', 'mean', 'median', 'nunique'}
        whole = pd.concat([train_transaction[feats], test_transaction[feats]], axis=0)
        df_aggs = whole.groupby(feats[0])[feats[1]].agg(aggs).reset_index()
        df_aggs.columns = feats[:1] + [f'{col}_{feats[1]}_each_{feats[0]}' for col in df_aggs.columns[1:]]
        for col in df_aggs.columns[1:]:
            self.train[col] = pd.merge(train_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]
            self.test[col] = pd.merge(test_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]


class addr1_aggs_with_v208(Feature):
    def create_features(self):
        feats = ['addr1', 'V208']
        aggs = {'sum', 'max', 'min', 'mean', 'median', 'nunique'}
        whole = pd.concat([train_transaction[feats], test_transaction[feats]], axis=0)
        df_aggs = whole.groupby(feats[0])[feats[1]].agg(aggs).reset_index()
        df_aggs.columns = feats[:1] + [f'{col}_{feats[1]}_each_{feats[0]}' for col in df_aggs.columns[1:]]
        for col in df_aggs.columns[1:]:
            self.train[col] = pd.merge(train_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]
            self.test[col] = pd.merge(test_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]


class addr1_aggs_with_v209(Feature):
    def create_features(self):
        feats = ['addr1', 'V209']
        aggs = {'sum', 'max', 'min', 'mean', 'median', 'nunique'}
        whole = pd.concat([train_transaction[feats], test_transaction[feats]], axis=0)
        df_aggs = whole.groupby(feats[0])[feats[1]].agg(aggs).reset_index()
        df_aggs.columns = feats[:1] + [f'{col}_{feats[1]}_each_{feats[0]}' for col in df_aggs.columns[1:]]
        for col in df_aggs.columns[1:]:
            self.train[col] = pd.merge(train_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]
            self.test[col] = pd.merge(test_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]


class addr1_aggs_with_v210(Feature):
    def create_features(self):
        feats = ['addr1', 'V210']
        aggs = {'sum', 'max', 'min', 'mean', 'median', 'nunique'}
        whole = pd.concat([train_transaction[feats], test_transaction[feats]], axis=0)
        df_aggs = whole.groupby(feats[0])[feats[1]].agg(aggs).reset_index()
        df_aggs.columns = feats[:1] + [f'{col}_{feats[1]}_each_{feats[0]}' for col in df_aggs.columns[1:]]
        for col in df_aggs.columns[1:]:
            self.train[col] = pd.merge(train_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]
            self.test[col] = pd.merge(test_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]


class addr1_aggs_with_v212(Feature):
    def create_features(self):
        feats = ['addr1', 'V212']
        aggs = {'sum', 'max', 'min', 'mean', 'median', 'nunique'}
        whole = pd.concat([train_transaction[feats], test_transaction[feats]], axis=0)
        df_aggs = whole.groupby(feats[0])[feats[1]].agg(aggs).reset_index()
        df_aggs.columns = feats[:1] + [f'{col}_{feats[1]}_each_{feats[0]}' for col in df_aggs.columns[1:]]
        for col in df_aggs.columns[1:]:
            self.train[col] = pd.merge(train_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]
            self.test[col] = pd.merge(test_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]

class addr1_aggs_with_v214(Feature):
    def create_features(self):
        feats = ['addr1', 'V214']
        aggs = {'sum', 'max', 'min', 'mean', 'median', 'nunique'}
        whole = pd.concat([train_transaction[feats], test_transaction[feats]], axis=0)
        df_aggs = whole.groupby(feats[0])[feats[1]].agg(aggs).reset_index()
        df_aggs.columns = feats[:1] + [f'{col}_{feats[1]}_each_{feats[0]}' for col in df_aggs.columns[1:]]
        for col in df_aggs.columns[1:]:
            self.train[col] = pd.merge(train_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]
            self.test[col] = pd.merge(test_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]


class addr1_aggs_with_v215(Feature):
    def create_features(self):
        feats = ['addr1', 'V215']
        aggs = {'sum', 'max', 'min', 'mean', 'median', 'nunique'}
        whole = pd.concat([train_transaction[feats], test_transaction[feats]], axis=0)
        df_aggs = whole.groupby(feats[0])[feats[1]].agg(aggs).reset_index()
        df_aggs.columns = feats[:1] + [f'{col}_{feats[1]}_each_{feats[0]}' for col in df_aggs.columns[1:]]
        for col in df_aggs.columns[1:]:
            self.train[col] = pd.merge(train_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]
            self.test[col] = pd.merge(test_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]


class addr1_aggs_with_v216(Feature):
    def create_features(self):
        feats = ['addr1', 'V216']
        aggs = {'sum', 'max', 'min', 'mean', 'median', 'nunique'}
        whole = pd.concat([train_transaction[feats], test_transaction[feats]], axis=0)
        df_aggs = whole.groupby(feats[0])[feats[1]].agg(aggs).reset_index()
        df_aggs.columns = feats[:1] + [f'{col}_{feats[1]}_each_{feats[0]}' for col in df_aggs.columns[1:]]
        for col in df_aggs.columns[1:]:
            self.train[col] = pd.merge(train_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]
            self.test[col] = pd.merge(test_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]


class addr1_aggs_with_v263(Feature):
    def create_features(self):
        feats = ['addr1', 'V263']
        aggs = {'sum', 'max', 'min', 'mean', 'median', 'nunique'}
        whole = pd.concat([train_transaction[feats], test_transaction[feats]], axis=0)
        df_aggs = whole.groupby(feats[0])[feats[1]].agg(aggs).reset_index()
        df_aggs.columns = feats[:1] + [f'{col}_{feats[1]}_each_{feats[0]}' for col in df_aggs.columns[1:]]
        for col in df_aggs.columns[1:]:
            self.train[col] = pd.merge(train_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]
            self.test[col] = pd.merge(test_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]


class addr1_aggs_with_v264(Feature):
    def create_features(self):
        feats = ['addr1', 'V264']
        aggs = {'sum', 'max', 'min', 'mean', 'median', 'nunique'}
        whole = pd.concat([train_transaction[feats], test_transaction[feats]], axis=0)
        df_aggs = whole.groupby(feats[0])[feats[1]].agg(aggs).reset_index()
        df_aggs.columns = feats[:1] + [f'{col}_{feats[1]}_each_{feats[0]}' for col in df_aggs.columns[1:]]
        for col in df_aggs.columns[1:]:
            self.train[col] = pd.merge(train_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]
            self.test[col] = pd.merge(test_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]


class addr1_aggs_with_v265(Feature):
    def create_features(self):
        feats = ['addr1', 'V265']
        aggs = {'sum', 'max', 'min', 'mean', 'median', 'nunique'}
        whole = pd.concat([train_transaction[feats], test_transaction[feats]], axis=0)
        df_aggs = whole.groupby(feats[0])[feats[1]].agg(aggs).reset_index()
        df_aggs.columns = feats[:1] + [f'{col}_{feats[1]}_each_{feats[0]}' for col in df_aggs.columns[1:]]
        for col in df_aggs.columns[1:]:
            self.train[col] = pd.merge(train_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]
            self.test[col] = pd.merge(test_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]


class addr1_aggs_with_v267(Feature):
    def create_features(self):
        feats = ['addr1', 'V267']
        aggs = {'sum', 'max', 'min', 'mean', 'median', 'nunique'}
        whole = pd.concat([train_transaction[feats], test_transaction[feats]], axis=0)
        df_aggs = whole.groupby(feats[0])[feats[1]].agg(aggs).reset_index()
        df_aggs.columns = feats[:1] + [f'{col}_{feats[1]}_each_{feats[0]}' for col in df_aggs.columns[1:]]
        for col in df_aggs.columns[1:]:
            self.train[col] = pd.merge(train_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]
            self.test[col] = pd.merge(test_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]


class addr1_aggs_with_v274(Feature):
    def create_features(self):
        feats = ['addr1', 'V274']
        aggs = {'sum', 'max', 'min', 'mean', 'median', 'nunique'}
        whole = pd.concat([train_transaction[feats], test_transaction[feats]], axis=0)
        df_aggs = whole.groupby(feats[0])[feats[1]].agg(aggs).reset_index()
        df_aggs.columns = feats[:1] + [f'{col}_{feats[1]}_each_{feats[0]}' for col in df_aggs.columns[1:]]
        for col in df_aggs.columns[1:]:
            self.train[col] = pd.merge(train_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]
            self.test[col] = pd.merge(test_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]


class addr1_aggs_with_v275(Feature):
    def create_features(self):
        feats = ['addr1', 'V275']
        aggs = {'sum', 'max', 'min', 'mean', 'median', 'nunique'}
        whole = pd.concat([train_transaction[feats], test_transaction[feats]], axis=0)
        df_aggs = whole.groupby(feats[0])[feats[1]].agg(aggs).reset_index()
        df_aggs.columns = feats[:1] + [f'{col}_{feats[1]}_each_{feats[0]}' for col in df_aggs.columns[1:]]
        for col in df_aggs.columns[1:]:
            self.train[col] = pd.merge(train_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]
            self.test[col] = pd.merge(test_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]


class addr1_aggs_with_v276(Feature):
    def create_features(self):
        feats = ['addr1', 'V276']
        aggs = {'sum', 'max', 'min', 'mean', 'median', 'nunique'}
        whole = pd.concat([train_transaction[feats], test_transaction[feats]], axis=0)
        df_aggs = whole.groupby(feats[0])[feats[1]].agg(aggs).reset_index()
        df_aggs.columns = feats[:1] + [f'{col}_{feats[1]}_each_{feats[0]}' for col in df_aggs.columns[1:]]
        for col in df_aggs.columns[1:]:
            self.train[col] = pd.merge(train_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]
            self.test[col] = pd.merge(test_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]


class addr1_aggs_with_v277(Feature):
    def create_features(self):
        feats = ['addr1', 'V277']
        aggs = {'sum', 'max', 'min', 'mean', 'median', 'nunique'}
        whole = pd.concat([train_transaction[feats], test_transaction[feats]], axis=0)
        df_aggs = whole.groupby(feats[0])[feats[1]].agg(aggs).reset_index()
        df_aggs.columns = feats[:1] + [f'{col}_{feats[1]}_each_{feats[0]}' for col in df_aggs.columns[1:]]
        for col in df_aggs.columns[1:]:
            self.train[col] = pd.merge(train_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]
            self.test[col] = pd.merge(test_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]


class addr1_aggs_with_v278(Feature):
    def create_features(self):
        feats = ['addr1', 'V278']
        aggs = {'sum', 'max', 'min', 'mean', 'median', 'nunique'}
        whole = pd.concat([train_transaction[feats], test_transaction[feats]], axis=0)
        df_aggs = whole.groupby(feats[0])[feats[1]].agg(aggs).reset_index()
        df_aggs.columns = feats[:1] + [f'{col}_{feats[1]}_each_{feats[0]}' for col in df_aggs.columns[1:]]
        for col in df_aggs.columns[1:]:
            self.train[col] = pd.merge(train_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]
            self.test[col] = pd.merge(test_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]


class addr1_aggs_with_v280(Feature):
    def create_features(self):
        feats = ['addr1', 'V280']
        aggs = {'sum', 'max', 'min', 'mean', 'median', 'nunique'}
        whole = pd.concat([train_transaction[feats], test_transaction[feats]], axis=0)
        df_aggs = whole.groupby(feats[0])[feats[1]].agg(aggs).reset_index()
        df_aggs.columns = feats[:1] + [f'{col}_{feats[1]}_each_{feats[0]}' for col in df_aggs.columns[1:]]
        for col in df_aggs.columns[1:]:
            self.train[col] = pd.merge(train_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]
            self.test[col] = pd.merge(test_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]


class addr1_aggs_with_v294(Feature):
    def create_features(self):
        feats = ['addr1', 'V294']
        aggs = {'sum', 'max', 'min', 'mean', 'median', 'nunique'}
        whole = pd.concat([train_transaction[feats], test_transaction[feats]], axis=0)
        df_aggs = whole.groupby(feats[0])[feats[1]].agg(aggs).reset_index()
        df_aggs.columns = feats[:1] + [f'{col}_{feats[1]}_each_{feats[0]}' for col in df_aggs.columns[1:]]
        for col in df_aggs.columns[1:]:
            self.train[col] = pd.merge(train_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]
            self.test[col] = pd.merge(test_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]


class addr1_aggs_with_v306(Feature):
    def create_features(self):
        feats = ['addr1', 'V306']
        aggs = {'sum', 'max', 'min', 'mean', 'median', 'nunique'}
        whole = pd.concat([train_transaction[feats], test_transaction[feats]], axis=0)
        df_aggs = whole.groupby(feats[0])[feats[1]].agg(aggs).reset_index()
        df_aggs.columns = feats[:1] + [f'{col}_{feats[1]}_each_{feats[0]}' for col in df_aggs.columns[1:]]
        for col in df_aggs.columns[1:]:
            self.train[col] = pd.merge(train_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]
            self.test[col] = pd.merge(test_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]


class addr1_aggs_with_v307(Feature):
    def create_features(self):
        feats = ['addr1', 'V307']
        aggs = {'sum', 'max', 'min', 'mean', 'median', 'nunique'}
        whole = pd.concat([train_transaction[feats], test_transaction[feats]], axis=0)
        df_aggs = whole.groupby(feats[0])[feats[1]].agg(aggs).reset_index()
        df_aggs.columns = feats[:1] + [f'{col}_{feats[1]}_each_{feats[0]}' for col in df_aggs.columns[1:]]
        for col in df_aggs.columns[1:]:
            self.train[col] = pd.merge(train_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]
            self.test[col] = pd.merge(test_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]


class addr1_aggs_with_v308(Feature):
    def create_features(self):
        feats = ['addr1', 'V308']
        aggs = {'sum', 'max', 'min', 'mean', 'median', 'nunique'}
        whole = pd.concat([train_transaction[feats], test_transaction[feats]], axis=0)
        df_aggs = whole.groupby(feats[0])[feats[1]].agg(aggs).reset_index()
        df_aggs.columns = feats[:1] + [f'{col}_{feats[1]}_each_{feats[0]}' for col in df_aggs.columns[1:]]
        for col in df_aggs.columns[1:]:
            self.train[col] = pd.merge(train_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]
            self.test[col] = pd.merge(test_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]


class addr1_aggs_with_v309(Feature):
    def create_features(self):
        feats = ['addr1', 'V309']
        aggs = {'sum', 'max', 'min', 'mean', 'median', 'nunique'}
        whole = pd.concat([train_transaction[feats], test_transaction[feats]], axis=0)
        df_aggs = whole.groupby(feats[0])[feats[1]].agg(aggs).reset_index()
        df_aggs.columns = feats[:1] + [f'{col}_{feats[1]}_each_{feats[0]}' for col in df_aggs.columns[1:]]
        for col in df_aggs.columns[1:]:
            self.train[col] = pd.merge(train_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]
            self.test[col] = pd.merge(test_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]

class addr1_aggs_with_v310(Feature):
    def create_features(self):
        feats = ['addr1', 'V310']
        aggs = {'sum', 'max', 'min', 'mean', 'median', 'nunique'}
        whole = pd.concat([train_transaction[feats], test_transaction[feats]], axis=0)
        df_aggs = whole.groupby(feats[0])[feats[1]].agg(aggs).reset_index()
        df_aggs.columns = feats[:1] + [f'{col}_{feats[1]}_each_{feats[0]}' for col in df_aggs.columns[1:]]
        for col in df_aggs.columns[1:]:
            self.train[col] = pd.merge(train_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]
            self.test[col] = pd.merge(test_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]


class addr1_aggs_with_v311(Feature):
    def create_features(self):
        feats = ['addr1', 'V311']
        aggs = {'sum', 'max', 'min', 'mean', 'median', 'nunique'}
        whole = pd.concat([train_transaction[feats], test_transaction[feats]], axis=0)
        df_aggs = whole.groupby(feats[0])[feats[1]].agg(aggs).reset_index()
        df_aggs.columns = feats[:1] + [f'{col}_{feats[1]}_each_{feats[0]}' for col in df_aggs.columns[1:]]
        for col in df_aggs.columns[1:]:
            self.train[col] = pd.merge(train_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]
            self.test[col] = pd.merge(test_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]


class addr1_aggs_with_v312(Feature):
    def create_features(self):
        feats = ['addr1', 'V312']
        aggs = {'sum', 'max', 'min', 'mean', 'median', 'nunique'}
        whole = pd.concat([train_transaction[feats], test_transaction[feats]], axis=0)
        df_aggs = whole.groupby(feats[0])[feats[1]].agg(aggs).reset_index()
        df_aggs.columns = feats[:1] + [f'{col}_{feats[1]}_each_{feats[0]}' for col in df_aggs.columns[1:]]
        for col in df_aggs.columns[1:]:
            self.train[col] = pd.merge(train_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]
            self.test[col] = pd.merge(test_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]


class addr1_aggs_with_v313(Feature):
    def create_features(self):
        feats = ['addr1', 'V313']
        aggs = {'sum', 'max', 'min', 'mean', 'median', 'nunique'}
        whole = pd.concat([train_transaction[feats], test_transaction[feats]], axis=0)
        df_aggs = whole.groupby(feats[0])[feats[1]].agg(aggs).reset_index()
        df_aggs.columns = feats[:1] + [f'{col}_{feats[1]}_each_{feats[0]}' for col in df_aggs.columns[1:]]
        for col in df_aggs.columns[1:]:
            self.train[col] = pd.merge(train_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]
            self.test[col] = pd.merge(test_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]


class addr1_aggs_with_v314(Feature):
    def create_features(self):
        feats = ['addr1', 'V314']
        aggs = {'sum', 'max', 'min', 'mean', 'median', 'nunique'}
        whole = pd.concat([train_transaction[feats], test_transaction[feats]], axis=0)
        df_aggs = whole.groupby(feats[0])[feats[1]].agg(aggs).reset_index()
        df_aggs.columns = feats[:1] + [f'{col}_{feats[1]}_each_{feats[0]}' for col in df_aggs.columns[1:]]
        for col in df_aggs.columns[1:]:
            self.train[col] = pd.merge(train_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]
            self.test[col] = pd.merge(test_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]


class addr1_aggs_with_v315(Feature):
    def create_features(self):
        feats = ['addr1', 'V315']
        aggs = {'sum', 'max', 'min', 'mean', 'median', 'nunique'}
        whole = pd.concat([train_transaction[feats], test_transaction[feats]], axis=0)
        df_aggs = whole.groupby(feats[0])[feats[1]].agg(aggs).reset_index()
        df_aggs.columns = feats[:1] + [f'{col}_{feats[1]}_each_{feats[0]}' for col in df_aggs.columns[1:]]
        for col in df_aggs.columns[1:]:
            self.train[col] = pd.merge(train_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]
            self.test[col] = pd.merge(test_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]


class addr1_aggs_with_v316(Feature):
    def create_features(self):
        feats = ['addr1', 'V316']
        aggs = {'sum', 'max', 'min', 'mean', 'median', 'nunique'}
        whole = pd.concat([train_transaction[feats], test_transaction[feats]], axis=0)
        df_aggs = whole.groupby(feats[0])[feats[1]].agg(aggs).reset_index()
        df_aggs.columns = feats[:1] + [f'{col}_{feats[1]}_each_{feats[0]}' for col in df_aggs.columns[1:]]
        for col in df_aggs.columns[1:]:
            self.train[col] = pd.merge(train_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]
            self.test[col] = pd.merge(test_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]


class addr1_aggs_with_v317(Feature):
    def create_features(self):
        feats = ['addr1', 'V317']
        aggs = {'sum', 'max', 'min', 'mean', 'median', 'nunique'}
        whole = pd.concat([train_transaction[feats], test_transaction[feats]], axis=0)
        df_aggs = whole.groupby(feats[0])[feats[1]].agg(aggs).reset_index()
        df_aggs.columns = feats[:1] + [f'{col}_{feats[1]}_each_{feats[0]}' for col in df_aggs.columns[1:]]
        for col in df_aggs.columns[1:]:
            self.train[col] = pd.merge(train_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]
            self.test[col] = pd.merge(test_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]


class addr1_aggs_with_v318(Feature):
    def create_features(self):
        feats = ['addr1', 'V318']
        aggs = {'sum', 'max', 'min', 'mean', 'median', 'nunique'}
        whole = pd.concat([train_transaction[feats], test_transaction[feats]], axis=0)
        df_aggs = whole.groupby(feats[0])[feats[1]].agg(aggs).reset_index()
        df_aggs.columns = feats[:1] + [f'{col}_{feats[1]}_each_{feats[0]}' for col in df_aggs.columns[1:]]
        for col in df_aggs.columns[1:]:
            self.train[col] = pd.merge(train_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]
            self.test[col] = pd.merge(test_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]


class addr1_aggs_with_v319(Feature):
    def create_features(self):
        feats = ['addr1', 'V319']
        aggs = {'sum', 'max', 'min', 'mean', 'median', 'nunique'}
        whole = pd.concat([train_transaction[feats], test_transaction[feats]], axis=0)
        df_aggs = whole.groupby(feats[0])[feats[1]].agg(aggs).reset_index()
        df_aggs.columns = feats[:1] + [f'{col}_{feats[1]}_each_{feats[0]}' for col in df_aggs.columns[1:]]
        for col in df_aggs.columns[1:]:
            self.train[col] = pd.merge(train_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]
            self.test[col] = pd.merge(test_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]


class addr1_aggs_with_v320(Feature):
    def create_features(self):
        feats = ['addr1', 'V320']
        aggs = {'sum', 'max', 'min', 'mean', 'median', 'nunique'}
        whole = pd.concat([train_transaction[feats], test_transaction[feats]], axis=0)
        df_aggs = whole.groupby(feats[0])[feats[1]].agg(aggs).reset_index()
        df_aggs.columns = feats[:1] + [f'{col}_{feats[1]}_each_{feats[0]}' for col in df_aggs.columns[1:]]
        for col in df_aggs.columns[1:]:
            self.train[col] = pd.merge(train_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]
            self.test[col] = pd.merge(test_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]


class addr1_aggs_with_v321(Feature):
    def create_features(self):
        feats = ['addr1', 'V321']
        aggs = {'sum', 'max', 'min', 'mean', 'median', 'nunique'}
        whole = pd.concat([train_transaction[feats], test_transaction[feats]], axis=0)
        df_aggs = whole.groupby(feats[0])[feats[1]].agg(aggs).reset_index()
        df_aggs.columns = feats[:1] + [f'{col}_{feats[1]}_each_{feats[0]}' for col in df_aggs.columns[1:]]
        for col in df_aggs.columns[1:]:
            self.train[col] = pd.merge(train_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]
            self.test[col] = pd.merge(test_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]


class addr1_aggs_with_v332(Feature):
    def create_features(self):
        feats = ['addr1', 'V332']
        aggs = {'sum', 'max', 'min', 'mean', 'median', 'nunique'}
        whole = pd.concat([train_transaction[feats], test_transaction[feats]], axis=0)
        df_aggs = whole.groupby(feats[0])[feats[1]].agg(aggs).reset_index()
        df_aggs.columns = feats[:1] + [f'{col}_{feats[1]}_each_{feats[0]}' for col in df_aggs.columns[1:]]
        for col in df_aggs.columns[1:]:
            self.train[col] = pd.merge(train_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]
            self.test[col] = pd.merge(test_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]


class addr1_aggs_with_v333(Feature):
    def create_features(self):
        feats = ['addr1', 'V333']
        aggs = {'sum', 'max', 'min', 'mean', 'median', 'nunique'}
        whole = pd.concat([train_transaction[feats], test_transaction[feats]], axis=0)
        df_aggs = whole.groupby(feats[0])[feats[1]].agg(aggs).reset_index()
        df_aggs.columns = feats[:1] + [f'{col}_{feats[1]}_each_{feats[0]}' for col in df_aggs.columns[1:]]
        for col in df_aggs.columns[1:]:
            self.train[col] = pd.merge(train_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]
            self.test[col] = pd.merge(test_transaction[feats[0]], df_aggs[[feats[0], col]], on=feats[0], how='left')[col]


class card1_card2_aggs_with_transaction_amt(Feature):
    def create_features(self):
        feats = ['card1', 'card2', 'TransactionAmt']
        by_ = f'{feats[0]}_{feats[1]}'
        len_ = len(train_transaction)
        aggs = {'sum', 'max', 'min', 'mean', 'median', 'nunique'}
        whole = pd.concat([train_transaction[feats], test_transaction[feats]], axis=0)
        whole['card1_card2'] = whole['card1'].astype(str) + '_' + whole['card2'].astype(str)
        df_aggs = whole.groupby(by_)[feats[-1]].agg(aggs).reset_index()
        df_aggs.columns = [by_] + [f'{col}_{feats[-1]}_each_{by_}' for col in df_aggs.columns[1:]]
        for col in df_aggs.columns[1:]:
            self.train[col] = pd.merge(whole, df_aggs, on=by_, how='left')[col].values[:len_]
            self.test[col] = pd.merge(whole, df_aggs, on=by_, how='left')[col].values[len_:]


class card1_card3_aggs_with_transaction_amt(Feature):
    def create_features(self):
        feats = ['card1', 'card3', 'TransactionAmt']
        by_ = f'{feats[0]}_{feats[1]}'
        len_ = len(train_transaction)
        aggs = {'sum', 'max', 'min', 'mean', 'median', 'nunique'}
        whole = pd.concat([train_transaction[feats], test_transaction[feats]], axis=0)
        whole['card1_card3'] = whole['card1'].astype(str) + '_' + whole['card3'].astype(str)
        df_aggs = whole.groupby(by_)[feats[-1]].agg(aggs).reset_index()
        df_aggs.columns = [by_] + [f'{col}_{feats[-1]}_each_{by_}' for col in df_aggs.columns[1:]]
        for col in df_aggs.columns[1:]:
            self.train[col] = pd.merge(whole, df_aggs, on=by_, how='left')[col].values[:len_]
            self.test[col] = pd.merge(whole, df_aggs, on=by_, how='left')[col].values[len_:]


class card1_card5_aggs_with_transaction_amt(Feature):
    def create_features(self):
        feats = ['card1', 'card5', 'TransactionAmt']
        by_ = f'{feats[0]}_{feats[1]}'
        len_ = len(train_transaction)
        aggs = {'sum', 'max', 'min', 'mean', 'median', 'nunique'}
        whole = pd.concat([train_transaction[feats], test_transaction[feats]], axis=0)
        whole['card1_card5'] = whole['card1'].astype(str) + '_' + whole['card5'].astype(str)
        df_aggs = whole.groupby(by_)[feats[-1]].agg(aggs).reset_index()
        df_aggs.columns = [by_] + [f'{col}_{feats[-1]}_each_{by_}' for col in df_aggs.columns[1:]]
        for col in df_aggs.columns[1:]:
            self.train[col] = pd.merge(whole, df_aggs, on=by_, how='left')[col].values[:len_]
            self.test[col] = pd.merge(whole, df_aggs, on=by_, how='left')[col].values[len_:]


class card1_P_emaildomain_aggs_with_transaction_amt(Feature):
    def create_features(self):
        feats = ['card1', 'P_emaildomain', 'TransactionAmt']
        by_ = f'{feats[0]}_{feats[1]}'
        len_ = len(train_transaction)
        aggs = {'sum', 'max', 'min', 'mean', 'median', 'nunique'}
        whole = pd.concat([train_transaction[feats], test_transaction[feats]], axis=0)
        whole['P_emaildomain'] = whole['P_emaildomain'].map(email['emails'])
        whole['card1_P_emaildomain'] = whole['card1'].astype(str) + '_' + whole['P_emaildomain'].astype(str)
        df_aggs = whole.groupby(by_)[feats[-1]].agg(aggs).reset_index()
        df_aggs.columns = [by_] + [f'{col}_{feats[-1]}_each_{by_}' for col in df_aggs.columns[1:]]
        for col in df_aggs.columns[1:]:
            self.train[col] = pd.merge(whole, df_aggs, on=by_, how='left')[col].values[:len_]
            self.test[col] = pd.merge(whole, df_aggs, on=by_, how='left')[col].values[len_:]


class card1_R_emaildomain_aggs_with_transaction_amt(Feature):
    def create_features(self):
        feats = ['card1', 'R_emaildomain', 'TransactionAmt']
        by_ = f'{feats[0]}_{feats[1]}'
        len_ = len(train_transaction)
        aggs = {'sum', 'max', 'min', 'mean', 'median', 'nunique'}
        whole = pd.concat([train_transaction[feats], test_transaction[feats]], axis=0)
        whole['R_emaildomain'] = whole['R_emaildomain'].map(email['emails'])
        whole['card1_R_emaildomain'] = whole['card1'].astype(str) + '_' + whole['R_emaildomain'].astype(str)
        df_aggs = whole.groupby(by_)[feats[-1]].agg(aggs).reset_index()
        df_aggs.columns = [by_] + [f'{col}_{feats[-1]}_each_{by_}' for col in df_aggs.columns[1:]]
        for col in df_aggs.columns[1:]:
            self.train[col] = pd.merge(whole, df_aggs, on=by_, how='left')[col].values[:len_]
            self.test[col] = pd.merge(whole, df_aggs, on=by_, how='left')[col].values[len_:]


class card2_card3_aggs_with_transaction_amt(Feature):
    def create_features(self):
        feats = ['card2', 'card3', 'TransactionAmt']
        by_ = f'{feats[0]}_{feats[1]}'
        len_ = len(train_transaction)
        aggs = {'sum', 'max', 'min', 'mean', 'median', 'nunique'}
        whole = pd.concat([train_transaction[feats], test_transaction[feats]], axis=0)
        whole['card2_card3'] = whole['card2'].astype(str) + '_' + whole['card3'].astype(str)
        df_aggs = whole.groupby(by_)[feats[-1]].agg(aggs).reset_index()
        df_aggs.columns = [by_] + [f'{col}_{feats[-1]}_each_{by_}' for col in df_aggs.columns[1:]]
        for col in df_aggs.columns[1:]:
            self.train[col] = pd.merge(whole, df_aggs, on=by_, how='left')[col].values[:len_]
            self.test[col] = pd.merge(whole, df_aggs, on=by_, how='left')[col].values[len_:]


class card2_card5_aggs_with_transaction_amt(Feature):
    def create_features(self):
        feats = ['card2', 'card5', 'TransactionAmt']
        by_ = f'{feats[0]}_{feats[1]}'
        len_ = len(train_transaction)
        aggs = {'sum', 'max', 'min', 'mean', 'median', 'nunique'}
        whole = pd.concat([train_transaction[feats], test_transaction[feats]], axis=0)
        whole['card2_card5'] = whole['card2'].astype(str) + '_' + whole['card5'].astype(str)
        df_aggs = whole.groupby(by_)[feats[-1]].agg(aggs).reset_index()
        df_aggs.columns = [by_] + [f'{col}_{feats[-1]}_each_{by_}' for col in df_aggs.columns[1:]]
        for col in df_aggs.columns[1:]:
            self.train[col] = pd.merge(whole, df_aggs, on=by_, how='left')[col].values[:len_]
            self.test[col] = pd.merge(whole, df_aggs, on=by_, how='left')[col].values[len_:]


class card2_P_emaildomain_aggs_with_transaction_amt(Feature):
    def create_features(self):
        feats = ['card2', 'P_emaildomain', 'TransactionAmt']
        by_ = f'{feats[0]}_{feats[1]}'
        len_ = len(train_transaction)
        aggs = {'sum', 'max', 'min', 'mean', 'median', 'nunique'}
        whole = pd.concat([train_transaction[feats], test_transaction[feats]], axis=0)
        whole['P_emaildomain'] = whole['P_emaildomain'].map(email['emails'])
        whole['card2_P_emaildomain'] = whole['card2'].astype(str) + '_' + whole['P_emaildomain'].astype(str)
        df_aggs = whole.groupby(by_)[feats[-1]].agg(aggs).reset_index()
        df_aggs.columns = [by_] + [f'{col}_{feats[-1]}_each_{by_}' for col in df_aggs.columns[1:]]
        for col in df_aggs.columns[1:]:
            self.train[col] = pd.merge(whole, df_aggs, on=by_, how='left')[col].values[:len_]
            self.test[col] = pd.merge(whole, df_aggs, on=by_, how='left')[col].values[len_:]


class card2_R_emaildomain_aggs_with_transaction_amt(Feature):
    def create_features(self):
        feats = ['card2', 'R_emaildomain', 'TransactionAmt']
        by_ = f'{feats[0]}_{feats[1]}'
        len_ = len(train_transaction)
        aggs = {'sum', 'max', 'min', 'mean', 'median', 'nunique'}
        whole = pd.concat([train_transaction[feats], test_transaction[feats]], axis=0)
        whole['R_emaildomain'] = whole['R_emaildomain'].map(email['emails'])
        whole['card2_R_emaildomain'] = whole['card2'].astype(str) + '_' + whole['R_emaildomain'].astype(str)
        df_aggs = whole.groupby(by_)[feats[-1]].agg(aggs).reset_index()
        df_aggs.columns = [by_] + [f'{col}_{feats[-1]}_each_{by_}' for col in df_aggs.columns[1:]]
        for col in df_aggs.columns[1:]:
            self.train[col] = pd.merge(whole, df_aggs, on=by_, how='left')[col].values[:len_]
            self.test[col] = pd.merge(whole, df_aggs, on=by_, how='left')[col].values[len_:]


class card3_card5_aggs_with_transaction_amt(Feature):
    def create_features(self):
        feats = ['card3', 'card5', 'TransactionAmt']
        by_ = f'{feats[0]}_{feats[1]}'
        len_ = len(train_transaction)
        aggs = {'sum', 'max', 'min', 'mean', 'median', 'nunique'}
        whole = pd.concat([train_transaction[feats], test_transaction[feats]], axis=0)
        whole['card3_card5'] = whole['card3'].astype(str) + '_' + whole['card5'].astype(str)
        df_aggs = whole.groupby(by_)[feats[-1]].agg(aggs).reset_index()
        df_aggs.columns = [by_] + [f'{col}_{feats[-1]}_each_{by_}' for col in df_aggs.columns[1:]]
        for col in df_aggs.columns[1:]:
            self.train[col] = pd.merge(whole, df_aggs, on=by_, how='left')[col].values[:len_]
            self.test[col] = pd.merge(whole, df_aggs, on=by_, how='left')[col].values[len_:]


class card3_P_emaildomain_aggs_with_transaction_amt(Feature):
    def create_features(self):
        feats = ['card3', 'P_emaildomain', 'TransactionAmt']
        by_ = f'{feats[0]}_{feats[1]}'
        len_ = len(train_transaction)
        aggs = {'sum', 'max', 'min', 'mean', 'median', 'nunique'}
        whole = pd.concat([train_transaction[feats], test_transaction[feats]], axis=0)
        whole['P_emaildomain'] = whole['P_emaildomain'].map(email['emails'])
        whole['card3_P_emaildomain'] = whole['card3'].astype(str) + '_' + whole['P_emaildomain'].astype(str)
        df_aggs = whole.groupby(by_)[feats[-1]].agg(aggs).reset_index()
        df_aggs.columns = [by_] + [f'{col}_{feats[-1]}_each_{by_}' for col in df_aggs.columns[1:]]
        for col in df_aggs.columns[1:]:
            self.train[col] = pd.merge(whole, df_aggs, on=by_, how='left')[col].values[:len_]
            self.test[col] = pd.merge(whole, df_aggs, on=by_, how='left')[col].values[len_:]


class card3_R_emaildomain_aggs_with_transaction_amt(Feature):
    def create_features(self):
        feats = ['card3', 'R_emaildomain', 'TransactionAmt']
        by_ = f'{feats[0]}_{feats[1]}'
        len_ = len(train_transaction)
        aggs = {'sum', 'max', 'min', 'mean', 'median', 'nunique'}
        whole = pd.concat([train_transaction[feats], test_transaction[feats]], axis=0)
        whole['R_emaildomain'] = whole['R_emaildomain'].map(email['emails'])
        whole['card3_R_emaildomain'] = whole['card3'].astype(str) + '_' + whole['R_emaildomain'].astype(str)
        df_aggs = whole.groupby(by_)[feats[-1]].agg(aggs).reset_index()
        df_aggs.columns = [by_] + [f'{col}_{feats[-1]}_each_{by_}' for col in df_aggs.columns[1:]]
        for col in df_aggs.columns[1:]:
            self.train[col] = pd.merge(whole, df_aggs, on=by_, how='left')[col].values[:len_]
            self.test[col] = pd.merge(whole, df_aggs, on=by_, how='left')[col].values[len_:]


class card5_P_emaildomain_aggs_with_transaction_amt(Feature):
    def create_features(self):
        feats = ['card5', 'P_emaildomain', 'TransactionAmt']
        by_ = f'{feats[0]}_{feats[1]}'
        len_ = len(train_transaction)
        aggs = {'sum', 'max', 'min', 'mean', 'median', 'nunique'}
        whole = pd.concat([train_transaction[feats], test_transaction[feats]], axis=0)
        whole['P_emaildomain'] = whole['P_emaildomain'].map(email['emails'])
        whole['card5_P_emaildomain'] = whole['card5'].astype(str) + '_' + whole['P_emaildomain'].astype(str)
        df_aggs = whole.groupby(by_)[feats[-1]].agg(aggs).reset_index()
        df_aggs.columns = [by_] + [f'{col}_{feats[-1]}_each_{by_}' for col in df_aggs.columns[1:]]
        for col in df_aggs.columns[1:]:
            self.train[col] = pd.merge(whole, df_aggs, on=by_, how='left')[col].values[:len_]
            self.test[col] = pd.merge(whole, df_aggs, on=by_, how='left')[col].values[len_:]


class card5_R_emaildomain_aggs_with_transaction_amt(Feature):
    def create_features(self):
        feats = ['card5', 'R_emaildomain', 'TransactionAmt']
        by_ = f'{feats[0]}_{feats[1]}'
        len_ = len(train_transaction)
        aggs = {'sum', 'max', 'min', 'mean', 'median', 'nunique'}
        whole = pd.concat([train_transaction[feats], test_transaction[feats]], axis=0)
        whole['R_emaildomain'] = whole['R_emaildomain'].map(email['emails'])
        whole['card5_R_emaildomain'] = whole['card5'].astype(str) + '_' + whole['R_emaildomain'].astype(str)
        df_aggs = whole.groupby(by_)[feats[-1]].agg(aggs).reset_index()
        df_aggs.columns = [by_] + [f'{col}_{feats[-1]}_each_{by_}' for col in df_aggs.columns[1:]]
        for col in df_aggs.columns[1:]:
            self.train[col] = pd.merge(whole, df_aggs, on=by_, how='left')[col].values[:len_]
            self.test[col] = pd.merge(whole, df_aggs, on=by_, how='left')[col].values[len_:]


class P_emaildomain_R_emaildomain_aggs_with_transaction_amt(Feature):
    def create_features(self):
        feats = ['P_emaildomain', 'R_emaildomain', 'TransactionAmt']
        by_ = f'{feats[0]}_{feats[1]}'
        len_ = len(train_transaction)
        aggs = {'sum', 'max', 'min', 'mean', 'median', 'nunique'}
        whole = pd.concat([train_transaction[feats], test_transaction[feats]], axis=0)
        whole['P_emaildomain'] = whole['P_emaildomain'].map(email['emails'])
        whole['R_emaildomain'] = whole['R_emaildomain'].map(email['emails'])
        whole['P_emaildomain_R_emaildomain'] = whole['P_emaildomain'].astype(str) + '_' + whole['R_emaildomain'].astype(str)
        df_aggs = whole.groupby(by_)[feats[-1]].agg(aggs).reset_index()
        df_aggs.columns = [by_] + [f'{col}_{feats[-1]}_each_{by_}' for col in df_aggs.columns[1:]]
        for col in df_aggs.columns[1:]:
            self.train[col] = pd.merge(whole, df_aggs, on=by_, how='left')[col].values[:len_]
            self.test[col] = pd.merge(whole, df_aggs, on=by_, how='left')[col].values[len_:]


if __name__ == '__main__':
    args = get_arguments()

    train_identity = pd.read_feather('./data/input/train_identity.feather')
    train_transaction = pd.read_feather('./data/input/train_transaction.feather')
    test_identity = pd.read_feather('./data/input/test_identity.feather')
    test_transaction = pd.read_feather('./data/input/test_transaction.feather')

    generate_features(globals(), args.force)
