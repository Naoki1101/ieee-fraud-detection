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

import warnings
warnings.filterwarnings('ignore')


SEED = 2019
START_DATE = '2017-11-30'

Feature.dir = 'features'


"""
その他
"""


class transaction_null_count(Feature):
    def create_features(self):
        self.train['transction_null_count'] = train_transaction.isnull().sum(axis=1).values
        self.test['transction_null_count'] = test_transaction.isnull().sum(axis=1).values


class card2_null_count(Feature):
    def create_features(self):
        self.train['card2_null_count'] = np.where(train_transaction['card2'].fillna(-999) != -999, 0, 1) 
        self.test['card2_null_count'] = np.where(test_transaction['card2'].fillna(-999) != -999, 0, 1)


class addr1_null_count(Feature):
    def create_features(self):
        self.train['addr1_null_count'] = np.where(train_transaction['addr1'].fillna(-999) != -999, 0, 1) 
        self.test['addr1_null_count'] = np.where(test_transaction['addr1'].fillna(-999) != -999, 0, 1)


class dist1_null_count(Feature):
    def create_features(self):
        self.train['dist1_null_count'] = np.where(train_transaction['dist1'].fillna(-999) != -999, 0, 1) 
        self.test['dist1_null_count'] = np.where(test_transaction['dist1'].fillna(-999) != -999, 0, 1)


class dist2_null_count(Feature):
    def create_features(self):
        self.train['dist2_null_count'] = np.where(train_transaction['dist2'].fillna(-999) != -999, 0, 1) 
        self.test['dist2_null_count'] = np.where(test_transaction['dist2'].fillna(-999) != -999, 0, 1)


class P_emaildomain_null_count(Feature):
    def create_features(self):
        self.train['P_emaildomain_null_count'] = np.where(train_transaction['P_emaildomain'].fillna(-999) != -999, 0, 1) 
        self.test['P_emaildomain_null_count'] = np.where(test_transaction['P_emaildomain'].fillna(-999) != -999, 0, 1)


class R_emaildomain_null_count(Feature):
    def create_features(self):
        self.train['R_emaildomain_null_count'] = np.where(train_transaction['R_emaildomain'].fillna(-999) != -999, 0, 1) 
        self.test['R_emaildomain_null_count'] = np.where(test_transaction['R_emaildomain'].fillna(-999) != -999, 0, 1)


class D2_null_count(Feature):
    def create_features(self):
        self.train['D2_null_count'] = np.where(train_transaction['D2'].fillna(-999) != -999, 0, 1) 
        self.test['D2_null_count'] = np.where(test_transaction['D2'].fillna(-999) != -999, 0, 1)


class D3_null_count(Feature):
    def create_features(self):
        self.train['D3_null_count'] = np.where(train_transaction['D3'].fillna(-999) != -999, 0, 1) 
        self.test['D3_null_count'] = np.where(test_transaction['D3'].fillna(-999) != -999, 0, 1)


class D4_null_count(Feature):
    def create_features(self):
        self.train['D4_null_count'] = np.where(train_transaction['D4'].fillna(-999) != -999, 0, 1) 
        self.test['D4_null_count'] = np.where(test_transaction['D4'].fillna(-999) != -999, 0, 1)


class D5_null_count(Feature):
    def create_features(self):
        self.train['D5_null_count'] = np.where(train_transaction['D5'].fillna(-999) != -999, 0, 1) 
        self.test['D5_null_count'] = np.where(test_transaction['D5'].fillna(-999) != -999, 0, 1)


class D6_null_count(Feature):
    def create_features(self):
        self.train['D6_null_count'] = np.where(train_transaction['D6'].fillna(-999) != -999, 0, 1) 
        self.test['D6_null_count'] = np.where(test_transaction['D6'].fillna(-999) != -999, 0, 1)


class D7_null_count(Feature):
    def create_features(self):
        self.train['D7_null_count'] = np.where(train_transaction['D7'].fillna(-999) != -999, 0, 1) 
        self.test['D7_null_count'] = np.where(test_transaction['D7'].fillna(-999) != -999, 0, 1)


class D8_null_count(Feature):
    def create_features(self):
        self.train['D8_null_count'] = np.where(train_transaction['D8'].fillna(-999) != -999, 0, 1) 
        self.test['D8_null_count'] = np.where(test_transaction['D8'].fillna(-999) != -999, 0, 1)


class D9_null_count(Feature):
    def create_features(self):
        self.train['D9_null_count'] = np.where(train_transaction['D9'].fillna(-999) != -999, 0, 1) 
        self.test['D9_null_count'] = np.where(test_transaction['D9'].fillna(-999) != -999, 0, 1)


class D10_null_count(Feature):
    def create_features(self):
        self.train['D10_null_count'] = np.where(train_transaction['D10'].fillna(-999) != -999, 0, 1) 
        self.test['D10_null_count'] = np.where(test_transaction['D10'].fillna(-999) != -999, 0, 1)


class D11_null_count(Feature):
    def create_features(self):
        self.train['D11_null_count'] = np.where(train_transaction['D11'].fillna(-999) != -999, 0, 1) 
        self.test['D11_null_count'] = np.where(test_transaction['D11'].fillna(-999) != -999, 0, 1)


class D12_null_count(Feature):
    def create_features(self):
        self.train['D12_null_count'] = np.where(train_transaction['D12'].fillna(-999) != -999, 0, 1) 
        self.test['D12_null_count'] = np.where(test_transaction['D12'].fillna(-999) != -999, 0, 1)


class D13_null_count(Feature):
    def create_features(self):
        self.train['D13_null_count'] = np.where(train_transaction['D13'].fillna(-999) != -999, 0, 1) 
        self.test['D13_null_count'] = np.where(test_transaction['D13'].fillna(-999) != -999, 0, 1)


class D14_null_count(Feature):
    def create_features(self):
        self.train['D14_null_count'] = np.where(train_transaction['D14'].fillna(-999) != -999, 0, 1) 
        self.test['D14_null_count'] = np.where(test_transaction['D14'].fillna(-999) != -999, 0, 1)


class D15_null_count(Feature):
    def create_features(self):
        self.train['D15_null_count'] = np.where(train_transaction['D15'].fillna(-999) != -999, 0, 1) 
        self.test['D15_null_count'] = np.where(test_transaction['D15'].fillna(-999) != -999, 0, 1)


class M2_null_count(Feature):
    def create_features(self):
        self.train['M2_null_count'] = np.where(train_transaction['M2'].fillna(-999) != -999, 0, 1) 
        self.test['M2_null_count'] = np.where(test_transaction['M2'].fillna(-999) != -999, 0, 1)


class M3_null_count(Feature):
    def create_features(self):
        self.train['M3_null_count'] = np.where(train_transaction['M3'].fillna(-999) != -999, 0, 1) 
        self.test['M3_null_count'] = np.where(test_transaction['M3'].fillna(-999) != -999, 0, 1)


class M4_null_count(Feature):
    def create_features(self):
        self.train['M4_null_count'] = np.where(train_transaction['M4'].fillna(-999) != -999, 0, 1) 
        self.test['M4_null_count'] = np.where(test_transaction['M4'].fillna(-999) != -999, 0, 1)


class M5_null_count(Feature):
    def create_features(self):
        self.train['M5_null_count'] = np.where(train_transaction['M5'].fillna(-999) != -999, 0, 1) 
        self.test['M5_null_count'] = np.where(test_transaction['M5'].fillna(-999) != -999, 0, 1)


class M6_null_count(Feature):
    def create_features(self):
        self.train['M6_null_count'] = np.where(train_transaction['M6'].fillna(-999) != -999, 0, 1) 
        self.test['M6_null_count'] = np.where(test_transaction['M6'].fillna(-999) != -999, 0, 1)


class M7_null_count(Feature):
    def create_features(self):
        self.train['M7_null_count'] = np.where(train_transaction['M7'].fillna(-999) != -999, 0, 1) 
        self.test['M7_null_count'] = np.where(test_transaction['M7'].fillna(-999) != -999, 0, 1)


class M8_null_count(Feature):
    def create_features(self):
        self.train['M8_null_count'] = np.where(train_transaction['M8'].fillna(-999) != -999, 0, 1) 
        self.test['M8_null_count'] = np.where(test_transaction['M8'].fillna(-999) != -999, 0, 1)


class V5_null_count(Feature):
    def create_features(self):
        self.train['V5_null_count'] = np.where(train_transaction['V5'].fillna(-999) != -999, 0, 1) 
        self.test['V5_null_count'] = np.where(test_transaction['V5'].fillna(-999) != -999, 0, 1)


class V10_null_count(Feature):
    def create_features(self):
        self.train['V10_null_count'] = np.where(train_transaction['V10'].fillna(-999) != -999, 0, 1) 
        self.test['V10_null_count'] = np.where(test_transaction['V10'].fillna(-999) != -999, 0, 1)


class V12_null_count(Feature):
    def create_features(self):
        self.train['V12_null_count'] = np.where(train_transaction['V12'].fillna(-999) != -999, 0, 1) 
        self.test['V12_null_count'] = np.where(test_transaction['V12'].fillna(-999) != -999, 0, 1)


class V20_null_count(Feature):
    def create_features(self):
        self.train['V20_null_count'] = np.where(train_transaction['V20'].fillna(-999) != -999, 0, 1) 
        self.test['V20_null_count'] = np.where(test_transaction['V20'].fillna(-999) != -999, 0, 1)


class V24_null_count(Feature):
    def create_features(self):
        self.train['V24_null_count'] = np.where(train_transaction['V24'].fillna(-999) != -999, 0, 1) 
        self.test['V24_null_count'] = np.where(test_transaction['V24'].fillna(-999) != -999, 0, 1)


class V29_null_count(Feature):
    def create_features(self):
        self.train['V29_null_count'] = np.where(train_transaction['V29'].fillna(-999) != -999, 0, 1) 
        self.test['V29_null_count'] = np.where(test_transaction['V29'].fillna(-999) != -999, 0, 1)


class V35_null_count(Feature):
    def create_features(self):
        self.train['V35_null_count'] = np.where(train_transaction['V35'].fillna(-999) != -999, 0, 1) 
        self.test['V35_null_count'] = np.where(test_transaction['V35'].fillna(-999) != -999, 0, 1)


class V37_null_count(Feature):
    def create_features(self):
        self.train['V37_null_count'] = np.where(train_transaction['V37'].fillna(-999) != -999, 0, 1) 
        self.test['V37_null_count'] = np.where(test_transaction['V37'].fillna(-999) != -999, 0, 1)


class V38_null_count(Feature):
    def create_features(self):
        self.train['V38_null_count'] = np.where(train_transaction['V38'].fillna(-999) != -999, 0, 1) 
        self.test['V38_null_count'] = np.where(test_transaction['V38'].fillna(-999) != -999, 0, 1)


class V44_null_count(Feature):
    def create_features(self):
        self.train['V44_null_count'] = np.where(train_transaction['V44'].fillna(-999) != -999, 0, 1) 
        self.test['V44_null_count'] = np.where(test_transaction['V44'].fillna(-999) != -999, 0, 1)


class V45_null_count(Feature):
    def create_features(self):
        self.train['V45_null_count'] = np.where(train_transaction['V45'].fillna(-999) != -999, 0, 1) 
        self.test['V45_null_count'] = np.where(test_transaction['V45'].fillna(-999) != -999, 0, 1)


class V48_null_count(Feature):
    def create_features(self):
        self.train['V48_null_count'] = np.where(train_transaction['V48'].fillna(-999) != -999, 0, 1) 
        self.test['V48_null_count'] = np.where(test_transaction['V48'].fillna(-999) != -999, 0, 1)


class V49_null_count(Feature):
    def create_features(self):
        self.train['V49_null_count'] = np.where(train_transaction['V49'].fillna(-999) != -999, 0, 1) 
        self.test['V49_null_count'] = np.where(test_transaction['V49'].fillna(-999) != -999, 0, 1)


class V53_null_count(Feature):
    def create_features(self):
        self.train['V53_null_count'] = np.where(train_transaction['V53'].fillna(-999) != -999, 0, 1) 
        self.test['V53_null_count'] = np.where(test_transaction['V53'].fillna(-999) != -999, 0, 1)


class V54_null_count(Feature):
    def create_features(self):
        self.train['V54_null_count'] = np.where(train_transaction['V54'].fillna(-999) != -999, 0, 1) 
        self.test['V54_null_count'] = np.where(test_transaction['V54'].fillna(-999) != -999, 0, 1)


class V55_null_count(Feature):
    def create_features(self):
        self.train['V55_null_count'] = np.where(train_transaction['V55'].fillna(-999) != -999, 0, 1) 
        self.test['V55_null_count'] = np.where(test_transaction['V55'].fillna(-999) != -999, 0, 1)


class V56_null_count(Feature):
    def create_features(self):
        self.train['V56_null_count'] = np.where(train_transaction['V56'].fillna(-999) != -999, 0, 1) 
        self.test['V56_null_count'] = np.where(test_transaction['V56'].fillna(-999) != -999, 0, 1)


class V61_null_count(Feature):
    def create_features(self):
        self.train['V61_null_count'] = np.where(train_transaction['V61'].fillna(-999) != -999, 0, 1) 
        self.test['V61_null_count'] = np.where(test_transaction['V61'].fillna(-999) != -999, 0, 1)


class V62_null_count(Feature):
    def create_features(self):
        self.train['V62_null_count'] = np.where(train_transaction['V62'].fillna(-999) != -999, 0, 1) 
        self.test['V62_null_count'] = np.where(test_transaction['V62'].fillna(-999) != -999, 0, 1)


class V67_null_count(Feature):
    def create_features(self):
        self.train['V67_null_count'] = np.where(train_transaction['V67'].fillna(-999) != -999, 0, 1) 
        self.test['V67_null_count'] = np.where(test_transaction['V67'].fillna(-999) != -999, 0, 1)


class V70_null_count(Feature):
    def create_features(self):
        self.train['V70_null_count'] = np.where(train_transaction['V70'].fillna(-999) != -999, 0, 1) 
        self.test['V70_null_count'] = np.where(test_transaction['V70'].fillna(-999) != -999, 0, 1)


class V75_null_count(Feature):
    def create_features(self):
        self.train['V75_null_count'] = np.where(train_transaction['V75'].fillna(-999) != -999, 0, 1) 
        self.test['V75_null_count'] = np.where(test_transaction['V75'].fillna(-999) != -999, 0, 1)


class V76_null_count(Feature):
    def create_features(self):
        self.train['V76_null_count'] = np.where(train_transaction['V76'].fillna(-999) != -999, 0, 1) 
        self.test['V76_null_count'] = np.where(test_transaction['V76'].fillna(-999) != -999, 0, 1)


class V77_null_count(Feature):
    def create_features(self):
        self.train['V77_null_count'] = np.where(train_transaction['V77'].fillna(-999) != -999, 0, 1) 
        self.test['V77_null_count'] = np.where(test_transaction['V77'].fillna(-999) != -999, 0, 1)


class V78_null_count(Feature):
    def create_features(self):
        self.train['V78_null_count'] = np.where(train_transaction['V78'].fillna(-999) != -999, 0, 1) 
        self.test['V78_null_count'] = np.where(test_transaction['V78'].fillna(-999) != -999, 0, 1)


class V82_null_count(Feature):
    def create_features(self):
        self.train['V82_null_count'] = np.where(train_transaction['V82'].fillna(-999) != -999, 0, 1) 
        self.test['V82_null_count'] = np.where(test_transaction['V82'].fillna(-999) != -999, 0, 1)


class V83_null_count(Feature):
    def create_features(self):
        self.train['V83_null_count'] = np.where(train_transaction['V83'].fillna(-999) != -999, 0, 1) 
        self.test['V83_null_count'] = np.where(test_transaction['V83'].fillna(-999) != -999, 0, 1)


class V86_null_count(Feature):
    def create_features(self):
        self.train['V86_null_count'] = np.where(train_transaction['V86'].fillna(-999) != -999, 0, 1) 
        self.test['V86_null_count'] = np.where(test_transaction['V86'].fillna(-999) != -999, 0, 1)


class V87_null_count(Feature):
    def create_features(self):
        self.train['V87_null_count'] = np.where(train_transaction['V87'].fillna(-999) != -999, 0, 1) 
        self.test['V87_null_count'] = np.where(test_transaction['V87'].fillna(-999) != -999, 0, 1)


class V91_null_count(Feature):
    def create_features(self):
        self.train['V91_null_count'] = np.where(train_transaction['V91'].fillna(-999) != -999, 0, 1) 
        self.test['V91_null_count'] = np.where(test_transaction['V91'].fillna(-999) != -999, 0, 1)


class V139_null_count(Feature):
    def create_features(self):
        self.train['V139_null_count'] = np.where(train_transaction['V139'].fillna(-999) != -999, 0, 1) 
        self.test['V139_null_count'] = np.where(test_transaction['V139'].fillna(-999) != -999, 0, 1)


class V165_null_count(Feature):
    def create_features(self):
        self.train['V165_null_count'] = np.where(train_transaction['V165'].fillna(-999) != -999, 0, 1) 
        self.test['V165_null_count'] = np.where(test_transaction['V165'].fillna(-999) != -999, 0, 1)


class V166_null_count(Feature):
    def create_features(self):
        self.train['V166_null_count'] = np.where(train_transaction['V166'].fillna(-999) != -999, 0, 1) 
        self.test['V166_null_count'] = np.where(test_transaction['V166'].fillna(-999) != -999, 0, 1)


class V202_null_count(Feature):
    def create_features(self):
        self.train['V202_null_count'] = np.where(train_transaction['V202'].fillna(-999) != -999, 0, 1) 
        self.test['V202_null_count'] = np.where(test_transaction['V202'].fillna(-999) != -999, 0, 1)


class V203_null_count(Feature):
    def create_features(self):
        self.train['V203_null_count'] = np.where(train_transaction['V203'].fillna(-999) != -999, 0, 1) 
        self.test['V203_null_count'] = np.where(test_transaction['V203'].fillna(-999) != -999, 0, 1)


class V204_null_count(Feature):
    def create_features(self):
        self.train['V204_null_count'] = np.where(train_transaction['V204'].fillna(-999) != -999, 0, 1) 
        self.test['V204_null_count'] = np.where(test_transaction['V204'].fillna(-999) != -999, 0, 1)


class V212_null_count(Feature):
    def create_features(self):
        self.train['V212_null_count'] = np.where(train_transaction['V212'].fillna(-999) != -999, 0, 1) 
        self.test['V212_null_count'] = np.where(test_transaction['V212'].fillna(-999) != -999, 0, 1)


class V221_null_count(Feature):
    def create_features(self):
        self.train['V221_null_count'] = np.where(train_transaction['V221'].fillna(-999) != -999, 0, 1) 
        self.test['V221_null_count'] = np.where(test_transaction['V221'].fillna(-999) != -999, 0, 1)


class V222_null_count(Feature):
    def create_features(self):
        self.train['V222_null_count'] = np.where(train_transaction['V222'].fillna(-999) != -999, 0, 1) 
        self.test['V222_null_count'] = np.where(test_transaction['V222'].fillna(-999) != -999, 0, 1)


class V261_null_count(Feature):
    def create_features(self):
        self.train['V261_null_count'] = np.where(train_transaction['V261'].fillna(-999) != -999, 0, 1) 
        self.test['V261_null_count'] = np.where(test_transaction['V261'].fillna(-999) != -999, 0, 1)


class V263_null_count(Feature):
    def create_features(self):
        self.train['V263_null_count'] = np.where(train_transaction['V263'].fillna(-999) != -999, 0, 1) 
        self.test['V263_null_count'] = np.where(test_transaction['V263'].fillna(-999) != -999, 0, 1)


class V264_null_count(Feature):
    def create_features(self):
        self.train['V264_null_count'] = np.where(train_transaction['V264'].fillna(-999) != -999, 0, 1) 
        self.test['V264_null_count'] = np.where(test_transaction['V264'].fillna(-999) != -999, 0, 1)


class V265_null_count(Feature):
    def create_features(self):
        self.train['V265_null_count'] = np.where(train_transaction['V265'].fillna(-999) != -999, 0, 1) 
        self.test['V265_null_count'] = np.where(test_transaction['V265'].fillna(-999) != -999, 0, 1)


class V266_null_count(Feature):
    def create_features(self):
        self.train['V266_null_count'] = np.where(train_transaction['V266'].fillna(-999) != -999, 0, 1) 
        self.test['V266_null_count'] = np.where(test_transaction['V266'].fillna(-999) != -999, 0, 1)


class V267_null_count(Feature):
    def create_features(self):
        self.train['V267_null_count'] = np.where(train_transaction['V267'].fillna(-999) != -999, 0, 1) 
        self.test['V267_null_count'] = np.where(test_transaction['V267'].fillna(-999) != -999, 0, 1)


class V274_null_count(Feature):
    def create_features(self):
        self.train['V274_null_count'] = np.where(train_transaction['V274'].fillna(-999) != -999, 0, 1) 
        self.test['V274_null_count'] = np.where(test_transaction['V274'].fillna(-999) != -999, 0, 1)


class V277_null_count(Feature):
    def create_features(self):
        self.train['V277_null_count'] = np.where(train_transaction['V277'].fillna(-999) != -999, 0, 1) 
        self.test['V277_null_count'] = np.where(test_transaction['V277'].fillna(-999) != -999, 0, 1)


if __name__ == '__main__':
    args = get_arguments()

    train_identity = pd.read_feather('./data/input/train_identity.feather')
    train_transaction = pd.read_feather('./data/input/train_transaction.feather')
    test_identity = pd.read_feather('./data/input/test_identity.feather')
    test_transaction = pd.read_feather('./data/input/test_transaction.feather')

    generate_features(globals(), args.force)