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
from sklearn.decomposition import PCA, NMF

from base import Feature, get_arguments, generate_features, sigmoid, minmaxscale, target_encoding

import MeCab
import lda
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer


SEED = 2019
PCA_COMP = 2
NMF_COMP1 = 2
NMF_COMP2 = 5
START_DATE = '2017-11-30'

Feature.dir = 'features'


"""
次元圧縮系
"""


def create_count_matrix(list_text):
    vectorizer = CountVectorizer(token_pattern=r"(?u)\b\w+\b")
    X = vectorizer.fit_transform(list_text)
    all_vocab_ar = [x[0] for x in sorted(vectorizer.vocabulary_.items(), key=lambda x: x[1])]
    all_vocab_ar
    return X, all_vocab_ar


def create_tfidf_matrix(list_text):
    vectorizer = TfidfVectorizer(token_pattern=r"(?u)\b\w+\b")
    X = vectorizer.fit_transform(list_text)
    all_vocab_ar = [x[0] for x in sorted(vectorizer.vocabulary_.items(), key=lambda x: x[1])]
    all_vocab_ar
    return X, all_vocab_ar


def do_lda(matrix, all_vocab, n_topics=5, seed=42, show_topics=False):
    model = lda.LDA(n_topics=n_topics, n_iter=1000, random_state=seed, alpha=0.5, eta=0.5)
    model.fit(matrix)

    if show_topics:
        topic_word = model.topic_word_
        n_top_words = 10
        for topic_idx, topic in enumerate(topic_word):
            print('Topic #%d:' % topic_idx)
            print(',  '.join([all_vocab[i] for i in topic.argsort()[:-n_top_words - 1:-1]]))
            print()

    doc_topic_data = pd.DataFrame(model.transform(matrix), columns=['Topic_{topic_num}'.format(topic_num=i) for i in range(n_topics)])

    return doc_topic_data


class pca_c(Feature):
    def create_features(self):
        feats = [f'C{i}' for i in range(1, 15)]
        whole = pd.concat([train_transaction[feats], test_transaction[feats]], axis=0)
        whole = whole.fillna(-999)
        pca = PCA(n_components=PCA_COMP, random_state=SEED)
        pca_comp = pca.fit_transform(whole.values)
        for i in range(pca_comp.shape[1]):
            self.train[f'pca_c_{i}'] = pca_comp[:len(train_transaction), i]
            self.test[f'pca_c_{i}'] = pca_comp[len(train_transaction):, i]


class nmf_c(Feature):
    def create_features(self):
        feats = [f'C{i}' for i in range(1, 15)]
        whole = pd.concat([train_transaction[feats], test_transaction[feats]], axis=0)
        whole = whole.fillna(999)
        nmf = NMF(n_components=NMF_COMP1, random_state=SEED)
        nmf_comp = nmf.fit_transform(whole.values)
        for i in range(nmf_comp.shape[1]):
            self.train[f'nmf_c_{i}'] = nmf_comp[:len(train_transaction), i]
            self.test[f'nmf_c_{i}'] = nmf_comp[len(train_transaction):, i]


class nmf_d(Feature):
    def create_features(self):
        feats = [f'D{i}' for i in range(1, 16)]
        whole = pd.concat([train_transaction[feats], test_transaction[feats]], axis=0)
        whole = whole.fillna(799) + 200
        nmf = NMF(n_components=NMF_COMP1, random_state=SEED)
        nmf_comp = nmf.fit_transform(whole.values)
        for i in range(nmf_comp.shape[1]):
            self.train[f'nmf_d_{i}'] = nmf_comp[:len(train_transaction), i]
            self.test[f'nmf_d_{i}'] = nmf_comp[len(train_transaction):, i]


class nmf_m(Feature):
    def create_features(self):
        feats = [f'M{i}' for i in range(1, 10)]
        whole = pd.concat([train_transaction[feats], test_transaction[feats]], axis=0).reset_index(drop=True)
        m_le= {'F': 0, 'T': 1, 'None': 2, 'M0': 3, 'M1': 4, 'M2': 5}
        whole = whole.astype(str).replace(m_le)
        nmf = NMF(n_components=NMF_COMP1, random_state=SEED)
        nmf_comp = nmf.fit_transform(whole.values)
        for i in range(nmf_comp.shape[1]):
            self.train[f'nmf_m_{i}'] = nmf_comp[:len(train_transaction), i]
            self.test[f'nmf_m_{i}'] = nmf_comp[len(train_transaction):, i]


class transaction_count_each_month_nmf(Feature):
    def create_features(self):
        feats = ['card1', 'TransactionDT', 'TransactionAmt']
        startdate = datetime.datetime.strptime(START_DATE, '%Y-%m-%d')
        whole = pd.concat([train_transaction[feats], test_transaction[feats]], axis=0).reset_index(drop=True)
        whole['datetime_'] = whole['TransactionDT'].apply(lambda x: (startdate + datetime.timedelta(seconds=x)))
        whole['month'] = whole['datetime_'].apply(lambda x: x.month)

        df_month_count = whole.pivot_table(index='card1', columns='month', values='TransactionDT', aggfunc='count', fill_value=0)

        nmf = NMF(n_components=NMF_COMP1, random_state=SEED)
        nmf_arr = nmf.fit_transform(df_month_count.values)
        for i in range(NMF_COMP1):
            df_month_count[f'nmf_{i}'] = nmf_arr[:, i]

        for i in range(NMF_COMP1):
            self.train[f'{self.__class__.__name__}_{i}'] = pd.merge(train_transaction['card1'], df_month_count[f'nmf_{i}'], on='card1', how='left')[f'nmf_{i}']
            self.test[f'{self.__class__.__name__}_{i}'] = pd.merge(test_transaction['card1'], df_month_count[f'nmf_{i}'], on='card1', how='left')[f'nmf_{i}']


class transaction_sum_amt_each_month_nmf(Feature):
    def create_features(self):
        feats = ['card1', 'TransactionDT', 'TransactionAmt']
        startdate = datetime.datetime.strptime(START_DATE, '%Y-%m-%d')
        whole = pd.concat([train_transaction[feats], test_transaction[feats]], axis=0).reset_index(drop=True)
        whole['datetime_'] = whole['TransactionDT'].apply(lambda x: (startdate + datetime.timedelta(seconds=x)))
        whole['month'] = whole['datetime_'].apply(lambda x: x.month)

        df_month_sum = whole.pivot_table(index='card1', columns='month', values='TransactionAmt', aggfunc='sum', fill_value=0)

        nmf = NMF(n_components=NMF_COMP1, random_state=SEED)
        nmf_arr = nmf.fit_transform(df_month_sum.values)
        for i in range(NMF_COMP1):
            df_month_sum[f'nmf_{i}'] = nmf_arr[:, i]

        for i in range(NMF_COMP1):
            self.train[f'{self.__class__.__name__}_{i}'] = pd.merge(train_transaction['card1'], df_month_sum[f'nmf_{i}'], on='card1', how='left')[f'nmf_{i}']
            self.test[f'{self.__class__.__name__}_{i}'] = pd.merge(test_transaction['card1'], df_month_sum[f'nmf_{i}'], on='card1', how='left')[f'nmf_{i}']


class transaction_mean_amt_each_month_nmf(Feature):
    def create_features(self):
        feats = ['card1', 'TransactionDT', 'TransactionAmt']
        startdate = datetime.datetime.strptime(START_DATE, '%Y-%m-%d')
        whole = pd.concat([train_transaction[feats], test_transaction[feats]], axis=0).reset_index(drop=True)
        whole['datetime_'] = whole['TransactionDT'].apply(lambda x: (startdate + datetime.timedelta(seconds=x)))
        whole['month'] = whole['datetime_'].apply(lambda x: x.month)

        df_month_mean = whole.pivot_table(index='card1', columns='month', values='TransactionAmt', aggfunc='mean', fill_value=0)

        nmf = NMF(n_components=NMF_COMP1, random_state=SEED)
        nmf_arr = nmf.fit_transform(df_month_mean.values)
        for i in range(NMF_COMP1):
            df_month_mean[f'nmf_{i}'] = nmf_arr[:, i]

        for i in range(NMF_COMP1):
            self.train[f'{self.__class__.__name__}_{i}'] = pd.merge(train_transaction['card1'], df_month_mean[f'nmf_{i}'], on='card1', how='left')[f'nmf_{i}']
            self.test[f'{self.__class__.__name__}_{i}'] = pd.merge(test_transaction['card1'], df_month_mean[f'nmf_{i}'], on='card1', how='left')[f'nmf_{i}']


class transaction_amt_div_mean_amt_each_month(Feature):
    def create_features(self):
        feats = ['card1', 'TransactionDT', 'TransactionAmt']
        startdate = datetime.datetime.strptime(START_DATE, '%Y-%m-%d')
        whole = pd.concat([train_transaction[feats], test_transaction[feats]], axis=0).reset_index(drop=True)
        whole['datetime_'] = whole['TransactionDT'].apply(lambda x: (startdate + datetime.timedelta(seconds=x)))
        whole['month'] = whole['datetime_'].apply(lambda x: x.month)
        month_num = whole['month'].unique().shape[0]

        df_month_sum = whole.pivot_table(index='card1', columns='month', values='TransactionAmt', aggfunc='sum', fill_value=0)
        df_month_sum['transaction_mean_amt_each_month'] = np.mean(df_month_sum.values, axis=1)

        for i in range(month_num):
            df_month_sum[f'{df_month_sum.columns[i]}_div_mean'] = df_month_sum[df_month_sum.columns[i]] / df_month_sum['transaction_mean_amt_each_month']

        nmf = NMF(n_components=NMF_COMP1, random_state=SEED)
        nmf_arr = nmf.fit_transform(df_month_sum.values[:, -month_num:])

        for i in range(NMF_COMP1):
            df_month_sum[f'nmf_{i}'] = nmf_arr[:, i]

        for i in range(NMF_COMP1):
            self.train[f'{self.__class__.__name__}_{i}'] = pd.merge(train_transaction['card1'], df_month_sum[f'nmf_{i}'], on='card1', how='left')[f'nmf_{i}']
            self.test[f'{self.__class__.__name__}_{i}'] = pd.merge(test_transaction['card1'], df_month_sum[f'nmf_{i}'], on='card1', how='left')[f'nmf_{i}']


class nmf_v1_v11(Feature):
    def create_features(self):
        feats = [f'V{i}' for i in range(1, 12)]
        whole = pd.concat([train_transaction[feats], test_transaction[feats]], axis=0)
        whole = whole.fillna(999)
        nmf = NMF(n_components=NMF_COMP2, random_state=SEED)
        nmf_comp = nmf.fit_transform(whole.values)
        for i in range(nmf_comp.shape[1]):
            self.train[f'{self.__class__.__name__}_{i}'] = nmf_comp[:len(train_transaction), i]
            self.test[f'{self.__class__.__name__}_{i}'] = nmf_comp[len(train_transaction):, i]


class nmf_v12_v34(Feature):
    def create_features(self):
        feats = [f'V{i}' for i in range(12, 35)]
        whole = pd.concat([train_transaction[feats], test_transaction[feats]], axis=0)
        whole = whole.fillna(999)
        nmf = NMF(n_components=NMF_COMP2, random_state=SEED)
        nmf_comp = nmf.fit_transform(whole.values)
        for i in range(nmf_comp.shape[1]):
            self.train[f'{self.__class__.__name__}_{i}'] = nmf_comp[:len(train_transaction), i]
            self.test[f'{self.__class__.__name__}_{i}'] = nmf_comp[len(train_transaction):, i]


class nmf_v35_v52(Feature):
    def create_features(self):
        feats = [f'V{i}' for i in range(35, 53)]
        whole = pd.concat([train_transaction[feats], test_transaction[feats]], axis=0)
        whole = whole.fillna(999)
        nmf = NMF(n_components=NMF_COMP2, random_state=SEED)
        nmf_comp = nmf.fit_transform(whole.values)
        for i in range(nmf_comp.shape[1]):
            self.train[f'{self.__class__.__name__}_{i}'] = nmf_comp[:len(train_transaction), i]
            self.test[f'{self.__class__.__name__}_{i}'] = nmf_comp[len(train_transaction):, i]


class nmf_v53_v74(Feature):
    def create_features(self):
        feats = [f'V{i}' for i in range(53, 75)]
        whole = pd.concat([train_transaction[feats], test_transaction[feats]], axis=0)
        whole = whole.fillna(999)
        nmf = NMF(n_components=NMF_COMP2, random_state=SEED)
        nmf_comp = nmf.fit_transform(whole.values)
        for i in range(nmf_comp.shape[1]):
            self.train[f'{self.__class__.__name__}_{i}'] = nmf_comp[:len(train_transaction), i]
            self.test[f'{self.__class__.__name__}_{i}'] = nmf_comp[len(train_transaction):, i]


class nmf_v75_v94(Feature):
    def create_features(self):
        feats = [f'V{i}' for i in range(75, 95)]
        whole = pd.concat([train_transaction[feats], test_transaction[feats]], axis=0)
        whole = whole.fillna(999)
        nmf = NMF(n_components=NMF_COMP2, random_state=SEED)
        nmf_comp = nmf.fit_transform(whole.values)
        for i in range(nmf_comp.shape[1]):
            self.train[f'{self.__class__.__name__}_{i}'] = nmf_comp[:len(train_transaction), i]
            self.test[f'{self.__class__.__name__}_{i}'] = nmf_comp[len(train_transaction):, i]


class nmf_v95_v137(Feature):
    def create_features(self):
        feats = [f'V{i}' for i in range(95, 138)]
        whole = pd.concat([train_transaction[feats], test_transaction[feats]], axis=0)
        whole = whole.fillna(999)
        nmf = NMF(n_components=NMF_COMP2, random_state=SEED)
        nmf_comp = nmf.fit_transform(whole.values)
        for i in range(nmf_comp.shape[1]):
            self.train[f'{self.__class__.__name__}_{i}'] = nmf_comp[:len(train_transaction), i]
            self.test[f'{self.__class__.__name__}_{i}'] = nmf_comp[len(train_transaction):, i]


class nmf_v138_v166(Feature):
    def create_features(self):
        feats = [f'V{i}' for i in range(138, 167)]
        whole = pd.concat([train_transaction[feats], test_transaction[feats]], axis=0)
        whole = whole.fillna(999)
        nmf = NMF(n_components=NMF_COMP2, random_state=SEED)
        nmf_comp = nmf.fit_transform(whole.values)
        for i in range(nmf_comp.shape[1]):
            self.train[f'{self.__class__.__name__}_{i}'] = nmf_comp[:len(train_transaction), i]
            self.test[f'{self.__class__.__name__}_{i}'] = nmf_comp[len(train_transaction):, i]


class nmf_v167_v278(Feature):
    def create_features(self):
        feats = [f'V{i}' for i in range(167, 279)]
        whole = pd.concat([train_transaction[feats], test_transaction[feats]], axis=0)
        whole = whole.fillna(999)
        nmf = NMF(n_components=NMF_COMP2, random_state=SEED)
        nmf_comp = nmf.fit_transform(whole.values)
        for i in range(nmf_comp.shape[1]):
            self.train[f'{self.__class__.__name__}_{i}'] = nmf_comp[:len(train_transaction), i]
            self.test[f'{self.__class__.__name__}_{i}'] = nmf_comp[len(train_transaction):, i]


class nmf_v279_v320(Feature):
    def create_features(self):
        feats =  [
            'V279', 'V280', 'V284', 'V285', 'V286', 'V287', 'V290', 'V291', 'V292', 'V293', 'V294',
            'V295', 'V297', 'V298', 'V299','V302', 'V303', 'V304', 'V305', 'V306', 'V307', 'V308',
            'V309', 'V310', 'V311', 'V312', 'V316', 'V317', 'V318', 'V319','V320', 'V320'
        ]
        whole = pd.concat([train_transaction[feats], test_transaction[feats]], axis=0)
        whole = whole.fillna(999)
        nmf = NMF(n_components=NMF_COMP2, random_state=SEED)
        nmf_comp = nmf.fit_transform(whole.values)
        for i in range(nmf_comp.shape[1]):
            self.train[f'{self.__class__.__name__}_{i}'] = nmf_comp[:len(train_transaction), i]
            self.test[f'{self.__class__.__name__}_{i}'] = nmf_comp[len(train_transaction):, i]


class card1_card5_count_pca(Feature):
    def create_features(self):
        feats = ['card1', 'card5', 'TransactionAmt']
        whole = pd.concat([train_transaction[feats], test_transaction[feats]], axis=0)
        df_count = whole.pivot_table(index='card1',
                                     columns='card5',
                                     values='TransactionAmt',
                                     aggfunc='count',
                                     fill_value=0)

        pca = PCA(n_components=PCA_COMP, random_state=SEED)
        pca_array = pca.fit_transform(df_count.values)

        col_names = [f'card1_card5_count_pca_{i}' for i in range(PCA_COMP)]

        df = pd.DataFrame(pca_array, index=df_count.index, columns=col_names).reset_index()
        whole = pd.merge(whole, df, on='card1', how='left')

        for col in col_names:
            self.train[col] = whole[col].values[:len(train_transaction)]
            self.test[col] = whole[col].values[len(train_transaction):]


class card1_card5_sum_pca(Feature):
    def create_features(self):
        feats = ['card1', 'card5', 'TransactionAmt']
        whole = pd.concat([train_transaction[feats], test_transaction[feats]], axis=0)
        df_sum = whole.pivot_table(index='card1',
                                   columns='card5',
                                   values='TransactionAmt',
                                   aggfunc='sum',
                                   fill_value=0)

        pca = PCA(n_components=PCA_COMP, random_state=SEED)
        pca_array = pca.fit_transform(df_sum.values)

        col_names = [f'card1_card5_sum_pca_{i}' for i in range(PCA_COMP)]

        df = pd.DataFrame(pca_array, index=df_sum.index, columns=col_names).reset_index()
        whole = pd.merge(whole, df, on='card1', how='left')

        for col in col_names:
            self.train[col] = whole[col].values[:len(train_transaction)]
            self.test[col] = whole[col].values[len(train_transaction):]


class card1_card5_std_pca(Feature):
    def create_features(self):
        feats = ['card1', 'card5', 'TransactionAmt']
        whole = pd.concat([train_transaction[feats], test_transaction[feats]], axis=0)
        df_std = whole.pivot_table(index='card1',
                                   columns='card5',
                                   values='TransactionAmt',
                                   aggfunc=np.std,
                                   fill_value=0)

        pca = PCA(n_components=PCA_COMP, random_state=SEED)
        pca_array = pca.fit_transform(df_std.values)

        col_names = [f'card1_card5_std_pca_{i}' for i in range(PCA_COMP)]

        df = pd.DataFrame(pca_array, index=df_std.index, columns=col_names).reset_index()
        whole = pd.merge(whole, df, on='card1', how='left')

        for col in col_names:
            self.train[col] = whole[col].values[:len(train_transaction)]
            self.test[col] = whole[col].values[len(train_transaction):]





if __name__ == '__main__':
    args = get_arguments()

    train_identity = pd.read_feather('./data/input/train_identity.feather')
    train_transaction = pd.read_feather('./data/input/train_transaction.feather')
    test_identity = pd.read_feather('./data/input/test_identity.feather')
    test_transaction = pd.read_feather('./data/input/test_transaction.feather')

    generate_features(globals(), args.force)
