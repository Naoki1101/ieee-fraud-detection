import os
import numpy as np
import pandas as pd
import datetime
import logging
from sklearn.model_selection import StratifiedKFold, KFold, GroupKFold, TimeSeriesSplit
from sklearn.metrics import roc_auc_score
import argparse
import pickle
import yaml
import time
import sys
import re

from scripts import load_datasets, load_target, timer
from scripts.utils import send_line, send_notion, submit_kaggle, transfar_dropbox, reduce_mem_usage, preds2rank, extract_drop_index, resample, calibration, div_col
from logs.logger import log_best
from models.lgbm import train_and_predict, save_importances, save_features
from fe_extract.base import get_arguments, generate_features

import warnings
warnings.filterwarnings('ignore')


# ===============
# Settings
# ===============
parser = argparse.ArgumentParser()
parser.add_argument('--config', default='./configs/default.yaml')
parser.add_argument('-m', '--massage')
options = parser.parse_args()

config = yaml.load(open(options.config))

COMPE_NAME = config['compe_name']
MODEL_NAME = 'lgbm'
SEED = config['seed']
NOW = datetime.datetime.now()
LOG_FILE_NAME = f'log_{MODEL_NAME}_{NOW:%Y%m%d%H%M%S}.log'
LOGGER_PATH = './logs/' + LOG_FILE_NAME
IMP_FILE_NAME = f'lgbm_importances_{NOW:%Y%m%d%H%M%S}.png'
IMP_PATH = './importances/' + IMP_FILE_NAME
SAMPLE_PATH = './data/input/sample_submission.csv'
OUTPUT_PATH = './data/output/'
FOLD_PATH = './folds/folds4.feather'
DROPBOX_PATH = f'/{COMPE_NAME}/'
CAT = config['categorical_features']
FEATURES = config['features']
TARGET_NAME = config['target_name']
DROP_IDX = config['drop_data']['outliers']
PARAMS = config[f'{MODEL_NAME}_params']
LOSS = config['loss']
ID_NAME = config['ID_name']

REDUCE = True
CALIBRATION = False
SAVE_OOF = False
KAGGLE_SUBMIT = False
DROPBOX = False

COMMENT = options.massage
if COMMENT == '':
    print('WARNING: Comment not found')

LINE_TOKEN = config['line_token']
DROPBOX_TOKEN = config['dropbox_token']
TOKEN_V2 = config['notion_params']['token_v2']
NOTION_URL = config['notion_params']['url']

logging.basicConfig(filename=LOGGER_PATH, level=logging.DEBUG)
logging.debug(config)

np.random.seed(SEED)


# ===============
# Main
# ===============
s = time.time()

with timer('load data', logging):
    with open('./data/else/col2path.pkl', 'rb') as f:
        col2path = pickle.load(f)
    X_train_all, X_test = load_datasets(FEATURES, col2path)
    y_train_all = load_target(TARGET_NAME)
    logging.debug(f'feature num: {len(X_train_all.columns)}')

with timer('reduce_mem_usage', logging):
    if REDUCE:
        X_train_all = reduce_mem_usage(X_train_all)
        X_test = reduce_mem_usage(X_test)

with timer(f'load {FOLD_PATH.split("/")[-1]}', logging):
    folds = pd.read_feather(FOLD_PATH)
    n_splits = folds['fold_id'].max() + 1

# with timer('concat oof', logging):
#     path = './features/lgbm_{data}.feather'
#     lgbm_pred_train = pd.read_feather(path.format(data='train'))
#     lgbm_pred_test = pd.read_feather(path.format(data='test'))

# with timer('concat diff', logging):
#     path = './features/lgbm_diff_{data}.feather'
#     lgbm_pred_train = pd.read_feather(path.format(data='train'))
#     lgbm_pred_test = pd.read_feather(path.format(data='test'))

#     X_train_all = pd.concat([X_train_all, lgbm_pred_train], axis=1)
#     X_test = pd.concat([X_test, lgbm_pred_test], axis=1)

# with timer(f'calculate sampling rate for calibration', logging):
#     if CALIBRATION:
#         list_sampling_rate = []
#         for fold_ in range(n_splits):
#             y_train = y_train_all[folds['fold_id'] != fold_]
#             list_sampling_rate.append(y_train.sum() / len(y_train))

# with timer('drop rows', logging):
#     X_train_all = X_train_all.drop(DROP_IDX, axis=0).reset_index(drop=True)
#     y_train_all = y_train_all.drop(DROP_IDX, axis=0).reset_index(drop=True)
#     folds = folds.drop(DROP_IDX, axis=0).reset_index(drop=True)

# with timer('down sampling', logging):
#     df_adversarial_val = pd.read_csv('./notebook/result_adversarial_validation.csv')
#     drop_idx = df_adversarial_val[(df_adversarial_val['isFraud'] == 0) & (df_adversarial_val['isTest'] < 0.1)].index
#     X_train_all = X_train_all.drop(drop_idx, axis=0).reset_index(drop=True)
#     y_train_all = y_train_all.drop(drop_idx, axis=0).reset_index(drop=True)
#     folds = folds.drop(drop_idx, axis=0).reset_index(drop=True)

# with timer('down sampling', logging):
#     drop_index = extract_drop_index(y_train_all, r=0.9)
#     X_train_all = X_train_all.drop(drop_index, axis=0).reset_index(drop=True)
#     y_train_all = y_train_all.drop(drop_index, axis=0).reset_index(drop=True)
#     folds = folds.drop(drop_index, axis=0).reset_index(drop=True)
#     with open('./data/else/random_downsampling_index.pkl', 'wb') as f:
#         pickle.dump(drop_index, f)

# with timer('resample', logging):
#     sample_size = int(len(X_train_all) * 1.5)
#     sample_index = resample(y_train_all, sample_size, r=5)
#     X_train_all = X_train_all.iloc[sample_index].reset_index(drop=True)
#     y_train_all = y_train_all.iloc[sample_index].reset_index(drop=True)
#     folds = folds.iloc[sample_index].reset_index(drop=True)
#     logging.debug(f'sample size: {sample_size}')

# with timer('down sampling', logging):
#     with open('./data/else/drop_idx_downsampling_few_card1.pkl', 'rb') as y:
#         drop_idx = pickle.load(y)
#     X_train_all = X_train_all.drop(drop_idx, axis=0).reset_index(drop=True)
#     y_train_all = y_train_all.drop(drop_idx, axis=0).reset_index(drop=True)
#     folds = folds.drop(drop_idx, axis=0).reset_index(drop=True)

# with timer('new features with high important features', logging):
#     col_list = [
#         'transaction_dt_bin_4368', 'transaction_dt_bin_182', 'transaction_dt_bin_91', 'elapsed_from_brawser_release', 
#         'transaction_amt_div_addr1_mean', 'nmf_c_0', 'pca_c_1', 'transaction_amt_div_card5_mean', 'mean_id_20_each_card1', 
#         'transaction_amt_div_card1_mean', 'C1_div_C14', 'TransactionAmt', 'transaction_amt_div_card5_median']
#     X_train_all = div_col(X_train_all, col_list)
#     X_test = div_col(X_test, col_list)
#     X_test = X_test[X_train_all.columns]

with timer('train and predict', logging):
    y_preds = []
    models = []
    oof = np.zeros(len(X_train_all))

    for fold_ in range(n_splits):
        print(f"=== fold{fold_} ===")
        with timer(f'fold{fold_}', logging):
            X_train, X_valid = X_train_all[folds['fold_id'] != fold_], X_train_all[folds['fold_id'] == fold_]
            y_train, y_valid = y_train_all[folds['fold_id'] != fold_], y_train_all[folds['fold_id'] == fold_]

            y_pred, model, oof = train_and_predict(
                X_train, X_valid, y_train, y_valid, X_test, PARAMS, CAT, oof
            )

            # if CALIBRATION:
            #     y_pred = calibration(y_pred, list_sampling_rate[fold_])

            log_best(model, LOSS)

            y_preds.append(y_pred)
            models.append(model)

with timer('save importances', logging):
    save_importances(models, X_train.columns, IMP_PATH, logging)

with timer('calculate score', logging):
    scores = [
        round(m.best_score['valid_1'][LOSS], 3) for m in models
    ]
    score = sum(scores) / len(scores)

# with timer('transform to rank', logging):
#     for i, preds in enumerate(y_preds):
#         y_preds[i] = preds2rank(preds) / len(preds)

with timer('create submission file', logging):
    sub = pd.read_csv(SAMPLE_PATH)
    y_sub = sum(y_preds) / len(y_preds)
    sub[TARGET_NAME] = y_sub
    fname = f'sub_{MODEL_NAME}_{NOW:%Y%m%d%H%M%S}_{score:.5f}.csv'
    OUTPUT_PATH += fname
    sub.to_csv(OUTPUT_PATH, index=False)

e = time.time()

with timer('save features', logging):
    save_features(oof, y_sub, overwrite=SAVE_OOF)

if DROPBOX:
    transfar_dropbox(OUTPUT_PATH, DROPBOX_PATH + f'data/output/{fname}', DROPBOX_TOKEN)
    transfar_dropbox(LOGGER_PATH, DROPBOX_PATH + f'logs/{LOG_FILE_NAME}', DROPBOX_TOKEN)
    transfar_dropbox(IMP_PATH, DROPBOX_PATH + f'importances/{IMP_FILE_NAME}', DROPBOX_TOKEN)

if KAGGLE_SUBMIT:
    submit_kaggle(
        path=OUTPUT_PATH,
        compe_name=COMPE_NAME,
        local_cv=score
    )

message = """{f}
cv: {cv:.4f}
scores: {s}
time: {t:.2f}[min]""".format(f=sys.argv[0], cv=score, s=scores, t=(e - s) / 60)

# 計算結果通知
send_line(LINE_TOKEN, message)

# 実験管理
send_notion(
    token_v2=TOKEN_V2,
    url=NOTION_URL,
    name=LOG_FILE_NAME,
    created=NOW,
    model=MODEL_NAME,
    local_cv=round(score, 4),
    time=round((e - s) / 60, 3),
    comment=COMMENT
)
