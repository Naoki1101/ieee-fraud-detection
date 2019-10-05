import datetime
import logging
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import argparse
import yaml
import pandas as pd
import numpy as np
import pickle

from scripts import load_datasets, load_target, timer

import optuna
import lightgbm as lgb

import warnings
warnings.filterwarnings('ignore')


# ===============
# Settings
# ===============
parser = argparse.ArgumentParser()
parser.add_argument('--config', default='./configs/default.yaml')
options = parser.parse_args()
config = yaml.load(open(options.config))

MODEL_NAME = 'lgbm'
SEED = config['seed']
NOW = datetime.datetime.now()
LOG_FILE_NAME = f'log_optuna_{MODEL_NAME}_{NOW:%Y%m%d%H%M%S}.log'
LOGGER_PATH = './logs/' + LOG_FILE_NAME
CAT = config['categorical_features']
FEATURES = config['features']
TARGET_NAME = config['target_name']

logging.basicConfig(filename=LOGGER_PATH, level=logging.DEBUG)
logging.debug(LOGGER_PATH)
logging.debug(CAT)
logging.debug(FEATURES)

np.random.seed(SEED)


# ===============
# Function
# ===============
def objective(trial):
    train_x, test_x, train_y, test_y = train_test_split(X_train_all,
                                                        y_train_all,
                                                        test_size=0.20,
                                                        random_state=SEED,
                                                        shuffle=True)

    dtrain = lgb.Dataset(train_x, label=train_y, categorical_feature=CAT)
    dval = lgb.Dataset(test_x, label=test_y, categorical_feature=CAT)

    num_round = 10000
    param = {
        'objective': 'binary',
        'metric': 'auc',
        'verbosity': -1,
        'max_depth': -1,
        'feature_fraction': trial.suggest_loguniform('feature_fraction', 0.5, 0.9),
        'bagging_fraction': trial.suggest_loguniform('bagging_fraction', 0.5, 0.9),
        'bagging_seed': SEED,
        'boosting_type': 'gbdt',
        'num_leaves': trial.suggest_int('num_leaves', 2**5, 2**10),
        'learning_rate': 0.01,
        'min_child_samples': trial.suggest_int('min_child_samples', 20, 1000),
        'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 10, 1000),
        'lambda_l1': trial.suggest_loguniform('lambda_l1', 1e-3, 1.0)
    }

    gbm = lgb.train(param, dtrain, num_round, valid_sets=[dtrain, dval], verbose_eval=100, early_stopping_rounds=200)
    preds = gbm.predict(test_x)
    auc = roc_auc_score(test_y, preds)
    return 1 - auc


# ===============
# Main
# ===============
with timer('load data', logging):
    with open('./data/else/col2path.pkl', 'rb') as f:
        col2path = pickle.load(f)
    X_train_all, X_test = load_datasets(FEATURES, col2path)
    y_train_all = load_target(TARGET_NAME)
    logging.debug(f'feature num: {len(X_train_all.columns)}')

with timer('concat oof', logging):
    path = './features/lgbm_{data}.feather'
    lgbm_pred_train = pd.read_feather(path.format(data='train'))
    lgbm_pred_test = pd.read_feather(path.format(data='test'))

    X_train_all = pd.concat([X_train_all, lgbm_pred_train], axis=1)
    X_test = pd.concat([X_test, lgbm_pred_test], axis=1)

with timer('optimize', logging):
    study = optuna.create_study()
    study.optimize(objective, n_trials=100)

print('Number of finished trials: {}'.format(len(study.trials)))
logging.debug('Number of finished trials: {}'.format(len(study.trials)))

print('Best trial:')
trial = study.best_trial
logging.debug('Best trial:')
logging.debug(study.best_trial)

print('  Value: {}'.format(1 - trial.value))
logging.debug('  Value: {}'.format(1 - trial.value))

print('  Params: ')
logging.debug('  Params: ')
for key, value in trial.params.items():
    print('    {}: {}'.format(key, value))
    logging.debug('    {}: {}'.format(key, value))
