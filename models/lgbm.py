import numpy as np
import pandas as pd
from pathlib import Path
import lightgbm as lgb
import logging

from logs.logger import log_evaluation

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings('ignore')


def train_and_predict(X_train, X_valid, y_train, y_valid, X_test, lgbm_params, cat, oof=None):

    lgb_train = lgb.Dataset(X_train, y_train, categorical_feature=cat)
    lgb_eval = lgb.Dataset(X_valid, y_valid, reference=lgb_train, categorical_feature=cat)

    logging.debug(lgbm_params)

    logger = logging.getLogger('main')
    callbacks = [log_evaluation(logger, period=100)]

    num_round = 10000
    model = lgb.train(
        lgbm_params,
        lgb_train,
        num_boost_round=num_round,
        valid_sets=[lgb_train, lgb_eval],
        verbose_eval=200,
        early_stopping_rounds=200,
        callbacks=callbacks
    )

    if oof is not None:
        oof[X_valid.index] = model.predict(X_valid, num_iteration=model.best_iteration)
    y_pred = model.predict(X_test, num_iteration=model.best_iteration)

    return y_pred, model, oof


def save_importances(models, tr_cols, path, logging):
    feature_importance_df = pd.DataFrame()
    for fold_, model in enumerate(models):
        fold_importance_df = pd.DataFrame()
        fold_importance_df["Feature"] = tr_cols
        fold_importance_df["importance"] = model.feature_importance()
        fold_importance_df["fold"] = fold_ + 1
        feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)

    unique_feature_importance_df = (feature_importance_df[["Feature", "importance"]]
                                    .groupby("Feature")
                                    .mean()
                                    .sort_values(by="importance", ascending=False))

    logging.debug(f'all features: {list(unique_feature_importance_df.index)}')
    logging.debug(f'low importance features: {list(unique_feature_importance_df[unique_feature_importance_df["importance"] <= 300].index)}')

    cols = (unique_feature_importance_df.index)

    best_features = feature_importance_df.loc[feature_importance_df.Feature.isin(cols)]

    plt.figure(figsize=(14, 100))
    sns.barplot(x="importance",
                y="Feature",
                data=best_features.sort_values(by="importance",
                                               ascending=False))
    plt.title('LightGBM Features (avg over folds)')
    plt.tight_layout()
    plt.savefig(path)


def save_features(oof, pred, overwrite=False, fname=None):
    root = Path('./features/')
    if fname is None:
        train_path = root.joinpath('lgbm_train.feather')
        test_path = root.joinpath('lgbm_test.feather')
    else:
        train_path = root.joinpath(f'{fname}_train.feather')
        test_path = root.joinpath(f'{fname}_test.feather')

    if train_path.exists() and test_path.exists() and not overwrite:
        print('lgbm was skipped')
    else:
        pd.DataFrame(oof, columns=['lgbm_pred']).to_feather(str(train_path))
        pd.DataFrame(pred, columns=['lgbm_pred']).to_feather(str(test_path))
