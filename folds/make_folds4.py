import pandas as pd
import numpy as np
import yaml
import datetime
from sklearn.model_selection import GroupKFold

import warnings
warnings.filterwarnings('ignore')


with open('./configs/default.yaml', 'r') as yf:
    config = yaml.load(yf)

# ===============
# Settings
# ===============
INPUT_PATH = './data/input/train_transaction.feather'
OUTPUT_PATH = './folds/folds4.feather'
N_FOLD = 6
SEED = config['seed']
TARGET = 'isFraud'
START_DATE = '2017-11-30'

# ===============
# Main
# ===============
df = pd.read_feather(INPUT_PATH)
startdate = datetime.datetime.strptime(START_DATE, '%Y-%m-%d')
df['datetime_'] = df['TransactionDT'].apply(lambda x: (startdate + datetime.timedelta(seconds=x)))
df['month'] = df['datetime_'].apply(lambda x: x.month)

split_groups = df['month']
month_unique = split_groups.unique()

df['fold_id'] = np.nan
for fold_, month in enumerate(month_unique):
    val_idx = df[df['month'] == month].index
    df.loc[val_idx, 'fold_id'] = fold_

df[['fold_id']].astype('int').to_feather(OUTPUT_PATH)
