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
INPUT_PATH = [
    './data/input/train_transaction.feather',
    './data/input/test_transaction.feather'
]
FEATURES = ['TransactionDT']
OUTPUT_PATH = './folds/folds8.feather'
N_FOLD = 6
SEED = config['seed']
START_DATE = config['start_date']


# ===============
# Main
# ===============
dfs = [pd.read_feather(f)[FEATURES] for f in INPUT_PATH]
df = pd.concat(dfs, axis=0).reset_index(drop=True)

df['datetime_'] = df['TransactionDT'].apply(lambda x: (START_DATE + datetime.timedelta(seconds=x)))
df['month'] = df['datetime_'].apply(lambda x: str(x)[:7])

split_groups = df['month']
month_unique = split_groups.unique()

df['fold_id'] = np.nan
for fold_, month in enumerate(month_unique):
    val_idx = df[df['month'] == month].index
    df.loc[val_idx, 'fold_id'] = fold_

df[['fold_id']].astype('int').to_feather(OUTPUT_PATH)
