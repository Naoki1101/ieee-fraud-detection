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
OUTPUT_PATH = './folds/folds7.feather'
SEED = config['seed']
TARGET = 'isFraud'
START_DATE = config['start_date']


# ===============
# Main
# ===============
df = pd.read_feather(INPUT_PATH)
df['datetime_'] = df['TransactionDT'].apply(lambda x: (START_DATE + datetime.timedelta(seconds=x)))
df['weekofyear'] = (df['TransactionDT'] - 86400) // (3600 * 24 * 7)

split_groups = df['weekofyear']
week_unique = split_groups.unique()

df['fold_id'] = np.nan
for week in week_unique:
    val_idx = df[df['weekofyear'] == week].index
    df.loc[val_idx, 'fold_id'] = week

df[['fold_id']].astype('int').to_feather(OUTPUT_PATH)