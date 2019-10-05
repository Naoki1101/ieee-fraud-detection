import pandas as pd
import numpy as np
import yaml
import datetime
from sklearn.model_selection import train_test_split

import warnings
warnings.filterwarnings('ignore')


with open('./configs/default.yaml', 'r') as yf:
    config = yaml.load(yf)

# ===============
# Settings
# ===============
INPUT_PATH = './data/input/train_transaction.feather'
OUTPUT_PATH = './folds/folds6.feather'
SEED = config['seed']
TARGET = 'isFraud'
START_DATE = config['start_date']


# ===============
# Main
# ===============
df = pd.read_feather(INPUT_PATH)
df['datetime_'] = df['TransactionDT'].apply(lambda x: (START_DATE + datetime.timedelta(seconds=x)))
df['month'] = df['datetime_'].apply(lambda x: x.month)

df_ = df[df['month'].isin([3, 4, 5])].iloc[:, :20]
x_train, x_val, y_train, y_val = train_test_split(df_,
                                                  df_['isFraud'],
                                                  test_size=0.33,
                                                  shuffle=True,
                                                  random_state=SEED,
                                                  stratify=df_['isFraud'])

df['fold_id'] = np.nan
df.loc[x_val.index, 'fold_id'] = 0
df['fold_id'].fillna(-1, inplace=True)

df[['fold_id']].astype('int').to_feather(OUTPUT_PATH)