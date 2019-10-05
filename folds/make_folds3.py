import pandas as pd
import numpy as np
import yaml
from sklearn.model_selection import StratifiedKFold

import warnings
warnings.filterwarnings('ignore')


with open('./configs/default.yaml', 'r') as yf:
    config = yaml.load(yf)

# ===============
# Settings
# ===============
INPUT_PATH = './data/input/train_transaction.feather'
OUTPUT_PATH = './folds/folds3.feather'
N_SPLITS = 5
SEED = config['seed']
ID_NAME = config['ID_name']
FOLD_COL = 'card1'


# ===============
# Main
# ===============
df = pd.read_feather(INPUT_PATH)

folds = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
df['fold_id'] = np.nan
for fold_, (train_index, valid_index) in enumerate(folds.split(df[ID_NAME], df[FOLD_COL])):
    df.loc[valid_index, 'fold_id'] = fold_

df[['fold_id']].astype('int').to_feather(OUTPUT_PATH)
