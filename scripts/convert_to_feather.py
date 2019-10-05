import pandas as pd
from pathlib import Path

target = [
    'train_identity',
    'train_transaction',
    'test_identity',
    'test_transaction'
]

extension = 'csv'

for t in target:
    (pd.read_csv('./data/input/' + t + '.' + extension, encoding="utf-8"))\
        .to_feather('./data/input/' + t + '.feather')
