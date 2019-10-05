import os
from tqdm import tqdm
from itertools import combinations
import numpy as np
from scipy.stats import rankdata
import requests
import dropbox
from notion.client import NotionClient


# =============================================================================
# XXXXX
# =============================================================================
def reduce_mem_usage(df):
    """ iterate through all the columns of a dataframe and modify the data type
        to reduce memory usage.
    """
    start_mem = df.memory_usage().sum() / 1024 ** 2
    # print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))
    # print("column = ", len(df.columns))
    for i, col in enumerate(df.columns):
        try:
            col_type = df[col].dtype

            if col_type != object:
                c_min = df[col].min()
                c_max = df[col].max()
                if str(col_type)[:3] == 'int':
                    if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                        df[col] = df[col].astype(np.int8)
                    elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                        df[col] = df[col].astype(np.int16)
                    elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                        df[col] = df[col].astype(np.int32)
                    elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                        df[col] = df[col].astype(np.int32)
                else:
                    if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                        df[col] = df[col].astype(np.float32)
                    elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                        df[col] = df[col].astype(np.float32)
                    else:
                        df[col] = df[col].astype(np.float32)
        except:
            continue

    end_mem = df.memory_usage().sum() / 1024 ** 2
    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    # print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))

    return df


def preds2rank(preds):
    rank_ = rankdata(preds)
    return rank_


def extract_drop_index(target, r=0.5):
    target_true_index = list(target[target == 1].index)
    target_false_index = list(target[target == 0].index)
    sample_size = int(len(target_false_index) * r)

    if len(target_true_index) >= sample_size:
        sample_size = len(target_true_index)

    dont_use_index = np.random.choice(target_false_index, sample_size, replace=False).tolist()

    return sorted(dont_use_index)


def resample(target_series, sample_size, r=5):
    arr = np.where(target_series == 0, 1, r)
    prob = arr / np.sum(arr)

    sample_result = np.random.choice(list(target_series.index), sample_size, p=prob, replace=True).tolist()

    return sorted(sample_result)


def calibration(y_proba, beta):
    return y_proba / (y_proba + (1 - y_proba) / beta)


def div_col(df, col_list):
    s = 1e-3
    comb = list(combinations(col_list, 2))

    for col1, col2 in comb:
        col_name = f'{col1}_div_{col2}'
        df[col_name] = df[col1] / (df[col2] + s)

    return df


# =============================================================================
# Other API
# =============================================================================
def submit_kaggle(path, compe_name, local_cv):
    cmd = f'kaggle competitions submit -c {compe_name} -f {path}  -m "{local_cv}"'
    os.system(cmd)


def transfar_dropbox(input_path, output_path, token):
    dbx = dropbox.Dropbox(token)
    dbx.users_get_current_account()
    with open(input_path, 'rb') as f:
        dbx.files_upload(f.read(), output_path)


# =============================================================================
# Notification
# =============================================================================
def send_line(line_token, message):
    endpoint = 'https://notify-api.line.me/api/notify'
    message = "\n{}".format(message)
    payload = {'message': message}
    headers = {'Authorization': 'Bearer {}'.format(line_token)}
    requests.post(endpoint, data=payload, headers=headers)


def send_notion(token_v2, url, name, created, model, local_cv, time, comment):
    client = client = NotionClient(token_v2=token_v2)
    cv = client.get_collection_view(url)
    row = cv.collection.add_row()
    row.name = name
    row.created = created
    row.model = model
    row.local_cv = local_cv
    row.time = time
    row.comment = comment
