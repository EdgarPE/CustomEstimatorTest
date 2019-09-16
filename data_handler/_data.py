# TODO: doc

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import resample


#
# Load data
#
def load_data(input_dir):
    dtypes = {
        'TransactionID': 'int32',
        'isFraud': 'int8',
        # 'TransactionDT': 'int32',
        'TransactionAmt': 'float32',
        'ProductCD': 'object',
        'card1': 'int32',
        'card2': 'float32',  # missing values
        'card3': 'float32',
        'card4': 'object',
        'card5': 'float32',  # missing values
        'card6': 'object',
        'addr1': 'float32',  # missing values
        'addr2': 'float32',  # missing values
        'dist1': 'float32',  # missing values
        'dist2': 'float32',  # missing values
    }
    train = pd.read_csv(input_dir + '/train_transaction_n20000.csv', index_col='TransactionID', dtype=dtypes)

    # drop lots of columns
    train = train[[col for col in dtypes if col != 'TransactionID']]

    # use only 1000 samples
    # train = train[0:500]

    return train


#
# Prepare data
#
def prepare_data(train, fix_imbalance=None):
    # Handle missing
    for col in [col for col in train.columns if train[col].dtype in ['int32', 'float32']]:
        train[col].fillna(0, inplace=True)

    for col in [col for col in train.columns if train[col].dtype in ['object']]:
        train[col].fillna('', inplace=True)

    # Label encoding
    labels = {}
    for col in ['ProductCD', 'card4', 'card6']:
        train.fillna('', inplace=True)
        labels[col] = LabelEncoder()
        labels[col].fit(train[col])
        train[col] = labels[col].transform(train[col]).astype('int32')

    # Fix imbalance
    if fix_imbalance == 'under':
        train = undersample_data(train)
    elif fix_imbalance == 'over':
        train = oversample_data(train)

    y = train['isFraud']
    X = train.drop(columns=['isFraud'])

    return X, y, labels


#
# Fix imbalance with under-sampling
#
def undersample_data(train):
    class_0 = train[train['isFraud'] == 0]
    class_1 = train[train['isFraud'] == 1]

    class_0_sub = class_0.sample(class_1.shape[0])

    return pd.concat([class_0_sub, class_1], axis=0)


#
# Fix imbalance with over-sampling
#
def oversample_data(train):
    class_0 = train[train['isFraud'] == 0]
    class_1 = train[train['isFraud'] == 1]

    class_1_up = resample(class_1, replace=True, n_samples=len(class_0), random_state=55)

    return pd.concat([class_0, class_1_up], axis=0)
