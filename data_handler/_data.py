import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import resample


def load_data(input_dir):
    """
    Data loader helper function.

    :param input_dir: The input file directory.
    :return: pandas.DataFrame
    """
    dtypes = {
        'isFraud': 'int8',
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
    train = pd.read_csv(input_dir + '/train_transaction_n20000.csv', dtype=dtypes)

    # drop lots of columns
    train = train[[col for col in dtypes if col != 'TransactionID']]

    return train


def prepare_data(train, fix_imbalance=None):
    """
    Some data preprocessing and optional imbalance fixing.

    :param train: pandas.DataFrame of imput data
    :param fix_imbalance: None, 'over' or 'under' for over- or under-sampling.
    :return: pandas.DataFrame preprocessed data.
    """
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


def undersample_data(train):
    """
    Fix imbalance with under-sampling

    :param train: pandas.Dataframe
    :return: pandas.Dataframe
    """
    class_0 = train[train['isFraud'] == 0]
    class_1 = train[train['isFraud'] == 1]

    class_0_sub = class_0.sample(class_1.shape[0])

    return pd.concat([class_0_sub, class_1], axis=0).sort_index().reset_index(drop=True)


def oversample_data(train):
    """
    Fix imbalance with obrt-sampling

    :param train: pandas.Dataframe
    :return: pandas.Dataframe
    """
    class_0 = train[train['isFraud'] == 0]
    class_1 = train[train['isFraud'] == 1]

    class_1_up = resample(class_1, replace=True, n_samples=len(class_0), random_state=55)

    return pd.concat([class_0, class_1_up], axis=0).sort_index().reset_index(drop=True)


def fast_auc(y_true, y_prob):
    """
    Fast roc_auc computation helper:
    https://www.kaggle.com/c/microsoft-malware-prediction/discussion/76013
    """
    y_true = np.asarray(y_true)
    y_true = y_true[np.argsort(y_prob)]
    nfalse = 0
    auc = 0
    n = len(y_true)
    for i in range(n):
        y_i = y_true[i]
        nfalse += (1 - y_i)
        auc += y_i * nfalse
    auc /= (nfalse * (n - nfalse))
    return auc
