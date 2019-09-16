import numpy as np
from aliz_estimator import CustomEstimator
from data_handler import load_data, prepare_data
from sklearn.metrics import accuracy_score, f1_score, fbeta_score, confusion_matrix, precision_score, recall_score, \
    balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold

import warnings
warnings.filterwarnings('ignore')


def false_positive_rate(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return fp / (fp + tn)


def false_negative_rate(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return fn / (fn + tp)


def fast_auc(y_true, y_prob):
    """
    fast roc_auc computation: https://www.kaggle.com/c/microsoft-malware-prediction/discussion/76013
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


def do_estimate(metrics, X, y, max_iter):
    metric_data = {}
    for m in metrics:
        metric_data[m] = {'train': 0, 'valid': 0}

    folds = StratifiedKFold(n_splits=5)
    for (fold_n, (train_idx, valid_idx)) in enumerate(folds.split(X, y)):
        X_train, X_valid = X.loc[train_idx], X.loc[valid_idx]
        y_train, y_valid = y.loc[train_idx], y.loc[valid_idx]

        print('Started fold %d of %d.' % ((fold_n + 1), folds.get_n_splits()))

        estimator = CustomEstimator(max_iter=max_iter, random_state=55)
        estimator.fit(X_train, y_train)

        pred_train = estimator.predict(X_train)
        pred_valid = estimator.predict(X_valid)
        proba_train = estimator.predict_proba(X_train)
        proba_valid = estimator.predict_proba(X_valid)

        for metric_name in metrics:
            metric_func = metrics[metric_name]
            metric_data[metric_name]['train'] += metric_func(y_train,
                                                             proba_train if metric_name == 'ROC AUC' else pred_train)
            metric_data[metric_name]['valid'] += metric_func(y_valid,
                                                             proba_valid if metric_name == 'ROC AUC' else pred_valid)

    for metric_name in sorted(metrics.keys()):
        # Mean of metrics
        metric_data[metric_name]['train'] /= folds.get_n_splits()
        metric_data[metric_name]['valid'] /= folds.get_n_splits()

        print('% -14s on train: %0.4f, on cross-validation: %0.4f' % (metric_name, metric_data[metric_name]['train'],
                                                                      metric_data[metric_name]['valid']))


#
# Config and hyper-params
#
input_dir = './input'
max_iter = 2000

#
# Cross validation metrics
#
metrics = {
    'Accuracy': accuracy_score,
    'Balanced acc.': balanced_accuracy_score,
    'F1': f1_score,
    'F2': lambda y_true, y_pred: fbeta_score(y_true, y_pred, 2),
    'False pos.': false_positive_rate,
    'False neg.': false_negative_rate,
    'Precision': precision_score,
    'Recall': recall_score,
    'ROC AUC': fast_auc,
}

#
# Load data and estimate
#
train = load_data(input_dir)

print('Do estimation without imbalance correction.')
X, y, _ = prepare_data(train, fix_imbalance=None)
do_estimate(metrics, X, y, max_iter)
print()

print('Do estimation with under-sampling.')
X, y, _ = prepare_data(train, fix_imbalance='under')
do_estimate(metrics, X, y, max_iter)
