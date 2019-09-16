import numpy as np
from aliz_estimator import ThresholdBinarizer, CustomEstimator
from data_handler import load_data, prepare_data, undersample_data
from sklearn.metrics import accuracy_score

import warnings
warnings.filterwarnings('ignore')

np.random.seed(55)

#
# Hyper parameters & config.
#
input_dir = './input'


#
# Load data
#
X, y, labels = prepare_data(load_data(input_dir), fix_imbalance='under')

#
# Pipeline
#
# steps = [('estimator', CustomEstimator(max_iter=1000)), ('binarizer', ThresholdBinarizer())]
# pipeline = Pipeline(steps)
# pipeline.fit(X, y)
# prediction = pipeline.predict(X)


#
# Estimator
#
estimator = CustomEstimator(max_iter=1000, binarize=True, random_state=55)
estimator.fit(X, y)
prediction = estimator.predict(X)

print('Accuracy %.3f' % accuracy_score(y, prediction))