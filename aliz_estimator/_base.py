"""
SciKit-Learn fashioned Custom Estimator.
"""
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import binarize


class CustomEstimator(BaseEstimator, ClassifierMixin):
    """
    This is a custom classifier, which is using sklearn LogisticRegression to calculate probabilities.

    Internally it uses the ThresholdBinarizer to optimize the threshold with which predicts class 0 or 1.
    """

    def __init__(self, random_state=None, solver='warn', max_iter=100):
        self.random_state = random_state
        self.solver = solver
        self.max_iter = max_iter

    def fit(self, X, y):
        """
        The usual scikit-learn fit method implementation from BaseEstimator.

        :param X: array-like, shape (n_samples, n_features) The input samples.
        :param y: array-like, shape (n_samples, ) The output classes
        :return:
        """


        # Check that X and y have correct shape
        X, y = check_X_y(X, y)
        # Store the classes seen during fit
        self.classes_ = unique_labels(y)

        self.X_ = X
        self.y_ = y

        # Setup LogisticRegression and call fit()
        self._logit = LogisticRegression(max_iter=self.max_iter, solver=self.solver, random_state=self.random_state)
        self._logit.fit(self.X_, self.y_)

        # Setup ThresholdBinarizer, fit() store y_true
        self._binarizer = ThresholdBinarizer()
        self._binarizer.fit(y.reshape(-1, 1))

        # Return the classifier
        return self

    def predict_proba(self, X):
        """
        Probability prediction, without binarization.

        :param X: array-like, shape (n_samples, n_features) The input samples.
        :return: ndarray, shape (n_samples,) The probility of class-1 for each sample.
        """

        # Check is fit had been called
        check_is_fitted(self, ['X_', 'y_'])

        # Input validation
        X = check_array(X)

        # Return Logistic Regression prediction probabilities (2nd column)
        return self._logit.predict_proba(X)[:, 1]

    def predict(self, X):
        """
        A reference implementation of a prediction for a classifier.

        This method uses the ThresholdBinarizer to make discrete predictions.

        :param X: array-like, shape (n_samples, n_features) The input samples.
        :return: ndarray, shape (n_samples,)
            The label for each sample is the label of the closest sample
            seen during fit.
        """
        pred = self.predict_proba(X)

        bin_pred = self._binarizer.transform(pred.reshape(-1, 1))
        pred = bin_pred.reshape(-1)

        return pred


class ThresholdBinarizer(BaseEstimator, TransformerMixin):
    """
    Binarize data (set feature values to 0 or 1) according to a threshold

    Decision threshold is computed internally based on best gini score.
    """

    def fit(self, X, y=None):
        """
        Just store X values as true values, needed later for gini computation.

        :param X: array-like
        :param y: ignored
        :return: self
        """

        check_array(X, accept_sparse='csr')

        self._X_true = X
        self._threshold = None

        return self

    def _gini_impurity(self, fact, pred, classes):
        """
        Gini impurity calculator

        :param fact: ndarray, shape (n_samples,)
        :param pred: ndarray, shape (n_samples,)
        :param classes: list of classes
        :return: float value of gini impurity
        """

        assert len(fact) == len(pred)
        len_ = len(fact)

        g_sum = 0.0
        for c in classes:
            pred_ = pred[fact == c]
            g_sum += len(fact[fact == c]) / len_ * len(pred_[pred[fact == c] != c]) / len_

        return g_sum


    def _find_threshold(self, fact, pred_prob):
        """
        Helper method to find the threshold.

        :param fact: ndarray, shape (n_samples,) Fact values
        :param pred_prob: ndarray, shape (n_samples,) Predicted probabilities
        :return: Float
        """

        # Sorted ndarray of probabilities, with extra 0.0 and 1.0 at the ends.
        tsh = np.vstack(([[0.]], np.sort(pred_prob, axis=0), [[1.]]))

        # tuples of (threshold, gini impurity value)
        tsh_gini = []
        for i in range(1, tsh.shape[0]):
            threshold = np.mean((tsh[i - 1, 0], tsh[i, 0]))
            pred = binarize(pred_prob, threshold=threshold, copy=True).astype('int')
            gini_value = self._gini_impurity(fact.reshape(-1), pred.reshape(-1), [0, 1])
            tsh_gini.append((threshold, gini_value))

        tsh_gini = np.array(tsh_gini)
        return tsh_gini[tsh_gini[:, 1] == np.amin(tsh_gini, axis=0)[1]][0][0]

    def transform(self, X_pred):
        """
        The scikit-learn fashioned transform() method. It does the binarization each element of X

        :param X_pred:
        :param copy:
        :return:
        """

        if self._threshold is None:
            self._threshold = self._find_threshold(self._X_true, X_pred)

        return binarize(X_pred, threshold=self._threshold)
