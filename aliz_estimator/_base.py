# TODO: doc

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from sklearn.metrics import euclidean_distances
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import binarize


# TODO: doc

class CustomEstimator(BaseEstimator, ClassifierMixin):
    """ An example classifier which implements a 1-NN algorithm.

    For more information regarding how to build your own classifier, read more
    in the :ref:`User Guide <user_guide>`.

    Parameters
    ----------
    demo_param : str, default='demo'
        A parameter used for demonstation of how to pass and store paramters.

    Attributes
    ----------
    X_ : ndarray, shape (n_samples, n_features)
        The input passed during :meth:`fit`.
    y_ : ndarray, shape (n_samples,)
        The labels passed during :meth:`fit`.
    classes_ : ndarray, shape (n_classes,)
        The classes seen at :meth:`fit`.
    """
    def __init__(self, random_state=None, solver='warn', max_iter=100, binarize=False):
        self.random_state = random_state
        self.solver = solver
        self.max_iter = max_iter
        self.binarize = binarize

    def fit(self, X, y):
        """A reference implementation of a fitting function for a classifier.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The training input samples.
        y : array-like, shape (n_samples,)
            The target values. An array of int.

        Returns
        -------
        self : object
            Returns self.
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
        if (self.binarize):
            self._binarizer = ThresholdBinarizer()
            self._binarizer.fit(y.reshape(-1, 1))

        # Return the classifier
        return self

    def predict(self, X):
        """ A reference implementation of a prediction for a classifier.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        y : ndarray, shape (n_samples,)
            The label for each sample is the label of the closest sample
            seen during fit.
        """
        # Check is fit had been called
        check_is_fitted(self, ['X_', 'y_'])

        # Input validation
        X = check_array(X)

        # Return Logistic Regression prediction probabilities (2nd column)
        pred = self._logit.predict_proba(X)[:, 1]

        if (self.binarize):
            bin_pred = self._binarizer.transform(pred.reshape(-1, 1), copy=True)
            pred = bin_pred.reshape(-1)

        return pred


# TODO: doc

class ThresholdBinarizer(BaseEstimator, TransformerMixin):
    """Binarize data (set feature values to 0 or 1) according to a threshold

    Values greater than the threshold map to 1, while values less than
    or equal to the threshold map to 0. With the default threshold of 0,
    only positive values map to 1.

    Binarization is a common operation on text count data where the
    analyst can decide to only consider the presence or absence of a
    feature rather than a quantified number of occurrences for instance.

    It can also be used as a pre-processing step for estimators that
    consider boolean random variables (e.g. modelled using the Bernoulli
    distribution in a Bayesian setting).

    Read more in the :ref:`User Guide <preprocessing_binarization>`.

    Parameters
    ----------
    threshold : float, optional (0.0 by default)
        Feature values below or equal to this are replaced by 0, above it by 1.
        Threshold may not be less than 0 for operations on sparse matrices.

    copy : boolean, optional, default True
        set to False to perform inplace binarization and avoid a copy (if
        the input is already a numpy array or a scipy.sparse CSR matrix).

    Examples
    --------
    >>> from sklearn.preprocessing import Binarizer
    >>> X = [[ 1., -1.,  2.],
    ...      [ 2.,  0.,  0.],
    ...      [ 0.,  1., -1.]]
    >>> transformer = Binarizer().fit(X)  # fit does nothing.
    >>> transformer
    Binarizer(copy=True, threshold=0.0)
    >>> transformer.transform(X)
    array([[1., 0., 1.],
           [1., 0., 0.],
           [0., 1., 0.]])

    Notes
    -----
    If the input is a sparse matrix, only the non-zero values are subject
    to update by the Binarizer class.

    This estimator is stateless (besides constructor parameters), the
    fit method does nothing but is useful when used in a pipeline.

    See also
    --------
    binarize: Equivalent function without the estimator API.
    """

    def __init__(self, threshold=0.0, copy=True):
        self.threshold = threshold
        self.copy = copy

    def fit(self, X, y=None):
        """Do nothing and return the estimator unchanged

        This method is just there to implement the usual API and hence
        work in pipelines.

        Parameters
        ----------
        X : array-like
        """
        check_array(X, accept_sparse='csr')

        self._X_true = X

        return self

    def _gini_impurity(self, fact, pred, classes):
        assert len(fact) == len(pred)
        len_ = len(fact)

        g_sum = 0.0
        for c in classes:
            pred_ = pred[fact == c]
            g_sum += (len(fact[fact == c]) / len_) * (len(pred_[pred_ != c]) / len_)

        return g_sum

    def _find_threshold(self, fact, pred_prob):
        # print(pred_prob)
        # print(np.amax(pred_prob, axis=None))

        def gini(fact, pred, classes):
            assert len(fact) == len(pred)
            len_ = len(fact)

            g_sum = 0.0
            for c in classes:
                pred_ = pred[fact == c]
                g_sum += len(fact[fact == c]) / len_ * len(pred_[pred[fact == c] != c]) / len_

            return g_sum

        # Sorted ndarray of probabilities, with extra 0.0 and 1.0 at the ends.
        tsh = np.vstack(([[0.]], np.sort(pred_prob, axis=0), [[1.]]))

        # tuples of (threshold, gini impurity value)
        tsh_gini = []
        for i in range(1, tsh.shape[0]):
            threshold = np.mean((tsh[i - 1, 0], tsh[i, 0]))
            pred = binarize(pred_prob, threshold=threshold, copy=True).astype('int')
            gini_value = gini(fact.reshape(-1), pred.reshape(-1), [0, 1])
            tsh_gini.append((threshold, gini_value))

        tsh_gini = np.array(tsh_gini)
        return tsh_gini[tsh_gini[:, 1] == np.amin(tsh_gini, axis=0)[1]][0][0]

    def transform(self, X_pred, copy=None):
        """Binarize each element of X

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape [n_samples, n_features]
            The data to binarize, element by element.
            scipy.sparse matrices should be in CSR format to avoid an
            un-necessary copy.

        copy : bool
            Copy the input X or not.
        """

        threshold = self._find_threshold(self._X_true, X_pred)

        return binarize(X_pred, threshold=threshold)
