from __future__ import annotations
from typing import Tuple, NoReturn
from ...base import BaseEstimator
import numpy as np
from itertools import product


def misclassification_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    sum = 0
    for i in range(len(y_true)):
        if np.sign(y_true[i]) != np.sign(y_pred[i]):
            sum += abs(y_true[i])
    return sum / len(y_true)


class DecisionStump(BaseEstimator):
    """
    A decision stump classifier for {-1,1} labels according to the CART algorithm

    Attributes
    ----------
    self.threshold_ : float
        The threshold by which the data is split

    self.j_ : int
        The index of the feature by which to split the data

    self.sign_: int
        The label to predict for samples where the value of the j'th feature is about the threshold
    """

    def __init__(self) -> DecisionStump:
        """
        Instantiate a Decision stump classifier
        """
        super().__init__()
        self.threshold_, self.j_, self.sign_ = None, None, None

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        fits a decision stump to the given data

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to
        """
        loss = 1
        for index in range(X.shape[1]):
            for sign in [-1, 1]:
                threshold, min_loss = self._find_threshold(X[:, index], y,
                                                           sign)
                if min_loss < loss:
                    loss = min_loss
                    self.threshold_, self.j_, self.sign_ = threshold, index, sign

    def _predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict responses for given samples using fitted estimator

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to predict responses for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to

        Returns
        -------
        responses : ndarray of shape (n_samples, )
            Predicted responses of given samples

        Notes
        -----
        Feature values strictly below threshold are predicted as `-sign` whereas values which equal
        to or above the threshold are predicted as `sign`
        """
        feature_to_predict_by = X.T[self.j_]
        predict = np.where(feature_to_predict_by < self.threshold_,
                           -self.sign_, self.sign_)
        return predict

    def _find_threshold(self, values: np.ndarray, labels: np.ndarray,
                        sign: int) -> Tuple[float, float]:
        """
        Given a feature vector and labels, find a threshold by which to perform a split
        The threshold is found according to the value minimizing the misclassification
        error along this feature

        Parameters
        ----------
        values: ndarray of shape (n_samples,)
            A feature vector to find a splitting threshold for

        labels: ndarray of shape (n_samples,)
            The labels to compare against

        sign: int
            Predicted label assigned to values equal to or above threshold

        Returns
        -------
        thr: float
            Threshold by which to perform split

        thr_err: float between 0 and 1
            Misclassificaiton error of returned threshold

        Notes
        -----
        For every tested threshold, values strictly below threshold are predicted as `-sign` whereas values
        which equal to or above the threshold are predicted as `sign`
        """
        sorted_indexes = np.argsort(values)
        values, labels = values[sorted_indexes], labels[sorted_indexes]
        thresholds = np.concatenate(
            [[-np.inf], (values[1:] + values[:-1]) / 2, [np.inf]])
        min_los = np.sum(labels)
        losses = np.append(min_los, min_los - np.cumsum(labels * sign))
        min_index = np.argmin(losses)
        return thresholds[min_index], losses[min_index]

    def _loss(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Evaluate performance under misclassification loss function

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Test samples

        y : ndarray of shape (n_samples, )
            True labels of test samples

        Returns
        -------
        loss : float
            Performance under missclassification loss function
        """
        sum = 0
        y_pred = self.predict(X)
        for i in range(len(y)):
            if y[i] != y_pred[i]:
                sum += 1
        return sum / len(y)
