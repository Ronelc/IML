from __future__ import annotations
from copy import deepcopy
from typing import Tuple, Callable
import numpy as np
from IMLearn import BaseEstimator


def MSE(estimator: BaseEstimator, X: np.ndarray, y: np.ndarray):
    y_hat = estimator.predict(X)
    return np.nanmean(((y_hat - y) ** 2))


def cross_validate(estimator: BaseEstimator, X: np.ndarray, y: np.ndarray,
                   scoring: Callable[[np.ndarray, np.ndarray, ...], float],
                   cv: int = 5) -> Tuple[float, float]:
    """
    Evaluate metric by cross-validation for given estimator

    Parameters
    ----------
    estimator: BaseEstimator
        Initialized estimator to use for fitting the data

    X: ndarray of shape (n_samples, n_features)
       Input data to fit

    y: ndarray of shape (n_samples, )
       Responses of input data to fit to

    scoring: Callable[[np.ndarray, np.ndarray, ...], float]
        Callable to use for evaluating the performance of the cross-validated model.
        When called, the scoring function receives the true- and predicted values for each sample
        and potentially additional arguments. The function returns the score for given input.

    cv: int
        Specify the number of folds.

    Returns
    -------
    train_score: float
        Average train score over folds

    validation_score: float
        Average validation score over folds
    """
    train_score_arr, validation_score_arr = [], []
    # randomize = np.arange(len(y))
    # np.random.shuffle(randomize)
    # X = X[randomize]
    # y = y[randomize]
    p = X.shape[0] // cv
    if X.shape[1] == 1:
        X, y = X.flatten(), y.flatten()
    train_X, train_Y, test_X, test_Y = None, None, None, None
    for i in range(cv):
        if i == 0:
            train_X, train_Y, test_X, test_Y = X[p:], y[p:], X[:p], y[:p]
        elif i + 1 < cv:
            train_X = np.concatenate([X[:i * p], X[(i + 1) * p:]])
            train_Y = np.concatenate([y[:i * p], y[(i + 1) * p:]])
            test_X, test_Y = X[i * p:(i + 1) * p], y[i * p:(i + 1) * p]
        else:
            train_X, train_Y = X[:i * p], y[:i * p]
            test_X, test_Y = X[i * p:], y[i * p:]

        estimator.fit(train_X, train_Y)
        y_hat_train = estimator.predict(train_X)
        train_score_arr.append(scoring(y_hat_train, train_Y))
        y_hat_test = estimator.predict(test_X)
        validation_score_arr.append(scoring(y_hat_test, test_Y))
    return np.mean(train_score_arr), np.mean(validation_score_arr)

    # from sklearn.model_selection import KFold
    # # X, y = X.flatten(), y.flatten()
    # kf = KFold(cv)
    # for train_index, test_index in kf.split(X):
    #     train_X, test_X = X[train_index], X[test_index]
    #     train_Y, test_Y = y[train_index], y[test_index]
    #     estimator.fit(train_X, train_Y)
    #     y_hat_train = estimator.predict(train_X)
    #     train_score_arr.append(scoring(y_hat_train, train_Y))
    #     y_hat_test = estimator.predict(test_X)
    #     validation_score_arr.append(scoring(y_hat_test, test_Y))
    # return np.mean(train_score_arr), np.mean(validation_score_arr)
