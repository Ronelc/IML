from typing import NoReturn
from ...base import BaseEstimator
import numpy as np
from numpy.linalg import det, inv
import math


class LDA(BaseEstimator):
    """
    Linear Discriminant Analysis (LDA) classifier

    Attributes
    ----------
    self.classes_ : np.ndarray of shape (n_classes,)
        The different labels classes. To be set in `LDA.fit`

    self.mu_ : np.ndarray of shape (n_classes,n_features)
        The estimated features means for each class. To be set in `LDA.fit`

    self.cov_ : np.ndarray of shape (n_features,n_features)
        The estimated features covariance. To be set in `LDA.fit`

    self._cov_inv : np.ndarray of shape (n_features,n_features)
        The inverse of the estimated features covariance. To be set in `LDA.fit`

    self.pi_: np.ndarray of shape (n_classes)
        The estimated class probabilities. To be set in `GaussianNaiveBayes.fit`
    """

    def __init__(self):
        """
        Instantiate an LDA classifier
        """
        super().__init__()
        self.classes_, self.mu_, self.cov_, self._cov_inv, self.pi_ = None, None, None, None, None

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        fits an LDA model.
        Estimates gaussian for each label class - Different mean vector, same covariance
        matrix with dependent features.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to
        """
        # calculate n_k - number of elements in each class
        self.classes_ = np.unique(y)
        n_k_arr = np.zeros(len(self.classes_))
        for k in range(len(self.classes_)):
            n_k_arr[k] = np.count_nonzero(y == k)

        # calculate self.pi_
        self.pi_ = [i / len(y) for i in n_k_arr]

        # calculate self.mu_
        self.mu_ = []
        for k in range(len(self.classes_)):
            self.mu_.append(np.sum(X[y == k], axis=0) / n_k_arr[k])
        self.mu_ = np.array(self.mu_)

        # calculate self.cov_
        self.cov_ = np.zeros((X.shape[1], X.shape[1]))
        for k in range(len(self.classes_)):
            x_minus_mu = np.array(X[y == k] - self.mu_[k])
            self.cov_ += np.matmul(np.transpose(x_minus_mu), x_minus_mu)
        self.cov_ /= (len(y) - len(self.classes_))

        # calculate self.cov_inv
        self._cov_inv = np.linalg.inv(self.cov_)

    def _predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict responses for given samples using fitted estimator

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to predict responses for

        Returns
        -------
        responses : ndarray of shape (n_samples, )
            Predicted responses of given samples
        """
        predict = []
        for likelihood in self.likelihood(X):
            predict.append(self.classes_[np.argmax(likelihood)])
        return np.array(predict)

    def likelihood(self, X: np.ndarray) -> np.ndarray:
        """
        Calculate the likelihood of a given data over the estimated model

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Input data to calculate its likelihood over the different classes.

        Returns
        -------
        likelihoods : np.ndarray of shape (n_samples, n_classes)
            The likelihood for each sample under each of the classes

        """
        if not self.fitted_:
            raise ValueError(
                "Estimator must first be fitted before calling `likelihood` function")
        likelihood_arr = []
        for x in X:
            x_likelihood = []
            for k in range(len(self.classes_)):
                a_k = np.matmul(self._cov_inv, self.mu_[k])
                b_k = math.log(self.pi_[k]) - 0.5 * (
                    np.matmul(np.matmul(self.mu_[k], self._cov_inv),
                              self.mu_[k]))
                likelihood = np.matmul(a_k.T, x) + b_k
                x_likelihood.append(likelihood)
            likelihood_arr.append(np.array(x_likelihood))
        return np.array(likelihood_arr)

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
        from ...metrics import misclassification_error
        if not self.fitted_:
            self.fit(X, y)
        return misclassification_error(y, self.predict(X))
