from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn import datasets
from IMLearn.metrics import mean_square_error
from IMLearn.utils import split_train_test
from IMLearn.model_selection import cross_validate
from IMLearn.learners.regressors import PolynomialFitting, LinearRegression, \
    RidgeRegression
from sklearn.linear_model import Lasso

from utils import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt


def MSE(y_hat: np.ndarray, y: np.ndarray):
    return np.mean(((y_hat - y) ** 2))


def f(x):
    return (x + 3) * (x + 2) * (x + 1) * (x - 1) * (x - 2)


def select_polynomial_degree(n_samples: int = 100, noise: float = 5):
    """
    Simulate data from a polynomial model and use cross-validation to select the best fitting degree

    Parameters
    ----------
    n_samples: int, default=100
        Number of samples to generate

    noise: float, default = 5
        Noise level to simulate in responses
    """

    # Question 1 - Generate dataset for model f(x)=(x+3)(x+2)(x+1)(x-1)(x-2) + eps for eps Gaussian noise
    # and split into training- and testing portions

    X = np.random.uniform(-1.2, 2, n_samples)
    eps = np.random.normal(0, noise, n_samples)
    y = np.zeros(n_samples)
    for i in range(n_samples):
        y[i] = (X[i] + 3) * (X[i] + 2) * (X[i] + 1) * (X[i] - 1) * (X[i] - 2) + \
               eps[i]
    train_X, train_Y, test_X, test_Y = split_train_test(pd.DataFrame(X),
                                                        pd.Series(y), 2 / 3)
    train_X, train_Y, test_X, test_Y = train_X.to_numpy(), train_Y.to_numpy(), test_X.to_numpy(), test_Y.to_numpy()

    # Plot
    x = np.linspace(-1.2, 2, 1000)
    plt.scatter(x, f(x), c="black", linestyle="-")
    plt.scatter(train_X, train_Y, c="red", alpha=0.5, label="train")
    plt.scatter(test_X, test_Y, c="blue", alpha=0.5, label="test")
    plt.title(
        f'Scatter plot of  true model and the two sets.\n noise level: {noise}')
    plt.xlabel('X')
    plt.ylabel('y')
    plt.show()

    # Question 2 - Perform CV for polynomial fitting with degrees 0,1,...,10
    train_score_arr, validation_score_arr = [], []
    for k in range(11):
        pf = PolynomialFitting(k + 1)
        train_score, validation_score = cross_validate(pf, train_X, train_Y,
                                                       MSE, 5)
        # print(train_score, validation_score)
        train_score_arr.append(train_score)
        validation_score_arr.append(validation_score)

    # plot
    polynomial_deg = [i for i in range(11)]
    plt.plot(polynomial_deg, train_score_arr)
    plt.plot(polynomial_deg, validation_score_arr)
    plt.title(
        f"validation error VS train error\n number of samples: {n_samples}, noise level: {noise}")
    plt.xlabel("polynomial deg")
    plt.ylabel("Error")
    plt.legend(["Train", "Validation"])
    plt.grid()
    plt.show()

    # Question 3 - Using best value of k, fit a k-degree polynomial model and report test error
    min_k = np.argmin(validation_score_arr)
    pf = PolynomialFitting(min_k)
    pf.fit(train_X.flatten(), train_Y.flatten())
    error = pf._loss(test_X.flatten(), test_Y.flatten())
    print(f"best k is: {min_k}", end="\n")
    print(f"test error is: {error}", end="\n")
    print(f"best validation error is: {validation_score_arr[min_k]}", end="\n")


def select_regularization_parameter(n_samples: int = 50,
                                    n_evaluations: int = 500):
    """
    Using sklearn's diabetes dataset use cross-validation to select the best fitting regularization parameter
    values for Ridge and Lasso regressions

    Parameters
    ----------
    n_samples: int, default=50
        Number of samples to generate

    n_evaluations: int, default = 500
        Number of regularization parameter values to evaluate for each of the algorithms
    """
    # Question 6 - Load diabetes dataset and split into training and testing portions
    X, y = datasets.load_diabetes(return_X_y=True)
    train_X, train_Y = X[:n_samples], y[:n_samples]
    test_X, test_Y = X[n_samples:], y[n_samples:]

    # Question 7 - Perform CV for different values of the regularization parameter for Ridge and Lasso regressions
    l_validation_lst, r_validation_lst = [], []
    l_train_lst, r_train_lst = [], []
    for lam in range(0, 6):
        lasso, ridge = Lasso(alpha=lam), RidgeRegression(lam)
        l_train_score, l_validation_score = cross_validate(lasso, train_X,
                                                           train_Y, MSE, 5)
        l_validation_lst.append(l_validation_score)
        l_train_lst.append(l_train_score)
        r_train_score, r_validation_score = cross_validate(ridge, train_X,
                                                           train_Y, MSE, 5)
        r_validation_lst.append(r_validation_score)
        r_train_lst.append(r_train_score)

    lambda_range = [i for i in range(0, 6)]
    plt.plot(lambda_range, r_validation_lst)
    plt.plot(lambda_range, r_train_lst)
    plt.title("find fit lambda for Ridge")
    plt.xlabel("lambda value")
    plt.ylabel("Error")
    plt.legend(["validation", "train"])
    plt.grid()
    plt.show()

    # lambda_range = [i for i in range(3000, 6000)]
    plt.plot(lambda_range, l_validation_lst)
    plt.plot(lambda_range, l_train_lst)
    plt.title("find fit lambda for  Lasso")
    plt.xlabel("lambda value")
    plt.ylabel("Error")
    plt.legend([" validation", "train"])
    plt.grid()
    plt.show()

    # Question 8 - Compare best Ridge model, best Lasso model and Least Squares model
    l_min_reg = np.argmin(l_validation_lst)
    r_min_reg = np.argmin(r_validation_lst)

    print(f"minimal regularization parameter for Ridge is: {r_min_reg}")
    print(f"minimal regularization parameter for Lasso is: {l_min_reg}")

    # fit all models
    lasso, ridge, lr = Lasso(alpha=l_min_reg), RidgeRegression(
        r_min_reg), LinearRegression()
    ridge.fit(train_X, train_Y)
    lasso.fit(train_X, train_Y)
    lr.fit(train_X, train_Y)

    # calculate the loss
    print(f"MSE of Lasso is: {MSE(lasso.predict(test_X), test_Y)}")
    print(f"MSE of Ridge is: {MSE(ridge.predict(test_X), test_Y)}")
    print(f"MSE of Linear Regression is: {MSE(lr.predict(test_X), test_Y)}")


if __name__ == '__main__':
    np.random.seed(0)

    # first part
    select_polynomial_degree(100, 5)
    select_polynomial_degree(100, 0)
    select_polynomial_degree(1500, 10)

    # second part
    select_regularization_parameter(50, 500)
