from IMLearn.utils import split_train_test
from IMLearn.learners.regressors import LinearRegression

from typing import NoReturn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio
import os.path

pio.templates.default = "simple_white"


def load_data(filename: str):
    """
    Load house prices dataset and preprocess data.
    Parameters
    ----------
    filename: str
        Path to house prices dataset

    Returns
    -------
    Design matrix and response vector (prices) - either as a single
    DataFrame or a Tuple[DataFrame, Series]
    """
    # extract df
    df = pd.read_csv(filename)

    # data process
    df = df[df['id'].notnull()]
    df = df[df['date'].notnull()]
    df = df[df['price'] > 0]
    df = df[df['sqft_lot15'] > 0]
    df = df[df['sqft_living'] > 0]
    df = df[df['floors'] > 0]
    df = df[df['view'] >= 0]
    df = df[df['view'] <= 4]
    df = df[df['condition'] > 0]
    df = df[df['condition'] < 6]
    df = df[df['yr_built'] <= 2015]
    df = df[df['yr_renovated'] <= 2015]
    df = df.drop_duplicates()


    df = pd.concat([df, pd.get_dummies(df['zipcode'])], axis=1)
    date = df['date']
    date = pd.concat(
        [date.str.slice(0, 4), date.str.slice(4, 6), date.str.slice(6, 8)],
        axis=1)
    date.columns = ['year', 'month', 'day']
    for i in ['year', 'month', 'day']:
        df = pd.concat([df, pd.get_dummies(date[i])], axis=1)

    # slice the Series vector
    Series = df['price'].reset_index(drop=True)
    # drop the categorical columns and the id
    df = df.drop(['id', 'date', 'price', 'zipcode'], axis=1)
    df = df.reset_index(drop=True)
    return df, Series


def feature_evaluation(X: pd.DataFrame, y: pd.Series,
                       output_path: str = ".") -> NoReturn:
    """
    Create scatter plot between each feature and the response.
        - Plot title specifies feature name
        - Plot title specifies Pearson Correlation between feature and response
        - Plot saved under given folder with file name including feature name
    Parameters
    ----------
    X : DataFrame of shape (n_samples, n_features)
        Design matrix of regression problem

    y : array-like of shape (n_samples, )
        Response vector to evaluate against

    output_path: str (default ".")
        Path to folder in which plots are saved
    """
    for feature in range(X.shape[1]):
        p = np.cov(X.iloc[:, feature], y)[1][0] / (
                np.std(X.iloc[:, feature]) * np.std(y))

        # create graphs
        title = "p= " + str(p) + ", " + "feature: " + str(
            X.iloc[:, feature].name)
        plt.scatter(X.iloc[:, feature], y)
        plt.title(title)
        plt.xlabel("feature")
        plt.ylabel("prices")

        # save graphs in folder
        plt.savefig(output_path + "/" + str(X.iloc[:, feature].name) + ".png")


if __name__ == '__main__':
    np.random.seed(0)

    # Question 1 - Load and preprocessing of housing prices dataset
    path = "../datasets/house_prices.csv"
    X, Y = load_data(path)

    # Question 2 - Feature evaluation with respect to response
    feature_evaluation(X, Y)

    # Question 3 - Split samples into training- and testing sets.
    train_X, train_Y, test_X, test_Y = split_train_test(X, Y)

    # Question 4 - Fit model over increasing percentages of the overall training data
    # For every percentage p in 10%, 11%, ..., 100%, repeat the following 10 times:
    #   1) Sample p% of the overall training data
    #   2) Fit linear model (including intercept) over sampled set
    #   3) Test fitted model over test set
    #   4) Store average and variance of loss over test set
    # Then plot average loss as function of training size with error ribbon of size (mean-2*std, mean+2*std)

    lr = LinearRegression()
    loss_arr, temp_loss, std_arr = [], [], []
    for p in range(10, 101):
        for iter in range(10):
            training_set_X = train_X.sample(frac=p * 0.01)
            training_set_y = train_Y[training_set_X.index]
            lr.fit(training_set_X, training_set_y)
            temp_loss.append(lr.loss(test_X, test_Y))
        loss_arr.append(np.mean(temp_loss))
        std_arr.append(np.std(temp_loss))
        temp_loss = []
    std_arr, loss_arr = np.asarray(std_arr), np.asarray(loss_arr)

    # print plot
    fig, ax = plt.subplots()
    ax.plot(np.arange(10, 101), loss_arr)
    plt.xlabel("percent of samples")
    plt.ylabel("loss")
    ax.fill_between(np.arange(10, 101), (loss_arr - 2 * std_arr),
                    (loss_arr + 2 * std_arr), alpha=.1)
    plt.show()
