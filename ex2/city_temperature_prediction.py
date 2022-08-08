import IMLearn.learners.regressors.linear_regression
from IMLearn.learners.regressors import PolynomialFitting
from IMLearn.utils import split_train_test

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.io as pio

pio.templates.default = "simple_white"


def load_data(filename: str) -> pd.DataFrame:
    """
    Load city daily temperature dataset and preprocess data.
    Parameters
    ----------
    filename: str
        Path to house prices dataset

    Returns
    -------
    Design matrix and response vector (Temp)
    """
    df = pd.read_csv(filename, parse_dates=['Date'])
    df = df[df['Country'].isin(
        ['South Africa', 'The Netherlands', 'Israel', 'Jordan'])]
    df = df[df['City'].notnull()]
    df = df[df['Year'] >= 1995]
    df = df[df['Year'] <= 2020]
    df = df[df['Temp'] > -20]
    df = df[df['Temp'] < 50]

    for i in ['Year', 'Month', 'Day']:
        df = pd.concat([df, pd.get_dummies(df[i])], axis=1)
    dates_arr = []
    for date in df['Date']:
        dates_arr.append(pd.Period(date, freq='H').day_of_year)
    df['DayOfYear'] = dates_arr
    df = df.drop(['Date', 'Day'], axis=1)
    df = df.drop_duplicates()
    df = df.reset_index(drop=True)
    return df


if __name__ == '__main__':
    np.random.seed(0)
    # Question 1 - Load and preprocessing of city temperature dataset
    path = "../datasets/City_Temperature.csv"
    data = load_data(path)
    response = data['Temp']

    # Question 2 - Exploring data for specific country
    israel_data = pd.DataFrame(data[data['Country'] == 'Israel'])

    # first plot
    fig_f = px.scatter(israel_data, x=israel_data.DayOfYear,
                       y=israel_data.Temp,
                       color=israel_data.Year, title="temp - day of year",
                       labels=dict(x="Day Of Year", y='temp'))
    fig_f.write_html('tmp.html', auto_open=True)

    # second plot
    g = israel_data.groupby('Month').agg('var').reset_index(drop=True)
    g['Month'] = [m for m in range(0, 12)]
    g.reset_index(drop=True)
    fig_s = px.bar(g, x=g.Month, y=g.Temp, title="var per month",
                   labels=dict(x="month", y='var'))
    fig_s.write_html('tmp.html', auto_open=True)

    # Question 3 - Exploring differences between countries
    var = data.groupby(['Country', 'Month']).agg('var').reset_index(drop=True)
    avg = data.groupby(['Country', 'Month']).agg('mean').reset_index(drop=True)
    new = []
    for c in ['Israel', 'Jordan', 'South Africa', 'The Netherlands']:
        for m in range(12):
            new.append(str(c) + "_" + str(m))
    avg['Country_Month'] = new
    avg.reset_index(drop=True)

    avg_plot = px.line(avg, x=avg.Country_Month, y=avg.Temp,
                       title="country per month",
                       labels=dict(x="month", y='var'), error_y=var['Temp'])
    avg_plot.write_html('tmp.html', auto_open=True)

    # Question 4 - Fitting model for different values of `k`
    y = israel_data['Temp']
    israel_data = israel_data.drop(['Temp'], axis=1)
    train_X, train_Y, test_X, test_Y = split_train_test(israel_data, y)
    train_X = train_X['DayOfYear']
    test_X = test_X['DayOfYear']

    loss = []
    for k in range(1, 11):
        pf = PolynomialFitting(k + 1)
        pf.fit(train_X, train_Y)
        loss.append(pf.loss(test_X, test_Y))
    print(loss)

    # print bar plot
    plt.bar(np.arange(1, 11), loss, color='maroon', width=0.4)
    plt.xlabel("degree")
    plt.ylabel("MSE")
    plt.title("MSE  with different K")
    plt.show()

    # Question 5 - Evaluating fitted model on different countries
    countries_lst = ['South Africa', 'The Netherlands', 'Jordan']
    pf = PolynomialFitting(5)
    pf.fit(train_X, train_Y)
    loss = []
    for country in countries_lst:
        test_data = pd.DataFrame(data[data['Country'] == country])
        test_Y = test_data['Temp']
        test_X = test_data['DayOfYear']
        loss.append(pf.loss(test_X, test_Y))

    # print plot
    plt.bar(countries_lst, loss, color='maroon', width=0.4)
    plt.xlabel("Country")
    plt.ylabel("MSE")
    plt.title("MSE  for different countries")
    plt.show()
