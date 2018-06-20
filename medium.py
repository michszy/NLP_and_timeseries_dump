import pandas as pd
import matplotlib.pyplot as plt
import math
import numpy as np
import seaborn as sns


from dateutil.relativedelta import relativedelta
from scipy.optimize import minimize

import statsmodels.formula.api as smf
import statsmodels.tsa.api as smt
import statsmodels.api as sm

from statsmodels.tsa.stattools import acf
from sklearn.model_selection import train_test_split

import scipy.stats as scs
from itertools import product
from tqdm import tqdm_notebook
from sklearn.metrics import r2_score, median_absolute_error, mean_absolute_error
from sklearn.metrics import median_absolute_error, mean_squared_error, mean_squared_log_error


import warnings
warnings.filterwarnings('ignore')




data = pd.read_csv('mlcourse_open/data/ads.csv', index_col=['Time'], parse_dates=['Time'])




# la moyenne de deplacement de la courbe sur les dernières indice de temps
def moving_average(series, n):
    return np.average(series[-n:])

print(moving_average(data, 24))

def mean_absolute_percentage_error(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


def auto_correlation(data):
    data = data['Ads']
    correlated_data = acf(data)
    y_axis = np.arange(len(correlated_data))
    plt.plot(y_axis, correlated_data)
    #plt.show()

#auto_correlation(data)


def plotMovingAverage(series, window, plot_intervals=True, scale=1.96, plot_anomalies= True ):
    plt.figure(figsize=(15, 7))
    plt.plot(data.Ads)
    plt.title('Ads watched (hourly data)')
    plt.grid(True)
    """
        series - dataframe with timeseries
        window - rolling window size
        plot_intervals - show confidence intervals
        plot_anomalies - show anomalies
    """

    rolling_mean = series.rolling(window=window).mean()
    plt.title("Moving average,  window size = {}".format(window))
    plt.plot(rolling_mean, "g", label='Rolling mean trend')

    if plot_intervals:
        mae = mean_absolute_error(series[window:],  rolling_mean[window:])
        deviation = np.std(series[window:] - rolling_mean[window:])
        lower_bond = rolling_mean - (mae + scale * deviation)
        upper_bond = rolling_mean + (mae + scale * deviation)
        plt.plot(upper_bond, "r--", label = "Upper Bond / Lower Bond")
        plt.plot(lower_bond, "r--")

        if plot_anomalies:
            anomalies = pd.DataFrame(index=series.index, columns=series.columns)
            anomalies[series < lower_bond] = series[series < lower_bond]
            anomalies[series > upper_bond] = series[series > upper_bond]
            plt.plot(anomalies, "ro", markersize=10)
    plt.plot(series[window:], label="Actual values")
    plt.legend(loc = "upper left")
    plt.grid(True)
    #plt.show()


def weighted_average(series, weights):
    """
    Calculate weighter average on series
    """
    result = 0.0
    weights.reverse()
    for n in range(len(weights)):
        result += series.iloc[-n-1] * weights[n]
    return float(result)


#plotMovingAverage(data, 4)
r= weighted_average(data, [0.6, 0.3, 0.1, 0.05])
# print(r)

print('______')

def exponenetial_smoothing(series, alpha):
    """
    :param series:  dataset with timestamps
    :param alpha: float [0.0, 1.0, smooting parameter
    :return:
    """

    result = [series[0]] # first value is same as series
    for n in range(1, len(series)):
        result.append(alpha * series[n] + (1 - alpha) * result[n-1])
    return result

def plotExponentialSmoothing(series, alphas):
    """
    Plot exponential smoothing with differente alphas
    :param series: dataset with timestamps
    :param alphas: list of floats, smoothing parameters
    :return:
    """

    with plt.style.context('seaborn-white'):
        plt.figure(figsize=(15,7))
        for alpha in alphas:
            plt.plot(exponenetial_smoothing(series, alpha), label="Alpha {}".format(alpha))
        plt.plot(series.values, "c", label = 'Actual')
        plt.legend(loc='best')
        plt.axis=('tight')
        plt.title("Exponential Smoothing")
        plt.grid(True)
        #plt.show();

#plotExponentialSmoothing(data.Ads, [0.01, 0.2])


def double_exponential_smoothing(series, alpha, beta):
    """

    :param series: dataset_with timeseries
    :param alpha: float between 0 and 1, smoothing parameter for level
    :param beta:  float between 0 and 1, smoothing parameter for trend
    :return:
    """

    result = [series[0]]
    for n in range(1, len(series)+1):
        if n == 1:
            level, trend = series[0], series[1] - series[0]
        if n >= len(series) :
            value = result[-1]
        else:
            value = series[n]
        last_level, level = level, alpha * value + (1 - alpha) * (level + trend)
        trend = beta * (level - last_level) + (1 - beta) * trend
        result.append(level + trend)
    return result


def plotDoubleExponentialSmoothing(series, alphas, betas):
    """
    Plots double exponential smoothing with different alphas and betas
    series - dataset with timestamps
    alphas - list of floats, smoothing parameters for level
    betas - list of floats, smoothing parameters for trend
    :return:
    """

    with plt.style.context("seaborn-white"):
        plt.figure(figsize=(20, 8))
        for alpha in alphas:
            for beta in betas:
                 plt.plot(double_exponential_smoothing(series, alpha, beta), label= "Alpha {}, beta {}".format(alpha, beta))
        plt.plot(series.values, label="Actual")
        plt.legend(loc="best")
        plt.axis("tight")
        plt.title("Double Exponential Smoothing")
        plt.grid(True)
        #plt.show()


#plotDoubleExponentialSmoothing(data.Ads, alphas=[0.09], betas=[0.9, 0.02])