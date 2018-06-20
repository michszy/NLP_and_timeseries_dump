import medium
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



X = data.index
y = data.Ads
X_train, X_test, y_train, y_test = train_test_split(X , y, test_size=0.2)


r = medium.moving_average(data, 24)
y_axis = np.arange(len(X))
plt.figure(figsize=(15,7))
plt.plot(data.index, data.Ads)
#plt.scatter(data.index[-1:], r)
#plt.show()


window = 5
rolling_mean = data.rolling(window=window).mean()
#plt.plot(rolling_mean)


result = 0.0
weigths = [0.8, 0.4, 0.05, 0.05]


# weight = last entry * first weight ; second from the last entry * second weight
for n in range(len(weigths)):
    result += data.iloc[-n-1] * weigths[n]

#plt.scatter("2017-09-22 00:00:00", result.Ads)
#plt.show()

alpha = 0.9
original_result = [data.Ads[0]]
for n in range(1,len(data.Ads)):
    r = alpha * data.Ads[n] + (1 - alpha) * original_result[n - 1]
    original_result.append(r)




to_train_data = data[:-2]
to_predict_data = data[-2:]

print(to_predict_data)
print("_____________")
print(to_train_data)

alpha = 0.9
result = [to_train_data.Ads[0]]
for n in range(1, len(to_train_data)):
    r = alpha * to_train_data.Ads[n] + (1 - alpha ) * result[n-1]
    result.append(r)


first_prediction_result = alpha * to_train_data['Ads'].iloc[-1] +  (1- alpha) * result[-1]
result.append(first_prediction_result)
second_prediction_result = alpha * first_prediction_result + (1 - alpha) * result[-1]

print('_________')
plt.plot(data.index[:-1], result)
plt.scatter('2017-09-21 22:00:00', first_prediction_result)
plt.scatter('2017-09-21 23:00:00', second_prediction_result)

print(first_prediction_result)
print(second_prediction_result)
print(data[-3:])

print('______double_exponential_smoothing_________')

alpha = 0.5
beta = 0.5
to_train_data = to_train_data.Ads
result = [to_train_data[0]]
for n in range(1, len(to_train_data)+1):
    if n == 1:
        level, trend = to_train_data[0], to_train_data[1] - to_train_data[0]
    if n >= len(to_train_data):
        value = result[-1]
    else:
        value = to_train_data[n]
    last_level, level = level, alpha * value + (1 - alpha) * (level + trend)
    trend = beta * (level - last_level) + (1 - beta) * trend
    result.append(level + trend)

print(result[-1:])
plt.plot(to_train_data.index, result[:-1])
plt.show()












plt.show()

