import pandas as pd
import matplotlib.pyplot as plt
data = pd.read_csv('mlcourse_open/data/ads.csv')
import math
import numpy as np

print(type(data['Ads']))

plt.plot(np.arange(0,len(data)),data["Ads"])



def first_solution_prediction(data, h, t):
    data = data[h[0]:h[1]+1]
    length = (h[1] - h[0])
    w = 1 /length
    sum = np.sum(data['Ads'])
    y = sum * w
    print(data['Ads'])
    print(length)
    print(w)
    print(sum)
    print(y)
    return y


plt.scatter(216, first_solution_prediction(data, [140, 215], 216))

def second_solution_prediction(data, h, t):
    data = data[h[0]:h[1] + 1]
    length = (h[1] - h[0])

    y = 0
    return y

r_second_solution = second_solution_prediction(data,[215, 215], 216)




plt.show()