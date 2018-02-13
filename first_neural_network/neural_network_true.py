#coding=utf-8

# %matplotib inline
# %config InlineBack.figure_format = 'retina'

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


data_path = 'Bike-Sharing-Dataset/hour.csv'
rides = pd.read_csv(data_path)
# print rides.head()
rides[:24*10].plot(x = 'dteday', y = 'cnt')
plt.show()




