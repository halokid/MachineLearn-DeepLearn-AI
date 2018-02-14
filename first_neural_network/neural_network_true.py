#coding=utf-8

# %matplotib inline
# %config InlineBack.figure_format = 'retina'

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


#设置pandas的显示宽度
pd.set_option('display.max_colwidth', 10000)

#获取数据
data_path = 'Bike-Sharing-Dataset/hour.csv'
rides = pd.read_csv(data_path)
# print rides.head()
rides[:24*10].plot(x = 'dteday', y = 'cnt')
#显示图形
# plt.show()


#虚拟变量
dummy_fields = ['season', 'weathersit', 'mnth', 'hr', 'weekday']
for each in dummy_fields:
  dummies = pd.get_dummies(rides[each], prefix=each, drop_first=False)
  rides = pd.concat([rides, dummies], axis=1)

fields_to_drop = ['instant', 'dteday', 'season', 'weathersit',
                  'weekday', 'atemp', 'mnth', 'workingday', 'hr']
data = rides.drop(fields_to_drop, axis=1)
print data.head()



