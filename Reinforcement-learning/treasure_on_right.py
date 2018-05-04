#!coding=utf-8
'''
A simple example for Reinforcement Learning using table lookup Q-learning method.
An agent "o" is on the left of a 1 dimensional world, the treasure is on the rightmost location.
Run this program and to see how the agent will improve its strategy of finding the treasure.
View more on my tutorial page: https://morvanzhou.github.io/tutorials/
'''


import numpy as np
import pandas as pd
import time

#reproducible
np.random.seed(2)


N_STATES = 6   # 1维世界的宽度
ACTIONS = ['left', 'right']     # 探索者的可用动作
EPSILON = 0.9   # 贪婪度 greedy
ALPHA = 0.1     # 学习率
GAMMA = 0.9    # 奖励递减值
MAX_EPISODES = 13   # 最大回合数
FRESH_TIME = 0.3    # 移动间隔时间



def build_q_table(n_states, actions):
  table = pd.DataFrame(np.zeros(n_states, len(actions)), columns=actions,)
  # print(table)
  return table

