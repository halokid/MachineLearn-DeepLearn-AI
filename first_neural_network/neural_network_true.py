#coding=utf-8

# %matplotib inline
# %config InlineBack.figure_format = 'retina'

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys

#设置pandas的显示宽度
pd.set_option('display.max_colwidth', 10000)

#获取数据
data_path = 'Bike-Sharing-Dataset/hour.csv'
rides = pd.read_csv(data_path)
# print rides.head()
rides[:24*10].plot(x = 'dteday', y = 'cnt')
#显示图形
# plt.show()


#虚拟变量(哑变量）
dummy_fields = ['season', 'weathersit', 'mnth', 'hr', 'weekday']
for each in dummy_fields:
  dummies = pd.get_dummies(rides[each], prefix=each, drop_first=False)
  rides = pd.concat([rides, dummies], axis=1)

fields_to_drop = ['instant', 'dteday', 'season', 'weathersit',
                  'weekday', 'atemp', 'mnth', 'workingday', 'hr']
data = rides.drop(fields_to_drop, axis=1)
# print data.head()


#调整目标变量
quant_features = ['casual', 'registered', 'cnt', 'temp', 'hum', 'widspeed']
scaled_features = {}
for each in quant_features:
  mean, std = data[each].mean(), data[each].std()
  data.loc[:, each] = (data[each] - mean) / std


#将数据拆分为训练、测试和验证数据集
#save the last 21 days
test_data = data[-21 * 24:]   #最后21行数据作为测试数据
data = data[:-21 * 24]        #最后21行其余的数据作为训练数据

#separate the data into features and targets
target_fields = ['cnt', 'casual', 'registered']
features, targets = data.drop(target_fields, axis=1), data[target_fields]
test_features, test_targets = test_data.drop(target_fields, axis=1), test_data[target_fields]


train_features, train_targets = features[:-60 * 24], targets[:-60 * 24]
val_features, val_targets = features[-60 * 24], targets[-60 * 24]


#============================= 数据治理阶段完毕 ========================================


class NeuralNetwork(object):
  def __init__(self, input_nodes, hidden_nodes, output_nodes, learning_rate):
    # Set number of nodes in input, hidden and output layers.
    self.input_nodes = input_nodes
    self.hidden_nodes = hidden_nodes
    self.output_nodes = output_nodes

    # Initialize weights
    self.weights_input_to_hidden = np.random.normal(0.0, self.hidden_nodes**-0.5,
                                                    (self.hidden_nodes, self.input_nodes))

    self.weights_hidden_to_output = np.random.normal(0.0, self.output_nodes**-0.5,
                                                     (self.output_nodes, self.hidden_nodes))
    self.lr = learning_rate

    #### Set this to your implemented sigmoid function ####
    # Activation function is the sigmoid function
    self.activation_function = self.sigmoid

  def sigmoid(self , x):
    return 1 / (1 + np.exp(-x))

  def train(self, inputs_list, targets_list):
    # Convert inputs list to 2d array
    inputs = np.array(inputs_list, ndmin=2).T
    targets = np.array(targets_list, ndmin=2).T

    # ------------- SHAPES NOTE --------------#
    # shape of inputs and targets will be (56,1) #
    # shape of weights_input_to_hidden will be (2, 56) #
    # shape of weights_hidden_to_output will be (1, 2) #
    # ------------- SHAPES NOTE --------------#

    #### Implement the forward pass here ####
    ### Forward pass ###
    # TODO: Hidden layer
    hidden_inputs = np.dot(self.weights_input_to_hidden , inputs)
    hidden_outputs = self.activation_function(hidden_inputs)

    # TODO: Output layer
    final_inputs = np.dot(self.weights_hidden_to_output , hidden_outputs)
    final_outputs = final_inputs

    #### Implement the backward pass here ####
    ### Backward pass ###
    output_errors = (targets - final_outputs) * 1

    # TODO: Backpropagated error
    hidden_errors = (self.weights_hidden_to_output).T * output_errors * hidden_outputs * (1-hidden_outputs)
    hidden_grad = np.dot( hidden_errors ,inputs.T )

    # TODO: Update the weights
    self.weights_hidden_to_output += self.lr * np.dot(output_errors , hidden_outputs.T)
    self.weights_input_to_hidden += self.lr * hidden_grad


  def run(self, inputs_list):
    # Run a forward pass through the network
    inputs = np.array(inputs_list, ndmin=2).T

    #### Implement the forward pass here ####
    # TODO: Hidden layer
    hidden_inputs = np.dot(self.weights_input_to_hidden , inputs)
    hidden_outputs = self.activation_function(hidden_inputs)

    # TODO: Output layer
    final_inputs = np.dot(self.weights_hidden_to_output , hidden_outputs)
    final_outputs = final_inputs

    return final_outputs


def MSE(y, Y):
  return np.mean((y-Y)**2)



### Set the hyperparameters here ###
epochs = 200
learning_rate = 0.6
hidden_nodes = 4
output_nodes = 1

N_i = train_features.shape[1]

network = NeuralNetwork(N_i, hidden_nodes, output_nodes, learning_rate)

losses = {'train':[], 'validation':[]}
for e in range(epochs):
  # Go through a random batch of 128 records from the training data set
  batch = np.random.choice(train_features.index, size=128)
  for record, target in zip(train_features.ix[batch].values,
                            train_targets.ix[batch]['cnt']):
    network.train(record, target)

    # shape of record is (56,)

  # Printing out the training progress
  train_loss = MSE(network.run(train_features), train_targets['cnt'].values)
  val_loss = MSE(network.run(val_features), val_targets['cnt'].values)
  sys.stdout.write("\rProgress: " + str(100 * e/float(epochs))[:4] \
                   + "% ... Training loss: " + str(train_loss)[:5] \
                   + " ... Validation loss: " + str(val_loss)[:5])

  losses['train'].append(train_loss)
  losses['validation'].append(val_loss)


plt.plot(losses['train'], label='Training loss')
plt.plot(losses['validation'], label='Validation loss')
plt.legend()
plt.ylim(ymax=0.5)


fig, ax = plt.subplots(figsize=(8,4))

mean, std = scaled_features['cnt']
predictions = network.run(test_features)*std + mean
ax.plot(predictions[0], label='Prediction')
ax.plot((test_targets['cnt']*std + mean).values, label='Data')
ax.set_xlim(right=len(predictions))
ax.legend()

dates = pd.to_datetime(rides.ix[test_data.index]['dteday'])
dates = dates.apply(lambda d: d.strftime('%b %d'))
ax.set_xticks(np.arange(len(dates))[12::24])
_ = ax.set_xticklabels(dates[12::24], rotation=45)













