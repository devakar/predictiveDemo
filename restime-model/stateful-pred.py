import csv

import numpy as np
import pandas as pd
from pandas import concat
import matplotlib.pyplot as plt

from pandas import DataFrame, read_csv
from numpy import log

from math import sqrt

from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error


from datetime import datetime
import time
import math

from keras.models import Sequential, load_model
from keras.layers import LSTM, Dense, Dropout, TimeDistributed
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.normalization import BatchNormalization
from keras.callbacks import EarlyStopping, TensorBoard, ReduceLROnPlateau, CSVLogger

np.random.seed(7)

dt = read_csv('streaming-responsetime-response-app-4days.csv', header=None, index_col=0)
dt.columns = ['value', 'instance']
dt.index.name = 'timestamp'
values = dt.values
print(values.shape)
print(type(dt))
print(dt.head())

resTime_dt = dt['value']
resTime_values = resTime_dt.values
print(resTime_values.shape)
print(type(resTime_dt))
print(resTime_dt.head())


resTime_values = resTime_values.reshape(len(resTime_values), 1)
print(type(resTime_values))
print(len(resTime_values))
print(resTime_values.shape)

# convert series to supervised learning
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
	n_vars = 1 if type(data) is list else data.shape[1]
	df = DataFrame(data)
	cols, names = list(), list()
	# input sequence (t-n, ... t-1)
	for i in range(n_in, 0, -1):
		cols.append(df.shift(i))
		names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
	# forecast sequence (t, t+1, ... t+n)
	for i in range(0, n_out):
		cols.append(df.shift(-i))
		if i == 0:
			names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
		else:
			names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
	# put it all together
	agg = concat(cols, axis=1)
	agg.columns = names
	# drop rows with NaN values
	if dropnan:
		agg.dropna(inplace=True)
	return agg

# frame as supervised learning
reframed = series_to_supervised(resTime_values, 5, 5)
print(reframed.head())


# split into train and test sets
reframed_values = reframed.values
#train = reframed_values[:-241, :]
test = reframed_values[:, :]/100
# split into input and outputs
#train_X, train_y = train[:, 0:5], train[:, 5:]
test_X, test_y = test[:, 0:5], test[:, 5:]
# reshape input to be 3D [samples, timesteps, features]
#train_X = train_X.reshape((train_X.shape[0], train_X.shape[1], 1))
test_X = test_X.reshape((test_X.shape[0], test_X.shape[1], 1))
#train_y = train_y.reshape((train_y.shape[0], train_y.shape[1], 1))
test_y = test_y.reshape((test_y.shape[0], test_y.shape[1], 1))
#print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)

#print(test_X[0:5,:,])
#print(test_y[0:5,:,])
model = load_model('res-1000epoch-scaled-26feb.h5')
yhat = model.predict(test_X, batch_size=1)
predicted = yhat*100
predicted =predicted.reshape(predicted.shape[0], predicted.shape[1])
predicted_5 = predicted[:,-1:]
print(predicted_5.shape)
#np.savetxt('predicted-stream-10day-2-32-LeakyReLU-scale1024-epoch1000-multi-stateful-all-1day.csv', predicted)
np.savetxt('predicted-stream-response-epoch1000model-scaled-26feb.csv', predicted_5)
test_y = test_y.reshape(test_y.shape[0], test_y.shape[1])
test_y = test_y*100 
print(test_y[0:5,:])
result = np.concatenate((test_y, predicted), axis=1)
np.savetxt('expected-predicted-stream-response-epoch1000model-scaled-26feb.csv', result, delimiter=',', fmt='%.5f')
# actual = memoryUsed_values.reshape(len(memoryUsed_values),1)
# actual = actual[12:,:]
# print(actual.shape)

# # actual_prediction= np.concatenate((actual, predicted), axis=1)
# # np.savetxt('actual-predicted-stream-2-32-LeakyReLU-scale1024-epoch2000.csv', actual_prediction)

dt = read_csv('expected-predicted-stream-response-epoch1000model-scaled-26feb.csv', header=None)
values = dt.values[:50,:]
print(values.shape)
print(type(dt))
print(dt.head())


l = values.shape[0]
rmse = np.ndarray(shape=(5,1), dtype=float)
mae = np.ndarray(shape=(5,1), dtype=float)
mape = np.ndarray(shape=(5,1), dtype=float)

for i in range(5):
    rmse[i][0] = sqrt(mean_squared_error(values[:,i], values[:,5+i]))
    mae[i][0] = mean_absolute_error(values[:,i], values[:,5+i])
    mape[i][0] = np.mean(np.abs((values[:,i] - values[:,5+i]) / values[:,i])) * 100
    plt.plot(values[:,i], label='Actual')
    plt.plot( values[:,5+i], label= 'Predicted')
    plt.legend(loc='upper right')
    plt.ylabel('Response Time (ms))')
    plt.xlabel('Time Instant')
    plt.title("Prediction of response time from LSTM Network")
    plt.savefig("expected-predicted-stream-response-epoch1000model-scaled-26feb-"+str(i+1)+".png")
    plt.show()

result = np.concatenate((rmse, mae, mape), axis=1)
np.savetxt('rmse-mae-mape-stream-response-epoch1000model-scaled-26feb.csv', result, delimiter=',', fmt='%.5f')

print("TEST RMSE: ", rmse)
print("TEST MAE: ", mae)
print("TEST MAPE: ", mape)


