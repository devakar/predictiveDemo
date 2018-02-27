import subprocess
import sys
import json
import csv

import numpy as np
import pandas as pd
from pandas import concat
import matplotlib.pyplot as plt

from pandas import DataFrame, read_csv
from numpy import log

from datetime import datetime
import time
import math

# from keras.backend import manual_variable_initialization
# manual_variable_initialization(True)

from keras.models import Sequential, load_model
from keras.layers import LSTM, Dense, Dropout, TimeDistributed
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.normalization import BatchNormalization
from keras.callbacks import EarlyStopping, TensorBoard, ReduceLROnPlateau, CSVLogger

#log_dir = "stream-1day-stateful-2-32-LeakyReLU-scale1024-epoch500-multi-stateful"

early_stopping = EarlyStopping(monitor='loss', min_delta=0, patience=20, verbose=0, mode='auto')
#logger = TensorBoard(log_dir='log/{}'.format(log_dir), write_graph=True, histogram_freq=1, batch_size=1)
reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.2, patience=10, min_lr=0.001)
#csv_logger = CSVLogger('train-stream-1day-stateful-2-32-LeakyReLU-scale1024-epoch500-multi-stateful.csv')

dt = read_csv('streaming-responsetime-response-app.csv', header=None, index_col=0)
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
resTime_values = resTime_values/100
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
train = reframed_values[:, :]
#test = reframed_values[-241:, :]
# split into input and outputs
train_X, train_y = train[:, 0:5], train[:, 5:]
#test_X, test_y = test[:, 0:5], test[:, 5:]
# reshape input to be 3D [samples, timesteps, features]
train_X = train_X.reshape((train_X.shape[0], train_X.shape[1], 1))
#test_X = test_X.reshape((test_X.shape[0], test_X.shape[1], 1))
train_y = train_y.reshape((train_y.shape[0], train_y.shape[1], 1))
#test_y = test_y.reshape((test_y.shape[0], test_y.shape[1], 1))
#print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)

# design network
model = Sequential()
model.add(LSTM(32, batch_input_shape=(1, train_X.shape[1], train_X.shape[2]), stateful=True, return_sequences=True, name='layer-1'))
model.add(LeakyReLU(alpha=.001))
model.add(LSTM(32, stateful=True, return_sequences=True, name='layer-2'))
model.add(LeakyReLU(alpha=.001))
model.add(Dense(1, name='output-layer'))
model.compile(loss='mse', optimizer='adam', metrics=['mse', 'mae', 'mape'])
for i in range(500):
    print("Number of iteration", i+1)
    model.fit(train_X, train_y, epochs=1, batch_size=1, verbose=2, shuffle=False, callbacks=[reduce_lr, early_stopping])
    model.reset_states()
model.save("res-500epoch-scaled-23feb.h5")





        
        
        












