import subprocess
import sys
import json
import csv

import requests

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import deque

from pandas import DataFrame, read_csv
from numpy import log

from datetime import datetime
import time
import math

from keras.models import Sequential, load_model
from keras.layers import LSTM, Dense, Dropout
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.normalization import BatchNormalization
from keras.callbacks import EarlyStopping, TensorBoard, ReduceLROnPlateau, CSVLogger

#log_dir = "streaming-update-noDiff"

#early_stopping = EarlyStopping(monitor='loss', min_delta=0, patience=5, verbose=0, mode='auto')
#logger = TensorBoard(log_dir='log/{}'.format(log_dir), write_graph=True, histogram_freq=1, batch_size=1)
#reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.2, patience=3, min_lr=0.001)
#csv_logger = CSVLogger('training-2-layer-32-32-LeakyReLU-streaming-update-noDiff-linear.csv')

requests.packages.urllib3.disable_warnings()

with open('config.json', 'r') as f:
    config = json.load(f)

metric = config['metric']
app_name = config['app_name']
app_id = config['app_id']
collector_url = config['collector_url']
down_threshold = config['down_threshold']
up_threshold = config['up_threshold']

URL = collector_url+"/v1/apps/"+app_id+"/metric_histories/"+metric
arg ="cf curl v2/apps/"+app_id+"/stats | jq 'with_entries(.value = .value.stats)'"



def mean(nums):
    return sum(nums, 0.0) / len(nums)

def predict():
    data = np.array(dq_input).reshape(1,5)
    print("in prediction function with data", data)
    train_X = data.reshape(data.shape[0],data.shape[1],1)
    train_X = train_X/100
    yhat= model.predict(train_X, batch_size=1)
    #model.reset_states()
    yhat = yhat.reshape(yhat.shape[0], yhat.shape[1])
    return yhat*100

def scale_decision():
    print("scaling decision function with response time for next 5 minutes, instance count and  provided thresold",  pred_val, instance_count, down_threshold, up_threshold)
    value = np.amax(pred_val)
    if value >= up_threshold:
        scaled_instance = math.ceil(value/up_threshold)
        if scaled_instance != instance_count:
            arg = 'cf scale ' + app_name + ' -i ' + str(scaled_instance)
            subprocess.check_output(arg, shell=True)
            print("the application is scaled")
    if value <= down_threshold:
        scaled_instance = math.ceil(value/down_threshold)
        if scaled_instance != instance_count:
            arg = 'cf scale ' + app_name + ' -i ' + str(scaled_instance)
            subprocess.check_output(arg, shell=True)
            print("the application is scaled")
    else:
        print("scaling of application is not required")
        
def streaming():
    r = requests.get(url = URL, verify=False)
    data = r.json()
    length = len(data)
    print(data)

    instance_stat = subprocess.check_output(arg, shell=True)
    instance_stat_json = json.loads(instance_stat)
    instance_count_cf = len(instance_stat_json)

    value = 0
    value_avg = 0
    for l in range(length):
        value = value + int(data[l]['value'])
    if length!=0:
        value_avg = math.ceil(float(value) / length)
    filename = "streaming-responsetime-"+config['app_name']+"-online-"+datetime.now().strftime('%Y-%m-%d')+".csv"
    f = open(filename,'a')
    handle=csv.writer(f)
    handle.writerow([datetime.now().strftime('%Y-%m-%d %H:%M'), value_avg, instance_count_cf])
    f.close()
    time.sleep(59)
    return value_avg, instance_count_cf


model = load_model('res-1000epoch-scaled-26feb.h5')

dq_input = deque([], 5)
dq_pred = deque([], 5)

for i in range(5):
    cur_val, instance_count =  streaming()
    dq_input.append(cur_val)
pred_val = predict()
scale_decision()
dq_pred.append(pred_val[0][-1])

for j in range(4):
    cur_val, instance_count =  streaming()
    dq_input.append(cur_val)
    pred_val = predict()
    scale_decision()
    dq_pred.append(pred_val[0][-1]) 

c=0
while(c != 100):
    f = open('26feb-expec-pred-resp-epoch1000scaled-model.csv','a')
    handle=csv.writer(f)
    print("in infinite loop with iteration: ", c)
    cur_val, instance_count =  streaming()
    predicted = dq_pred.popleft()
    print("expected and predicted value", cur_val, predicted)
    handle.writerow([datetime.now().strftime('%Y-%m-%d %H:%M:%S'), cur_val, predicted,instance_count])
    dq_input.append(cur_val)
    pred_val = predict()
    scale_decision()
    dq_pred.append(pred_val[0][-1]) 
    
    c = c+1
    f.close()

        
        
        












