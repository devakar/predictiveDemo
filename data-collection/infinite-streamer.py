import requests
import math
import csv
import subprocess
import json
from datetime import datetime
import numpy as np
import time

requests.packages.urllib3.disable_warnings()

with open('config.json', 'r') as f:
    config = json.load(f)

URL = config['collector_url']+"/v1/apps/"+config['app_id']+"/metric_histories/"+config['metric']
arg ="cf curl v2/apps/"+config['app_id']+"/stats | jq 'with_entries(.value = .value.stats)'"

flag=True
mem_quota  = 1024
while(flag):
    r = requests.get(url = URL, verify=False)
    data = r.json()
    length = len(data)
    print(data)
    
    try:
        instance_stat = subprocess.check_output(arg, shell=True)
        instance_stat_json = json.loads(instance_stat)
        instance_count_cf = len(instance_stat_json)
        mem_quota = instance_stat_json['0']['mem_quota']/(1024*1024)
    except subprocess.CalledProcessError as e:
        instance_count_cf = -1

    if config['metric'] == "memoryused":
        value = np.zeros(shape=(instance_count_cf), dtype=float)
        instances = np.zeros(shape=(instance_count_cf), dtype=float)
        for l in range(length):
            value[data[l]['instance_index']] = value[data[l]['instance_index']] + int(data[l]['value'])
            instances[data[l]['instance_index']] = instances[data[l]['instance_index']] + 1
        value_per_instance = value/instances
        value_total = np.sum(value_per_instance)
        filename = "streaming-"+config['metric']+"-"+config['app_name']+datetime.now().strftime('%Y-%m-%d')+".csv"
        f = open(filename,'a')
        handle=csv.writer(f)
        handle.writerow([datetime.now().strftime('%Y-%m-%d %H:%M'), value_total, mem_quota, instance_count_cf])
        f.close()

    if config['metric'] == "responsetime":
        value = 0
        value_avg = 0
        for l in range(length):
            value = value + int(data[l]['value'])
        if length!=0:
            value_avg = math.ceil(float(value) / length)
        filename = "streaming-"+config['metric']+"-"+config['app_name']+datetime.now().strftime('%Y-%m-%d')+".csv"
        f = open(filename,'a')
        handle=csv.writer(f)
        handle.writerow([datetime.now().strftime('%Y-%m-%d %H:%M'), value_avg, instance_count_cf])
        f.close()
    
    time.sleep(59)