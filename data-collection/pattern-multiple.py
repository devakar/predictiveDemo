import json
import subprocess
import time
import random

with open('config.json', 'r') as f:
    config = json.load(f)

base_url=config['app_url']
fast_url=base_url+"/fast"
slow_url=base_url+"/slow/"

flag=True
while(flag):
    for i in range(1,10,1):
        arg = "curl "+fast_url+ " -k"
        subprocess.check_output(arg, stderr=subprocess.STDOUT, shell=True)
        print("fast", arg)
        time.sleep(30)
    for i in range(1,12,1):
        times = (5*i)+4+random.randint(3,5)
        arg = "curl "+slow_url+str(times) + " -k"
        subprocess.check_output(arg, stderr=subprocess.STDOUT, shell=True)
        print("slow", arg)
        time.sleep(30)
