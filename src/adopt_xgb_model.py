import numpy as np
import pandas as pd
#import tensorflow as tf
#from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import xgboost
#matplotlib.rcParams['font.family'] ='Malgun Gothic'
#matplotlib.rcParams['axes.unicode_minus'] =False
import matplotlib.pyplot as plt
import sys

X_test = pd.read_csv('temp.csv',encoding='cp949')

model_ip = xgboost.XGBRegressor()
model_dt = xgboost.XGBRegressor()

#model_ip.load_model('ipmodel_nodt2.json')
#model_dt.load_model('dtmodel2.json')
model_ip.load_model('ipbest_47.json')
model_dt.load_model('dtbest_47.json')


keylist = X_test.keys()
values = np.array([420.0, 3614.346, 383.5744, 600.0226, 1721.424,  0.0   ,  0.0  ,  0.053055  , 0.000000, 0    , 1    ,  -1984.84 ,  0.828 , 32.545  , 5.059 ])
#                ([  TF ,    PF1  ,  PF1_2  ,    PF6  ,    PF9  , LFS_t0 ,LFS_dt , HFS_t0,  HFS_dt , EC_2G, EC_7G, NBI_t0, NBI_dt, NBI_PW, wall ])


for i in range(15):  # without wall condition
	X_test[keylist[i]]=values[i]

#-----Single wall Part---------------------
Ip_predict = model_ip.predict(X_test)

X_test.insert(len(X_test.columns), 'Ip', Ip_predict)
dt_predict = model_dt.predict(X_test)
X_test.pop('Ip')

print("-------------Result----------------------")
print('>>>', Ip_predict[0], dt_predict[0])
print("-----------------------------------------")
#------Single wall part end----------------
