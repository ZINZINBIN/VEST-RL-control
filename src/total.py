#!/APP/anaconda3-2020.11/bin/python

import os, sys, prof
import numpy             as np
import pandas            as pd
import matplotlib.pyplot as plt
# XGB
from sklearn.model_selection import train_test_split
import xgboost
# LSTM
from tensorflow              import convert_to_tensor
from tensorflow.keras.models import load_model
from joblib                  import load


# sys.argv description
# python adopt_lstm_model.py dir_name

# Input list
TF 		= 420.0
PF1		= 3614.346
PF1_2	= 383.5744
PF6		= 600.0226
PF9		= 1721.424
LFS_t0	= 0.0   
LFS_dt	= 0.0  
HFS_t0	= 0.053055  
HFS_dt	= 0.000000
EC_2G	= 0    
EC_7G	= 1    
NBI_t0	= -1984.84 
NBI_dt	= 0.828 
NBI_PW	= 32.545  
wall	= 5.059 

rmajor0	= 0.45			# [m] 0.3-0.5  usually around ~0.42
rmajor1	= 0.35			# 
aminor0	= 0.30			# [m] 0.2-0.35 usually around ~0.30
aminor1	= 0.20			# 
elong0	= 1.40			#     1.4-1.8 decrease
elong1	= 1.40
tria0	= 0.20			#     0.2-0.5 decrease
tria1	= 0.20
#qa0		= 3.7			#     2.0-5.0 decrease
#qa1		= 2.0

# XGB part
X_test = pd.read_csv('temp.csv',encoding='cp949')

model_ip = xgboost.XGBRegressor()
model_dt = xgboost.XGBRegressor()

model_ip.load_model('./xgb_model/ipbest_47.json')
model_dt.load_model('./xgb_model/dtbest_47.json')

keylist = X_test.keys()
values = [TF, PF1, PF1_2, PF6, PF9, LFS_t0, LFS_dt, HFS_t0, HFS_dt, EC_2G, EC_7G, NBI_t0, NBI_dt, NBI_PW, wall]
values = np.array(values)

for i in range(15):  # without wall condition
	X_test[keylist[i]]=values[i]

Ip_predict = model_ip.predict(X_test)					# kA
X_test.insert(len(X_test.columns), 'Ip', Ip_predict)
dt_predict = model_dt.predict(X_test)					# ms
X_test.pop('Ip')

print("-------------Result----------------------")
print('>>>', Ip_predict[0], dt_predict[0])
print("-----------------------------------------")

# LSTM part

# File options
dir_name  = sys.argv[1]
#path_top  = "/home/hcho/2023/vest-nn/works"
path_top    = "./"
path_home   = path_top + dir_name
path_input  = path_home + "/input.csv"
path_output = path_home + "/output.csv"
path_time   = path_home + "/time.csv"

if not os.path.exists(path_home): os.mkdir(path_home)

# Input settings, use XGB results
shot   = 10000 		  # Dummy
dt     = 1e-5         # [s]
t0     = 0.0
t1     = dt_predict[0] / 1e3
trange = np.arange(t0, t1, dt)
ip0     =  6.0e4		# [A] >= 30kA
ip1     = Ip_predict[0] * 1e3 # [A]


# Write input file
with open(path_input, "w") as f:
	
	f.write("SHOT,TIME,IP,RMAJOR,AMINOR,ELONG,TRIA,QA\n")

	for t in trange:

		ip     = prof.f_ip(t, ip0, ip1, t0, t1)
		rmajor = prof.quadratic_decrease(t, rmajor0, rmajor1, t0, t1)
		aminor = prof.quadratic_decrease(t, aminor0, aminor1, t0, t1)
		elong  = prof.linear_decrease(t, elong0, elong1, t0, t1)
		tria   = prof.linear_decrease(t,  tria0,  tria1, t0, t1)
		qa     = prof.f_qa2(aminor, rmajor, 0.18, ip, elong)

		f.write("%5d,%e,%e,%e,%e,%e,%e,%e\n"%(shot, t, ip,
		                                      rmajor,aminor,
		                                      elong,tria, qa))

print("Finished writing input file [%s]"%dir_name)
# /// END OF MAKE INPUT ///


# Model Directory
path_nn      = "./lstm_model"
path_models  = path_nn + "/models"
path_scalers = path_nn + "/scalers"

# Inputs
INPUT_PARAMS  = ["IP", "RMAJOR", "AMINOR", "ELONG", "TRIA", "QA"]

# Split into Sequences
dfi = pd.read_csv(path_input)

input_shotnums = list(dfi["SHOT"])
shot_list = list(set(dfi["SHOT"]))
shot_list.sort() 

time_data   = dfi["TIME"].to_numpy()
input_data  = dfi[INPUT_PARAMS].to_numpy()

window_size = 15

# Declare Ensemble Arrays
IDs = list(range(1,5))
pred_output_seq  = []
sigma_output_seq = []

# Load best model
MODEL = 'wdia_mid_simple'

# Predictions
for i, ID in enumerate(IDs):

	# Load Scaler 
	path_scaler = path_scalers + "/scaler_%s"%MODEL + str(ID)
	scaler = load(open(path_scaler, "rb"))

	# Transform Input
	input_nosplit = scaler.transform(input_data)# Scaling inputs

	# Make datasets 
	time_seq  = []
	input_seq = []

	for shot in shot_list:  # Shot numbers loop

		tmp_times   = []
		tmp_inputs  = []
		tmp_outputs = []

		for j in range(len(input_shotnums)):# Add rows with the 
											# same shot numbers 
			if int(input_shotnums[j]) == int(shot):
				tmp_times.append(time_data[j])
				tmp_inputs.append(input_nosplit[j])

		# Equal-shot-number data arrays
		shot_times  = np.array(tmp_times)
		shot_inputs = np.array(tmp_inputs)

		for k in range(shot_times.shape[0] - window_size + 1): 
			time_seq.append(shot_times[k+window_size-1])
			input_seq.append(shot_inputs[k:k+window_size])

	time_seq  = np.array(time_seq)
	input_seq = np.array(input_seq)

	inputs_tensor = convert_to_tensor(input_seq)# Convert inputs into tensor form

	# Load Submodel
	path_submodel = path_models + "/%s"%MODEL + str(ID) + ".h5"
	submodel  = load_model(path_submodel)

	# Predict Output
	output = submodel.predict(input_seq)
	pred_output_seq.append(output)

# Ensemble Averaging
pred_output_seq = np.array(pred_output_seq)
pred_output_seq_mean = np.mean(pred_output_seq, axis=0).reshape(1,-1)[0]
pred_output_sigma = np.std(pred_output_seq, axis=0).reshape(1,-1)[0]

save_array = np.array([pred_output_seq_mean, pred_output_sigma])
save_array = np.transpose(save_array)

# Save Results
dfo = pd.DataFrame(save_array, columns=["PRED", "SIGMA"])
dfo.to_csv(path_output)
dft = pd.DataFrame(time_seq, columns=["TIME"])
dft.to_csv(path_time)


# Load Results
dfi = pd.read_csv(path_input)
dfo = pd.read_csv(path_output)
dft = pd.read_csv(path_time)

t0 = dft["TIME"].iloc[0]
for j,t in enumerate(list(dfi["TIME"])):
	if t == t0:
		dfi = dfi.iloc[j:]
		break

# Read input
t  = dfi["TIME"]
ip = dfi["IP"]
R  = dfi["RMAJOR"]
a  = dfi["AMINOR"]
k  = dfi["ELONG"]
d  = dfi["TRIA"]
qa = dfi["QA"]

# Read output
Wmin = dfo["PRED"] - dfo["SIGMA"]
Wmean= dfo["PRED"]
Wmax = dfo["PRED"] + dfo["SIGMA"]

bt = np.array([0.18] * len(dfi["TIME"]) )

# Plot
plt.subplot(511)
plt.plot(t, ip/1e6, label='Ip [MA]') #[kA]
plt.plot(t, bt    , label='BT [T]') #[T]
plt.legend()
plt.subplot(512)
plt.plot(t, R     , label='R [m]') #[m]
plt.plot(t, a     , label='a [m]') #[m]
plt.ylabel('Radius')
plt.legend()
plt.subplot(513)
plt.plot(t, k     , label='Elongation')
plt.plot(t, d     , label='Triangularity')
plt.ylabel('Shape')
plt.legend()
plt.subplot(514)
plt.plot(t, qa    , label='qa')
plt.ylabel('qa')
plt.legend()
plt.subplot(515)
plt.plot(t, Wmean, 'r')
plt.fill_between(t, Wmin, Wmax, alpha=0.2, color='r')
plt.ylabel('Wdia [kJ]')
plt.xlabel('Time [s]')
plt.tight_layout()
plt.show()
