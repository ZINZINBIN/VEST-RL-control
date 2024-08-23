#!/APP/anaconda3-2020.11/bin/python

import os
import sys

import numpy             as np
import pandas            as pd
import matplotlib.pyplot as plt
from tensorflow              import convert_to_tensor
from tensorflow.keras.models import load_model
from joblib                  import load


# sys.argv description
# python adopt_lstm_model.py dir_name dt[ms/s] ip[kA/A]


# File options
dir_name  = sys.argv[1]
#path_top  = "/home/hcho/2023/vest-nn/works"
path_top    = "./"
path_home   = path_top + dir_name
path_input  = path_home + "/input.csv"
path_output = path_home + "/output.csv"
path_time   = path_home + "/time.csv"

if not os.path.exists(path_home): os.mkdir(path_home)

# Input settings
shot   = 37441
#dt     = 1e-5		# [s]
dt     = sys.argv[2]
if dt > 1: dt /= 1e3  #[ms -> s]
t0     = 0.0
t1     = 8e-3		# duration ~ 3ms-10ms=3e-3-10e-3s
trange = np.arange(t0, t1, dt)

# -----------------------------------------------------
ip0     =  6.0e4		# [A] >= 30kA
#ip1     = 10.0e4		# [A] ~= 100kA
ip1     = sys.argv[3]
if ip1 < 1000: ip1 *= 1e3  #[kA -> A]
rmajor0 = 0.45			# [m] 0.3-0.5  usually around ~0.42
rmajor1 = 0.35			# 
aminor0 = 0.30			# [m] 0.2-0.35 usually around ~0.30
aminor1 = 0.20			# 
elong0 = 1.40			#     1.4-1.8 decrease
elong1 = 1.40
tria0  = 0.20			#     0.2-0.5 decrease
tria1  = 0.20
qa0    = 3.7			#     2.0-5.0 decrease
qa1    = 2.0
# -----------------------------------------------------


# ====================================================================
def linear_decrease(t, y0, y1):
	
	y = y0 - (y0-y1)/(t1-t0) * (t - t0)

	return y

def quadratic_decrease(t, y0, y1):

	y = (y1-y0)/(t1-t0)**2 * (t - t0)**2 + y0

	return y

def exponential_decrease(t, y0, y1):

	kexp = -1.0/(t1-t0) * np.log(y1/y0)

	y = y0 * np.exp(-kexp * (t - t0))

	return y

def const_change(t, y0, y1, tc, ftype):
	
	if t <= tc:
		y = y0
	else:
		if ftype == 'lin':
			y = y0 + (y1-y0)/(t1-tc) * (t - tc)
		elif ftype == 'quad':
			y = (y1-y0)/(t1-tc)**2 * (t - tc)**2 + y0
		elif ftype == 'exp':
			kexp = -1.0/(t1-tc) * np.log(y1/y0)
			y = y0 * np.exp(-kexp * (t - tc))
	
	return y

def change_const(t, y0, y1, tc, ftype):
	
	if t > tc:
		y = y1
	else:
		if ftype == 'lin':
			y = y0 + (y1-y0)/(tc-t0) * (t - t0)
		elif ftype == 'quad':
			y = (y1-y0)/(tc-t0)**2 * (t - t0)**2 + y0
		elif ftype == 'exp':
			kexp = -1.0/(tc-t0) * np.log(y1/y0)
			y = y0 * np.exp(-kexp * (t - t0))
	
	return y
	
def f_ip(t):
	
	# Sine type
	T = 2.0*(t1-t0)
	v_ip = (ip1-ip0)*np.sin(2.0*np.pi/T * (t - t0)) + ip0

	# Maybe other types...?

	return v_ip

def f_rmajor(t):

	v_rmajor = quadratic_decrease(t, rmajor0, rmajor1)

	return v_rmajor

def f_aminor(t):

	v_aminor = quadratic_decrease(t, aminor0, aminor1)

	return v_aminor

def f_elong(t):

	v_elong = linear_decrease(t, elong0, elong1)

	return v_elong

def f_tria(t):

	v_tria = linear_decrease(t, tria0, tria1)

	return v_tria

def f_qa(t):

	v_qa = exponential_decrease(t, qa0, qa1)

	return v_qa

def f_qa2(a, r0, bt, ip, kappa):
	"""
	qa = a / R0 * B0 / (mu0 Ip / 2pi sqrt(1 + K^2) a) 
	
	"""

	v_qa = np.sqrt(1.0 + kappa**2.0) * a**2.0 * bt / (np.sqrt(2.0) * 2.0e-7) / r0 / ip

	return v_qa
# ====================================================================

# Write input file
with open(path_input, "w") as f:
	
	f.write("SHOT,TIME,IP,RMAJOR,AMINOR,ELONG,TRIA,QA\n")

	for t in trange:

		ip     = f_ip(t)
		rmajor = f_rmajor(t)
		aminor = f_aminor(t)
		elong  = f_elong(t)
		tria   = f_tria(t)
		qa     = f_qa2(aminor, rmajor, 0.18, ip, elong)

		#ip     = f_ip(t)
		#rmajor = const_change(t, rmajor0, rmajor1, 6e-3, 'lin')
		#aminor = const_change(t, aminor0, aminor1, 6e-3, 'lin')
		#elong  = change_const(t,  elong0,  elong1, 6e-3, 'lin')
		#tria   = const_change(t,   tria0,   tria1, 6e-3, 'lin')
		#qa     = f_qa2(aminor, rmajor, 0.18, ip, elong)

		f.write("%5d,%e,%e,%e,%e,%e,%e,%e\n"%(shot, t, ip,
		                                      rmajor,aminor,
		                                      elong,tria, qa))

print("Finished writing input file [%s]"%dir_name)
# /// END OF MAKE INPUT ///


# DO PREDICT
# Model Directory
#path_nn      = "/home/hcho/2022-2/vest-nn/nn/models/VEST-NN/" + MODEL
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

# Predictions
for i, ID in enumerate(IDs):

	# Load Scaler 
	path_scaler = path_scalers + "/scaler_wdia_mid_simple" + str(ID)
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
	path_submodel = path_models + "/wdia_mid_simple" + str(ID) + ".h5"
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

# ------------------------------------------------------------------------
# Print Results
# ------------------------------------------------------------------------
dfo = pd.DataFrame(save_array, columns=["PRED", "SIGMA"])
print(dfo)
dfo.to_csv(path_output)
dft = pd.DataFrame(time_seq, columns=["TIME"])
dft.to_csv(path_time)
