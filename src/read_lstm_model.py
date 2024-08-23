import os, sys
import numpy             as np
import pandas            as pd
import matplotlib.pyplot as plt

# File options
dir_name  = sys.argv[1]
#path_top  = "/home/hcho/2023/vest-nn/works"
path_top    = "./"
path_home   = path_top + dir_name
path_input  = path_home + "/input.csv"
path_output = path_home + "/output.csv"
path_time   = path_home + "/time.csv"

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

# Plot input
plt.subplot(411)
plt.plot(t, ip/1e6, label='Ip [MA]') #[kA]
plt.plot(t, bt    , label='BT [T]') #[T]
plt.subplot(412)
plt.plot(t, R     , label='R [m]') #[m]
plt.plot(t, a     , label='a [m]') #[m]
plt.subplot(413)
plt.plot(t, k     , label='Elongation')
plt.plot(t, d     , label='Triangularity')
plt.subplot(414)
plt.plot(t, qa    , label='qa')
plt.legend()

plt.figure()
plt.plot(t, Wmean, 'r')
plt.fill_between(t, Wmin, Wmax, alpha=0.2, color='r')
plt.show()
