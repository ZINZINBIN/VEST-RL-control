import numpy as np

def linear_decrease(t, y0, y1, t0, t1):
	
	y = y0 - (y0-y1)/(t1-t0) * (t - t0)

	return y

def quadratic_decrease(t, y0, y1, t0, t1):

	y = (y1-y0)/(t1-t0)**2 * (t - t0)**2 + y0

	return y

def exponential_decrease(t, y0, y1, t0, t1):

	kexp = -1.0/(t1-t0) * np.log(y1/y0)

	y = y0 * np.exp(-kexp * (t - t0))

	return y

	
def f_ip(t, ip0, ip1, t0, t1):
	
	# Sine type
	T = 2.0*(t1-t0)
	v_ip = (ip1-ip0)*np.sin(2.0*np.pi/T * (t - t0)) + ip0

	# Maybe other types...?

	return v_ip


def f_qa2(a, r0, bt, ip, kappa):
	"""
	qa = a / R0 * B0 / (mu0 Ip / 2pi sqrt(1 + K^2) a) 
	
	"""

	v_qa = np.sqrt(1.0 + kappa**2.0) * a**2.0 * bt / (np.sqrt(2.0) * 2.0e-7) / r0 / ip

	return v_qa
