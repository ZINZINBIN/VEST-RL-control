from src.env.simulator import Simulator
import numpy as np

if __name__ == "__main__":
    
    # example input list
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
    
    values = [TF, PF1, PF1_2, PF6, PF9, LFS_t0, LFS_dt, HFS_t0, HFS_dt, EC_2G, EC_7G, NBI_t0, NBI_dt, NBI_PW, wall]
    input_data = np.array(values)
    
    sim = Simulator()
    
    t1, ip1, wdia_t = sim.predict(input_data)
    
    print("=======================")
    print("t1:{:.3f}".format(t1))
    print("ip1:{:.3f}".format(ip1))
    print("wdia-t:{:.3f}".format(wdia_t))