import numpy as np
import xgboost

from tensorflow import convert_to_tensor
from tensorflow.keras.models import load_model
from joblib import load

from collections import namedtuple
from typing import Dict
import src.env.prof as prof

input_converter = namedtuple(
    'Input', ['TF', 'PF1', 'PF1_2', 'PF6', 'PF9', 'LFS_t0', 'LFS_dt', 'HFS_t0',
              'HFS_dt','EC_2G', 'EC_7G', 'NBI_t0', 'NBI_dt', 'NBI_PW', 'wall']
)

action_range = {
    'TF':[], 
    'PF1':[], 
    'PF1_2':[], 
    'PF6':[], 
    'PF9':[], 
    'LFS_t0':[], 
    'LFS_dt':[], 
    'HFS_t0':[],
    'HFS_dt':[],
    'EC_2G':[], 
    'EC_7G':[], 
    'NBI_t0':[], 
    'NBI_dt':[], 
    'NBI_PW':[], 
    'wall':[],
}

class Simulator:
    def __init__(self, dt:float = 1e-5, ip0:float = 6.0e4):
        # Ip and operation time predicter
        self.model_ip = xgboost.XGBRegressor()
        self.model_dt = xgboost.XGBRegressor()

        self.model_ip.load_model('./xgb_model/ipbest_47.json')
        self.model_dt.load_model('./xgb_model/dtbest_47.json')
        
        self.dt = dt
        self.ip0 = ip0
        
        self.set_default_geometriy()
        
        # diamagnetic energy predicter
        # model directory
        path_nn      = "./src/lstm_model"
        path_models  = path_nn + "/models"
        path_scalers = path_nn + "/scalers"
        MODEL = 'wdia_mid_simple'
        
        self.scalers = []
        self.models_wdia = []
        
        for id in range(1,5):
            path_scaler = path_scalers + "/scaler_%s"%MODEL + str(id)
            self.scalers[id] = load(open(path_scaler, "rb"))
            path_submodel = path_models + "/%s"%MODEL + str(id) + ".h5"
            self.models_wdia[id]  = load_model(path_submodel)
            
    def generate_wdia_input(self, t1, ip1):
        shot   = 10000
        trange = np.arange(t0, t1, self.dt)
        x_input = np.zeros((1, len(trange), 8))

        for i, t in enumerate(trange):
            ip     = prof.f_ip(t, ip0, ip1, t0, t1)
            rmajor = prof.quadratic_decrease(t, self.rmajor0, self.rmajor1, t0, t1)
            aminor = prof.quadratic_decrease(t, self.aminor0, self.aminor1, t0, t1)
            elong  = prof.linear_decrease(t, self.elong0, self.elong1, t0, t1)
            tria   = prof.linear_decrease(t,  self.tria0,  self.tria1, t0, t1)
            qa     = prof.f_qa2(aminor, rmajor, 0.18, ip, elong)
            
            x_input[:,i,0] = shot
            x_input[:,i,1] = t
            x_input[:,i,2] = ip
            x_input[:,i,3] = rmajor
            x_input[:,i,4] = aminor
            x_input[:,i,5] = elong
            x_input[:,i,6] = tria
            x_input[:,i,7] = qa
        return x_input
     
    def set_default_geometriy(self):
        self.rmajor0	= 0.45			# [m] 0.3-0.5  usually around ~0.42
        self.rmajor1	= 0.35			# 
        self.aminor0	= 0.30			# [m] 0.2-0.35 usually around ~0.30
        self.aminor1	= 0.20			# 
        self.elong0	= 1.40			#     1.4-1.8 decrease
        self.elong1	= 1.40
        self.tria0	= 0.20			#     0.2-0.5 decrease
        self.tria1	= 0.20
        
    def set_geometry(self, args:Dict):
        self.rmajor0 = args['rmajor0']
        self.rmajor1 = args['rmajor1']
        self.aminor0 = args['aminor0']
        self.aminor1 = args['aminor1']
        
        self.elong0 = args['elong0']
        self.elong1 = args['elong1']
        self.tria0 = args['tria0']
        self.tria1 = args['tria1']
        
    def adjust_scale(self, inputs:np.array):
        inputs_list = []
        for scaler in self.scalers:
            inputs_list.append(scaler.transform(inputs))
        
        return inputs_list
    
    def predict(self, inputs:np.array):
        
        t1 = self.model_dt.predict(inputs)
        ip1 = self.model_ip.predict(inputs)
        
        dia_input = self.generate_wdia_input(t1, ip1)
        dia_input_scaled_list = self.adjust_scale(dia_input)
        
        dia_list = []
        for dia_input, model in zip(dia_input_scaled_list, self.models_wdia):
            dia_predicted = model.predict(dia_input)
            dia_list.append(dia_predicted)
        
        dias_integral = np.mean(dia_list).sum(axis = 1).reshape(-1,)
        
        return t1, ip1, dias_integral



if __name__ == "__main__":
    sim = Simulator()
    