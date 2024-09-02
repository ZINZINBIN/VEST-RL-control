from typing import Dict, Literal
import math

class RewardSender:
    def __init__(
        self,
        w_wdia : float,
        w_dt : float, 
        wdia_r:float,
        dt_r:float,
        a : float = 3.0,
        ):
        self.w_wdia = w_wdia
        self.w_dt = w_dt
        
        self.wdia_r = wdia_r
        self.dt_r = dt_r
        self.a = a
        
    def _compute_tanh(self, x):
        return math.tanh(x)
    
    def _compute_performance_reward(self, x, x_criteria, x_scale, a : float = 3.0):
        xn = x / x_criteria - x_scale
        reward = self._compute_tanh(a * xn)
        return reward
        
    def _compute_reward(self, state:Dict):
        
        wdia = state['wdia']
        dt1 = state['t1']
        
        reward_wdia = self._compute_performance_reward(wdia, self.wdia_r, 1, self.a)
        reward_dt = self._compute_performance_reward(dt1, self.dt_r, 1, self.a)
        
        reward = reward_dt * self.w_dt + reward_wdia * self.w_wdia 
        reward /= (self.w_dt + self.w_wdia)
        
        return reward
    
    def _compute_reward_dict(self, state:Dict):
        
        wdia = state['wdia']
        dt1 = state['t1']
        
        reward_wdia = self._compute_performance_reward(wdia, self.wdia_r, 1, self.a)
        reward_dt = self._compute_performance_reward(dt1, self.dt_r, 1, self.a)
        
        reward = reward_dt * self.w_dt + reward_wdia * self.w_wdia 
        reward /= (self.w_dt + self.w_wdia)
        
        reward_dict = {
            "total":reward,
            "wdia":reward_wdia,
            "dt":reward_dt,
        }
        
        return reward_dict
        
    def __call__(self, state : Dict):
        reward = self._compute_reward(state)
        return reward