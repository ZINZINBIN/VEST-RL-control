from typing import Dict, Literal
import math

class RewardSender:
    def __init__(
        self,
        w_wdia : float,
        w_ipdt : float, 
        wdia_r:float,
        ipdt_r:float,
        a : float = 3.0,
        ):
        self.w_wdia = w_wdia
        self.w_ipdt = w_ipdt
        
        self.wdia_r = wdia_r
        self.ipdt_r = ipdt_r
        self.a = a
        
    def _compute_tanh(self, x):
        return math.tanh(x)
    
    def _compute_performance_reward(self, x, x_criteria, x_scale, a : float = 3.0):
        xn = x / x_criteria - x_scale
        reward = self._compute_tanh(a * xn)
        return reward
        
    def _compute_reward(self, state:Dict):
        
        wdia = state['wdia']
        ip1 = state['ip1']
        dt1 = state['t1']
        
        ipdt = ip1/dt1
        reward_wdia = self._compute_performance_reward(wdia, self.wdia_r, 1, self.a)
        reward_ipdt = self._compute_performance_reward(1 / ipdt, 1 / self.ipdt_r, 1, self.a)
        
        reward = reward_ipdt * self.w_ipdt + reward_wdia * self.w_wdia 
        reward /= (self.w_ipdt + self.w_wdia)
        
        return reward
    
    def _compute_reward_dict(self, state:Dict):
        
        wdia = state['wdia']
        ip1 = state['ip1']
        dt1 = state['t1']
        
        ipdt = ip1/dt1
        reward_wdia = self._compute_performance_reward(wdia, self.wdia_r, 1, self.a)
        reward_ipdt = self._compute_performance_reward(1 / ipdt, 1 / self.ipdt_r, 1, self.a)
        
        reward = reward_ipdt * self.w_ipdt + reward_wdia * self.w_wdia 
        reward /= (self.w_ipdt + self.w_wdia)
        
        reward_dict = {
            "total":reward,
            "wdia":reward_wdia,
            "ipdt":reward_ipdt,
        }
        
        return reward_dict
        
    def __call__(self, state : Dict):
        reward = self._compute_reward(state)
        return reward