import gym
import numpy as np
from src.env.simulator import Simulator
from src.rl.reward import RewardSender

INIT_ACTION = {
    "TF" 		: 420.0,
    "PF1"		: 3614.346,
    "PF1_2"	: 383.5744,
    "PF6"		: 600.0226,
    "PF9"		: 1721.424,
    "LFS_t0"	: 0.0, 
    "LFS_dt"	: 0.0, 
    "HFS_t0"	: 0.053055,  
    "HFS_dt"	: 0.000000,
    "EC_2G"	:0,    
    "EC_7G"	:1,   
    "NBI_t0"	: -1984.84, 
    "NBI_dt"	: 0.828, 
    "NBI_PW"	: 32.545,  
    "wall"	: 5.059, 
}

INIT_STATE = {
    "t1":17.47,
    "ip1":157.04, 
    "wdia":274.55
}

class Emulator(gym.Env):
    def __init__(self, emulator:Simulator, reward_sender:RewardSender):
        self.emulator = emulator
        self.reward_sender = reward_sender
        
        self.actions = []
        self.states = []
        self.rewards = []
        
        self.current_reward = None
        self.current_state = None
        self.current_action = None
        
        self.init_state = None
        self.init_action = None
        self.init_reward = None
        
        self.wdia = []
        self.ip1 = []
        self.dt1 = []
        
        # optimization status
        self.optim_status = {}
            
        self.init_env(INIT_ACTION)
    
    
    def init_env(self, init_action):
        init_state, init_reward, _, _ = self.step(init_action)
        
        self.init_state = init_state
        self.init_reward = init_reward
        self.init_action = init_action
    
    def step(self, action):
        
        # predict next state
        t1, ip1, wdia = self.emulator.predict(action)
        
        state = {
            "t1":t1 * 1e3, # s -> ms
            "ip1":ip1 * 1e-3, # A->kA
            "wdia":wdia
        }
        
        optim_status = self.reward_sender._compute_reward_dict(state)
        
        # compute reward
        reward = self.reward_sender(state)
        
        # update state and action
        self.current_state = state
        self.current_action = action
        self.current_reward = reward
        
        # save history
        self.wdia.append(wdia)
        self.ip1.append(ip1 * 1e-3)
        self.dt1.append(t1 * 1e3)
        
        self.actions.append(action)
        self.states.append(state)
        self.rewards.append(reward)
        
        # optimization process status logging
        if optim_status is not None:
            
            for key, value in optim_status.items():
            
                if key not in self.optim_status.keys():
                    self.optim_status[key] = []
                
                self.optim_status[key].append(value)
        
        return state, reward, False, {}
    
    def close(self):
        self.actions.clear()
        self.states.clear()
        self.rewards.clear()
        
        self.current_action = None
        self.current_state = None
        self.current_reward = None
        
        self.init_state = None
        self.init_action = None
        self.init_reward = None
    
    def reset(self):
        self.current_action = None
        self.current_state = None
        self.current_rward = None