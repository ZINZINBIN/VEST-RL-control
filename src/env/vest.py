import gym
import numpy as np
from src.env.simulator import Simulator
from src.rl.reward import RewardSender

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
    
    def init_env(self, init_action):
        init_state, init_reward, _, _ = self.step(init_action)
        
        self.init_state = init_state
        self.init_reward = init_reward
        self.init_action = init_action
    
    def step(self, action):
        
        # predict next state
        t1, ip1, wdia = self.emulator.predict(action)
        
        state = {
            "t1":t1,
            "ip1":ip1,
            "wdia":wdia
        }
        
        # compute reward
        reward = self.reward_sender(state)
        
        # update state and action
        self.current_state = state
        self.current_action = action
        self.current_reward = reward
        
        # save history
        self.wdia.append(wdia)
        self.ip1.append(ip1)
        self.dt1.append(t1)
        
        self.actions.append(action)
        self.states.append(state)
        self.rewards.append(reward)
        
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
        self.current_reward = None