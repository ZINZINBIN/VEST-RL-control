import gym
import numpy as np

class Vest(gym.Env):
    def __init__(self, emulator):
        self.emulator = emulator
        self.reward_sender = None
        
        self.actions = []
        self.states = []
        self.rewards = []
        
        self.done = False
        self.current_reward = None
        self.current_state = None
        self.current_action = None
        
        self.init_state = None
        self.init_action = None
        self.init_reward = None
    
    def step(self, action):
        pass
    
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