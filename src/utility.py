import numpy as np
import pandas as pd
import os

import matplotlib.pyplot as plt
from typing import List, Optional, Literal, Dict, Union

def temperal_average(X:np.array, Y:np.array, k:int):
    
    clip_length = X.shape[0] // k
    
    X_mean = np.zeros(clip_length)
    
    Y_mean = np.zeros(clip_length)
    Y_lower = np.zeros(clip_length)
    Y_upper = np.zeros(clip_length)
    
    for i in range(clip_length):
        
        idx_start = i * k 
        idx_end = (i+1) * k 
        
        if idx_end >= X.shape[0]:
            idx_end = X.shape[0] - 1
            
        X_mean[i] = int(np.mean(X[idx_start:idx_end]))
        Y_mean[i] = np.mean(Y[idx_start:idx_end])
        Y_lower[i] = np.min(Y[idx_start:idx_end])
        Y_upper[i] = np.max(Y[idx_start:idx_end])
    
    return X_mean, Y_mean, Y_lower, Y_upper

def plot_policy_loss(
    loss_list:List, 
    buffer_size : int, 
    temporal_length:int = 8, 
    save_dir : Optional[str] = None,
    ):
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    loss = np.repeat(np.array(loss_list).reshape(-1,1), repeats = buffer_size, axis = 1).reshape(-1,)
    episode = np.array(range(1, len(loss)+1, 1))
    
    # clip the invalid value
    loss = np.clip(loss, a_min = -2.0, a_max = 5.0)
    
    # temperal average
    x_mean, loss_mean, loss_lower, loss_upper = temperal_average(episode, loss, temporal_length)
    
    fig = plt.figure(figsize = (8,4))
    
    clr = plt.cm.Purples(0.9)
    
    plt.plot(x_mean, loss_mean, c = 'r', label = '$<loss_t>$')
    plt.fill_between(x_mean, loss_lower, loss_upper, alpha = 0.3, edgecolor = clr, facecolor = clr)
    
    plt.xlabel("Episodes")
    plt.ylabel("Policy loss")
    plt.legend(loc = 'upper right')
    
    fig.tight_layout()
    plt.savefig(os.path.join(save_dir, "policy_loss.png"), facecolor = fig.get_facecolor(), edgecolor = 'none', transparent = False)
    
    fig.clear()
    

# print the result of overall optimization process
def plot_optimization_status(
    optimization_status:Dict, 
    temporal_length:int = 8, 
    save_dir : Optional[str] = None,
    ):
    
    '''
        optimization_status: Dict[key, value]
        - key: obj-1, obj-2, .... (ex) q95, fbs, beta, ...., total
        - value: List type of reward with respect to each episode
        
        smoothing_k : n_points for moving average process
        smoothing_method: backward or center
    '''
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    if temporal_length == 1:
        print("buffer size = 1 | the default value 8 is automatically selected")
        temporal_length = 8
    
    for idx, key in enumerate(optimization_status.keys()):
        
        obj_reward = np.array(optimization_status[key])
        episode = np.array(range(1, len(obj_reward)+1, 1))
        
        x_mean, obj_reward_mean, obj_reward_lower, obj_reward_upper = temperal_average(episode, obj_reward, temporal_length)
        
        fig = plt.figure(figsize = (8,4))
       
        clr = plt.cm.Purples(0.9)
       
        plt.plot(x_mean, obj_reward_mean, c = 'r', label = '$<r_t>$:{}'.format(key))
        plt.fill_between(x_mean, obj_reward_lower, obj_reward_upper, alpha = 0.3, edgecolor = clr, facecolor = clr)
        
        plt.xlabel("Episodes")
        plt.ylabel("Reward:{}".format(key))
        plt.legend(loc = 'upper right')
        
        fig.tight_layout()
        plt.savefig(os.path.join(save_dir, "reward_history_{}.png".format(key)), facecolor = fig.get_facecolor(), edgecolor = 'none', transparent = False)
        
        fig.clear()