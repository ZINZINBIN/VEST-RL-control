from src.env.vest import Emulator
from src.rl.ppo import train_ppo, ActorCritic, ReplayBuffer
from src.env.simulator import Simulator
from src.rl.reward import RewardSender
from src.utility import plot_policy_loss, plot_optimization_status, find_optimal_case
import torch
import pickle
import argparse, os, warnings

warnings.filterwarnings(action = 'ignore')

def parsing():
    parser = argparse.ArgumentParser(description="VEST control configuration optimization through DRL")
    
    # tag for labeling the optimization process
    parser.add_argument("--tag", type = str, default = "")
    
    # GPU allocation
    parser.add_argument("--gpu_num", type = int, default = 0)
    
    # PPO setup
    parser.add_argument("--buffer_size", type = int, default = 4)
    parser.add_argument("--num_episode", type = int, default = 10000)
    parser.add_argument("--verbose", type = int, default = 1000)
    parser.add_argument("--lr", type = float, default = 1e-3)
    parser.add_argument("--gamma", type = float, default = 0.999)
    parser.add_argument("--eps_clip", type = float, default = 0.2)
    parser.add_argument("--entropy_coeff", type = float, default = 0.05)
    
    # Control env setup
    parser.add_argument("--n_target", type = int, default = 3)
    parser.add_argument("--n_control", type = int, default = 15)
    
    # Reward setup
    parser.add_argument("--w_wdia", type = float, default = 0.5)
    parser.add_argument("--w_dt", type = float, default = 0.5)
    parser.add_argument("--wdia_r", type = float, default = 0.05)
    parser.add_argument("--dt_r", type = float, default = 17) # kA/ms
    parser.add_argument("--a", type = float, default = 1.0)
    
    # Visualization
    parser.add_argument("--smoothing_temporal_length", type = int, default = 16)
    
    args = vars(parser.parse_args()) 

    return args

# torch device state
print("=============== Device setup ===============")
print("torch device avaliable : ", torch.cuda.is_available())
print("torch current device : ", torch.cuda.current_device())
print("torch device num : ", torch.cuda.device_count())
print("torch version : ", torch.__version__)
    
if __name__ == "__main__":
    
    args = parsing()
    
    # device allocation
    if(torch.cuda.device_count() >= 1):
        device = "cuda:{}".format(args['gpu_num'])
    else:
        device = 'cpu'
    
    reward_sender = RewardSender(
        w_wdia = args['w_wdia'],
        w_dt = args['w_dt'],
        wdia_r = args['wdia_r'],
        dt_r = args['dt_r'],
        a = args['a'],
    )
    
    simulator = Simulator(dt = 0.00001, ip0 = 6e4)
    env = Emulator(simulator, reward_sender)
    
    output_dim = args['n_target']
    ctrl_dim = args['n_control']
    
    # policy and value network
    policy_network = ActorCritic(input_dim = ctrl_dim + output_dim, mlp_dim = 64, n_actions = ctrl_dim, std = 0.5)
    
    # gpu allocation
    policy_network.to(device)
    
    # optimizer    
    policy_optimizer = torch.optim.Adam(policy_network.parameters(), lr = args['lr'])
    
    # loss function for critic network
    value_loss_fn = torch.nn.SmoothL1Loss(reduction = 'none')
    
    # memory
    memory = ReplayBuffer(args['buffer_size'])
    
    # directory
    if not os.path.exists("./weights"):
        os.makedirs("./weights")
    
    if not os.path.exists("./results"):
        os.makedirs("./results")
        
    if len(args['tag']) > 0:
        tag = "PPO_{}".format(args['tag'])
    else:
        tag = "PPO"
    
    save_best = "./weights/{}_best.pt".format(tag)
    save_last = "./weights/{}_last.pt".format(tag)
    save_result = "./results/params_search_{}.pkl".format(tag)
    
    # Design optimization
    print("============ Design optimization ============")
    result = train_ppo(
        env, 
        memory,
        policy_network,
        policy_optimizer,
        value_loss_fn,
        args['gamma'],
        args['eps_clip'],
        args['entropy_coeff'],
        device,
        args['num_episode'],
        args['verbose'],
        save_best,
        save_last,
    )
    
    print("======== Logging optimization process ========")
    # save optimal case
    find_optimal_case(result, {"filename":os.path.join("./results", "{}_stat.txt".format(args['tag']))})
    
    # save optimization process
    optimization_status = env.optim_status
    plot_optimization_status(optimization_status, args['smoothing_temporal_length'], "./results/{}_optimization".format(tag))
    
    # save policy loss change
    plot_policy_loss(result['loss'], args['buffer_size'], args['smoothing_temporal_length'], "./results/{}_optimization".format(tag))
    
    with open(save_result, 'wb') as file:
        pickle.dump(result, file)

    # exit
    env.close()