# VEST Control Parameters Optimization through Deep Reinforcement Learning
## Introduction
This is the repository for the optimization code of VEST control parameters. The code is based on Proximal Policy Optimization, while finding the optimial configuration satisfying the maximum diamagnetic energy generated during the operation. 

## How to Use
- Yon can add modules for executing the code with pip3 install as below. 
    ```
        pip3 install -r requirements.txt
    ```

- Then, yon can optimize control parameters for VEST with PPO algorithm with this command below.
    ```
        python3 main.py --num_episode {# of episodes} --buffer_size {buffer size} --lr {learning rate}
    ```