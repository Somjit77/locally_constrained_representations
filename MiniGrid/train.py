import datetime
import os
import argparse
import time
import pandas as pd
import logging
logging.getLogger('tensorflow').disabled = True

os.environ['TF_CPP_MIN_LOG_LEVEL'] = "3"  # Suppress Tensorflow Messages
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Set CPU/GPU
import numpy as np
from agent import *
from env_minigrid import GymMiniGridEnv

parser = argparse.ArgumentParser()
parser.add_argument("--algorithm", help='algorithm', default='dqn')  # dqn, dqn_lcr
parser.add_argument("--save", help='log files', default='True')
parser.add_argument("--full_state", help='Use Full State', default='True')
args = parser.parse_args()
algorithm = args.algorithm
full_state = eval(args.full_state)
save = eval(args.save)

'''Log directory'''
if save:
    ver = '1.0'
    model_dir = 'Results/Result-{}/models/'.format(ver)
    output_dir = 'Results/Result-{}/outputs/'.format(ver)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    log_file_name = 'Results/Result-{}/'.format(ver) + 'parameters.txt'
    test_file_name = 'Results/Result-{}/'.format(ver) + 'test.txt'

train_columns = ['Reward', 'Steps', 'Epsilon']
train_df = pd.DataFrame(columns=train_columns)

'''Environment Parameters'''
game = 'MiniGrid-RandomGoal-6x6-v0'
length = 11
breadth = 11
seed = 0  # Seed for Env, TF, Numpy
num_episodes = 5000
logs = {'log_interval': 1,  # Number of Episodes after which to print output/save batch output
        'output_dir': output_dir if save else '',
        'model_dir': model_dir if save else '',
        'log_file': train_df if save else '',  # Log data
        'test_file_name': test_file_name if save else ''
        }

'''Parameters of Algorithm'''
algorithm_params = {'algorithm': algorithm,
                    'use_gpu': False,
                    'full_state': full_state,
                    'batch_size': 32,
                    'gamma': 0.99,
                    'learning_rate': 1e-3,
                    'start_epsilon': 1.0,
                    'stop_epsilon': 1e-3,
                    'epsilon_decay': 1e-3**(1/num_episodes),
                    'seed': seed,
                    'K': 10,
                    'lcr_batch_size': 5000,
                    'gradient_steps': 100,
                    'lcr_learning_rate': 1e-4,
                    }
model_params = {'hidden_units': [64, 64],  # model architecture
                'max_buffer_size': 10000,
                'min_buffer_size': 1000,
                'copy_step': 5,  # number of episodes, 1 means no target network
                }

'''Runs'''
runs = 1
rewards = {}
losses = {}
time_taken = np.zeros(runs)
if save:
    time_file_name = output_dir + '/time_{}'.format(algorithm)
for run in range(runs):
    '''Set seed'''
    seed = run
    algorithm_params['seed'] = seed
    logs['run'] = run
    '''Write Parameters to log_file'''
    if save:
        with open(log_file_name, "w") as f:
            if 'MiniGrid' in game:
                f.write('Environment: {}, Episodes: {}\n'.format(game, num_episodes))
            else:
                f.write(f'Environment: {game}-{length}x{breadth}, Episodes: {num_episodes}\n')
            f.write('Algorithm Parameters: {} \n'.format(algorithm_params))
            f.write('Model Parameters: {} \n'.format(model_params))
            f.write('Run: {} \n'.format(run))
            f.flush()
    '''Initialize Environment & Model'''
    if 'MiniGrid' in game:
        env = GymMiniGridEnv(game, seed, full_state)
    else:
        raise Exception('Environment Not Defined')
    '''Train the Agent'''
    start_time = time.time()
    reward_history = train_agent(env, num_episodes, model_params, algorithm_params, logs, save)
    end_time = time.time()
    time_taken[run] = end_time - start_time
    if save:
        with open(log_file_name, "a") as f:
            f.write('Time taken: {}\n'.format(time_taken))
            f.flush()
    '''Store the results'''
    rewards = reward_history
    '''Save Files'''
    if save:
        np.save(time_file_name, time_taken)
