import argparse
import os
import time

os.environ['TF_CPP_MIN_LOG_LEVEL'] = "3"  # Suppress Tensorflow Messages
from agent import *
from env import Environment

parser = argparse.ArgumentParser()
parser.add_argument("--version", help="Version", default='1.0')
parser.add_argument("--algorithm", help="algorithm", default='dqn')  # dqn, dqn_lcr
parser.add_argument("--verbose", help="log files", default='True')
parser.add_argument("--use_gpu", help="GPU/CPU", default='False')
parser.add_argument("--lcr_batch_size", help="batch size for lcr", default=256)
parser.add_argument("--K", help="# nearest neighbours", default=5)
parser.add_argument("--lcr_lr", help="LR for lcr", default=1e-4)
parser.add_argument("--gradient_steps", help="#gradient_steps", default=100)
args = parser.parse_args()
algorithm = args.algorithm
use_gpu = eval(args.use_gpu)
verbose = eval(args.verbose)
lcr_batch_size = int(args.lcr_batch_size)
K = int(args.K)
lcr_lr = float(args.lcr_lr)
gradient_steps = int(args.gradient_steps)

'''Log directory'''
if verbose:
    ver = args.version
    result_dir = 'Results/Result-{}/'.format(ver)
    model_dir = result_dir + 'models/'
    output_dir = result_dir + 'outputs/'
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    log_file_name = result_dir + '{}_parameters.txt'.format(algorithm)
    logs = {'log_file_name': log_file_name,
            'model_dir': model_dir,
            'output_dir': output_dir,
            }
else:
    logs = {'log_interval': 1,  # Number of Episodes after which to print output
            }

'''Environment Parameters'''
game = 'CartPole-v1'
seed = 0  # Seed for Env, TF, Numpy
# num_frames = 10000  # Million Frames
num_episodes = 1000
runs = 10

'''Parameters of Algorithm'''
algorithm_params = {'algorithm': algorithm,  # dqn
                    'use_gpu': use_gpu,
                    'num_episodes': num_episodes,
                    'batch_size': 64,
                    'gamma': 0.99,
                    'learning_rate': 1e-3,
                    'start_epsilon': 1.0,
                    'stop_epsilon': 1e-3,
                    'epsilon_decay': 1e-3,
                    'seed': seed
                    }
'''lcr algorithm parameters'''
if 'lcr' in algorithm:
    algorithm_params['K'] = K
    algorithm_params['lcr_batch_size'] = lcr_batch_size
    algorithm_params['X_gradient_steps'] = gradient_steps
    algorithm_params['lcr_learning_rate'] = lcr_lr
    algorithm_params['Phi_gradient_steps'] = gradient_steps

model_params = {'hidden_units': [32],  # model architecture
                'max_buffer_size': 5000,
                'min_buffer_size': 100,  # must be more than batch size
                'copy_step': 25,  # 1 means no target network
                }

'''Runs'''
rewards = {}
losses = {}
time_taken = np.zeros(runs)
if verbose:
    time_file_name = log_file_name
for run in range(runs):
    '''Write Parameters to log_file'''
    if verbose:
        if run == 0:
            with open(log_file_name, "w") as f:
                f.write('Algorithm Parameters: {} \n'.format(algorithm_params))
                f.write('Model Parameters: {} \n'.format(model_params))
                f.write('Number of Episodes: {} \n'.format(num_episodes))
                f.write('Run: {} \n'.format(run))
                f.flush()
        else:
            with open(log_file_name, "a") as f:
                f.write('Run: {} \n'.format(run))
                f.flush()
    '''Set seed'''
    seed = run
    algorithm_params['seed'] = seed
    '''Initialize Environment & Model'''
    env = Environment(seed, game)
    '''Train the Agent'''
    start_time = time.time()
    train_agent(env, num_episodes, model_params, algorithm_params, logs, verbose)
    end_time = time.time()
    time_taken[run] = end_time - start_time
    '''Time Taken'''
    if verbose:
        with open(log_file_name, "a") as f:
            f.write('Time taken: {} seconds\n'.format(time_taken[run]))
            f.flush()
        np.save(result_dir+'time_taken.npy', time_taken)

