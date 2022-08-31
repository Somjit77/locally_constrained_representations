from rlzoo.common.env_wrappers import *
from rlzoo.common.utils import *
from rlzoo.algorithms import *
import argparse

import tensorboard

parser = argparse.ArgumentParser()
parser.add_argument("--lle", help="Use LCR", default='False')
parser.add_argument("--version", help="Version", default='10.0')
parser.add_argument("--env", default='Humanoid-v2')
parser.add_argument("--max_steps", default=1000)
args = parser.parse_args()

#Only use SAC, other algorithms have not been modified for LCR
AlgName = 'SAC'
envs = [
        'Ant-v2',
        'HalfCheetah-v2',
        'Humanoid-v2',
        'HumanoidStandup-v2',
        'Reacher-v2',
        'Striker-v2',
        'Thrower-v2',
        'Pusher-v2'
    ]
EnvName = args.env
envs = [EnvName]
EnvType = 'mujoco'
lcr = eval(args.lcr)
ver = args.version
if lcr:
    lcr_str = '_L_'
else:
    lcr_str = '_'
max_episode_steps = int(args.max_steps)
for EnvName in envs:
    runs = 5
    episodes = 5000
    for run_no in range(runs):
        env = build_env(EnvName, EnvType, seed=run_no, max_episode_steps=max_episode_steps)
        set_seed(run_no)
        if lcr:
            lcr_params = {'K': 10,
                          'Phi_gradient_steps': 100,
                          'lcr_lr': 3e-4,
                          'lcr_batch_size': 5000}
        else:
            lcr_params = {}
        alg_params, learn_params = call_default_params(env, EnvType, AlgName, lcr, lcr_params, default_seed=False)
        alg = eval(AlgName+'(**alg_params)')
        learn_params['train_episodes'] = episodes
        learn_params['save_interval'] = 20
        learn_params['version'] = ver + lcr_str + str(run_no)
        alg.learn(env=env, mode='test', render=True, **learn_params)
        env.close()