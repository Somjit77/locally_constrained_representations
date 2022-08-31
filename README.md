# Locally Constrained Representations
Implementation of the LCR algorithm on the following environments.

Environments
============
Atari
-----
All Default Hyper-Parameters have been set. Select game with --game and run main.py. The codebase has been adapted from [here](https://github.com/Kaixhin/Rainbow)

Gym-Control
-----------
Default Hyper-Paremeters is for the main plot. Choose appropriate hyper-parameters for ablation studies while setting the default ones as constant. Run main.py. Change game between 'CartPole-v1' and 'Acrobot-v1'

Mujoco
------
This is an extension of the [RLZoo](https://github.com/tensorlayer/RLzoo) library. The LCR code is built on top of RLZoo with only SAC being modified. The code will not work for other algorithms. The default hyper-parameters are set to the default ones for RLZoo SAC. Run train.py to reproduce the results from the paper. Change the max_steps for each environment based on the table provided in the supplementary material.

MiniGrid
--------
The new proposed environments are implemented in env_minigrid.py. Run train.py with default hyper-paramters to recreate the plots from the paper.