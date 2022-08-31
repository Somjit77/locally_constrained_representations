from abc import ABC

import tensorflow as tf
import numpy as np
import random
import pandas as pd
from statistics import mean
from tqdm import tqdm
from collections import deque

GPUs = tf.config.experimental.list_physical_devices('GPU')

if GPUs:
    try:
        for gpu in GPUs:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)


def to_onehot(size, value):
    """1 hot encoding for observed state"""
    return np.eye(size)[value]


class Model(tf.keras.Model):
    def __init__(self, num_states, hidden_units, num_actions):
        super(Model, self).__init__()
        self.input_layer = tf.keras.layers.InputLayer(input_shape=(num_states,))
        self.hidden_layers = []
        for i in hidden_units:
            self.hidden_layers.append(tf.keras.layers.Dense(
                i, activation='tanh', kernel_initializer='RandomNormal'))
        self.output_layer = tf.keras.layers.Dense(
            num_actions, activation='linear', kernel_initializer='RandomNormal')

    @tf.function
    def call(self, inputs):
        z = self.input_layer(inputs)
        for layer in self.hidden_layers:
            z = layer(z)
        output = self.output_layer(z)
        return output, z


class lcr_Model(tf.keras.Model):
    def __init__(self, K, input_dim, batch_size):
        super(lcr_Model, self).__init__()
        self.input_layer = tf.keras.layers.InputLayer(input_shape=(input_dim, K), batch_size=batch_size)
        '''Constrain to positive weights to eliminate over fitting'''
        self.output_layer = tf.keras.layers.Dense(1, use_bias=False, kernel_constraint=tf.keras.constraints.NonNeg())

    @tf.function
    def call(self, inputs):
        z = self.input_layer(inputs)
        output = self.output_layer(z)
        return output


class DQN:
    def __init__(self, num_states, num_actions, model_params, alg_params):
        np.random.seed(alg_params['seed'])
        tf.random.set_seed(alg_params['seed'])
        random.seed(alg_params['seed'])
        use_gpu = alg_params['use_gpu']
        if use_gpu:
            tf.config.experimental.set_visible_devices([GPUs[0]], 'GPU')
        else:
            tf.config.experimental.set_visible_devices([], 'GPU')
        '''Hyper Parameters'''
        self.alg_params = alg_params
        self.num_actions = num_actions
        self.num_states = num_states
        self.alg = alg_params['algorithm']
        self.batch_size = alg_params['batch_size']
        self.optimizer = tf.optimizers.Adam(alg_params['learning_rate'])
        self.hidden_units = model_params['hidden_units']
        self.gamma = alg_params['gamma']
        '''Define the Model'''
        self.model = Model(num_states, self.hidden_units, num_actions)
        '''Define the Experience Replay Buffer'''
        self.experience = {'s': [], 'a': [], 'r': [], 's2': [], 'done': []}
        self.max_experiences = model_params['max_buffer_size']
        self.min_experiences = model_params['min_buffer_size']

    def predict(self, inputs):
        output, _ = self.model(np.atleast_2d(inputs.astype('float32')))
        return output

    def train_lcr(self):
        total_experiences = len(self.experience['s'])
        batch_size = self.alg_params['lcr_batch_size']
        K = self.alg_params['K']
        losses_lcr = []
        X_dim = self.num_states
        ids = np.random.randint(low=0, high=len(self.experience['s']), size=batch_size)
        idxes = np.expand_dims(ids, axis=1)
        for index_id in range(1):
            lcr_model = lcr_Model(K=K, input_dim=self.hidden_units[-1], batch_size=batch_size)
            x_actual = np.zeros(shape=[batch_size, X_dim])
            x_nearest = np.zeros(shape=[batch_size, K, X_dim])
            for batch_id in range(batch_size):
                # Find W's and R's w.r.t the nearest points
                if idxes[batch_id][index_id] >= K // 2 and total_experiences - idxes[batch_id][index_id] > K // 2 + (
                        K % 2):
                    x_all = np.asarray([self.experience['s'][j] for j in range(idxes[batch_id][index_id] - K // 2,
                                                                               idxes[batch_id][index_id] + K // 2 + (
                                                                                       K % 2) + 1)])
                    x_actual[batch_id, :] = x_all[K // 2, :]
                    # Calculate nearest X's
                    x_nearest[batch_id] = np.delete(x_all, K // 2, axis=0)
                elif idxes[batch_id][index_id] < K // 2:
                    x_all = np.asarray([self.experience['s'][j] for j in range(0, K + 1)])
                    x_actual[batch_id] = x_all[0]
                    # Calculate nearest X's
                    x_nearest[batch_id] = np.delete(x_all, 0, axis=0)
                else:
                    x_all = np.asarray(
                        [self.experience['s'][j] for j in range(total_experiences - K - 1, total_experiences)])
                    x_actual[batch_id] = x_all[-1]
                    # Calculate nearest X's
                    x_nearest[batch_id] = np.delete(x_all, -1, axis=0)
            # Calculate R
            _, r_actual = self.model(np.atleast_2d(x_actual.astype('float32')))
            # Run Gradient Descent on W and Phi together
            loss_lcr = 0
            for _ in range(self.alg_params['Phi_gradient_steps']):
                with tf.GradientTape() as tape:
                    # Calculate nearest R
                    _, r_nearest = self.model(np.atleast_2d(x_nearest.astype('float32')))
                    r_predicted = lcr_model(tf.reshape(r_nearest, [batch_size, self.hidden_units[-1], K]))
                    loss = tf.keras.losses.mean_squared_error(np.atleast_3d(r_actual), r_predicted)
                    optimizer = tf.optimizers.Adam(learning_rate=self.alg_params['lcr_learning_rate'])
                    var_list = self.model.trainable_weights + lcr_model.trainable_weights
                    grads = tape.gradient(loss, var_list)
                    optimizer.apply_gradients(zip(grads, var_list))
                    loss_lcr += loss
            losses_lcr.append(loss_lcr / self.alg_params['Phi_gradient_steps'])
        return np.mean(losses_lcr), np.mean(losses_lcr)

    def train(self, TargetNet):
        ids = np.random.randint(low=0, high=len(self.experience['s']), size=self.batch_size)
        states = np.asarray([self.experience['s'][i] for i in ids])
        actions = np.asarray([self.experience['a'][i] for i in ids])
        rewards = np.asarray([self.experience['r'][i] for i in ids])
        states_next = np.asarray([self.experience['s2'][i] for i in ids])
        dones = np.asarray([self.experience['done'][i] for i in ids])
        value_next = np.max(TargetNet.predict(states_next), axis=1)
        actual_values = np.where(dones, rewards, rewards + self.gamma * value_next)

        with tf.GradientTape() as tape:
            selected_action_values = tf.math.reduce_sum(
                self.predict(states) * tf.one_hot(actions, self.num_actions), axis=1)
            loss = tf.math.reduce_mean(tf.square(actual_values - selected_action_values))
        variables = self.model.trainable_variables
        gradients = tape.gradient(loss, variables)
        self.optimizer.apply_gradients(zip(gradients, variables))
        return loss, ids

    def get_action(self, states, epsilon):
        if np.random.random() < epsilon:
            return np.random.choice(self.num_actions)
        else:
            return np.argmax(self.predict(np.atleast_2d(states))[0])

    def add_experience(self, exp):
        if len(self.experience['s']) >= self.max_experiences:
            for key in self.experience.keys():
                self.experience[key].pop(0)
        for key, value in exp.items():
            self.experience[key].append(value)

    def copy_weights(self, TrainNet):
        variables1 = self.model.trainable_variables
        variables2 = TrainNet.model.trainable_variables
        for v1, v2 in zip(variables1, variables2):
            v1.assign(v2.numpy())


def play_game(global_step, env, TrainNet, TargetNet, epsilon, copy_step):
    rewards = 0
    done = False
    observations = env.reset()
    losses = list()
    if 'lcr' in TrainNet.alg:
        losses_lcr_x = list()
        losses_lcr_x.append(0)
        losses_lcr_r = list()
        losses_lcr_r.append(0)
        idxes = []
    while not done:
        action = TrainNet.get_action(observations, epsilon)
        prev_observations = observations
        observations, reward, done, _ = env.step(action)
        rewards += reward
        if done:
            env.reset()
        exp = {'s': prev_observations, 'a': action, 'r': reward, 's2': observations, 'done': done}
        TrainNet.add_experience(exp)
        if len(TrainNet.experience['s']) < TrainNet.min_experiences:
            loss = 0
            if 'lcr' in TrainNet.alg:
                loss_x = 0
                loss_r = 0
                losses_lcr_x.append(loss_x)
                losses_lcr_r.append(loss_r)
        else:
            loss, ids = TrainNet.train(TargetNet)
            if 'lcr' in TrainNet.alg:
                if global_step % TrainNet.alg_params['lcr_batch_size'] == 0:
                    loss_x, loss_r = TrainNet.train_lcr()
                    losses_lcr_x.append(loss_x)
                    losses_lcr_r.append(loss_r)

        if isinstance(loss, int):
            losses.append(loss)
        else:
            losses.append(loss.numpy())
        if global_step % copy_step == 0:
            TargetNet.copy_weights(TrainNet)
        global_step += 1
    if 'lcr' in TrainNet.alg:
        return global_step, rewards, mean(losses), mean(losses_lcr_x), mean(losses_lcr_r)
    else:
        return global_step, rewards, mean(losses)


def test(env, TrainNet, logs, num_episodes):
    for _ in range(num_episodes):
        observation = env.reset()
        rewards = 0
        steps = 0
        done = False
        while not done:
            action = TrainNet.get_action(observation, 0)
            observation, reward, done, _ = env.step(action)
            steps += 1
            rewards += reward
        with open(logs['log_file_name'], "a") as f:
            print("Testing steps: {} rewards :{} ".format(steps, rewards), file=f)
        print("Testing steps: {} rewards :{} ".format(steps, rewards))


def train_agent(env, num_episodes, model_params, algorithm_params, logs, verbose):
    num_actions = env.number_of_actions
    try:
        state_space = len(env.state_space.sample())
    except TypeError:
        state_space = env.state_space.n
    if verbose:
        train_columns = ['Episode', 'Epsilon', 'Rewards', 'TD Error']
        train_df = pd.DataFrame(columns=train_columns)
    copy_step = model_params['copy_step']
    alg = algorithm_params['algorithm']
    TrainNet = DQN(state_space, num_actions, model_params, algorithm_params)
    TargetNet = DQN(state_space, num_actions, model_params, algorithm_params)
    epsilon_start = algorithm_params['start_epsilon']
    epsilon = epsilon_start
    decay = algorithm_params['epsilon_decay']**(1/num_episodes)
    min_epsilon = algorithm_params['stop_epsilon']
    global_step = 1
    n = 0
    pbar = tqdm(total=num_episodes)
    while True:
        epsilon = max(min_epsilon, epsilon*decay)
        if 'lcr' in alg:
            global_step, total_reward, losses, loss_x, loss_r = \
                play_game(global_step, env, TrainNet, TargetNet, epsilon, copy_step)
        else:
            global_step, total_reward, losses = play_game(global_step, env, TrainNet, TargetNet, epsilon, copy_step)
        '''Store the results'''
        if verbose:
            '''Save Rewards and Losses'''
            train_df.loc[n, 'Episode'] = n + 1
            train_df.loc[n, 'Epsilon'] = epsilon
            train_df.loc[n, 'Rewards'] = total_reward
            train_df.loc[n, 'TD Error'] = losses
            if 'lcr' in alg:
                train_df.loc[n, 'X-loss'] = float(loss_x)
                train_df.loc[n, 'R-loss'] = float(loss_r)
            train_df.to_csv(logs['output_dir'] + '{}_training_run_{}.csv'.format(alg, algorithm_params['seed']),
                            encoding='utf-8', index=False)
            TrainNet.model.save_weights(logs['model_dir'] + '{}_output/'.format(alg))
        else:
            if n % logs['log_interval'] == 0:
                print("episode:{}, eps:{:.3f}, reward:{:.2f}, loss:{:.2f}".format(n, epsilon, total_reward, losses))
        n += 1
        pbar.update(1)
        if n == num_episodes:
            break
    env.close()
