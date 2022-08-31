import tensorflow as tf
import numpy as np
import random
import os
import copy
from statistics import mean
from collections import deque
from tqdm import tqdm
import psutil
from psutil._common import bytes2human

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


def randargmax(b, **kw):
    """ a random tie-breaking argmax"""
    return np.argmax(np.random.random(b.shape) * (b == b.max()), **kw)


class Model(tf.keras.Model):
    def __init__(self, num_states, is_full_state, hidden_units, num_actions, alg):
        super(Model, self).__init__()
        self.alg = alg
        self.full_state = is_full_state
        if is_full_state:
            self.input_layer = tf.keras.layers.InputLayer(input_shape=(num_states[0], num_states[1], num_states[2]), name='input')
            self.cnn_layer_1 = tf.keras.layers.Conv2D(filters=16, kernel_size=2, activation='relu', name='cnn_1')
            self.max_pool_1 = tf.keras.layers.MaxPool2D(pool_size=(2, 2), name='maxpool_1')
            self.cnn_layer_2 = tf.keras.layers.Conv2D(filters=32, kernel_size=2, activation='relu', name='cnn_2')
            self.cnn_layer_3 = tf.keras.layers.Conv2D(filters=64, kernel_size=2, activation='relu', name='cnn_3')
            self.flatten_layer = tf.keras.layers.Flatten(name='flatten')
            self.common_layer = tf.keras.layers.Dense(hidden_units[0], activation='tanh', kernel_initializer='RandomNormal', name='common_layer')
        else:
            self.input_layer = tf.keras.layers.InputLayer(input_shape=(num_states, ), name="input")
            self.flatten_layer = tf.keras.layers.Flatten(name='flatten')
            self.common_layer_1 = tf.keras.layers.Dense(hidden_units[0], activation='tanh', kernel_initializer='RandomNormal', name='common_layer_1')
            self.common_layer_2 = tf.keras.layers.Dense(hidden_units[1], activation='tanh', kernel_initializer='RandomNormal', name='common_layer_2')
        
        self.dqn_model = tf.keras.Sequential(name='DQN')
        if is_full_state:
            self.dqn_model.add(self.input_layer)
            self.dqn_model.add(self.cnn_layer_1)
            self.dqn_model.add(self.max_pool_1)
            self.dqn_model.add(self.cnn_layer_2)
            self.dqn_model.add(self.cnn_layer_3)
            self.dqn_model.add(self.flatten_layer)
            self.dqn_model.add(self.common_layer)
            for i in range(1, len(hidden_units)):
                self.dqn_model.add(
                    tf.keras.layers.Dense(hidden_units[i], activation='tanh', kernel_initializer='RandomNormal',
                                            name=f'DQN_layer_{i}'))
        else:
            self.dqn_model.add(self.input_layer)
            self.dqn_model.add(self.common_layer_1)
            self.dqn_model.add(self.common_layer_2)

            for i in range(2, len(hidden_units)):
                self.dqn_model.add(
                    tf.keras.layers.Dense(hidden_units[i], activation='tanh', kernel_initializer='RandomNormal',
                                            name=f'DQN_layer_{i}'))
        self.dqn_model.add(tf.keras.layers.Dense(num_actions, activation='linear', kernel_initializer='RandomNormal',
                                                name='DQN_output_layer'))


    def representation_out(self, inputs):
        if self.full_state:
            x = self.input_layer(inputs)
            x = self.cnn_layer_1(x)
            x = self.max_pool_1(x)
            x = self.cnn_layer_2(x)
            x = self.cnn_layer_3(x)
            x = self.flatten_layer(x)
            x = self.common_layer(x)
        else:
            x = self.input_layer(inputs)
            x = self.common_layer_1(x)
            x = self.common_layer_2(x)
        return x

    @tf.function
    def call(self, inputs):
        output = self.dqn_model(inputs)
        return output



class LCR_Model(tf.keras.Model):
    def __init__(self, K, input_dim, batch_size):
        super(LCR_Model, self).__init__()
        self.input_layer = tf.keras.layers.InputLayer(input_shape=(input_dim, K), batch_size=batch_size)
        '''Constrain to positive weights to eliminate over fitting'''
        self.output_layer = tf.keras.layers.Dense(1, use_bias=False, kernel_constraint=tf.keras.constraints.NonNeg())


    @tf.function
    def call(self, inputs):
        z = self.input_layer(inputs)
        output = self.output_layer(z)
        return output

    def reinitialize(self):
        for l in self.layers:
            if hasattr(l,"kernel_initializer"):
                l.kernel.assign(l.kernel_initializer(tf.shape(l.kernel)))
            if hasattr(l,"bias_initializer"):
                l.bias.assign(l.bias_initializer(tf.shape(l.bias)))
            if hasattr(l,"recurrent_initializer"):
                l.recurrent_kernel.assign(l.recurrent_initializer(tf.shape(l.recurrent_kernel)))
        return self


class DQN:
    def __init__(self, env, num_states, num_actions, model_params, alg_params):
        np.random.seed(alg_params['seed'])
        tf.random.set_seed(alg_params['seed'])
        random.seed(alg_params['seed'])
        if alg_params['use_gpu']:
            tf.config.experimental.set_visible_devices([GPUs[0]], 'GPU')
        else:
            tf.config.experimental.set_visible_devices([], 'GPU')
        self.env = env
        self.num_actions = num_actions
        self.num_states = num_states
        self.alg = alg_params['algorithm']
        self.batch_size = alg_params['batch_size']
        self.hidden_units = model_params['hidden_units']
        self.gamma = alg_params['gamma']
        self.is_full_state = alg_params['full_state']
        self.lcr_batch_size = alg_params['lcr_batch_size']
        self.lcr_learning_rate = alg_params['lcr_learning_rate']
        self.K = alg_params['K']
        self.gradient_steps = alg_params['gradient_steps']
        self.model = Model(num_states, self.is_full_state, self.hidden_units, num_actions, self.alg)
        self.dqn_optimizer = tf.optimizers.Adam(alg_params['learning_rate'])
        self.experience = {'s': [], 'a': [], 'r': [], 's2': [], 'done': []}
        self.max_experiences = model_params['max_buffer_size']
        self.min_experiences = model_params['min_buffer_size']

    def predict(self, inputs):
        return self.model(np.atleast_2d(inputs.astype('float32')))

    def train_dqn(self, TargetNet, selected_gvf=0):
        if len(self.experience['s']) < self.min_experiences:
            return 0
        ids = np.random.randint(low=0, high=len(self.experience['s']), size=self.batch_size)
        states = np.asarray([self.experience['s'][i] for i in ids])
        actions = np.asarray([self.experience['a'][i] for i in ids])
        states_next = np.asarray([self.experience['s2'][i] for i in ids])
        rewards = np.asarray([self.experience['r'][i] for i in ids])
        dones = np.asarray([self.experience['done'][i] for i in ids])

        if self.alg == 'dqn' or self.alg == 'dqn_lcr':
            value_next = np.max(TargetNet.model(states_next), axis=1)
        else:
            raise Exception('Algorithm undefined')
        actual_values = np.where(dones, rewards, rewards + self.gamma * value_next)

        with tf.GradientTape() as tape:
            selected_action_values = tf.math.reduce_sum(
                self.model(states) * tf.one_hot(actions, self.num_actions), axis=1)
            loss = tf.math.reduce_mean(tf.square(actual_values - selected_action_values))
        variables = self.model.dqn_model.trainable_variables
        gradients = tape.gradient(loss, variables)
        self.dqn_optimizer.apply_gradients(zip(gradients, variables))
        return loss.numpy()

    def train_lcr(self):
        total_experiences = len(self.experience['s'])
        K = self.K
        losses_lcr = []
        if self.is_full_state:
            X_dim = self.num_states
        else:
            X_dim = (self.num_states, )
        x_actual = np.zeros(shape=(self.lcr_batch_size,) + X_dim)
        x_nearest = np.zeros(shape=(self.lcr_batch_size,) + (self.K, ) + X_dim)
        idxs = np.arange(total_experiences - self.lcr_batch_size, total_experiences)
        # idxs = np.random.randint(low=0,high=total_experiences, size=self.lcr_batch_size)
        lcr_model = LCR_Model(K=self.K, input_dim=self.hidden_units[0], batch_size=self.lcr_batch_size)
        for batch_no, id in enumerate(idxs):
            x_all = np.zeros(shape=(self.K + 1,) + X_dim)
            if id >= self.K // 2 and total_experiences - id > self.K // 2 + (self.K % 2):
                for count, j in enumerate(range(id - self.K // 2, id + self.K // 2 + (self.K % 2) + 1)):
                    x_all[count] = self.experience['s'][j]
                x_actual[batch_no] = x_all[self.K // 2]
                # Calculate nearest X's
                x_nearest[batch_no] = np.concatenate((x_all[0:self.K//2], x_all[self.K//2+1:]), axis=0)
            elif id < self.K // 2:
                for j in range(0, self.K + 1):
                    x_all[j] = self.experience['s'][j]
                x_actual[batch_no] = x_all[0]
                # Calculate nearest X's
                x_nearest[batch_no] = x_all[1:]
            else:
                for count, j in enumerate(range(total_experiences - self.K - 1, total_experiences)):
                    x_all[count] = self.experience['s'][j]
                x_actual[batch_no] = x_all[-1]
                # Calculate nearest X's
                x_nearest[batch_no] = x_all[:-1]
        # Calculate R
        r_actual = self.model.representation_out(np.atleast_2d(x_actual.astype('float32')))
        # Run Gradient Descent on W and Phi together
        loss_lcr = 0
        for _ in range(self.gradient_steps):
            with tf.GradientTape() as tape:
                # Calculate nearest R
                r_nearest = []
                # r_nearest = tf.zeros(shape=(self.lcr_batch_size, self.hidden_units[0], self.K))
                for k in range(self.K):
                    r_nearest.append(self.model.representation_out(x_nearest[:, k, :]))
                r_nearest = tf.stack(r_nearest)
                r_predicted = lcr_model(tf.reshape(r_nearest, [self.lcr_batch_size, self.hidden_units[0], K]))
                loss = tf.keras.losses.mean_squared_error(np.atleast_3d(r_actual), r_predicted)
                optimizer = tf.optimizers.Adam(learning_rate=self.lcr_learning_rate)
                var_list = self.model.trainable_weights + lcr_model.trainable_weights
                grads = tape.gradient(loss, var_list)
                optimizer.apply_gradients(zip(grads, var_list))
                loss_lcr += loss
        losses_lcr.append(loss_lcr / self.gradient_steps)
        return np.mean(losses_lcr)

    def get_action(self, states):
        # states = np.array(states, ndmin=4)
        if self.is_full_state:
            states = np.array(states, ndmin=4)
        else:
            states = np.atleast_2d(states)
        action_values = np.asarray(self.model(states)[0])
        return randargmax(action_values)

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


def play_game(is_full_state, global_step, num_episodes, episode, env, alg, TrainNet, TargetNet, epsilon, copy_step):
    rewards = 0
    episode_step = 0
    done = False
    observations = env.reset()
    if is_full_state:
        observations = np.atleast_3d(observations)

    losses_dqn = list()
    losses_lcr = list()
    while not done:
        if np.random.random() <= 1-epsilon:
            action = TrainNet.get_action(observations)
        else:
            action = np.random.choice(TrainNet.num_actions)

        prev_observations = observations
        observations, reward, done = env.step(action)
        if is_full_state:
            observations = np.atleast_3d(observations)

        episode_step += 1
        rewards += reward
        if done:
            if episode % 10 == 0:
                env.full_reset()
            else:
                env.reset()

        global_step += 1
        exp = {'s': prev_observations, 'a': action, 'r': reward, 's2': observations, 'done': done}

        TrainNet.add_experience(exp)

        if alg == 'dqn' or alg == 'dqn_lcr':
            loss_dqn = TrainNet.train_dqn(TargetNet)
            losses_dqn.append(loss_dqn)
        if alg == 'dqn_lcr':
            if global_step % TrainNet.lcr_batch_size == 0:
                loss_lcr = TrainNet.train_lcr()
                losses_lcr.append(loss_lcr)
        

        if episode % copy_step == 0:
            TargetNet.copy_weights(TrainNet)
    if alg == 'dqn':
        losses = [np.mean(np.asarray(losses_dqn))]
    elif alg == 'dqn_lcr':
        losses = [np.mean(np.asarray(losses_dqn))]
    else:
        raise Exception('Algorithm Undefined')
    return global_step, episode_step, rewards, losses


def train_agent(env, num_episodes, model_params, algorithm_params, logs, save):
    num_actions = env.num_actions
    is_full_state = algorithm_params['full_state']
    try:
        if is_full_state:
            sample_state = np.atleast_3d(env.reset())
            state_space = sample_state.shape
        else:
            state_space = env.num_states
    except AttributeError:
        state_space = env.obs_length

    copy_step = model_params['copy_step']
    saved_model_dir = logs['model_dir'] + 'Run_' + str(logs['run']) + '_' + algorithm_params['algorithm'] + '/'
    TrainNet = DQN(env, state_space, num_actions, model_params, algorithm_params)
    TargetNet = DQN(env, state_space, num_actions, model_params, algorithm_params)
    
    total_rewards_list = []
    total_losses_list = []
    alg = algorithm_params['algorithm']
    epsilon_start = algorithm_params['start_epsilon']
    episodes_after_min_epsilon = algorithm_params['epsilon_decay']
    min_epsilon = algorithm_params['stop_epsilon']
    global_step = 1
    epsilon_decay = algorithm_params['epsilon_decay']
    n = 0
    
    pbar = tqdm(total=num_episodes)
    epsilon = epsilon_start
    while True:
        global_step, episode_step, total_reward, losses = play_game(is_full_state, global_step, num_episodes, n,
                                                                                           env, alg, TrainNet, TargetNet, epsilon, copy_step)
        total_rewards_list.append(total_reward)
        total_rewards = np.array(total_rewards_list)
        avg_rewards = total_rewards[max(0, n - 100):(n + 1)].mean()
        if n % logs['log_interval'] == 0:
            if save:
                train_df = logs['log_file']
                train_df.loc[n, 'Reward'] = np.round(total_reward, 4)
                train_df.loc[n, 'Steps'] = episode_step
                train_df.loc[n, 'Epsilon'] = np.round(epsilon, 4)
                if alg == 'dqn' or alg == 'dqn_lcr':
                    train_df.loc[n, 'DQN_Loss'] = np.round(np.mean(np.asarray(losses[0])), 4)
                train_df.loc[n, 'Ram Usage'] = psutil.Process(os.getpid()).memory_info().rss / 1024 ** 3
                train_df.to_csv(logs['output_dir'] + alg + '_training_run_{}.csv'.format(logs['run']), encoding='utf-8',
                                index=False)
            if not save:
                print("episode:{}, eps:{:.3f}, reward:{:.2f}".format(n, epsilon, total_reward))
        n += 1
        epsilon *= epsilon_decay
        epsilon = max(min_epsilon, epsilon)
        
        pbar.update(1)
        if n == num_episodes:
            break
        
    if not os.path.exists(saved_model_dir):
        os.makedirs(saved_model_dir)
    TrainNet.model.save_weights(saved_model_dir)
    return total_rewards