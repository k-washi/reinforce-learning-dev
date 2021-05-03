"""
1. ステップで変更されるε-Greedy方策を導入
2. Huber Lossに変更
3. Experiment replay：溜まるまで学習しないようにする(自己相関の低減のため)
"""

import sys
import pathlib

import random
import gym
import time
import numpy as np

from collections import deque # スレッドセーフなキュー
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.python.util.tf_decorator import rewrap


# utilsのパスを通す
module_path = pathlib.Path(__file__, "..", "..").resolve()
if module_path not in sys.path:
    sys.path.append(str(module_path))

from utils.tfboard import RLSummaryWriter, RLModelSaver
from gym_env.packman import GymPackman

class QNet(tf.keras.Model):

    def __init__(self, actions_space):
        super(QNet, self).__init__()
        self.action_space = actions_space
        self.conv1 = Conv2D(32, 8, strides=4, activation="relu", kernel_initializer="he_normal")
        self.conv2 = Conv2D(64, 4, strides=2, activation="relu", kernel_initializer="he_normal")
        self.conv3 = Conv2D(64, 3, strides=1, activation="relu", kernel_initializer="he_normal")
        self.flatten1 = Flatten()
        self.dense1 = Dense(512, activation="relu", kernel_initializer="he_normal")
        self.qvalues = Dense(self.action_space, kernel_initializer="he_normal")

    @tf.function
    def call(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.flatten1(x)
        x = self.dense1(x)
        qvalues = self.qvalues(x)
        return qvalues

class DQN():
    def __init__(self, state_size, action_size, max_episode=500, replay_buffer_num=5000, batch_size=8, n_step_size=4) -> None:
        self._state_size = state_size
        self._action_size = action_size

        self._replay_buffer = deque(maxlen=replay_buffer_num)

        self._q_n_step_size = n_step_size
        self._tmp_replay_buffer = deque(maxlen=self._q_n_step_size)

        self.gamma = 0.9
        self.update_rate = 1000 #* batch_size

        self._max_episode = max_episode
        self._min_epsilon_rate = 0.1

        self._batch_size = batch_size
        self._reward_clip = True
        
        self.main_net = self.build_net()
        self.target_net = self.build_net()

        self.optimizer = tf.keras.optimizers.Adam(lr=0.00025, epsilon=0.01/self._batch_size)
        self.loss = tf.keras.losses.Huber(delta=1.0)

        self.target_net.set_weights(self.main_net.get_weights())

    
    def build_net(self):
        return QNet(self._action_size)
    
    def store_transistion(self, state, action, reward, next_state, done):
        # 遅延報酬の伝搬のため
        self._tmp_replay_buffer.append((state, action, reward, next_state, done))
        if len(self._tmp_replay_buffer) == self._q_n_step_size:
            nstep_return = 0
            has_done = 0

            for i, (state, action, reward, next_state, done) in enumerate(self._tmp_replay_buffer):
                reward = np.clip(reward, -1, 1) if self._reward_clip else reward
                nstep_return += self.gamma ** i * (1 - done) * reward
                if done:
                    has_done = True
                    break
            state = self._tmp_replay_buffer[0][0]
            action = self._tmp_replay_buffer[0][1]
            next_state = self._tmp_replay_buffer[-1][3]
            self._replay_buffer.append((state, action, nstep_return, next_state, has_done))

            self._tmp_replay_buffer = deque(maxlen=self._q_n_step_size)
            return True
        return False

    def epsilon_greedy(self, state, step):
        epsilon = max(1 - (1 - self._min_epsilon_rate) * step / self._max_episode, self._min_epsilon_rate)
        if random.uniform(0, 1) < epsilon:
            return np.random.randint(self._action_size)
        Q_value = self.main_net.predict(state) 
        return np.argmax(Q_value[0])
    
    def train(self, batch_size):
        minibatch = random.sample(self._replay_buffer, batch_size)
        #b_Q_values = np.zeros((batch_size, 1, self._action_size))
        #b_state = np.zeros((batch_size, *self._state_size))
    
        states, actions, rewards, next_states, dones = zip(*minibatch)
        states, actions, rewards, next_states, dones = np.vstack(states), np.vstack(actions), np.vstack(rewards), np.vstack(next_states), np.vstack(dones)
        not_dones = np.logical_not(dones).astype(np.int32)
        # print(states.shape, actions.shape, rewards.shape, next_states.shape, dones.shape)
        # (8, 88, 80, 1) (8, 1) (8, 1) (8, 88, 80, 1) (8, 1)
        
        target_Qs = rewards + not_dones * self.gamma ** self._q_n_step_size * np.amax(self.target_net.predict(next_states), axis=1) # (8, 1)
        with tf.GradientTape() as tape:
            Q_values = self.main_net(states) # (8, 9)
            actions_onehot = tf.one_hot(actions.flatten().astype(np.int32), self._action_size)
            q = tf.reduce_sum(Q_values * actions_onehot, axis=1, keepdims=True)

            loss = self.loss(target_Qs, q)
        
        grad = tape.gradient(loss, self.main_net.trainable_variables)
        self.optimizer.apply_gradients(zip(grad, self.main_net.trainable_variables))

        return loss

    def update_target_net(self):
        self.target_net.set_weights(self.main_net.get_weights())

def main():
    import time

    env = GymPackman()
    state_size = env.state_size
    action_size = env.action_size

    print(state_size, action_size)

    num_epi = 2000
    num_timesteps = 5000
    replay_buffer_num = 50000
    batch_size = 8
    save_epi_rate = 50
    q_n_step_size = 2

    dqn = DQN(state_size, action_size, num_epi, replay_buffer_num, batch_size, n_step_size=q_n_step_size)
    done = False

    rl_summary = RLSummaryWriter()
    writer = RLModelSaver("./models")

    time_step = 0
    for epi in range(num_epi):
        Return = 0
        state = env.reset()
        state = env.preprocess_state(state)
        state = env.state_stack(state)

        start_time = time.time()
        learning = False
        for t in range(num_timesteps):
            env.render()
            
            if time_step % dqn.update_rate == 0 and len(dqn._replay_buffer) > replay_buffer_num:
                dqn.update_target_net()

            state = env.get_state()
            a = dqn.epsilon_greedy(state, epi)
            
            

            next_state, reward, done = env.step(a)
            next_state = env.preprocess_state(next_state)
            env.state_stack(next_state)
            next_state = env.get_state()
            set_buffer = dqn.store_transistion(state, a, reward, next_state, done)

            state = next_state
            Return += reward
            if done:
                print(f'Episode:{epi}, timesteps: {time_step}: {len(dqn._replay_buffer)}, Return: {Return}, time: {time.time() - start_time}')
                break
            
            if len(dqn._replay_buffer) >= replay_buffer_num and set_buffer:
                learning = True
                time_step += 1
                dqn.train(batch_size)

        rl_summary.set_frame_id(epi)
        rl_summary.add_return(Return)
        if learning:
            model_path = writer.reward_save(dqn.main_net, epi, Return)
            if model_path is not None:
                print(f"{model_path} is saved.")

            if epi % save_epi_rate == 0:
                model_path = writer.save(dqn.main_net, epi)


    model_path = writer.save(dqn.main_net, epi)



if __name__ == "__main__":
    main()
        