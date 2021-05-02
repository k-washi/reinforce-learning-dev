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


class DuelingQNet(tf.keras.Model):
    def __init__(self, actions_space):

        super(DuelingQNet, self).__init__()
        self.action_space = actions_space
        self.conv1 = Conv2D(32, 8, strides=4, activation="relu", kernel_initializer="he_normal")
        self.conv2 = Conv2D(64, 4, strides=2, activation="relu", kernel_initializer="he_normal")
        self.conv3 = Conv2D(64, 3, strides=1, activation="relu", kernel_initializer="he_normal")
        self.flatten1 = Flatten()
        self.dense1 = Dense(512, activation="relu", kernel_initializer="he_normal")
        self.value = Dense(1, kernel_initializer="he_normal")
        self.dense2 = Dense(512, activation="relu", kernel_initializer="he_normal")
        self.advantages = Dense(self.action_space, kernel_initializer="he_normal")
        
    @tf.function
    def call(self, x):

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.flatten1(x)

        x1 = self.dense1(x)
        value = self.value(x1)

        x2 = self.dense2(x)
        advantages = self.advantages(x2)
        advantages_scaled = advantages - tf.reduce_mean(advantages, axis=1, keepdims=True)
        
        q = value + advantages_scaled

        return q


class DQN():
    def __init__(self, state_size, action_size, max_episode=500, replay_buffer_num=5000, batch_size=8) -> None:
        self._state_size = state_size
        self._action_size = action_size

        self._replay_buffer = deque(maxlen=replay_buffer_num)
        self.gamma = 0.9
        self.update_rate = 1000

        self._max_episode = max_episode
        self._min_epsilon_rate = 0.001

        self._batch_size = batch_size

        
        self.main_net = self.build_net()
        self.target_net = self.build_net()
        
        self.optimizer = tf.keras.optimizers.Adam(lr=0.00025, epsilon=0.01/self._batch_size)
        self.loss = tf.keras.losses.Huber(delta=1.0)

        self.target_net.set_weights(self.main_net.get_weights())

    
    def build_net(self):
        model = DuelingQNet(self._action_size)

        return model
    
    def store_transistion(self, state, action, reward, next_state, done):
        self._replay_buffer.append((state, action, reward, next_state, done))
    

    def epsilon_greedy(self, state, step):
        epsilon = max(1 - (1 - self._min_epsilon_rate) * step / self._max_episode, self._min_epsilon_rate)
        if random.uniform(0, 1) < epsilon:
            return np.random.randint(self._action_size)
        Q_value = self.main_net(state) 
        return np.argmax(Q_value[0])
    
    def train(self, batch_size):
        minibatch = random.sample(self._replay_buffer, batch_size)
        #b_Q_values = np.zeros((batch_size, 1, self._action_size))
        #b_state = np.zeros((batch_size, *self._state_size))
    
        states, actions, rewards, next_states, dones = zip(*minibatch)
        states, actions, rewards, next_states, dones = np.vstack(states), np.vstack(actions), np.vstack(rewards), np.vstack(next_states), np.vstack(dones)
        rewards = np.clip(rewards, -1, 1)
        not_dones = np.logical_not(dones).astype(np.int32)
        #print(states.shape, actions.shape, rewards.shape, next_states.shape, dones.shape)
        # (8, 88, 80, 1) (8, 1) (8, 1) (8, 88, 80, 1) (8, 1)
        
        target_Qs = rewards + not_dones * self.gamma * np.amax(self.target_net(next_states), axis=1) # (8, 1)
        with tf.GradientTape() as tape:
            Q_values = self.main_net(states) # (8, 9)
            actions_onehot = tf.one_hot(actions.flatten().astype(np.int32), self._action_size)
            q = tf.reduce_sum(Q_values * actions_onehot, axis=1, keepdims=True)

            loss = self.loss(target_Qs, q)
        
        grad = tape.gradient(loss, self.main_net.trainable_variables)
        self.optimizer.apply_gradients(zip(grad, self.main_net.trainable_variables))

        return loss
        
        #self.main_net.fit(states, Q_values, epochs=1, verbose=0) # intput, result
        
    
    def update_target_net(self):
        self.target_net.set_weights(self.main_net.get_weights())

def main():
    import time

    env = GymPackman()
    state_size = env.state_size
    action_size = env.action_size

    print(state_size, action_size)

    num_epi = 1200
    num_timesteps = 20000
    replay_buffer_num = 500
    batch_size = 8
    save_epi_rate = 50

    dqn = DQN(state_size, action_size, num_epi, replay_buffer_num, batch_size)
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
        total_loss = 0
        for t in range(num_timesteps):
            env.render()
            time_step += 1
            if time_step % dqn.update_rate == 0 and time_step > replay_buffer_num:
                dqn.update_target_net()

            state = env.get_state()
            a = dqn.epsilon_greedy(state, epi)
            
            

            next_state, reward, done = env.step(a)
            next_state = env.preprocess_state(next_state)
            env.state_stack(next_state)
            next_state = env.get_state()
            dqn.store_transistion(state, a, reward, next_state, done)

            state = next_state
            Return += reward
            if done:
                print(f'Episode:{epi}, timesteps: {t}, Return: {Return}, time: {time.time() - start_time}')
                break
            
            if len(dqn._replay_buffer) > batch_size and t % env._learning_frame_length == 0 and time_step > replay_buffer_num:
                loss = dqn.train(batch_size)
                total_loss += loss
        model_path = writer.reward_save(dqn.main_net, epi, Return)
        if model_path is not None:
            print(f"{model_path} is saved.") 
        
        if epi % save_epi_rate == 0:
            model_path = writer.save(dqn.main_net, epi)

        rl_summary.set_frame_id(epi)
        rl_summary.add_return(Return)
    
    model_path = writer.save(dqn.main_net, epi)


            
            







if __name__ == "__main__":
    main()
        