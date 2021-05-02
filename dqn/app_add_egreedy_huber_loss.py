"""
1. ステップで変更されるε-Greedy方策を導入
2. Huber Lossに変更
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

from utils.tfboard import RLSummaryWriter

color = np.array([210, 164, 74]).mean() # for change image contrast

# フレームをX個繋げる

class GymPackman():
    def __init__(self) -> None:
        self._env = gym.make("MsPacman-v0")
        self._learning_frame_length = 4
        self._height = 88
        self._width = 80
        self._state_size = (self._height, self._width, self._learning_frame_length)
        self._action_size = self._env.action_space.n

        print(f"Create gym env, s:{self._state_size}, a:{self._action_size}")

        
    
    @property
    def state_size(self):
        return self._state_size
    
    @property
    def action_size(self):
        return self._action_size
        
    def preprocess_state(self, state):
        image = state[1:176:2, ::2] # 210x160x3 => 88x80x3
        image = image.mean(axis=2) # gray scale
        #image[image == color] = 0
        image = (image - 128) / 128 - 1
        
        image = image.reshape(self._height, self._width)
        image = np.expand_dims(image, axis=0)
        
        return image


    def reset(self):
        state = self._env.reset()
        frame = self.preprocess_state(state)
        self._frames = deque([frame] * 4, maxlen=self._learning_frame_length) 
        return state
    
    def state_stack(self, state):
        self._frames.append(state)
        return np.stack(self._frames, axis=3)
    
    def get_state(self):
        return np.stack(self._frames, axis=3)

    def render(self):
        return self._env.render()

    def step(self, action):
        next_state, reward, done, _ = self._env.step(action)
        
        return next_state, reward, done

class DQN():
    def __init__(self, state_size, action_size, max_episode=500) -> None:
        self._state_size = state_size
        self._action_size = action_size

        self._replay_buffer = deque(maxlen=5000)
        self.gamma = 0.9
        self.update_rate = 1000

        self._max_episode = max_episode
        self._min_epsilon_rate = 0.1

        
        self.main_net = self.build_net()
        self.target_net = self.build_net()

        self.target_net.set_weights(self.main_net.get_weights())

    
    def build_net(self):
        model = Sequential()
        model.add(Conv2D(32, (8, 8), strides=4, padding='same', input_shape=self._state_size))
        model.add(Activation('relu'))
        model.add(Conv2D(64, (4, 4), strides=2, padding='same'))
        model.add(Activation('relu'))
        model.add(Conv2D(64, (4, 4), strides=2, padding='same'))
        model.add(Activation('relu'))

        model.add(Flatten())
        model.add(Dense(512, activation='relu'))
        model.add(Dense(self._action_size, activation='linear'))

        loss = tf.keras.losses.Huber(delta=1.0)
        model.compile(loss=loss, optimizer=Adam())

        return model
    
    def store_transistion(self, state, action, reward, next_state, done):
        self._replay_buffer.append((state, action, reward, next_state, done))
    

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
        
        target_Qs = rewards + not_dones * self.gamma * np.amax(self.target_net.predict(next_states), axis=1) # (8, 1)
        Q_values = self.main_net.predict(states) # (8, 9)
        for ind in range(batch_size):
            a = actions[ind][0]
            Q_values[ind][a] = target_Qs[ind][0]
        
        self.main_net.fit(states, Q_values, epochs=1, verbose=0) # intput, result
    
    def update_target_net(self):
        self.target_net.set_weights(self.main_net.get_weights())

def main():
    import time

    env = GymPackman()
    state_size = env.state_size
    action_size = env.action_size

    print(state_size, action_size)

    num_epi = 500
    num_timesteps = 20000
    batch_size = 8
    num_screen = 4

    dqn = DQN(state_size, action_size, num_epi)
    done = False

    rl_summary = RLSummaryWriter()

    time_step = 0
    for epi in range(num_epi):
        Return = 0
        state = env.reset()
        state = env.preprocess_state(state)
        state = env.state_stack(state)

        start_time = time.time()
        for t in range(num_timesteps):
            env.render()
            time_step += 1
            if time_step % dqn.update_rate == 0:
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
            
            if len(dqn._replay_buffer) > batch_size and t % env._learning_frame_length == 0:
                dqn.train(batch_size)
        
        rl_summary.set_frame_id(epi)
        rl_summary.add_return(Return)
            
            







if __name__ == "__main__":
    main()
        