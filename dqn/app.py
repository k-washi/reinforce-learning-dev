import random
import gym
import numpy as np

from collections import deque # スレッドセーフなキュー

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.python.util.tf_decorator import rewrap

color = np.array([210, 164, 74]).mean()

class GymPackman():
    def __init__(self) -> None:
        self._env = gym.make("MsPacman-v0")

        self._state_size = (88, 80, 1)
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
        image[image == color] = 0
        image = (image - 128) / 128 - 1
        image = np.expand_dims(image.reshape(88, 80, 1), axis=0)
        return image


    def reset(self):
        return self._env.reset()

    def render(self):
        return self._env.render()

    def step(self, action):
        next_state, reward, done, _ = self._env.step(action)
        return next_state, reward, done



class DQN():
    def __init__(self, state_size, action_size) -> None:
        self._state_size = state_size
        self._action_size = action_size

        self._replay_buffer = deque(maxlen=5000)
        self.gamma = 0.9
        self.epsilon = 0.8
        self.update_rate = 1000
        
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

        model.compile(loss='mse', optimizer=Adam())

        return model
    
    def store_transistion(self, state, action, reward, next_state, done):
        self._replay_buffer.append((state, action, reward, next_state, done))
    
    def epsilon_greedy(self, state):
        if random.uniform(0, 1) < self.epsilon:
            return np.random.randint(self._action_size)
        Q_value = self.main_net.predict(state)
        return np.argmax(Q_value[0])
    
    def train(self, batch_size):
        minibatch = random.sample(self._replay_buffer, batch_size)
        for state, action, reward, next_state, done in minibatch:
            if not done:
                target_Q = (reward + self.gamma * np.amax(self.target_net.predict(next_state)))
            else:
                target_Q = reward
        
            Q_values = self.main_net.predict(state)
            Q_values[0][action] = target_Q
            self.main_net.fit(state, Q_values, epochs=1, verbose=0)
    
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

    dqn = DQN(state_size, action_size)
    done = False

    time_step = 0
    for i in range(num_epi):
        Return = 0
        state = env.reset()
        state = env.preprocess_state(state)
        for t in range(num_timesteps):
            s = time.time()
            env.render()
            time_step += 1
            if time_step % dqn.update_rate == 0:
                dqn.update_target_net()

            a = dqn.epsilon_greedy(state)
            next_state, reward, done = env.step(a)
            next_state = env.preprocess_state(next_state)

            dqn.store_transistion(state, a, reward, next_state, done)

            state = next_state
            Return += reward
            if done:
                print(f'Episode:{i}, Return: {Return}')
                break
            
            if len(dqn._replay_buffer) > batch_size:
                dqn.train(batch_size)
            
            print(time.time() - s)







if __name__ == "__main__":
    main()
        