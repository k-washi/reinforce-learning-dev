import sys
import pathlib

import random
import gym
import time
import numpy as np

from collections import deque # スレッドセーフなキュー

# utilsのパスを通す
module_path = pathlib.Path(__file__, "..", "..").resolve()
if module_path not in sys.path:
    sys.path.append(str(module_path))

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