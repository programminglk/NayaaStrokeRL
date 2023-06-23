import gymnasium as gym
from gymnasium import spaces
import numpy as np
from gym_game.envs.pygame_2d import PyGame2D

class CustomEnv(gym.Env):
    #metadata = {'render.modes' : ['human']}
    def __init__(self):
        self.pygame = PyGame2D()
        # self.action_space = spaces.Discrete(3)

        # actions =
            # 0. move left
            # 1. move right
            # 2. move up
            # 3. move down

        self.action_space = spaces.Discrete(4)
        
        # observations = 
            # 1. within snake body (0-yes, 1-no)
            # 2. y distance from start
            # 3. x distance to goal
        self.observation_space = spaces.MultiDiscrete([2, 800, 1500])


    def reset(self, seed=None, options=None, render=False):
        del self.pygame
        self.pygame = PyGame2D()
        obs = self.pygame.observe()
        return obs, {}

    def step(self, action):
        self.pygame.action(action)
        obs = self.pygame.observe()
        reward = self.pygame.evaluate()
        done = self.pygame.is_done()
        print("step done ---- see below for rewards and obs")
        return obs, reward, done, False, {}

    def render(self, mode="human", close=False):
        self.pygame.view()
