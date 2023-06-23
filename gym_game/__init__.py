import gymnasium as gym
from gymnasium import register

register(
    id='NayaaStroke-v0',
    entry_point='gym_game.envs:CustomEnv',
    max_episode_steps=2000,
)

