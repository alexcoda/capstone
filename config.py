"""Parameters for running DQN experiments."""
from dataclasses import dataclass
import torch
import gym


@dataclass(repr=False)
class Params:
    n_episodes: int
    action_frame_freq: int
    target_update_freq: int
    batch_size: int
    gamma: float
    eps_start: float
    eps_end: float
    eps_decay: int
    env_name: str

    def __post_init__(self):
        # Setup for cuda and gym environment
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.env = gym.make(self.env_name).unwrapped

params = Params(
    n_episodes=1000,
    action_frame_freq=1,
    target_update_freq=1,
    batch_size=128,
    gamma=0.999,
    eps_start=0.9,
    eps_end=0.05,
    eps_decay=200,
    env_name='CartPole-v0')
