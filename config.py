"""Parameters for running DQN experiments."""
from dataclasses import dataclass
import torch
import gym


@dataclass(repr=False)
class Params:
    """Parameters for running the RL system.

    n_episodes: The number of episodes to train for.
    action_frame_freq: How often the agent will select an action. If >1, then
        the agent will only select actions on the nth frame, and chosen actions
        will be repeated on intervening frames.
    target_update_freq: How often to update the weights of our target network.
    batch_size: The batch size of events to sample from the memory replay for
        each update.
    gamma: The weighting factor to apply towards expected future rewards.
    eps_start: The starting epsilon value to use for sampling actions using an
        epsilon greedy strategy.
    eps_end: The final epsilon value to anneal towards.
    eps_decay: The period (in episodes) over which to anneal the epsilon to the
        end value.
    env_name: The name of the gym environment to train on.
    """
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
