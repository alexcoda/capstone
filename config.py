"""Parameters for running DQN experiments."""
import torch
import gym

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
env_name = 'Pong-v0'
# env_name = 'CartPole-v0'
env = gym.make(env_name).unwrapped

N_EPISODES = 50
BATCH_SIZE = 128
GAMMA = 0.999
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 200
TARGET_UPDATE = 10

N_ACTIONS = 6 # 6 for Atari, 2 for Cartpole
N_LINEAR = 128 #128 for atari, 448 for Cartpole

config = {'device': device,
          'env': env,
          'env_name': env_name,
          'N_EPISODES': N_EPISODES,
          'BATCH_SIZE': BATCH_SIZE,
          'GAMMA': GAMMA,
          'EPS_START': EPS_START,
          'EPS_END': EPS_END,
          'EPS_DECAY': EPS_DECAY,
          'TARGET_UPDATE': TARGET_UPDATE,
          'N_ACTIONS': N_ACTIONS,
          'N_LINEAR': N_LINEAR}
