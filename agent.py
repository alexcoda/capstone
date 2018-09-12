"""Code for the DQN agent."""
import torch.nn.functional as F
import torch.optim as optim
import random
import torch
import math

from collections import namedtuple


# Local imports
from model import DQN


Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class DQNAgent:

    def __init__(self, config):
        self.config = config
        self.env = config['env']
        self.device = config['device']
        self.EPS_END = config['EPS_END']
        self.EPS_START = config['EPS_START']
        self.EPS_DECAY = config['EPS_DECAY']
        self.BATCH_SIZE = config['BATCH_SIZE']
        self.GAMMA = config['GAMMA']

        # Set up our networks
        self.policy_net = DQN(config).to(self.device)
        self.target_net = DQN(config).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.RMSprop(self.policy_net.parameters())
        self.memory = ReplayMemory(10000)

        self.steps_done = 0
        # Add config params
        # Add method for selecting actions
        # Add method for optiomizing the model

    def select_action(self, state):
        """Select an action given the state according to the current policy."""
        sample = random.random()
        eps_threshold = self.EPS_END + (self.EPS_START - self.EPS_END) * \
            math.exp(-1. * self.steps_done / self.EPS_DECAY)
        self.steps_done += 1
        if sample > eps_threshold:
            with torch.no_grad():
                return self.policy_net(state).max(1)[1].view(1, 1)
        else:
            return torch.tensor([[random.randrange(2)]],
                                device=self.device, dtype=torch.long)

    def step(self):
        """Take a single step in optimizing the model."""
        if len(self.memory) < self.BATCH_SIZE:
            return  # Memory has too few instances
        transitions = self.memory.sample(self.BATCH_SIZE)
        # Transpose the batch
        batch = Transition(*zip(*transitions))

        # Compute a mask of non-final states and concatenate the batch elements
        non_final_mask = torch.tensor(
            tuple(map(lambda s: s is not None,
                  batch.next_state)), device=self.device, dtype=torch.uint8)
        non_final_next_states = torch.cat(
            [s for s in batch.next_state if s is not None])
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        # Compute V(s_{t+1}) for all next states.
        next_state_values = torch.zeros(self.BATCH_SIZE, device=self.device)
        next_state_values[non_final_mask] = \
            self.target_net(non_final_next_states).max(1)[0].detach()
        # Compute the expected Q values
        expected_state_action_values = \
            (next_state_values * self.GAMMA) + reward_batch

        # Compute Huber loss
        loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()
