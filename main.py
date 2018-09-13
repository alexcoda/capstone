"""Main script for running RL experiments."""
import matplotlib.pyplot as plt


# Local imports
from vis import show_example_screen
from utils import get_logger
from agent import DQNAgent
from config import params
from train import train


def main():
    logger = get_logger(__name__, 4)

    env = params.env
    device = params.device
    env.reset()

    # Show an example screen
    # print('Showing example screen.')
    # show_example_screen(env, device)

    # Setup the agent
    print('Setting up the agent.')
    agent = DQNAgent(params)

    # Run the training loop
    print('Training.')
    train(agent, params)


if __name__ == "__main__":
    main()
    print('Done!')
    params.env.close()
