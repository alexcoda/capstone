"""Main script for running RL experiments."""
import matplotlib.pyplot as plt


# Local imports
from vis import show_example_screen
from agent import DQNAgent
from config import config
from train import train


def main():
    env = config['env']
    device = config['device']
    plt.ion()
    env.reset()

    # Show an example screen
    print('Showing example screen.')
    show_example_screen(env, device)

    # Setup the agent
    print('Setting up the agent.')
    agent = DQNAgent(config)

    # Run the training loop
    print('Training.')
    train(agent, config)


if __name__ == "__main__":
    main()
    print('Done!')
    config['env'].close()
