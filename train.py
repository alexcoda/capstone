"""Code for training a DQN agent"""
import matplotlib.pyplot as plt
import torch

from itertools import count

# Local imports
from vis import get_screen, plot_durations


def train(agent, config):
    env = config['env']
    device = config['device']
    N_EPISODES = config['N_EPISODES']
    TARGET_UPDATE = config['TARGET_UPDATE']

    episode_durations = []

    for i_episode in range(N_EPISODES):
        print(i_episode)
        # Initialize the environment and state
        env.reset()
        last_screen = get_screen(env, device)
        current_screen = get_screen(env, device)
        state = current_screen - last_screen
        for t in count():
            # Select and perform an action
            action = agent.select_action(state)
            _, reward, done, _ = env.step(action.item())
            reward = torch.tensor([reward], device=device)

            # Observe new state
            last_screen = current_screen
            current_screen = get_screen(env, device)
            if not done:
                next_state = current_screen - last_screen
            else:
                next_state = None

            # Store the transition in memory
            agent.memory.push(state, action, next_state, reward)

            # Move to the next state
            state = next_state

            # Perform one step of the optimization (on the target network)
            agent.step()
            if done:
                episode_durations.append(t + 1)
                plot_durations(episode_durations)
                break
        # Update the target network
        if i_episode % TARGET_UPDATE == 0:
            agent.target_net.load_state_dict(agent.policy_net.state_dict())

    print('Complete')
    env.render()
    env.close()
    plt.ioff()
    plt.show()
