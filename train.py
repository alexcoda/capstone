"""Code for training a DQN agent"""
import matplotlib.pyplot as plt
import torch

from itertools import count

# Local imports
from vis import get_screen, plot_durations


def train(agent, params):
    # Unpack the params we plan to use
    env = params.env
    device = params.device
    n_episodes = params.n_episodes
    target_update_freq = params.target_update_freq
    action_frame_freq = params.action_frame_freq

    episode_durations = []

    for i_episode in range(n_episodes):
        print(f"Episode {i_episode}")
        # Initialize the environment and state
        env.reset()
        last_screen = get_screen(env, device)
        current_screen = get_screen(env, device)
        state = current_screen - last_screen

        # Execute the current episode
        for t in count():
            # Select the next action
            if t % action_frame_freq == 0:
                action = agent.select_action(state)
                print(f"new action: {action}")
            # Perform the action
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
                print(f"  Episode Duration: {t + 1}")
                episode_durations.append(t + 1)
                if len(episode_durations) % 10 == 0:
                    print(f"  Last 10 Avg: {sum(episode_durations[-10:]) / 10}")
                    print(f"  Memory Size: {len(agent.memory)}")
                    plot_durations(episode_durations)
                break
        # Update the target network
        if i_episode % target_update_freq == 0:
            agent.target_net.load_state_dict(agent.policy_net.state_dict())

    print('Complete')
    env.render()
    env.close()
    # plt.ioff()
