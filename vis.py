"""Code for visualizing the model while running."""
import torchvision.transforms as T
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
import preprocess

resize = T.Compose([T.ToPILImage(),
                    T.Resize(40, interpolation=Image.CUBIC),
                    T.ToTensor()])


def show_example_screen(config):
    plt.figure()
    plt.imshow(get_screen(config).cpu().squeeze(0).permute(1, 2, 0).numpy(),
               interpolation='none')
    plt.title('Example extracted screen')
    plt.show()


def get_screen(config):
    env = config['env']
    device = config['device']
    env_name = config['env_name']
    # Strip off the top and bottom of the screen
    if env_name=='Pong-v0':
        screen = env.render(mode='rgb_array')
        screen = preprocess.prepro_pong(screen)
    else:
        screen = env.render(mode='rgb_array').transpose(
        (2, 0, 1))  # transpose into torch order (CHW)
        screen = preprocess.prepro_cartpole(screen, env)
    # screen = torch.from_numpy(screen)
    print(type(screen), resize(screen).shape)
    # Resize, and add a batch dimension (BCHW)
    screen = resize(screen).unsqueeze(0).to(device)
    return screen


def plot_durations(episode_durations):
    plt.figure(2)
    plt.clf()
    durations_t = torch.tensor(episode_durations, dtype=torch.float)
    plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(durations_t.numpy())
    # Take 100 episode averages and plot them too
    if len(durations_t) >= 100:
        means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())

    plt.pause(0.001)  # pause a bit so that plots are updated
