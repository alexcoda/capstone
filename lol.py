import gym
env = gym.make('Pong-v0')
print(env.reset())

for i in range(1000):
    env.step(env.action_space.sample())
    env.render()