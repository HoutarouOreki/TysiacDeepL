import gymnasium
import random

def start(name):
    env = gymnasium.make('CartPole-v0')
    states = env.observation_space.shape[0]
    actions = n.action_space.n

if __name__ == '__main__':
    start
