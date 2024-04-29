import numpy as np
from keras import Sequential
from keras.src.layers import Flatten, Dense
from keras.src.optimizers import Adam
from rl.agents import DQNAgent
from rl.memory import SequentialMemory
from rl.policy import BoltzmannQPolicy

from GameEnv.env.tysiac_game_env import TysiacGameEnv


def start():
    env = TysiacGameEnv(render_mode="h")
    env.reset()
    env.render()

    for agent in env.agent_iter():
        observation, reward, termination, truncation, info = env.last()

        if termination or truncation:
            action = None
        else:
            if "action_mask" in info:
                mask = info["action_mask"]
            else:
                mask = None
            action = env.action_space(agent).sample(mask)  # this is where you would insert your policy

        observation, reward, termination, truncation, info = env.step(action)
        if "turn_completed" in info and info["turn_completed"]:
            env.render()

    env.close()


def build_model(states, actions):
    model = Sequential()
    model.add(Flatten(input_shape=(1, states)))
    model.add(Dense(24, activation="relu"))
    model.add(Dense(24, activation="relu"))
    model.add(Dense(actions, activation="linear"))
    return model


def build_agent(model, actions):
    policy = BoltzmannQPolicy()
    memory = SequentialMemory(limit=50000, window_length=1)
    return DQNAgent(model=model, memory=memory, policy=policy,
                    nb_actions=actions, nb_steps_warmup=10, target_model_update=1e-2)


def train():
    env = TysiacGameEnv(render_mode="h")
    env.reset()

    states = 2
    actions = 7
    print(actions)
    print(actions)
    model = build_model(states, actions)
    agents = {}
    for agent_id in env.possible_agents:
        agent = build_agent(model, actions)
        agent.compile(Adam(lr=1e-3), metrics=["mae"])
        agents[agent_id] = agent

    env.render()

    for agent in env.agent_iter():
        observation, reward, termination, truncation, info = env.last()

        if termination or truncation:
            action = None
        else:
            if "action_mask" in info:
                mask = info["action_mask"]
            else:
                mask = None
            print(env.observations[agent])
            action = agents[agent].forward(env.observations[agent])
            print(agent, action)
            # action = env.action_space(agent).sample(mask)  # this is where you would insert your policy

        observation, reward, termination, truncation, info = env.step(action)
        if "turn_completed" in info and info["turn_completed"]:
            env.render()

    env.close()


if __name__ == '__main__':
    train()
