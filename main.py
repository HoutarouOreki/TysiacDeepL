from GameEnv.env.tysiac_game_env import TysiacGameEnv


def start():
    env = TysiacGameEnv()
    env.reset()

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
        print(observation)

    env.close()


if __name__ == '__main__':
    start()
