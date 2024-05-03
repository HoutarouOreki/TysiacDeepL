import warnings
from copy import deepcopy

import numpy as np
import rl.core
from keras.src.callbacks import History
from rl.callbacks import TrainIntervalLogger, TrainEpisodeLogger, Visualizer, CallbackList


def select_action(agent, observation, mask):
    actions = agent.compute_q_values([observation])
    if mask is None:
        return np.argmax(actions)
    action_mask = [float('-inf') if x == 0 else 0 for x in mask]
    masked_actions = np.add(action_mask, actions)
    if np.count_nonzero(mask) == 0:
        return None
    return np.argmax(masked_actions)


def fit(agents: dict[str, rl.core.Agent], env, nb_steps, action_repetition=1, callbacks=None, verbose=1,
        visualize=False, nb_max_start_steps=0, start_step_policy=None, log_interval=10000,
        nb_max_episode_steps=None):
    """Trains the agent on the given environment.

    # Arguments
        env: (`Env` instance): Environment that the agent interacts with. See [Env](#env) for details.
        nb_steps (integer): Number of training steps to be performed.
        action_repetition (integer): Number of times the agent repeats the same action without
            observing the environment again. Setting this to a value > 1 can be useful
            if a single action only has a very small effect on the environment.
        callbacks (list of `keras.callbacks.Callback` or `rl.callbacks.Callback` instances):
            List of callbacks to apply during training. See [callbacks](/callbacks) for details.
        verbose (integer): 0 for no logging, 1 for interval logging (compare `log_interval`), 2 for episode logging
        visualize (boolean): If `True`, the environment is visualized during training. However,
            this is likely going to slow down training significantly and is thus intended to be
            a debugging instrument.
        nb_max_start_steps (integer): Number of maximum steps that the agent performs at the beginning
            of each episode using `start_step_policy`. Notice that this is an upper limit since
            the exact number of steps to be performed is sampled uniformly from [0, max_start_steps]
            at the beginning of each episode.
        start_step_policy (`lambda observation: action`): The policy
            to follow if `nb_max_start_steps` > 0. If set to `None`, a random action is performed.
        log_interval (integer): If `verbose` = 1, the number of steps that are considered to be an interval.
        nb_max_episode_steps (integer): Number of steps per episode that the agent performs before
            automatically resetting the environment. Set to `None` if each episode should run
            (potentially indefinitely) until the environment signals a terminal state.

    # Returns
        A `keras.callbacks.History` instance that recorded the entire training process.
    """

    first_agent = next(iter(agents.values()))

    for agent in agents.values():
        if not agent.compiled:
            raise RuntimeError(
                'You tried to fit your agent but it hasn\'t been compiled yet. Please call `compile()` before `fit()`.')
        if action_repetition < 1:
            raise ValueError(f'action_repetition must be >= 1, is {action_repetition}')

        agent.training = True

    callbacks = [] if not callbacks else callbacks[:]

    if verbose == 1:
        callbacks += [TrainIntervalLogger(interval=log_interval)]
    elif verbose > 1:
        callbacks += [TrainEpisodeLogger()]
    if visualize:
        callbacks += [Visualizer()]
    history = History()
    callbacks += [history]
    callbacks = CallbackList(callbacks)
    if hasattr(callbacks, 'set_model'):
        callbacks.set_model(first_agent)
    else:
        callbacks._set_model(first_agent)
    callbacks._set_env(env)
    params = {
        'nb_steps': nb_steps,
    }
    if hasattr(callbacks, 'set_params'):
        callbacks.set_params(params)
    else:
        callbacks._set_params(params)

    for agent in agents.values():
        agent._on_train_begin()
    callbacks.on_train_begin()

    episode = np.int16(0)
    for agent in agents.values():
        agent.step = np.int16(0)
    observations = {a: None for a in env.agents}
    episode_rewards = {a: None for a in env.agents}
    episode_steps = {a: None for a in env.agents}
    did_abort = False
    try:
        for agent_id in env.agent_iter():
            agent = agents[agent_id]
            if observations[agent_id] is None:  # start of a new episode
                callbacks.on_episode_begin(episode)
                episode_steps[agent_id] = np.int16(0)
                episode_rewards[agent_id] = np.float32(0)

                # Obtain the initial observation by resetting the environment.
                agent.reset_states()
                observations[agent_id] = env.get_observations_as_number_array(agent_id)
                if agent.processor is not None:
                    observations[agent_id] = agent.processor.process_observation(observations[agent_id])
                assert observations[agent_id] is not None

                # Perform random starts at beginning of episode and do not record them into the experience.
                # This slightly changes the start position between games.
                nb_random_start_steps = 0 if nb_max_start_steps == 0 else np.random.randint(nb_max_start_steps)
                for _ in range(nb_random_start_steps):
                    if start_step_policy is None:
                        action = env.action_space.sample()
                    else:
                        action = start_step_policy(observations[agent_id])
                    if agent.processor is not None:
                        action = agent.processor.process_action(action)
                    callbacks.on_action_begin(action)

                    env.step(action)
                    observations[agent_id], reward, termination, truncation, info = env.last()
                    observations[agent_id] = env.get_observations_as_number_array(agent_id)

                    observations[agent_id] = deepcopy(observations[agent_id])
                    if agent.processor is not None:
                        observations[agent_id], reward, termination, truncation, info = agent.processor.process_step(observations[agent_id], reward, termination, truncation, info)
                    callbacks.on_action_end(action)
                    if truncation or termination:
                        warnings.warn(
                            f'Env ended before {nb_random_start_steps} random steps could be performed at the start. You should probably lower the `nb_max_start_steps` parameter.')
                        observations[agent_id] = deepcopy(env.reset())
                        if agent.processor is not None:
                            observations[agent_id] = agent.processor.process_observation(observations[agent_id])
                        break
                continue

            # At this point, we expect to be fully initialized.
            assert episode_rewards[agent_id] is not None
            assert episode_steps[agent_id] is not None
            assert observations[agent_id] is not None

            # Run a single step.
            callbacks.on_step_begin(episode_steps[agent_id])
            # This is where all the work happens. We first perceive and compute the action
            # (forward step) and then use the reward to improve (backward step).
            mask = env.compute_action_mask(agent_id)
            action = select_action(agent, observations[agent_id], mask)
            if agent.processor is not None:
                action = agent.processor.process_action(action)
            reward = np.float32(0)
            accumulated_info = {}
            done = False
            for _ in range(action_repetition):
                callbacks.on_action_begin(action)
                env.step(action)
                observations[agent_id], r, termination, truncation, info = env.last()
                observations[agent_id] = env.get_observations_as_number_array(agent_id)
                observations[agent_id] = deepcopy(observations[agent_id])
                if agent.processor is not None:
                    observations[agent_id], r, termination, truncation, info = agent.processor.process_step(observations[agent_id], r, termination, truncation, info)
                for key, value in info.items():
                    if not np.isrealobj(value):
                        continue
                    if key not in accumulated_info:
                        accumulated_info[key] = np.zeros_like(value)
                    accumulated_info[key] += value
                callbacks.on_action_end(action)
                reward += r
                if done:
                    break
            if nb_max_episode_steps and episode_steps[agent_id] >= nb_max_episode_steps - 1:
                # Force a terminal state.
                done = True
            # metrics = agent.backward(reward, terminal=done)
            episode_rewards[agent_id] += reward

            step_logs = {
                'action': action,
                'observation': observations[agent_id],
                'reward': reward,
                # 'metrics': metrics,
                'episode': episode,
                'info': accumulated_info,
            }
            callbacks.on_step_end(episode_steps[agent_id], step_logs)
            episode_steps[agent_id] += 1
            agent.step += 1

            if done:
                # We are in a terminal state but the agent hasn't yet seen it. We therefore
                # perform one more forward-backward call and simply ignore the action before
                # resetting the environment. We need to pass in `terminal=False` here since
                # the *next* state, that is the state of the newly reset environment, is
                # always non-terminal by convention.
                agent.forward(observations[agent_id])
                agent.backward(0., terminal=False)

                # This episode is finished, report and reset.
                episode_logs = {
                    'episode_reward': episode_rewards[agent_id],
                    'nb_episode_steps': episode_steps[agent_id],
                    'nb_steps': agent.step,
                }
                callbacks.on_episode_end(episode, episode_logs)

                episode += 1
                observations[agent_id] = None
                episode_steps[agent_id] = None
                episode_rewards[agent_id] = None
    except KeyboardInterrupt:
        # We catch keyboard interrupts here so that training can be be safely aborted.
        # This is so common that we've built this right into this function, which ensures that
        # the `on_train_end` method is properly called.
        did_abort = True
    callbacks.on_train_end(logs={'did_abort': did_abort})
    for agent in agents.values():
        agent._on_train_end()

    return history