"""
   Agent
"""

import argparse
import logging
import numpy as np
from tqdm import tqdm

from env import FogEnvDiscrete

logger = logging.getLogger(__name__)

Action = np.array
State = np.array

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--number-of-nodes", default=2, type=int, required=False)
    parser.add_argument("--number-of-devices", default=2, type=int, required=False)
    parser.add_argument("--max-timesteps", default=1000, type=int, required=False)

    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s %(levelname)-8s %(message)s",
        datefmt="%d-%b-%y %H:%M:%S"
    )

    logger.info(f"Start testing")

    args = parser.parse_args()

    number_of_nodes = args.number_of_nodes
    number_of_devices = args.number_of_devices
    max_timesteps = args.max_timesteps

    # ================= Environment =================
    env = FogEnvDiscrete(
        n_nodes=number_of_nodes,
        max_steps=max_timesteps
    )
    # ========================================================

    # ================= Hyper-parameters =================
    # Number of episode we will run
    n_episodes = 100

    # maximum of iteration per episode
    max_iter_episode = 100

    # initialize the exploration probability to 1
    exploration_proba = 1

    # exploration decreasing decay for exponential decreasing
    exploration_decreasing_decay = 0.00001

    # minimum of exploration probability
    min_exploration_proba = 0.01

    # discounted factor
    gamma = 0.999

    # learning rate
    lr = 0.1
    # ===================================================

    # ================= Initialization =================
    # Number of observations
    obs_space_load = env.observation_space_load
    n_observations_load = int(
        np.prod(obs_space_load['node_load'].shape[0])
    )
    logger.info(f"n_observations: {n_observations_load}")

    obs_space_distance = env.observation_space_distance
    n_observations_distance = int(
        np.prod(obs_space_distance['node_distance'].shape[0])
    )
    logger.info(f"n_observations: {n_observations_distance}")

    obs_space_priority = env.observation_space_priority
    n_observations_priority = int(
        np.prod(obs_space_priority['task_priority'].shape[0])
    )
    logger.info(f"n_observations: {n_observations_priority}")

    # Number of actions
    n_actions = int(env.action_space.n)
    logger.info(f"n_actions: {n_actions}")

    # Initialize the tables to 0
    load_table = np.zeros((n_observations_load, n_actions))
    distance_table = np.zeros((n_observations_distance, n_actions))
    priority_table = np.zeros((n_observations_priority, n_actions))

    rewards_per_episode = list()
    # ===================================================

    # ================= Main loop =================
    # we iterate over episodes
    for e in tqdm(range(n_episodes), desc="N° episodes"):
        # we initialize the first state of the episode
        current_state = env.reset()
        done = False

        # sum the rewards that the agent gets from the environment
        total_episode_reward = 0

        for i in tqdm(range(max_iter_episode), desc="N° iterations"):
            # we sample a float from a uniform distribution over 0 and 1
            # if the sampled float is less than the exploration proba
            #     the agent selects a random action
            # else
            #     he exploits his knowledge using the bellman equation

            if np.random.uniform(0, 1) < exploration_proba:
                action = env.action_space.sample()
            else:
                action = np.argmax(load_table[current_state, :])

            logger.debug(f"selected action: {action}")

            # The environment runs the chosen action and returns
            # the next state, a reward and true if the episode is ended.
            load_state, distance_state, priority_state, reward, done = env.step(action)

            load_state_index = (load_state[0]-1)*obs_space_load['node_load'].shape[0] + (load_state[1]-1)
            distance_state_index = (distance_state[0]-1)*obs_space_distance['node_distance'].shape[0] + (distance_state[1]-1)
            priority_state_index = (priority_state[0]-1)*obs_space_priority['task_priority'].shape[0] + (priority_state[1]-1)

            logger.debug(f"next_state: {load_state}. {distance_state}, {priority_state}")
            logger.debug(f"reward: {reward[0]}. {reward[1]}, {reward[2]}")
            logger.debug(f"done: {done}")

            # We update our tables using the Q-learning iteration
            # logger.debug(f"load_table before: {load_table[load_state, action]}")
            # load_table[current_state[0], action] = (1 - lr) * load_table[current_state[0], action] + lr * (
            #            reward[0] + gamma * max(load_table[load_state_index, :]))
            # logger.debug(f"load_table after: {load_table[current_state[0], action]}")

            total_episode_reward += reward[0]
            # If the episode is finished, we leave the for loop
            if done:
                env.system.reset()
                break
            current_state = [load_state, distance_state, priority_state]

        # logging
        logger.info(f"Episode has terminated after: {i} steps")
        logger.info(f"Episode has total reward: {total_episode_reward}")

        # We update the exploration proba using exponential decay formula
        exploration_proba = max(min_exploration_proba, np.exp(-exploration_decreasing_decay * e))
        rewards_per_episode.append(total_episode_reward)
    # ==============================================

    # ================= Evaluation =================
    logger.info("Mean reward per thousand episodes")
    for i in range(10):
        logger.info(f"{(i+1)*1000},mean episodic reward: ,{np.mean(rewards_per_episode[1000*i:1000*(i+1)])}")
    # ==============================================

