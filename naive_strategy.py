import argparse
import logging
import time

import torch

import numpy as np
import random

import matplotlib.pyplot as plt
from tqdm import tqdm
from pathlib import Path
from tensorboardX import SummaryWriter

from env import FogEnvDiscrete

plt.style.use('ggplot')
logger = logging.getLogger(__name__)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = "cpu"

# reproducibility
SEED = 7
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

##############################################################################
# Agent Hyperparameters
##############################################################################
# Number of training episodes
N_EPISODES = 50
# Number of iterations (where we keep only the replay buffer)
NUMBER_ITERATIONS = 1
##############################################################################

##############################################################################
# System Hyperparameters
##############################################################################
# NUMBER_OF_NODES = 2
# REWARD_MODE = 'load'
# MAX_TIMESTEPS = 20
PLOT_ALL = False
##############################################################################


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--number-of-nodes", default=5, type=int, required=False)
    parser.add_argument("--max-timesteps", default=100, type=int, required=False)
    parser.add_argument("--logging", default="INFO", type=str, required=False)
    parser.add_argument("--save-dir", default="results", type=str, required=False)
    parser.add_argument("--plot-all", default=0, type=int, required=False)

    args = parser.parse_args()

    NUMBER_OF_NODES = args.number_of_nodes
    MAX_TIMESTEPS = args.max_timesteps
    PLOT_ALL = bool(args.plot_all)

    logging.basicConfig(
        level=eval(f"logging.{args.logging}"),
        format="%(asctime)s %(levelname)-8s %(message)s",
        datefmt="%d-%b-%y %H:%M:%S"
    )
    logger.info(f"Start testing")

    # tensorboard logger
    tensorboard_writer = SummaryWriter(str(Path(args.save_dir, "naive")))

    # ================= Environment =================
    env = FogEnvDiscrete(
        n_nodes=NUMBER_OF_NODES,
        max_steps=MAX_TIMESTEPS
    )
    # ========================================================

    # create network and target network
    obs_load = env.observation_space_load["node_load"].shape[0]
    n_load_actions = env.action_space.n
    logger.info(f"DRL1(LOAD) observation space shape: {obs_load}")
    logger.info(f"n_actions: {n_load_actions}")

    # DRL 2 (distance)
    obs_distance = env.observation_space_distance["node_distance"].shape[0]
    n_distance_actions = env.action_space.n
    logger.info(f"DRL2(DISTANCE) observation space shape: {obs_distance}")
    logger.info(f"n_actions: {n_distance_actions}")

    # DRL 3 (priority)
    obs_priority = env.observation_space_priority["task_priority"].shape[0]
    n_priority_actions = env.action_space.n
    logger.info(f"DRL3(PRIORITY) observation space shape: {obs_priority}")
    logger.info(f"n_actions: {n_load_actions}")

    total_time = 0
    total_ep = 0

    for _ in tqdm(range(NUMBER_ITERATIONS), desc="Learning iterations"):
        # Algorithm
        state = env.reset()
        ep = 0
        episode_reward = 0
        episode_reward_dict = {
            'reward_load': 0,
            'reward_distance': 0,
            'reward_priority': 0
        }
        ep_steps = 0

        logger.debug(f" ## Start episode")

        episodes_bar = tqdm(total=N_EPISODES, desc="Episodes")
        start_time = time.time()
        with open(Path(args.save_dir, "naive", "statistics.txt"), 'w') as f:
            f.write(f"task_number,time(sec),load,distance,priority\n")
        while ep < N_EPISODES:
            # ====================================================
            load_state = state[0]
            load_list = load_state[:NUMBER_OF_NODES]

            # We select the least loaded node, if two nodes have the same minimal load, we select the closest (first)
            action = int(np.argmin(load_list))

            # Send to cloud if all the nodes are fully occupied
            cloud_threshold = 0.95
            if np.all(load_state[:NUMBER_OF_NODES - 1] > cloud_threshold):
                action = -1
            # ====================================================

            logger.debug(f"selected action: {action}")
            if PLOT_ALL:
                tensorboard_writer.add_scalar(
                    tag="Action/action",
                    scalar_value=action,
                    global_step=total_time,
                )

            load_state, distance_state, priority_state, reward, done = env.step(action)
            episode_reward += reward[0]
            total_time += 1
            ep_steps += 1

            # We extract reward functions from the info dict
            for key, value in enumerate(["reward_load", "reward_distance", "reward_priority"]):
                tensorboard_writer.add_scalar(
                    tag=f"Reward/{value}",
                    scalar_value=reward[key],
                    global_step=total_time,
                )
                episode_reward_dict[value] += reward[key]

            if ep == 0 and ep_steps in [1, 20, 60, 200, 600, 999]:
                with open(Path(args.save_dir, "naive", "statistics.txt"), 'a') as f:
                    f.write(f"{ep_steps},{time.time()-start_time},{episode_reward_dict['reward_load']},"
                            f"{episode_reward_dict['reward_distance']},{episode_reward_dict['reward_priority']}\n")

            # log metrics from the system
            env.system.log(tensorboard_writer=tensorboard_writer, global_step=total_time, plot=PLOT_ALL)

            logger.debug(f"next_state: [load_State, distance_state, priority_state]")
            logger.debug(f"next_state: {load_state}, {distance_state}, {priority_state}")
            logger.debug(f"reward: {reward}")
            logger.debug(f"done: {done}")

            state = [load_state, distance_state, priority_state]
            if done:
                state = env.reset()
                # logger.info(f"We finished episode {ep} with reward {episode_reward}")
                for key, value in episode_reward_dict.items():
                    tensorboard_writer.add_scalar(
                        tag=f"Episode/Return_{key.split('_')[1]}",
                        scalar_value=value,
                        global_step=total_ep,
                    )
                ep += 1
                total_ep += 1
                episodes_bar.update(n=1)
                episode_reward = 0
                episode_reward_dict = {
                    'reward_load': 0,
                    'reward_distance': 0,
                    'reward_priority': 0
                }
                ep_steps = 0

        # Save current run's results
        episodes_bar.close()
