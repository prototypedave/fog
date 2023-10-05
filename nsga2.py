"""

"""
import argparse
import logging
import os
import random
import time
from pathlib import Path

import gym
import numpy as np
import torch.cuda
from matplotlib import pyplot as plt
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.problem import Problem
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.optimize import minimize
from tensorboardX import SummaryWriter
from tqdm import tqdm

from agent_dqn import DRLModel
from env import FogEnvDiscrete
from rnd import IntegerRandomSampling
from rounding import RoundingRepair

plt.style.use("ggplot")
logger = logging.getLogger(__name__)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

SEED = 7
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

N_EPISODES = 100
NUMBER_ITERATIONS = 1

PLOT_ALL = False


class MultiObjectiveOptimizationProblem(Problem):
    def __init__(
            self,
            load_action_values: np.array,
            distance_action_values: np.array,
            priority_action_values: np.array,
            number_of_nodes: int
    ):
        super().__init__(n_var=1, n_obj=3, n_ieq_constr=0, xl=0, xu=number_of_nodes - 1)
        self.load_action_values = load_action_values
        self.distance_action_values = distance_action_values
        self.priority_action_values = priority_action_values

    def _evaluate(self, x, out, *args, **kwargs):
        f1 = -self.load_action_values[x]
        f2 = -self.distance_action_values[x]
        f3 = -self.priority_action_values[x]

        out["F"] = [[f1, f2, f3]]


def min_max_rescale(array: np.array):
    return (array - array.min()) / (array.max() - array.min())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--number-of-nodes", default=5, type=int, required=False)
    parser.add_argument("--max-timesteps", default=100, type=int, required=False)
    parser.add_argument("--logging", default="INFO", type=str, required=False)
    parser.add_argument("--save-dir", default="~/results_last", type=str, required=False)
    parser.add_argument("--iteration", default=0, type=int, required=False)
    parser.add_argument("--plot-all", default=0, type=int, required=False)

    args = parser.parse_args()

    NUMBER_OF_NODES = args.number_of_nodes
    MAX_STEPS = args.max_timesteps
    PLOT_ALL = bool(args.plot_all)

    logging.basicConfig(
        level=eval(f"logging.{args.logging}"),
        format="%(asctime)s %(levelname)-8s %(message)s",
        datefmt="%d-%b-%y %H:%M:%S"
    )
    logger.info(f"start testing")

    # tensorboard logger
    tensorboard_writer = SummaryWriter(str(Path(args.save_dir, "nsga2")))

    # Environment
    env = FogEnvDiscrete(n_nodes=NUMBER_OF_NODES, max_steps=MAX_STEPS)

    # CREATE NETWORK
    # DRL 1 (load)
    obs_size_load = env.observation_space_load["node_load"].shape[0]
    n_load_actions = env.action_space.n
    logger.info(f"DRL1(LOAD) observation space shape: {env.observation_space_load.shape}")
    logger.info(f"n_actions: {n_load_actions}")

    # DRL 2 (distance)
    obs_size_distance = env.observation_space_distance.shape
    n_distance_actions = env.action_space.n
    logger.info(f"DRL2(DISTANCE) observation space shape: {env.observation_space_distance.shape}")
    logger.info(f"n_actions: {n_distance_actions}")

    # DRL 3 (priority)
    obs_size_priority = env.observation_space_priority.shape
    n_priority_actions = env.action_space.n
    logger.info(f"DRL3(PRIORITY) observation space shape: {env.observation_space_load.shape}")
    logger.info(f"n_actions: {n_load_actions}")

    # DRL FUNCTIONS
    # load
    DRL_load = DRLModel(input_size=obs_size_load, output_size=n_load_actions)
    DRL_load.load_state_dict(
        torch.load(Path(args.save_dir, 'load', f'saved_DRL_itr-{args.iteration + 1}_reward-load.pt'))
    )
    DRL_load.to(device).eval()

    # distance
    DRL_distance = DRLModel(input_size=obs_size_distance, output_size=n_distance_actions)
    DRL_distance.load_state_dict(
        torch.load(Path(args.save_dir, 'distance', f'saved_DRL_itr-{args.iteration + 1}_reward-load.pt'))
    )
    DRL_distance.to(device).eval()

    # priority
    DRL_priority = DRLModel(input_size=obs_size_priority, output_size=n_priority_actions)
    DRL_distance.load_state_dict(
        torch.load(Path(args.save_dir, 'priority', f'saved_DRL_itr-{args.iteration + 1}_reward-load.pt'))
    )
    DRL_priority.to(device).eval()

    total_time = 0
    total_ep = 0

    for _ in tqdm(range(NUMBER_ITERATIONS), desc="Learning iterations"):
        # Algorithm
        state = env.reset()

        state_load = state[0]
        state_distance = state[1]
        state_priority = state[2]

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
        with open(Path(args.save_dir, "nsga2", "statistics.txt"), 'w') as f:
            f.write(f"task_number,time(sec),load,distance,priority\n")
        while ep < N_EPISODES:
            # sample all best actions
            # LOAD
            tensor_state_load = torch.FloatTensor(state_load).unsqueeze(0).to(device)
            action_load = DRL_load.forward(tensor_state_load).detach().cpu().numpy().flatten()
            # min-max normalize to [0,1]
            action_values_load = min_max_rescale(array=action_load)

            # DISTANCE
            tensor_state_distance = torch.FloatTensor(state_distance).unsqueeze(0).to(device)
            action_distance = DRL_distance.forward(tensor_state_distance).detach().cpu().numpy().flatten()
            # min-max normalize to [0,1]
            action_values_distance = min_max_rescale(array=action_distance)

            # Priority
            tensor_state_priority = torch.FloatTensor(state_priority).unsqueeze(0).to(device)
            action_priority = DRL_priority.forward(tensor_state_priority).detach().cpu().numpy().flatten()
            # min-max normalize to [0,1]
            action_values_priority = min_max_rescale(array=action_priority)

            # Log best action with respect to the three objectives
            if PLOT_ALL:
                for mode in ['load', 'distance', 'priority']:
                    tensorboard_writer.add_scalar(
                        tag=f"Action/action_{mode}",
                        scalar_value=int(np.argmax(eval(f"action_values_{mode}"))),
                        global_step=total_time,
                    )

            MODRL_problem = MultiObjectiveOptimizationProblem(
                load_action_values=action_values_load,
                distance_action_values=action_values_distance,
                priority_action_values=action_values_priority,
                number_of_nodes=NUMBER_OF_NODES
            )

            algorithm = NSGA2(
                pop_size=20,
                sampling=IntegerRandomSampling(),
                crossover=SBX(prob=1.0, eta=3.0, vtype=float, repair=RoundingRepair()),
                mutation=PM(prob=1.0, eta=3.0, vtype=float, repair=RoundingRepair()),
                eliminate_duplicates=True,
            )

            res = minimize(
                MODRL_problem,
                algorithm,
                termination=('n_gen', 10),
                seed=SEED,
                save_history=False
            )

            logger.debug(f"Best solution found: {res.X}")
            logger.debug(f"Function value: {res.F}")

            action = np.random.choice(res.X.flatten())

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

            for idx, key in enumerate(['reward_load', 'reward_distance', 'reward_priority']):
                tensorboard_writer.add_scalar(
                    tag=f"Reward/{key}",
                    scalar_value=reward[idx],
                    global_step=total_time,
                )
                episode_reward_dict[key] += reward[idx]

            if ep == 0 and ep_steps in [1, 20, 60, 200, 600, 999]:
                with open(Path(args.save_dir, "nsga2", "statistics.txt"), 'a') as f:
                    f.write(f"{ep_steps},{time.time() - start_time},{episode_reward_dict['reward_load']},"
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



