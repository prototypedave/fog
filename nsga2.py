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
#torch.manual_seed(SEED)
#np.random.seed(SEED)
#random.seed(SEED)

N_EPISODES = 100
NUMBER_ITERATIONS = 5

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
        if x.ndim == 2:
            x = x[:, 0]  # Take the first column if X has multiple columns

        num_solutions = x.shape[0]  # Number of solutions to evaluate
        objectives = np.zeros((num_solutions, self.n_obj))  # Initialize array for objectives

        for i in range(num_solutions):
            x_int = int(x[i])  # Convert each element of X to integer
            f1 = -self.load_action_values[x_int]
            f2 = -self.distance_action_values[x_int]
            f3 = -self.priority_action_values[x_int]
            objectives[i, :] = [f1, f2, f3]  # Store objectives for this solution

        out["F"] = objectives


def min_max_rescale(array: np.array):
    return (array - array.min()) / (array.max() - array.min())


def check_pt_model(model_path, DRL):
    # Check if the model file exists
    if model_path.exists():
        # If the model file exists, load the model
        DRL.load_state_dict(torch.load(model_path))
    else:
        # If the model file does not exist, create a new model and save it
        torch.save(DRL.state_dict(), model_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--number-of-nodes", default=250, type=int, required=False)
    parser.add_argument("--max-timesteps", default=10, type=int, required=False)
    parser.add_argument("--logging", default="INFO", type=str, required=False)
    parser.add_argument("--save-dir", default="results", type=str, required=False)
    parser.add_argument("--iteration", default=100, type=int, required=False)
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

    # Retrieve the observation sizes for each objective
    obs_size_load = NUMBER_OF_NODES
    obs_size_distance = NUMBER_OF_NODES
    obs_size_priority = NUMBER_OF_NODES

    actions_size = env.action_space.n
    logger.info(f"DRL1(LOAD) observation space shape: {obs_size_load}")
    logger.info(f"n_actions: {actions_size}")

    # DRL 2 (distance)
    logger.info(f"DRL2(DISTANCE) observation space shape: {obs_size_distance}")
    logger.info(f"n_actions: {actions_size}")

    # DRL 3 (priority)
    logger.info(f"DRL3(PRIORITY) observation space shape: {obs_size_priority}")
    logger.info(f"n_actions: {actions_size}")

    # DRL FUNCTIONS
    # load
    DRL_load = DRLModel(input_size=obs_size_load, output_size=actions_size)

    # distance
    DRL_distance = DRLModel(input_size=obs_size_distance, output_size=actions_size)

    # priority
    DRL_priority = DRLModel(input_size=obs_size_priority, output_size=actions_size)

    total_time = 0
    total_ep = 0

    for _ in tqdm(range(NUMBER_ITERATIONS), desc="Learning iterations"):
        # Algorithm
        state = env.reset()
        state_load = state[0]

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
            tensor_load = torch.FloatTensor(state[0]).unsqueeze(0).to(device)
            action_load = DRL_load.forward(tensor_load).detach().cpu().numpy().flatten()
            action_load = min_max_rescale(action_load)

            # DISTANCE
            tensor_distance = torch.FloatTensor(state[1]).unsqueeze(0).to(device)
            action_distance = DRL_distance.forward(tensor_distance).detach().cpu().numpy().flatten()
            action_distance = min_max_rescale(action_distance)

            # Priority
            tensor_priority = torch.FloatTensor(state[2]).unsqueeze(0).to(device)
            action_priority = DRL_priority.forward(tensor_priority).detach().cpu().numpy().flatten()
            tensor_priority = min_max_rescale(action_priority)

            # Log best action with respect to the three objectives
            if PLOT_ALL:
                for mode in ['load', 'distance', 'priority']:
                    tensorboard_writer.add_scalar(
                        tag=f"Action/action_{mode}",
                        scalar_value=int(np.argmax(eval(f"action_values_{mode}"))),
                        global_step=total_time,
                    )

            MODRL_problem = MultiObjectiveOptimizationProblem(
                load_action_values=action_load,
                distance_action_values=action_distance,
                priority_action_values=action_priority,
                number_of_nodes=NUMBER_OF_NODES
            )

            algorithm = NSGA2(
                pop_size=20,  # Set your desired population size
                sampling=IntegerRandomSampling(),
                crossover=SBX(prob=1.0, eta=3.0, vtype=float, repair=RoundingRepair()),
                mutation=PM(prob=1.0, eta=3.0),
                eliminate_duplicates=True
            )

            res = minimize(
                problem=MODRL_problem,
                algorithm=algorithm,
                termination=('n_gen', 10),  # Set your desired number of generations
                seed=SEED,
                save_history=True,
            )

            logger.debug(f"Best solution found: {res.X}")
            logger.debug(f"Function value: {res.F}")

            best_solution_index = np.argmax(res.F.sum(axis=1))
            action = res.X[best_solution_index]

            logger.debug(f"selected action: {action}")
            if PLOT_ALL:
                tensorboard_writer.add_scalar(
                    tag="Action/action",
                    scalar_value=action,
                    global_step=total_time,
                )

            next_state, reward, done, info = env.step(action)

            episode_reward += reward
            total_time += 1
            ep_steps += 1

            reward_list = [info["reward_load"], info["reward_distance"], info["reward_priority"]]

            for idx, key in enumerate(['reward_load', 'reward_distance', 'reward_priority']):
                tensorboard_writer.add_scalar(
                    tag=f"Reward/{key}",
                    scalar_value=reward_list[idx],
                    global_step=total_time,
                )
                episode_reward_dict[key] += reward_list[idx]

            if ep == 0 and ep_steps in [1, 20, 60, 200, 600, 999]:
                with open(Path(args.save_dir, "nsga2", "statistics.txt"), 'a') as f:
                    f.write(f"{ep_steps},{time.time() - start_time},{episode_reward_dict['reward_load']},"
                            f"{episode_reward_dict['reward_distance']},{episode_reward_dict['reward_priority']}\n")

            # log metrics from the system
            env.system.log(tensorboard_writer=tensorboard_writer, global_step=total_time, plot=PLOT_ALL)

            logger.debug(f"next_state: [load_State, distance_state, priority_state]")
            logger.debug(f"next_state: {next_state}")
            logger.debug(f"reward: {reward}")
            logger.debug(f"done: {done}")

            state = next_state
            if done:
                env.system.reset()
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



