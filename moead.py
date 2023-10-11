import argparse
import logging
import time

import torch

import numpy as np
import random

import matplotlib.pyplot as plt
from pymoo.factory import get_reference_directions
from tqdm import tqdm
from pathlib import Path
from tensorboardX import SummaryWriter

from agent_dqn import DRLModel

from pymoo.core.problem import Problem
from pymoo.algorithms.moo.moead import MOEAD
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
# from pymoo.operators.repair.rounding import RoundingRepair
# from pymoo.operators.sampling.rnd import IntegerRandomSampling
from pymoo.optimize import minimize

from env import FogEnvDiscrete
from rnd import IntegerRandomSampling
from rounding import RoundingRepair

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
N_EPISODES = 100

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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--number-of-nodes", default=5, type=int, required=False)
    parser.add_argument("--max-timesteps", default=100, type=int, required=False)
    parser.add_argument("--logging", default="INFO", type=str, required=False)
    parser.add_argument("--save-dir", type=str, required=True)
    parser.add_argument("--iteration", default=0, type=int, required=False)
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
    tensorboard_writer = SummaryWriter(str(Path(args.save_dir, "moead")))

    # ================= Environment =================
    env = FogEnvDiscrete(
        n_nodes=NUMBER_OF_NODES,
        max_steps=MAX_TIMESTEPS
    )
    # ========================================================

    # create network and target network
    obs_load = NUMBER_OF_NODES
    n_load_actions = env.action_space.n
    logger.info(f"DRL1(LOAD) observation space shape: {obs_load}")
    logger.info(f"n_actions: {n_load_actions}")

    obs_distance = NUMBER_OF_NODES
    n_distance_actions = env.action_space.n
    logger.info(f"DRL2(DISTANCE) observation space shape: {obs_distance}")
    logger.info(f"n_actions: {n_distance_actions}")

    obs_priority = NUMBER_OF_NODES
    n_priority_actions = env.action_space.n
    logger.info(f"DRL3(PRIORITY) observation space shape: {obs_priority}")
    logger.info(f"n_actions: {n_load_actions}")

    # ================= Get Q functions =================
    # load q function
    q_net_load = DRLModel(input_size=obs_load, output_size=n_load_actions)
    #q_net_load.load_state_dict(
     #   torch.load(Path(args.save_dir, 'load', f'saved_DRL_itr-{args.iteration + 1}_reward-load.pt')))
    q_net_load.to(device).eval()

    # distance q function
    q_net_distance = DRLModel(input_size=obs_distance, output_size=n_distance_actions)
    #q_net_distance.load_state_dict(
    #    torch.load(Path(args.save_dir, 'distance', f'saved_DRL_itr-{args.iteration + 1}_reward-distance.pt')))
    q_net_distance.to(device).eval()

    # priority q function
    q_net_priority = DRLModel(input_size=obs_priority, output_size=n_priority_actions)
    #q_net_priority.load_state_dict(
     #   torch.load(Path(args.save_dir, 'priority', f'saved_DRL_itr-{args.iteration + 1}_reward-priority.pt')))
    q_net_priority.to(device).eval()
    # ===================================================

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
        with open(Path(args.save_dir, "moead", "statistics.txt"), 'w') as f:
            f.write(f"task_number,time(sec),load,distance,priority\n")

        while ep < N_EPISODES:
            # ====================================================
            # sample best action from all three q networks

            tensor_state_load = torch.FloatTensor(state[0]).unsqueeze(0).to(device)

            # Infer load q network
            action_values_load = q_net_load.forward(tensor_state_load).detach().cpu().numpy().flatten()
            # min-max normalize to [0,1]
            action_values_load = min_max_rescale(array=action_values_load)

            tensor_state_distance = torch.FloatTensor(state[1]).unsqueeze(0).to(device)

            # Infer distance q network
            action_values_distance = q_net_distance.forward(tensor_state_distance).detach().cpu().numpy().flatten()
            action_values_distance = min_max_rescale(array=action_values_distance)

            tensor_state_priority = torch.FloatTensor(state[2]).unsqueeze(0).to(device)
            # Infer priority q network
            action_values_priority = q_net_priority.forward(tensor_state_priority).detach().cpu().numpy().flatten()
            action_values_priority = min_max_rescale(array=action_values_priority)

            # Log best action with respect to the three objectives
            if PLOT_ALL:
                for mode in ['load', 'distance', 'priority']:
                    tensorboard_writer.add_scalar(
                        tag=f"Action/action_{mode}",
                        scalar_value=int(np.argmax(eval(f"action_values_{mode}"))),
                        global_step=total_time,
                    )

            # ================ Create Multi-objective optimization problem ================
            moo_problem = MultiObjectiveOptimizationProblem(
                load_action_values=action_values_load,
                distance_action_values=action_values_distance,
                priority_action_values=action_values_priority,
                number_of_nodes=NUMBER_OF_NODES
            )

            # algorithm = GA(
            #     pop_size=20,
            #     sampling=IntegerRandomSampling(),
            #     crossover=SBX(prob=1.0, eta=3.0, vtype=float, repair=RoundingRepair()),
            #     mutation=PM(prob=1.0, eta=3.0, vtype=float, repair=RoundingRepair()),
            #     eliminate_duplicates=True,
            # )
            ref_dirs = get_reference_directions("uniform", 3, n_partitions=12)
            algorithm = MOEAD(
                ref_dirs,
                n_neighbors=20,
                prob_neighbor_mating=0.7,
                sampling=IntegerRandomSampling(),
                crossover=SBX(prob=1.0, eta=3.0, vtype=float, repair=RoundingRepair()),
                mutation=PM(prob=1.0, eta=3.0, vtype=float, repair=RoundingRepair()),
            )

            res = minimize(
                moo_problem,
                algorithm,
                termination=('n_gen', 10),
                seed=SEED,
                save_history=False
            )

            logger.debug(f"Best solution found: {res.X}")
            logger.debug(f"Function value: {res.F}")

            action = np.random.choice(res.X.flatten())
            # ====================================================

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

            # We extract reward functions from the info dict
            reward_list = [info["reward_load"], info["reward_distance"], info["reward_priority"]]

            for idx, key in enumerate(['reward_load', 'reward_distance', 'reward_priority']):
                tensorboard_writer.add_scalar(
                    tag=f"Reward/{key}",
                    scalar_value=reward_list[idx],
                    global_step=total_time,
                )
                episode_reward_dict[key] += reward_list[idx]

            if ep == 0 and ep_steps in [1, 20, 60, 200, 600, 999]:
                with open(Path(args.save_dir, "moead", "statistics.txt"), 'a') as f:
                    f.write(f"{ep_steps},{time.time()-start_time},{episode_reward_dict['reward_load']},"
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
