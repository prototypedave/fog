"""
QDN Agent:
   class: DRLModel
   class: ReplayBuffer
   func: eval_dqn
"""
import argparse
import logging
import random
from copy import deepcopy
from pathlib import Path

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib import pyplot as plt
from tensorboardX import SummaryWriter
import torch.optim as optim
from tqdm import tqdm

from env import FogEnvDiscrete

plt.style.use('ggplot')
logger = logging.getLogger(__name__)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class DRLModel(nn.Module):
    def __init__(self, input_size, output_size):
        super(DRLModel, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)  # Input size is the dimensionality of your state vector
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, output_size)  # Output size is the number of possible actions

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        q_values = self.fc3(x)
        return q_values

    def select_greedyaction(self, state):
        with torch.no_grad():
            Q = self.forward(state)

            # Greedy action
            action_index = Q.max(1)[1].unsqueeze(1)

        return action_index.item()


class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, sample):
        """Saves a transition.
            sample is a tuple (state, next_state, action, reward, done)
        """
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = sample
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        batch_size = min(len(self.memory), batch_size)
        samples = random.sample(self.memory, batch_size)
        return map(np.asarray, zip(*samples))

    def __len__(self):
        return len(self.memory)


# Discount factor
GAMMA = 0.99
EVAL_EVERY = 2

# Batch size
BATCH_SIZE = 256
# Capacity of the replay buffer
BUFFER_CAPACITY = 500000
# Update target net every 'C' episodes
UPDATE_TARGET_EVERY = 100

# Initial value of epsilon
EPSILON_START = 1
# Parameter to decrease epsilon
DECREASE_EPSILON = 10000
# Minimum value of epsilon
EPSILON_MIN = 0.2
# Step at which to strat decreasing epsilon
START_DECREASE = 150000
# Episode at which to stop exploring
STOP_EXPLORING = 5000

# Number of training episodes
N_EPISODES = 7500
# Number of iterations (where we keep only the replay buffer)
NUMBER_ITERATIONS = 1

# Learning rate
LEARNING_RATE = 0.001

NUMBER_OF_NODES = 250
MAX_TIMESTEPS = 1000
PLOT_ALL = False

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--number-of-nodes", default=250, type=int, required=False)
    parser.add_argument("--max-timesteps", default=100, type=int, required=False)
    parser.add_argument("--logging", default="INFO", type=str, required=False)
    parser.add_argument("--save-dir", default="results", type=str, required=False)

    args = parser.parse_args()

    NUMBER_OF_NODES = args.number_of_nodes
    MAX_TIMESTEPS = args.max_timesteps

    logging.basicConfig(
        level=eval(f"logging.{args.logging}"),
        format="%(asctime)s %(levelname)-8s %(message)s",
        datefmt="%d-%b-%y %H:%M:%S"
    )
    logger.info(f"Start testing")

    # tensorboard logger
    tensorboard_writer = SummaryWriter(args.save_dir)

    env = FogEnvDiscrete(
        n_nodes=NUMBER_OF_NODES,
        max_steps=MAX_TIMESTEPS
    )

    # initialize replay buffer
    replay_buffer_load = ReplayBuffer(BUFFER_CAPACITY)
    replay_buffer_distance = ReplayBuffer(BUFFER_CAPACITY)
    replay_buffer_priority = ReplayBuffer(BUFFER_CAPACITY)

    # create network and target network
    # DRL 1 (load)
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

    for iterat in tqdm(range(NUMBER_ITERATIONS), desc="Learning iterations"):
        logger.info(f" ## Create DRL-networks")

        # main drls
        DRL_load = DRLModel(input_size=obs_load, output_size=n_load_actions)
        DRL_distance = DRLModel(input_size=obs_distance, output_size=n_distance_actions)
        DRL_priority = DRLModel(input_size=obs_priority, output_size=n_priority_actions)

        logger.info(f" ## Create target DRL-network")
        # The target network initialized with the same weights
        target_load = DRLModel(input_size=obs_load, output_size=n_load_actions).to(device)
        target_load.load_state_dict(DRL_load.state_dict())
        target_load.eval()

        target_distance = DRLModel(input_size=obs_distance, output_size=n_distance_actions).to(device)
        target_distance.load_state_dict(DRL_distance.state_dict())
        target_distance.eval()

        target_priority = DRLModel(input_size=obs_load, output_size=n_priority_actions).to(device)
        target_priority.load_state_dict(DRL_priority.state_dict())
        target_priority.eval()

        logger.info(f" ## Create Optimizer")
        # objective and optimizer
        optimizer_L = optim.Adam(params=DRL_load.parameters(), lr=LEARNING_RATE)
        optimizer_D = optim.Adam(params=DRL_distance.parameters(), lr=LEARNING_RATE)
        optimizer_P = optim.Adam(params=DRL_priority.parameters(), lr=LEARNING_RATE)

        # Algorithm
        state = env.reset()
        epsilon = EPSILON_START
        ep = 0
        learn_steps = 0
        episode_reward = 0
        episode_reward_dict = {
            'reward_load': 0,
            'reward_distance': 0,
            'reward_priority': 0
        }

        logger.debug(f" ## Start episode")
        episode_rewards = np.zeros((N_EPISODES, 3))

        episodes_bar = tqdm(total=N_EPISODES, desc="Episodes")
        # env_steps_bar = tqdm(total=env.max_timesteps, desc="env timesteps")
        while ep < N_EPISODES:
            logger.debug(f"\n============ step: {learn_steps} ============")
            # ====================================================
            # sample epsilon-greedy action
            p = random.random()
            # We keep the last N_EPISODES - STOP_EXPLORING episodes for pure exploitation (sort of evaluation)
            if (p < epsilon) and (ep <= STOP_EXPLORING):
                # Select an action with uniform probability
                action = np.random.randint(low=0, high=n_load_actions)
                action_L = action
                action_D = action
                action_P = action
                tensorboard_writer.add_scalar(
                    tag="exploration/exploration",
                    scalar_value=1.0,
                    global_step=total_time,
                )
            else:
                # Select greedy_action
                tensor_state_L = torch.FloatTensor(state[0]).unsqueeze(0).to(device)
                action_L = DRL_load.select_greedyaction(tensor_state_L)
                tensor_state_D = torch.FloatTensor(state[1]).unsqueeze(0).to(device)
                action_D = DRL_load.select_greedyaction(tensor_state_D)
                tensor_state_P = torch.FloatTensor(state[2]).unsqueeze(0).to(device)
                action_P = DRL_load.select_greedyaction(tensor_state_P)
                tensorboard_writer.add_scalar(
                    tag="exploration/exploration",
                    scalar_value=0.0,
                    global_step=total_time,
                )

            logger.debug(f"selected action: {action_L}, {action_D}, {action_P}")
            actions = [action_L, action_D, action_P]
            if PLOT_ALL:
                for action in actions:
                    tensorboard_writer.add_scalar(
                        tag="Action/action",
                        scalar_value=action,
                        global_step=total_time,
                    )
            action = random.choice(actions)
            load_state, distance_state, priority_state, reward, done = env.step(action)

            episode_reward += reward[0]
            total_time += 1
            print(total_time)

            for idx, key in enumerate(['reward_load', 'reward_distance', 'reward_priority']):
                tensorboard_writer.add_scalar(
                    tag=f"Reward/{key}",
                    scalar_value=reward[idx],
                    global_step=total_time,
                )
                episode_reward_dict[key] += reward[idx]

            # log metrics from the system
            env.system.log(tensorboard_writer=tensorboard_writer, global_step=total_time, plot=PLOT_ALL)

            logger.debug(f"next_state: [load_State, distance_state, priority_state]")
            logger.debug(f"next_state: {load_state}, {distance_state}, {priority_state}")
            logger.debug(f"reward: {reward}")
            logger.debug(f"done: {done}")

            if actions[0] > -1:
                sample_tuple_load = (state[0], load_state, action, reward[0], done)
                replay_buffer_load.push(sample_tuple_load)
            if actions[1] > -1:
                sample_tuple_distance = (state[1], distance_state, action, reward[1], done)
                replay_buffer_distance.push(sample_tuple_distance)
            if actions[2] > -1:
                sample_tuple_priority = (state[2], priority_state, action, reward[2], done)
                replay_buffer_priority.push(sample_tuple_priority)

            logger.debug(f"replay_buffer size: {len(replay_buffer_load)}")
            logger.debug(f"replay_buffer size: {len(replay_buffer_distance)}")
            logger.debug(f"replay_buffer size: {len(replay_buffer_priority)}")

            if PLOT_ALL:
                tensorboard_writer.add_scalar(
                    tag=f"Replay_Buffer/Size",
                    scalar_value=len(replay_buffer_load),
                    global_step=total_time,
                )
            if len(replay_buffer_load) > BATCH_SIZE:
                learn_steps += 1

                # UPDATE MODEL
                # get batch
                batch_state, batch_next_state, batch_action, batch_reward, batch_done = replay_buffer_load.sample(
                        BATCH_SIZE)

                batch_state = torch.FloatTensor(batch_state).to(device)
                batch_next_state = torch.FloatTensor(batch_next_state).to(device)
                batch_action = torch.FloatTensor(batch_action).unsqueeze(1).to(device)
                batch_reward = torch.FloatTensor(batch_reward).unsqueeze(1).to(device)
                batch_done = torch.FloatTensor(batch_done).unsqueeze(1).to(device)

                with torch.no_grad():
                    # Greedy action from the target network
                    target_L = target_load(batch_next_state)
                    value_next = target_L.max(dim=1, keepdim=False)[0]

                    # Eliminate terminal states
                    mask = (1 - batch_done).reshape(value_next.shape)
                    value_next = (value_next * mask).unsqueeze(1).to(device)

                    # Target values
                    targets = batch_reward + GAMMA * value_next

                # current predictions
                DRL_load.train()
                values = DRL_load(batch_state).gather(1, batch_action.long())

                # compute loss and update model (loss and optimizer)
                optimizer_L.zero_grad()
                loss = F.mse_loss(values, targets)
                loss.backward()
                optimizer_L.step()

                tensorboard_writer.add_scalar(
                    tag=f"QNet/Loss",
                    scalar_value=float(loss),
                    global_step=total_time,
                )

                if epsilon > EPSILON_MIN and learn_steps > START_DECREASE:
                    epsilon -= (EPSILON_START - EPSILON_MIN) / DECREASE_EPSILON

            if len(replay_buffer_distance) > BATCH_SIZE:
                learn_steps += 1

                # UPDATE MODEL
                # get batch
                batch_state, batch_next_state, batch_action, batch_reward, batch_done = replay_buffer_load.sample(
                        BATCH_SIZE)

                batch_state = torch.FloatTensor(batch_state).to(device)
                batch_next_state = torch.FloatTensor(batch_next_state).to(device)
                batch_action = torch.FloatTensor(batch_action).unsqueeze(1).to(device)
                batch_reward = torch.FloatTensor(batch_reward).unsqueeze(1).to(device)
                batch_done = torch.FloatTensor(batch_done).unsqueeze(1).to(device)

                with torch.no_grad():
                    # Greedy action from the target network
                    target_L = target_distance(batch_next_state)
                    value_next = target_L.max(dim=1, keepdim=False)[0]

                    # Eliminate terminal states
                    mask = (1 - batch_done).reshape(value_next.shape)
                    value_next = (value_next * mask).unsqueeze(1).to(device)

                    # Target values
                    targets = batch_reward + GAMMA * value_next

                # current predictions
                DRL_distance.train()
                values = DRL_distance(batch_state).gather(1, batch_action.long())

                # compute loss and update model (loss and optimizer)
                optimizer_D.zero_grad()
                loss = F.mse_loss(values, targets)
                loss.backward()
                optimizer_D.step()

                tensorboard_writer.add_scalar(
                    tag=f"QNet/Loss",
                    scalar_value=float(loss),
                    global_step=total_time,
                )

                if epsilon > EPSILON_MIN and learn_steps > START_DECREASE:
                    epsilon -= (EPSILON_START - EPSILON_MIN) / DECREASE_EPSILON

            if len(replay_buffer_priority) > BATCH_SIZE:
                learn_steps += 1

                # UPDATE MODEL
                # get batch
                batch_state, batch_next_state, batch_action, batch_reward, batch_done = replay_buffer_load.sample(
                        BATCH_SIZE)

                batch_state = torch.FloatTensor(batch_state).to(device)
                batch_next_state = torch.FloatTensor(batch_next_state).to(device)
                batch_action = torch.FloatTensor(batch_action).unsqueeze(1).to(device)
                batch_reward = torch.FloatTensor(batch_reward).unsqueeze(1).to(device)
                batch_done = torch.FloatTensor(batch_done).unsqueeze(1).to(device)

                with torch.no_grad():
                    # Greedy action from the target network
                    target_L = target_priority(batch_next_state)
                    value_next = target_L.max(dim=1, keepdim=False)[0]

                    # Eliminate terminal states
                    mask = (1 - batch_done).reshape(value_next.shape)
                    value_next = (value_next * mask).unsqueeze(1).to(device)

                    # Target values
                    targets = batch_reward + GAMMA * value_next

                # current predictions
                DRL_priority.train()
                values = DRL_load(batch_state).gather(1, batch_action.long())

                # compute loss and update model (loss and optimizer)
                optimizer_P.zero_grad()
                loss = F.mse_loss(values, targets)
                loss.backward()
                optimizer_P.step()

                tensorboard_writer.add_scalar(
                    tag=f"QNet/Loss",
                    scalar_value=float(loss),
                    global_step=total_time,
                )

                if epsilon > EPSILON_MIN and learn_steps > START_DECREASE:
                    epsilon -= (EPSILON_START - EPSILON_MIN) / DECREASE_EPSILON

            # update target network
            if learn_steps % UPDATE_TARGET_EVERY == 0:
                target_load.train()
                target_load.load_state_dict(DRL_load.state_dict())
                target_load.eval()

                target_distance.train()
                target_distance.load_state_dict(DRL_distance.state_dict())
                target_distance.eval()

                target_priority.train()
                target_priority.load_state_dict(DRL_priority.state_dict())
                target_priority.eval()

            state = [load_state, distance_state, priority_state]
            if done:
                mean_rewards = -1
                episode_rewards[ep] = [total_time, episode_reward, mean_rewards]
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

        # Save current run's results
        episodes_bar.close()

        # save model state
        DRL_load.eval()
        torch.save(DRL_load.state_dict(), Path(args.save_dir, f'saved_DRL_load_itr-.pt'))
        DRL_distance.eval()
        torch.save(DRL_distance.state_dict(), Path(args.save_dir, f'saved_DRL_distance_itr-.pt'))
        DRL_priority.eval()
        torch.save(DRL_priority.state_dict(), Path(args.save_dir, f'saved_DRL_priority_itr-.pt'))



