"""
"""
import logging
import time
from typing import List

import gym
import numpy as np
from gym.spaces import Dict, Discrete, MultiDiscrete, Box
from gym import spaces

from system import System

Action = np.array
State = np.array


class FogEnvDiscrete(gym.Env):
    """ Custom environment """
    metadata = {'render.modes': ['human']}

    def __init__(self, n_nodes: int, max_steps: int):
        super(FogEnvDiscrete, self).__init__()
        self.n_nodes = n_nodes

        # Define action space (each agent selects one of the nodes)
        self.action_space = spaces.Discrete(n_nodes)

        # Define observation space for node states (node load, node distance, task priority)
        self.observation_space = spaces.Box(
            low=np.array([0.0, 0.0, 0.0]),  # Minimum values for node_load, distance, and priority
            high=np.array([1.0, 1.0, 1.0]),  # Maximum values for node_load, distance, and priority (normalized to 1)
            dtype=np.float32
        )

        # Initialize the system with assigning different nodes
        self.system = System(n_iot=100, n_fog=n_nodes, n_cloud=1)
        self.system.run()

        self.task: tuple = ()
        self.max_steps = max_steps
        self.steps = 0

    def reset(self):
        # get the list of nodes current load
        node_loads = self.system.controller.get_current_load()

        # get the first task to be processed
        task = self.system.controller.get_task()
        # save the task to the env
        self.task = task[1]

        # use the task index to retrieve the node sending position
        pos = self.system.layer1.devices[task[0]].position
        # calculate the distance of the node from each fog in the network
        node_distances = self.system.controller.get_distance(pos, self.task)

        # return a list of priorities
        task_priorities = self.system.controller.get_node_priority()

        # normalized lists
        loads = self._normalize(node_loads)
        distance = self._normalize(node_distances)
        priority = self._normalize(task_priorities)

        obs = np.concatenate([loads, distance, priority])
        return [loads, distance, priority]

    def step(self, action):
        self.steps += 1

        # assign task to the node
        self.system.controller.send_task(self.task, int(action))
        # process all the nodes
        self.system.controller.process_task()

        # get the list of nodes current load
        node_loads = self.system.controller.get_current_load()

        # get the first task to be processed
        task = self.system.controller.get_task()
        # save the task to the env
        self.task = task[1]

        # use the task index to retrieve the node sending position
        pos = self.system.layer1.devices[task[0]].position
        # calculate the distance of the node from each fog in the network
        node_distances = self.system.controller.get_distance(pos, self.task)

        # return a list of priorities
        task_priorities = self.system.controller.get_node_priority()

        # normalized lists
        loads = self._normalize(node_loads)
        distance = self._normalize(node_distances)
        priority = self._normalize(task_priorities)

        obs = np.concatenate([loads, distance, priority])

        reward, info = self.reward(obs)

        state = [loads, distance, priority]

        done = False
        if self.steps >= self.max_steps:
            done = True

        return state, reward, done, info

    def reward(self, state):
        load = state[0]
        distance = state[1]
        priority = state[2]

        load_reward = 1 - load
        distance_reward = 1 - distance
        priority_reward = 1 - priority

        info = {
            "reward_load": load_reward,
            "reward_distance": distance_reward,
            "reward_priority": priority_reward
        }

        reward = load_reward + distance_reward + priority_reward

        return reward, info

    def render(self, mode='human', close=False):
        # Render the environment to the screen
        return

    def _normalize(self, values):
        # Perform min-max normalization to [0, 1] for the given list of values
        if all(x == values[0] for x in values):
            return [0.0] * len(values)  # Set normalized values to 0 or any other default value

        min_value = min(values)
        max_value = max(values)
        return [(x - min_value) / (max_value - min_value) for x in values]


logger = logging.getLogger(__name__)


if __name__ == "__main__":
    env = FogEnvDiscrete(100, 10)
    env.reset()
    print(env.task_priorities)



