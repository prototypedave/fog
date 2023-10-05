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
        #self.reward_mode = reward_mode

        # Define action space (each agent selects one of the nodes)
        self.action_space = spaces.Discrete(n_nodes)

        # Define observation space for node states (node load, node distance, task priority)
        self.observation_space_load = spaces.Dict({
            "node_load": spaces.Box(low=0, high=1, shape=(n_nodes,), dtype=np.float32)
        })
        self.observation_space_distance = spaces.Dict({
            "node_distance": spaces.Box(low=0, high=1, shape=(n_nodes,), dtype=np.float32)
        })
        self.observation_space_priority = spaces.Dict({
            "task_priority": spaces.Box(low=0, high=1, shape=(n_nodes,), dtype=np.float32)
        })

        # Initialize the system with assigning different nodes
        self.system = System(n_iot=2, n_fog=n_nodes, n_cloud=1)

        # Initialize node states (random float values between 0 and 1)
        self.node_loads: List[float] = []
        self.node_distances: List[float] = []
        self.task_priorities: List[float] = []
        self.task: tuple = ()

        self.steps = 0
        self.max_steps = max_steps

    def reset(self):
        # get the tasks
        task = self.system.controller.get_task()
        print(task)
        self.task = task[1]
        self.node_loads = self.system.controller.get_current_load()

        pos = self.system.layer1.devices[task[0]].position
        self.node_distances = self.system.controller.get_distance(pos)

        self.task_priorities = self.system.controller.get_node_priority()

        state1 = np.array(self.node_loads)
        state2 = np.array(self.node_distances)
        state3 = np.array(self.task_priorities)

        return [state1, state2, state3]

    def step(self, action: Action):
        self.steps += 1

        # assign task to the node
        self.system.controller.send_task(self.task, int(action))

        # process all the nodes
        self.system.controller.process_task()

        reward = self.reward(action)

        self.system.run()
        task = self.system.controller.get_task()
        self.task = task[1]
        self.node_loads = self.system.controller.get_current_load()

        pos = self.system.layer1.devices[task[0]].position
        self.node_distances = self.system.controller.get_distance(pos)

        self.task_priorities = self.system.controller.get_node_priority()

        state1 = np.array(self.node_loads)
        state2 = np.array(self.node_distances)
        state3 = np.array(self.task_priorities)

        if self.steps >= self.max_steps:
            done = True
        else:
            done = False

        return state1, state2, state3, reward, done

    def reward(self, action):
        # ========= Load Reward =========
        reward1 = -np.linalg.norm(np.array(self.node_loads))

        # ========= Distance Reward =========
        reward2 = -np.linalg.norm(np.array(self.node_distances))

        # ========= Priority =========
        reward3 = -np.linalg.norm(np.array(self.task_priorities))

        reward = [reward1, reward2, reward3]
        return reward

    def render(self, mode='human', close=False):
        # Render the environment to the screen
        return

logger = logging.getLogger(__name__)


if __name__ == "__main__":
    env = FogEnvDiscrete(100, 10)
    env.reset()
    print(env.task_priorities)



