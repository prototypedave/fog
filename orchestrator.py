"""

"""
import time
from typing import List, Dict

from components import FogNode
from iot_devices import IoTDevice


class Orchestrator:
    def __init__(self, num_iots: int):
        # initialize
        self.tasks: List[tuple] = []
        self.tasks_id: List[int] = []
        self.running: bool = False

        self.fogs: List[FogNode] = []

        self.iots = num_iots

    def add_task(self, idx: int, task: tuple) -> None:
        # get task from iot device
        tup: tuple = (idx, task)
        self.tasks.append(tup)

    def get_task(self) -> List:
        # prepare a task to be sent to the fog
        if len(self.tasks) == 0:
            # regenerate tasks
            for n in range(self.iots):
                iot = IoTDevice(n)
                iot.generate_task()
                task: List[tuple] = iot.tasks
                for p in task:
                    self.tasks.append(p)
                    self.tasks_id.append(n)

        # remove the task as soon as received
        task: tuple = self.tasks.pop(0)
        idx: int = self.tasks_id.pop(0)
        t: List = [idx, task]
        return t

    def get_node(self, ind: int) -> FogNode:
        # returns a fog node in the network using its index
        return self.fogs[ind]

    def process_task(self) -> None:
        # run the network to process each node
        for fog in self.fogs:
            fog.compute_tasks()

    def send_task(self, task: tuple, idx: int) -> None:
        # send tasks to the fog node
        fog: FogNode = self.get_node(idx)
        fog.add_task_in_queue(task)

    def get_current_load(self) -> List[float]:
        # get a list of nodes
        node_load: List[float] = []
        for fog in self.fogs:
            load = fog.current_load / 100
            node_load.append(load)
        return node_load

    def get_distance(self, pos: tuple, task: tuple) -> List[float]:
        dist_list: List[float] = []
        # get the distances
        for fog in self.fogs:
            distance = fog.calculate_distance(pos, task)
            dist_list.append(distance)
        return dist_list

    def get_node_priority(self) -> List[float]:
        priority_list: List[float] = []
        for fog in self.fogs:
            prior = fog.calculate_task_priority()
            priority_list.append(prior)
        return priority_list

    def generate_tasks(self, devices: List[IoTDevice]) -> None:
        for iot in devices:
            iot.generate_task()
            for task in iot.tasks:
                self.tasks.append(task)
                self.tasks_id.append(iot.index)




