"""

"""
import time
from typing import List, Dict

from components import FogNode
from iot_devices import IoTDevice


class Orchestrator:
    def __init__(self):
        self.tasks: List[tuple] = []
        self.tasks_id: List[int] = []
        self.running: bool = False

        self.fogs: List[FogNode] = []

    def add_task(self, idx: int, task: tuple):
        tup: tuple = (idx, task)
        self.tasks.append(tup)

    def get_task(self) -> List:
        task: tuple = self.tasks.pop(0)
        idx: int = self.tasks_id.pop(0)
        t: List = [idx, task]
        return t

    def get_node(self, ind: int):
        return self.fogs[ind]

    def process_task(self):
        for fog in self.fogs:
            process = fog.compute_tasks()
            print(process)

    def send_task(self, task: tuple, idx: int):
        fog: FogNode = self.get_node(idx)
        fog.add_task_in_queue(task)

    def get_current_load(self):
        node_load: List[float] = []
        for fog in self.fogs:
            load = fog.current_load / 100
            node_load.append(load)
        return node_load

    def get_distance(self, pos: tuple):
        dist_list: List[float] = []
        # get the distances
        for fog in self.fogs:
            distance = fog.calculate_distance(pos)
            dist_list.append(distance)
        return dist_list

    def get_node_priority(self):
        priority_list: List[float] = []
        for fog in self.fogs:
            prior = fog.calculate_task_priority()
            priority_list.append(prior)
        return priority_list

    def generate_tasks(self, devices: List[IoTDevice]):
        for iot in devices:
            iot.generate_task()
            for task in iot.tasks:
                self.tasks.append(task)
                self.tasks_id.append(iot.index)




