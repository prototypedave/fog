"""

"""
import math
import time
from queue import PriorityQueue
from typing import List

import psutil


class ComputingNode:
    def __init__(self, idx: int, x: int, y: int):
        # initialize
        self.index = idx
        self.max_cpu_load: int = 40
        self.min_cpu_frequency: float = 2.0

        self.position: tuple = (x, y)
        self.task_processed: bool = False

        # store tasks in this que
        self.fog_tasks: PriorityQueue = PriorityQueue()
        self.tasks_to_be_processed = 0
        self.processed_tasks = 0

        # communication / transmission
        self.transmit: bool = False

        self.current_load = 0

    def add_task_in_queue(self, job: tuple) -> None:
        # get current time and add the scheduled time
        scheduled_time = self.calculate_task_delay(job)

        # add task to the que with a negative priority
        self.fog_tasks.put((-job[2], scheduled_time, job))
        print(f"task qued, waiting time: {scheduled_time}ms, node {self.index}")

    def classify_priorities(self):
        # get the number of different priorities
        priorities: List[int] = [0, 0, 0]
        tasks: PriorityQueue = PriorityQueue()

        while not self.fog_tasks.empty():
            prior, s, task = self.fog_tasks.get()
            prior *= -1
            if prior == 0:
                priorities[0] += 1
            elif prior == 1:
                priorities[1] += 1
            elif prior == 2:
                priorities[2] += 1
            else:
                raise ValueError(f"Invalid priority")

            tasks.put((-prior, s, task))
        self.fog_tasks = tasks

    def get_scheduled_time(self, prior: int) -> List[int]:
        # create a list to store scheduled time per given priority
        s_time: List[int] = []
        copy_qued_tasks = PriorityQueue()

        while not self.fog_tasks.empty():
            priority, sched, job = self.fog_tasks.get()
            p = priority * -1
            if p == prior:
                s_time.append(sched)
            # add task to the duplicate que storage
            copy_qued_tasks.put((priority, sched, job))

        # return the que stack back
        self.fog_tasks = copy_qued_tasks
        return s_time

    def update_waiting_time(self, schedule: int):
        # updates the scheduled time everytime a task is removed from que
        copy_tasks = PriorityQueue()

        while not self.fog_tasks.empty():
            prior, scheduled_time, job = self.fog_tasks.get()
            scheduled_time = scheduled_time - schedule
            # check for negative values
            if scheduled_time < 0:
                scheduled_time = 0
            copy_tasks.put((prior, scheduled_time, job))

        # copy back the tasks to the que
        self.fog_tasks = copy_tasks

    def calculate_task_delay(self, job: tuple) -> float:
        # assign schedule time based on priority
        if job[2] == 2:
            return sum(self.get_scheduled_time(2)) + 2.0
        elif job[2] == 1:
            high: float = sum(self.get_scheduled_time(2))
            return sum(self.get_scheduled_time(1)) + 5.0 + high
        elif job[2] == 0:
            high: float = sum(self.get_scheduled_time(2))
            mid: float = sum(self.get_scheduled_time(1))
            return sum(self.get_scheduled_time(0)) + 10.0 + high + mid
        else:
            raise ValueError(f"unknown priority {job[2]}")

    def compute_tasks(self) -> float:
        # Process tasks and return the processed task
        process_time: float = 0.0

        while not self.fog_tasks.empty():
            self.tasks_to_be_processed = self.fog_tasks.qsize()
            priority, sched, job = self.fog_tasks.get()
            process_time = self.calculate_computing_time(job[1])
            self.processed_tasks += 1

            # update the scheduled time
            self.update_waiting_time(sched)
        return process_time

    def calculate_computing_time(self, cpu_instr: float) -> float:
        # Check the current CPU load and frequency
        current_cpu_load = psutil.cpu_percent(interval=1)
        current_cpu_frequency = psutil.cpu_freq().current  # Current CPU frequency in GHz

        # Check if CPU load and frequency exceed specified thresholds
        if current_cpu_load > self.max_cpu_load:
            print(f"CPU load exceeds the allowed limit.")

        if current_cpu_frequency < self.min_cpu_frequency:
            print(f"CPU frequency is below the required minimum.")

        # Record the start time
        start_time = time.time()

        # Perform some CPU-intensive operations (read and process tasks)
        completed_instructions = 0
        for i in range(int(cpu_instr)):
            # Simulate processing by incrementing the completed_instructions counter
            completed_instructions += 1

            # Print progress as a percentage
            progress = (completed_instructions / cpu_instr) * 100
            print(f"Progress: {progress:.2f}%", end="\r")  # Print on the same line

        # Record the end time
        end_time = time.time()

        # Calculate the elapsed time
        elapsed_time = end_time - start_time

        if not self.task_processed:
            self.current_load += current_cpu_load

        self.current_load = current_cpu_load

        # return [current_cpu_load, current_cpu_frequency, elapsed_time]
        return elapsed_time

    def update_transmission_state(self):
        if self.transmit:
            self.transmit = False
        else:
            self.transmit = True

    def calculate_distance(self, dist: tuple):
        x1: int = dist[0]
        y1: int = dist[1]
        x2: int = self.position[0]
        y2: int = self.position[1]

        distance: float = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        return distance

    def reset(self):
        while not self.fog_tasks.empty():
            self.fog_tasks.get()
        self.tasks_to_be_processed = 0
        self.processed_tasks = 0

    def calculate_task_priority(self):
        copy_tasks = PriorityQueue()
        priority: float = 0.0
        count: int = 0

        while not self.fog_tasks.empty():
            prior, scheduled_time, job = self.fog_tasks.get()
            priority += (prior * -1)
            count += 1
            copy_tasks.put((prior, scheduled_time, job))
        if count != 0:
            priority = priority / count

        # copy back the tasks to the que
        self.fog_tasks = copy_tasks
        return priority


class FogNode(ComputingNode):
    def __init__(self, idx: int, x: int, y: int, cpu_load: float = 80.0, cpu_freq: float = 2.0):
        super().__init__(idx=idx, x=x, y=y)
        self.max_cpu_load = cpu_load
        self.min_cpu_frequency = cpu_freq


class CloudNode(ComputingNode):
    def __init__(self, idx: int, x: int, y: int, cpu_load: float = 100.0, cpu_freq: float = 3.0):
        super().__init__(idx=idx, x=x, y=y)
        self.max_cpu_load = cpu_load
        self.min_cpu_frequency = cpu_freq


if __name__ == "__main__":
    fog = FogNode(1, 230, 290)
    print(fog.calculate_distance((500, 40)))
