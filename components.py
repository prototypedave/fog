"""
    desc: Computing nodes for the tasks generated
    class: ComputingNode
        : parent classes with relevant functions
    class: FogNode
    class: CloudNode
"""
import csv
import math
import time
from queue import PriorityQueue
from typing import List, Dict

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

        # task completion
        self.submission_time: Dict = {}
        self.n_tasks: int = 0

        # storage
        self.memory = 10240  # 10GB
        self.count = 0

        self.span: bool = False
        self.start_time: float = 0.0   # time for scheduling

        self.sched_time = 0.0

    def add_task_in_queue(self, job: tuple) -> None:
        if not self.span:
            self.start_time = time.time()
            self.span = True

        # get current time
        sub_time = time.time()
        self.submission_time[job] = sub_time

        # get scheduled time based on the task
        scheduled_time = self.calculate_task_delay(job)
        self.sched_time += scheduled_time

        # count the nunmber of tasks added to que
        self.tasks_to_be_processed += 1

        # add task to the que with a negative priority
        self.fog_tasks.put((-job[2], scheduled_time, job))
        print(f"task qued, waiting time: {scheduled_time}ms, node {self.index}")

    def classify_priorities(self) -> None:
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

    def update_waiting_time(self, schedule: int) -> None:
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
            return sum(self.get_scheduled_time(2)) + 1.0
        elif job[2] == 1:
            high: float = sum(self.get_scheduled_time(2))
            return sum(self.get_scheduled_time(1)) + 2.0 + high
        elif job[2] == 0:
            high: float = sum(self.get_scheduled_time(2))
            mid: float = sum(self.get_scheduled_time(1))
            return sum(self.get_scheduled_time(0)) + 3.0 + high + mid
        else:
            raise ValueError(f"unknown priority {job[2]}")

    def compute_tasks(self) -> float:
        # Process tasks and return the processed task
        process_time: float = 0.0
        r_memory = self.memory

        while not self.fog_tasks.empty():
            self.tasks_to_be_processed = self.fog_tasks.qsize()
            priority, sched, job = self.fog_tasks.get()
            process_time = self.calculate_computing_time(job[1])
            self.processed_tasks += 1

            # save processing time to file
            file_path_1 = "results/processing_time.csv"
            #self.write_to_csv(job[0], process_time, file_path_1)

            # get the remaining storage
            r_memory -= job[0]
            self.count += 1

            # calculate task completion time from submission
            current = time.time()
            start = self.submission_time[job]

            c_time = current - start
            self.n_tasks += 1

            # update the csv file
            file_path = "results/task_completion_time.csv"

            #self.write_to_csv(self.n_tasks, c_time, file_path)

            # calculate computation delay
            comp_del = sched + process_time

            # write to file
            file = "results/computational_delay.csv"
            #self.write_to_csv(self.index, comp_del, file)

            # calculate propagation
            trans = job[0] / 10  # Mbps
            propagation = comp_del + trans

            filep = "results/propagation.csv"
            self.write_to_csv(job[0], propagation, filep)

            # update the scheduled time
            self.update_waiting_time(sched)

        # assume now the q is empty
        # save the remaining memory in csv
        file_path = "results/storage.csv"
        self.write_to_csv(self.index, r_memory, file_path)

        self.span = False
        if self.start_time > 0:
            makespan: float = time.time() - self.start_time

            filepath = "results/makespan.csv"
            # self.write_to_csv(self.index, makespan, filepath)

        return process_time

    def calculate_computing_time(self, cpu_instr: float) -> float:
        # Check the current CPU load and frequency
        current_cpu_load = psutil.cpu_percent(interval=1)
        current_cpu_frequency = psutil.cpu_freq().current  # Current CPU frequency in GHz

        # write to csv the load per number of processed tasks
        file_path = "results/load.csv"
        # self.write_to_csv(self.processed_tasks, current_cpu_load, file_path)

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

        return elapsed_time

    def update_transmission_state(self):
        if self.transmit:
            self.transmit = False
        else:
            self.transmit = True

    def calculate_distance(self, dist: tuple, task: tuple):
        x1: int = dist[0]
        y1: int = dist[1]
        x2: int = self.position[0]
        y2: int = self.position[1]

        distance: float = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

        # calculate latency
        lat: float = task[0] / 10240  # assume an average of 10mbps for every node
        latency: float = lat * distance  # total time it takes to transmit the task to the destination

        file_path = "results/latency.csv"
        #self.write_to_csv(int(distance), latency, file_path)

        return distance

    def reset(self):
        # before a reset calculate the throughput
        throughput: float = 0.0

        if self.processed_tasks != 0 and self.tasks_to_be_processed != 0:
            throughput = self.processed_tasks / self.tasks_to_be_processed

        # save to file
        file_path = "results/throughput.csv"
        #self.write_to_csv(self.index, throughput, file_path)

        # write to file
        filepath = "results/scheduled_time.csv"
        self.write_to_csv(self.tasks_to_be_processed, self.sched_time, filepath)

        # save to file
        file = "results/congestion.csv"
        self.write_to_csv(self.index, self.tasks_to_be_processed, file)

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

    def write_to_csv(self, num, tim: float, file_path: str) -> None:
        # Open the file in append mode
        with open(file_path, 'a', newline='') as file:
            # Create a CSV writer object
            writer = csv.writer(file)
            # Write the updated values to the CSV file
            writer.writerow([num, tim])


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
