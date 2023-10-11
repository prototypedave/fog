"""
   class: IoTDevice
   desc: task generation nodes
"""
import csv
import time
from typing import List

import numpy as np


class IoTDevice:
    def __init__(self, idx: int):
        # initialize the following variables on start up
        self.index: int = idx
        self.storage: int = 1024  # Mbs
        self.position: tuple = (np.random.randint(0, 500), np.random.randint(0, 200))

        # task related
        self.task_list: List[str, int] = []
        self.n_tasks: int = self.storage // 25  # assume an average of 25mbs per task
        self.generated_times: List[float] = [0.0]

        self.rate: float = 10.0  # kbs

        self.tasks: List[tuple] = []

    def generate_poisson_tasks(self) -> None:
        # Generate random inter-arrival times based on Poisson process
        inter_generate_times = np.random.exponential(1.0 / self.rate, self.n_tasks)

        # Calculate arrival times based on inter-arrival times
        for i in range(1, self.n_tasks):
            self.generated_times.append(self.generated_times[i - 1] + inter_generate_times[i])

    def generate_random_numbers(self, limit: int) -> int:
        # Generate random numbers within the specified limit
        return np.random.randint(1, limit)

    def populate_tasks_size(self) -> None:
        total: int = 0
        while total < self.storage - 100:
            # Randomly select the application (high, mid, or low)
            category: str = np.random.choice(["high", "mid", "low"])

            # Set the upper limit based on the selected application
            if category == "high":
                upper_limit: int = np.random.randint(200, 250)
            elif category == "mid":
                upper_limit: int = np.random.randint(100, 200)
            else:
                upper_limit: int = np.random.randint(10, 100)

            # Generate a random number within the selected application limit
            number: int = self.generate_random_numbers(upper_limit)

            # Check if adding the number exceeds the total limit
            if total + number > self.storage - 100:
                break

            # Add the generated number to the list and update the total
            self.task_list.append((category, number))
            total += number

    def calculate_cpu_instructions(self, size: int) -> float:
        start_time = time.process_time()

        # Code block you want to measure
        for i in range(size * 1024):  # Get the number of reads in the given size
            # task generation
            # Calculate progress as a percentage
            progress = (i + 1) / (size * 1024) * 100

            # Print the progress (overwrite the previous line using '\r')
            print(f"Progress: {progress:.2f}%", end="\r")

        end_time = time.process_time()

        elapsed_time = end_time - start_time

        # Estimate instructions per second (adjust this value)
        instructions_per_second = (size * 1024) / elapsed_time

        return instructions_per_second

    def generate_task(self) -> None:
        # get the scheduling and size generation first
        self.generate_poisson_tasks()
        self.populate_tasks_size()

        # based on the lists generated,schedule task generation
        time_idx = 0
        while time_idx < len(self.generated_times) and time_idx < len(self.task_list):
            time.sleep(self.generated_times[time_idx])
            priority, size = self.task_list[time_idx]

            # calculate the transmission time of sending this data
            c_time = size / 10 # Mbps

            # write to csv
            filepath = "results/transmission_delay.csv"
            #self.write_to_csv(size, c_time, filepath)

            # assign priorities in values 0 being low and 2 high
            if priority == "high":
                priority = 2
            elif priority == "mid":
                priority = 1
            elif priority == "low":
                priority = 0
            else:
                raise ValueError(f"Priority not valid")
            cpu_req = self.calculate_cpu_instructions(size)
            task = (size, cpu_req, priority)
            self.tasks.append(task)
            print(f"{task} generated at {time.time()}, scheduled time: {self.generated_times[time_idx]}")
            time_idx += 1

    def get_device_task(self) -> tuple:
        # custom function that generates tasks when all the tasks are processed
        self.generate_task()
        task: tuple = self.tasks[0]
        return task

    def write_to_csv(self, num, tim: float, file_path: str) -> None:
        # Open the file in append mode
        with open(file_path, 'a', newline='') as file:
            # Create a CSV writer object
            writer = csv.writer(file)
            # Write the updated values to the CSV file
            writer.writerow([num, tim])
