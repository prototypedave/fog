"""
    Fog Network
"""
from typing import List

import numpy as np
from tensorboardX import SummaryWriter

from components import FogNode, CloudNode
from iot_devices import IoTDevice
from orchestrator import Orchestrator


class IoTLayer:
    def __init__(self, num: int, controller: Orchestrator):
        self.devices: List[IoTDevice] = []
        self.orch: Orchestrator = controller

        # populate IoTDevices in this layer
        for n in range(num):
            iot = IoTDevice(n)
            self.devices.append(iot)


class FogLayer:
    def __init__(self, num: int, controller: Orchestrator):
        self.fogs: List[FogNode] = []
        self.orch: Orchestrator = controller

        # populate FogNodes in this layer
        for m in range(num):
            x: int = np.random.randint(100, 400)
            y: int = np.random.randint(200, 400)
            fg = FogNode(m, x, y)
            self.fogs.append(fg)
            self.orch.fogs.append(fg)


class CloudLayer:
    def __init__(self, num: int):
        self.clouds: List[CloudNode] = []

        # populate CLoudNodes
        for k in range(num):
            x: int = np.random.randint(200, 300)
            y: int = np.random.randint(400, 500)
            cloud = CloudNode(k, x, y)
            self.clouds.append(cloud)


class System:
    def __init__(self, n_iot: int, n_fog: int, n_cloud: int):
        self.controller = Orchestrator()
        self.layer1 = IoTLayer(n_iot, self.controller)
        self.layer2 = FogLayer(n_fog, self.controller)
        self.layer3 = CloudLayer(n_cloud)

        self.run()

    def run(self):
        self.controller.running = True
        self.controller.generate_tasks(self.layer1.devices)

    def log(self, tensorboard_writer: SummaryWriter, global_step: int, plot: bool = False):
        if plot:
            for i, node in enumerate(self.layer2.fogs):
                # Load
                tensorboard_writer.add_scalar(
                    tag=f"Load/Node_{i}",
                    scalar_value=node.get_current_load(mode='percent'),
                    global_step=global_step,
                )
                # Waiting list size in nodes
                tensorboard_writer.add_scalar(
                    tag=f"Waiting_list_size/Node_{i}",
                    scalar_value=node.tasks.qsize(),
                    global_step=global_step,
                )
                # Number of tasks in process in nodes
                tensorboard_writer.add_scalar(
                    tag=f"Number_tasks_in_process/Node_{i}",
                    scalar_value=node.number_tasks_in_process,
                    global_step=global_step,
                )
                # Number of completed tasks in nodes
                tensorboard_writer.add_scalar(
                    tag=f"Number_completed_tasks/Node_{i}",
                    scalar_value=node.number_completed_tasks,
                    global_step=global_step,
                )

            # Cloud
            # Load
            tensorboard_writer.add_scalar(
                tag=f"Load/cloud",
                scalar_value=self.layer3.clouds[0].current_load,
                global_step=global_step,
            )
            # Waiting list size in nodes
            tensorboard_writer.add_scalar(
                tag=f"Waiting_list_size/cloud",
                scalar_value=self.layer3.clouds[0].fog_tasks.qsize(),
                global_step=global_step,
            )
            # Number of tasks in process in nodes
            tensorboard_writer.add_scalar(
                tag=f"Number_tasks_in_process/cloud",
                scalar_value=self.layer3.clouds[0].tasks_to_be_processed,
                global_step=global_step,
            )
            # Number of completed tasks in nodes
            tensorboard_writer.add_scalar(
                tag=f"Number_completed_tasks/cloud",
                scalar_value=self.layer3.clouds[0].processed_tasks,
                global_step=global_step,
            )

            # System waiting list size
            tensorboard_writer.add_scalar(
                tag=f"Waiting_list_size/Orchestrator",
                scalar_value=len(self.controller.tasks),
                global_step=global_step,
            )

if __name__ == "__main__":
    sys = System(3,50, 1)
    print(sys.controller.tasks)
    sys.controller.get_task()