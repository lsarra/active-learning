from multiprocessing.managers import RemoteError
from sqlite3 import Time
import numpy as np
from tqdm import tqdm
import ipyparallel as ipp
import IPython
import time
import logging
import asyncio
import activelearning as al
from activelearning.config import cfg as cfg
import nest_asyncio

nest_asyncio.apply()


class Parallel:
    """
    Helps handling ipyparallel instances.
    """
    def __init__(self, CLUSTER_ID, n_nodes=5, init_variables=None, init_callback=None):
        """

        Args:
            CLUSTER_ID (str): name of the ipyparallel instance to connect to
            init_variables (dict, optional): Variables to send to the remote nodes. Defaults to None.
            init_callback (func, optional): Function to run after initialization to load required libraries. Defaults to None.

        """

        self.cluster = ipp.Cluster(
            profile=al.config.cfg.PARALLEL.PROFILE_ID,
            cluster_id=CLUSTER_ID,
            n=n_nodes,
            engine_timeout=7200,
        )
        self.CLUSTER_ID = CLUSTER_ID
        # self.stop()
        try:
            self.client = asyncio.run(self.cluster.start_and_connect())
        except TimeoutError:
            al.logger.critical("Cannot allocate workers.")
            raise TimeoutError
        self.dview = self.client[:]

        # Change to current directory
        import sys
        import os

        library_path = os.path.abspath(f"{sys.path[0]}")
        self.execute("import sys, os")
        self.execute(f"sys.path.insert(0, ('{library_path}'))")
        self.execute(f"os.chdir('{library_path}')")

        self.execute(f"import activelearning as al")
        self.execute(f"from activelearning.config import cfg as cfg")

        config_path = (
            f"{al.config.Directories().PATH_OUTPUT_CONFIGS}/{cfg.PATH.SUBLABEL}.yaml"
        )
        self.execute(f"al.utils.init('{config_path}')")

        # Push initial variables to the remote nodes
        # Execute initialization commands
        self.push(init_variables, block=True)
        if init_callback is not None:
            self.dview.push(dict([[init_callback.__name__, init_callback]]), block=True)
            self.dview[f"{init_callback.__name__}()"]

        self.execute("cfg.PARALLEL.LOG=True")
        al.logger.info(
            f"Cluster {al.config.cfg.PARALLEL.PROFILE_ID}/{CLUSTER_ID} has been initialized with {len(self)} nodes."
        )

    def log(self, message, log_level):
        self.dview[f'al.logger.log({log_level},"{message}")']

    def stop(self):
        asyncio.run(self.cluster.stop_cluster())
        al.logger.info(
            f"Cluster {al.config.cfg.PARALLEL.PROFILE_ID}/{self.CLUSTER_ID} has been stopped."
        )

    def push(self, variable_dict, block=True):
        if variable_dict is not None:
            return self.dview.push(variable_dict, block=block)

    def execute(self, execute_commands, block=True):
        self.dview.execute(execute_commands, block=block)

    def assign_task(
        self,
        function_name_and_args,
        aggregator_func=lambda x: x,
        show_progress=False,
        title=None,
    ):
        al.logger.info(f"Requested parallel task {function_name_and_args}")

        execute_string = f"result = {function_name_and_args}"
        if show_progress:
            self._track_task_progress(self.dview.execute(execute_string), title)
        else:
            self.dview.execute(execute_string, block=True)
        return aggregator_func(self["result"])

    def _track_task_progress(self, task, title=None):
        al.logger.info(f"Parallel running {title}")

        while not task.ready():
            IPython.display.clear_output(wait=True)
            if title is not None:
                print(title)
                print("-------------------------------------")
            fracs = []
            stdout = [
                self.client.metadata[worker_id]["stdout"] for worker_id in task.msg_ids
            ]
            try:
                for i in range(len(stdout)):
                    try:
                        last_line = stdout[i].split("\n")[-2]
                        last_vals = np.array(last_line.split(" "), dtype=np.int)

                        fracs.append(last_vals[0] / last_vals[1])
                        if not np.isnan(last_vals).any():
                            tqdm(
                                initial=last_vals[0],
                                total=last_vals[1],
                                desc=f"worker {i:02d}",
                                position=0,
                                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt}",
                            )

                    except IndexError as e:
                        ...

                if len(fracs) > 0:
                    print("\n")
                    tqdm(
                        initial=np.mean(fracs),
                        total=1,
                        desc=f"overall",
                        position=0,
                        bar_format="{l_bar}{bar}| ",
                    )
                else:
                    print("Waiting for response from workers...")

            except ValueError:
                ...

            time.sleep(1)
        IPython.display.clear_output()
        print("Execution finished.")

        for e in task.error:
            if e is not None:
                al.logger.error("Unfortunately there was an error:")
                raise RemoteError(e)

    def __len__(self):
        return len(self.dview)

    def __getitem__(self, str):
        return self.dview[str]


def init_active_learner(parallel, last_save_path):
    parallel.execute(
        f"system, learner, advisor = al.utils.load_all('{last_save_path}')"
    )


def log_progress(n, n_max):
    """Helper function to show progress bar interactively."""
    if n % 50 == 0 and al.config.cfg.PARALLEL.LOG:
        print(n, n_max)
