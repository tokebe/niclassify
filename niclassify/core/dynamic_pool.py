from multiprocessing import cpu_count, pool
from threading import Semaphore
import psutil
from typing import Callable, Tuple, Dict, Literal
import math


class DynamicPool:
    def __init__(
        self,
        pool_type: Literal["thread", "process"] = "thread",
        pool_size: int = cpu_count(),
        resources: int = math.ceil((psutil.virtual_memory().total - 2e9) / 1e6),  # leaves 2GB to system
    ) -> None:
        if pool_type not in ["thread", "process"]:
            raise ValueError("pool type must be one of (thread, process).")
        self.pool = (
            pool.ThreadPool(processes=pool_size)
            if pool_type == "thread"
            else pool.Pool(processes=pool_size)
        )
        self.resources_size = resources
        self.resources = Semaphore(resources)

    def add_task(self, task: Callable, cost: int = 1, *args, **kwargs):
        # block until enough resources are available
        for _ in range(min(cost, self.resources_size)):
            self.resources.acquire()

        return self.pool.apply_async(
            task, args, kwargs, lambda x: self.task_complete(cost)
        )

    def task_complete(self, cost: int):
        self.resources.release(min(cost, self.resources_size))

    def map(self, tasks: Tuple[Callable, int, Tuple[any, ...], Dict[str, any]]):
        queue = []
        for func, cost, args, kwargs in [
            (list(task) + [None] * 2)[:4] for task in tasks
        ]:
            args = () if args is None else args
            kwargs = {} if kwargs is None else {}
            queue.append(self.add_task(func, cost, *args, *kwargs))
        return [result.get() for result in queue]
