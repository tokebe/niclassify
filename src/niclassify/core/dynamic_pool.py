import math
from collections.abc import Callable
from multiprocessing import cpu_count, pool
from threading import Semaphore
from typing import Any, Literal, TypeVar

import psutil

CPU_COUNT = cpu_count()
RESOURCES_DEFAULT = math.ceil(
    (psutil.virtual_memory().total - 2e9) / 1e6
)  # leaves 2GB to system

R = TypeVar("R")


class DynamicPool:
    """A configurable pool which will only work on as many tasks as can fit its resources at a time."""

    def __init__(
        self,
        pool_type: Literal["thread", "process"] = "thread",
        pool_size: int = CPU_COUNT,
        resources: int = RESOURCES_DEFAULT,
    ) -> None:
        """Instantiate a DynamicPool instance."""
        if pool_type not in ["thread", "process"]:
            raise ValueError("pool type must be one of (thread, process).")
        self.pool: pool.ThreadPool | pool.Pool = (
            pool.ThreadPool(processes=pool_size)
            if pool_type == "thread"
            else pool.Pool(processes=pool_size)
        )
        self.resources_size: int = resources
        self.resources: Semaphore = Semaphore(resources)
        self.queue: list[pool.AsyncResult[Any]] = []
        self.failed: bool = False

    def add_task(
        self, task: Callable[..., R], cost: int = 1, *args: Any, **kwargs: Any
    ) -> pool.AsyncResult[R]:
        """Schedule a task to be added to the pool as soon as resources are available."""
        # block until enough resources are available
        if self.failed:
            raise RuntimeError("A task in the pool failed.")
        for _ in range(min(cost, self.resources_size)):
            self.resources.acquire()

        return self.pool.apply_async(
            task,
            args,
            kwargs,
            callback=lambda x: self.task_complete(cost),
            error_callback=lambda x: self.task_failed(),
        )

    def task_complete(self, cost: int) -> None:
        """Release the resources used by a task.

        Primarily used as a callback.
        """
        self.resources.release(min(cost, self.resources_size))

    # FIX: doesn't let pool.map() call stop blocking
    def task_failed(self) -> None:
        """Cancel all tasks and close the pool."""
        self.failed = True
        self.pool.terminate()
        raise RuntimeError("A task in the pool failed.")

    def map(
        self, tasks: list[tuple[Callable[..., R], int, tuple[Any, ...], dict[str, Any]]]
    ) -> list[R]:
        """Schedule a number of tasks and await their completion."""
        for func, cost, args, kwargs in tasks:
            self.queue.append(self.add_task(func, cost, *args, *kwargs))
        return [result.get() for result in self.queue]

    def close(self) -> None:
        self.pool.close()
        self.pool.join()
