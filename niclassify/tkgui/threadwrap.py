"""A wrapper for threading a function."""

import threading


def threaded(func):
    """
    Thread a given function.

    Threads are made as daemons so they are closed when the main thread closes.

    Args:
        func (func): A function to be threaded.
    """
    def wrapper(*args, **kwargs):
        thread = threading.Thread(
            target=func, args=args, kwargs=kwargs, daemon=True)
        thread.start()
        return thread
    return wrapper
