"""A wrapper for threading a function."""

import threading


def threaded(func):
    """Thread a given function.

    Args:
        func (func): A function to be threaded.
    """
    def wrapper(*args, **kwargs):
        thread = threading.Thread(target=func, args=args, kwargs=kwargs)
        thread.start()
        return thread
    return wrapper
