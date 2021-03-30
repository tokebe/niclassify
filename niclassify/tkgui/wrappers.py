"""Wrapping decorator functions used by the program."""

import os
import threading
import traceback

from tkinter import messagebox


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


def report_uncaught(func):
    """Execute a function, catching and logging any uncaught exceptions.

    Calls back to class instance to handle reporting to user.

    Args:
        func (Function): A function to be run.
    """

    def wrapper(self, *args, **kwargs):
        try:
            func(self, *args, **kwargs)
        except:
            error_trace = traceback.format_exc()
            logfile = os.path.join(
                self.util.USER_PATH,
                "logs/error_traceback.log"
            )
            open(logfile, "w").close()

            with open(logfile, "w") as error_log:
                error_log.write(error_trace)

            self.uncaught_exception(error_trace, logfile)
    return wrapper
