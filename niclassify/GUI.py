"""The main GUI script controlling the GUI version of niclassify.

Technically extensible by subclassing and copying the main() function.
If you have to do that, I'm sorry. It probably won't be fun.
"""
import matplotlib
import threading

import tkinter as tk

import matplotlib.pyplot as plt

from tkinter import ttk

from core import utilities
from core.StandardProgram import StandardProgram
from core.classifiers import RandomForestAC
from tkui.clftool import ClassifierTool

matplotlib.use('Agg')  # this makes threading not break

# NOW
# TODO make a new module for getting title/message for dialogs, make json
# can just be functions replacing the message functions, will make things easy
# TODO look for everywhere a file is interacted with and add failure warnings
# TODO do the same for web requests
# TODO see if you can increase timeout time for requests
# TODO misc. todo's throughout files

# LATER
# TODO implement matrix and measures generation
# TODO implement backend for PTP

# FIXES & MINOR FEATURES
# TODO go searching for problems, unexpected behaviors, etc


def main():
    """Run the GUI."""
    utilities.assure_path()

    root = tk.Tk()

    root.style = ttk.Style()
    app = ClassifierTool(root, StandardProgram, RandomForestAC, utilities)
    root.update()
    root.minsize(root.winfo_width(), root.winfo_height())

    def graceful_exit():
        """
        Exit the program gracefully.

        This includes cleaning tempfiles and closing any processes.
        """
        try:
            app.tempdir.cleanup()
        except PermissionError:
            None
        print(threading.active_count())
        plt.close("all")
        root.quit()
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", graceful_exit)

    root.mainloop()


if __name__ == "__main__":
    main()
