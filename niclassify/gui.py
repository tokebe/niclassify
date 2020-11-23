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
from tkgui.clftool import ClassifierTool


matplotlib.use('Agg')  # this makes threading not break

# NOW
# TODO metric (consider/read about/test out-of-bag and F1)

# LATER
# TODO properly implement logging for most steps
# TODO implement commandline version, use for tests
# TODO File import history for merging?
# TODO use pGMYC?

# LATERER
# TODO add new tool for merging prepared data
# TODO support mac/linux
# filedialog asking for paths to required executables (rscript, etc)
# config file to keep these paths
# TODO attempt to make an .exe file
# TODO go back and implement TUI with support for linux+mac if possible
# this will require minor changes to some utility functions for system checking
# TODO create secondary scripts for checking what caused a failure?

# FIXES & MINOR FEATURES
# TODO go searching for problems, unexpected behaviors, etc
# TODO minor issues:
# main progressbar freezes on certain errors, such as not enough labels


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
