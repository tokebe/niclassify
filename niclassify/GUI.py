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
# TODO there appears to be a species delimitation problem
# some files getting to create_measures don't having any groupings
# some delims fail because of max windows path length error
# this leaves a non-zero size file with no actual delimitations
# > need to investigate why/when it's failing
# it's failing from max windows path length in bPTP, outside of my control
# this is theoretically not 100% avoidable and so will have to be compensated
# for somehow.
# TODO check minor fixes/features in datatool.py
# just count up on finding a dupe
# Will have to ask Jeremy about this
# TODO automate trimming sequences
# TODO check for agreement of status in delimited species
# TODO add new tool for merging prepared data
# TODO test running on new system without running setup of bPTP
# You'll probably have to make a contained python env with everything installed

# LATER
# TODO generate a requirements.txt
# TODO options to split by lower taxon levels
# TODO File import history for merging?
# TODO use pGMYC

# LATERER
# TODO attempt to make an .exe file
# TODO go back and implement TUI with support for linux+mac if possible
# this will require minor changes to some utility functions for system checking
# TODO create secondary scripts for checking what caused a failure

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
