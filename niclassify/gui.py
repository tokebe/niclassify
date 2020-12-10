"""The main GUI script controlling the GUI version of niclassify.

Technically extensible by subclassing and copying the main() function.
If you have to do that, I'm sorry. It probably won't be fun.
"""
import sys
import multiprocessing
if getattr(sys, 'frozen', False):  # required for pyinstaller mp
    multiprocessing.set_start_method('forkserver', force=True)
    multiprocessing.freeze_support()

import matplotlib
import threading

import tkinter as tk

import matplotlib.pyplot as plt

from tkinter import ttk

from .core import utilities
from .core.StandardProgram import StandardProgram
from .core.classifiers import RandomForestAC
from .tkgui.clftool import ClassifierTool
matplotlib.use('Agg')  # this makes threading not break

def main():
    """Run the GUI."""

    root = tk.Tk()

    root.style = ttk.Style()
    app = ClassifierTool(root, StandardProgram, RandomForestAC, utilities)
    root.update()
    root.minsize(root.winfo_width(), root.winfo_height())
    root.iconbitmap(utilities.PROGRAM_ICON)

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
