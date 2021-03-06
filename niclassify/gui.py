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
matplotlib.use('agg')  # this makes threading not break on exit


def main():
    """Run the GUI."""

    root = tk.Tk(className="NIClassify")

    root.style = ttk.Style()
    app = ClassifierTool(root, StandardProgram, RandomForestAC, utilities)
    root.update()
    root.minsize(root.winfo_width(), root.winfo_height())
    root.iconbitmap(utilities.PROGRAM_ICON)
    if utilities.PLATFORM == 'Linux':
        img = tk.Image("photo", utilities.PLATFORM[1:-4] + ".png")
        root.tk.call('wm', 'iconphoto', root._w, img)

    def graceful_exit():
        """
        Exit the program gracefully.

        This includes cleaning tempfiles and closing any processes.
        """
        try:
            app.tempdir.cleanup()
        except PermissionError:
            None
        plt.close("all")
        root.quit()
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", graceful_exit)

    root.mainloop()


if __name__ == "__main__":
    main()
