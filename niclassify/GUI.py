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
# TODO use Jeremy's test files. Ensure that undelimited species are ignored.
# TODO add checks after each stage of data preparation to ensure it worked
# add appropriate warnings for failures, for instance alignment failures
# when generating measures

# LATER
# TODO ask about joining species status by delimitation rather than noted name
# this could give us significantly more samples with a status, but is at the
# mercy of the delimitation's performance
# overall it would be rather difficult to measure the effects on 'true' BA
# oversplitting would actually be better in this case, as fewer groups would
# contain multiple species, though oversplitting also hurts our ability to get
# intraspecies similarity information, which is rather key to have
# that said, some information might be regained in comparison between
# min distance and median distance, etc - for that reason
# we may wish to favor oversplitting rather than overlumping

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
