import tkinter as tk
from tkinter import ttk
import core
import tkui
import logging
import matplotlib.pyplot as plt
import shutil

# TODO TOP PRIORITY: RENAME ALL DATA TO FEATURES

# TODO grab your dad to push buttons until something breaks
# TODO after you've done that see if there are ways to make this whole thing
# more professional and easily maintainable:
# - methods that should be combined/deleted/decombined/made private/properties
# - things that should be objects/objects that should be re-parented
# - look at a guide for making a good python application or something I guess
# - prefer to not keep dataframes and do inplace=True when possible
# - try to make everything more object oriented instead of proceedural
# - change most functions to take arguments and return things rather than
# relying on class variables
# TODO try to optimize idle RAM usage
# this will be a huge task and will definitely involve rewriting core
# you'll have to try and keep as little data stored as possible
# esp. the actual data files.
# TODO add threading to basically everything
# TODO add progress bars while doing threading
# TODO add status updates to everything while doing threading
# TODO add logging to everything
# TODO add popup notifications for things such as completing training, etc
# TODO possible thing to do later: automatic wizard creation


class MainRoot(tk.Tk):
    None


def main():
    core.assure_path()

    root = MainRoot()

    def graceful_exit():
        plt.close("all")
        root.quit()
        root.destroy()
        shutil.rmtree('tkui/temp')

    root.protocol("WM_DELETE_WINDOW", graceful_exit)
    root.style = ttk.Style()
    root.style.theme_use("winnative")
    # print(root.style.theme_names())
    app = tkui.MainApp(root)
    app.core = core
    root.update()
    root.minsize(root.winfo_width(), root.winfo_height())
    root.mainloop()


if __name__ == "__main__":
    main()
