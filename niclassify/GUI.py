import tkinter as tk
from tkinter import ttk
import core
import tkui
import logging
import matplotlib.pyplot as plt
import shutil

# TODO fix error in sheet selection:
# non-existent sheet sheet1 listed
# XLRDError: No sheet names <''> when selecting any sheet
# likely has to do with the use of a stringvar
# TODO continue refactoring from get_sheet_cols()
# TODO add dialogs asking if user wants to lose current classifier/etc
# triggered when attempting to load new data/change sheet selection
# TODO refactor and reorganize everything
# TODO while you're doing that, make sure every function enables/disables all
# buttons that it should appropriately
# TODO check for errors while predicting and warn user if they don't have the
# right columns selected
# TODO add threading to basically everything
# TODO add progress bars while doing threading
# TODO add status updates to everything while doing threading
# TODO add logging to everything
# TODO add popup notifications for things such as completing training, etc


class MainRoot(tk.Tk):
    None

    # def get_data(self):
    #     print("welp")

    # def get_data(self):

    # def train_classifier(self):

    #     # convert class labels to lower if classes are in str format
    #     if not np.issubdtype(metadata[class_column].dtype, np.number):
    #         metadata[class_column] = metadata[class_column].str.lower()

    #     # get only known data and metadata
    #     data_known, metadata_known = core.get_known(
    #         data_norm, metadata, class_column)

    #     # train classifier
    #     logging.info("training random forest...")
    #     forest = core.train_forest(data_known, metadata_known,
    #                                class_column, multirun)


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
