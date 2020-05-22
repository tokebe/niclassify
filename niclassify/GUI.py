import tkinter as tk
from tkinter import ttk
import core
import tkui
import logging
import matplotlib.pyplot as plt

# TODO enable make predictions on data load and give warning if attempting
# to train without having selected columns
# TODO fix pairplot viewer
# TODO add status updates to everything
# TODO add threading to basically everything
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

    root = MainRoot()

    def graceful_exit():
        plt.close("all")
        root.quit()
        root.destroy()

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
