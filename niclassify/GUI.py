import logging
import os
import shutil
import subprocess
import sys
import threading

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tkinter as tk


from joblib import dump, load
from tkinter import filedialog
from tkinter import ttk
from xlrd import XLRDError

# TODO remove the leading . before testing
from core import utilities
from core.StandardProgram import StandardProgram
from core.classifiers import RandomForestAC
from tkui.elements import DataPanel, TrainPanel, PredictPanel, StatusBar
from tkui.utilities import view_open_file


class MainApp(tk.Frame):
    def __init__(self, parent, *args, **kwargs):
        tk.Frame.__init__(self, parent, *args, **kwargs)

        self.parent = parent

        # set up program utilities
        self.sp = StandardProgram(RandomForestAC())
        self.sp.boilerplate()

        # TODO decide if there are any values you need to store
        # preferably as temp files

        # set up elements of main window
        parent.title("Random Forest Classifier Tool")
        # panel for most controls, so statusbar stays on bottom
        self.panels = tk.Frame(
            self.parent
        )
        self.panels.pack(fill=tk.BOTH, expand=True)
        # data import/column selection section
        self.data_section = DataPanel(
            self.panels,
            self,
            text="Data",
            labelanchor=tk.N
        )
        self.data_section.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        # section holds operational controls
        self.operate_section = tk.Frame(
            self.panels
        )
        self.operate_section.pack(side=tk.RIGHT, anchor=tk.N)
        # training controls
        self.train_section = TrainPanel(
            self.operate_section,
            self,
            text="Train",
            labelanchor=tk.N
        )
        self.train_section.pack(fill=tk.X)
        # prediction controls
        self.predict_section = PredictPanel(
            self.operate_section,
            self,
            text="Predict",
            labelanchor=tk.N
        )
        self.predict_section.pack(fill=tk.X)
        # statusbar
        self.status_bar = StatusBar(
            self.parent,
            self
        )
        self.status_bar.pack(fill=tk.X)

    def check_enable_predictions(self):
        conditions = [
            self.raw_data is not None,
            self.clf is not None
        ]
        print(conditions)
        if all(conditions):
            self.predict_section.prediction_make.config(state=tk.ACTIVE)
            return True
        else:
            return False

    def enable_train(self, event):
        self.train_section.train_button.config(state=tk.ACTIVE)

    def get_data_file(self):
        # check if user is overwriting and ask if they're ok with it
        if self.clf is not None and self.raw_data is not None:
            if (tk.messagebox.askokcancel(
                title="Existing Classifier",
                message="Opening a new data file will delete unsaved classifier and results."
            )) is False:
                return

        # prompt user for file
        data_file = filedialog.askopenfilename(
            title="Open Data File",
            initialdir="data/",
            filetypes=[
                ("All Files", ".*"),
                ("Excel file", ".xlsx .xlsm .xlsb .xltx .xltm .xls .xlt .xml"),
                ("Comma separated values", ".csv .txt"),
                ("Tab separated values", ".tsv .txt"),
                ("Standard deliniated text file", ".txt")
            ]
        )
        # if user cancels don't try to open nothing
        if len(data_file) <= 0:
            self.status_bar.progress.stop()
            self.status_bar.progress.config(mode="determinate")
            return

        # assuming user chooses a proper file:

        self.sp.data_file = data_file

        # update window title to reflect chosen file
        self.parent.title(
            "Random Forest Classifier Tool: {}".format(
                os.path.basename(data_file))
        )

        # reset things
        self.reset_controls()
        self.raw_data = None

        # handle file import given whether file is excel or text
        if (  # some sort of excel file
            os.path.splitext(data_file)[1]
            in ["xlsx", "xlsm", "xlsb", "xltx", "xltm", "xls", "xlt", "xml"]
        ):
            # enable sheet selection and get list of sheets for dropdown
            self.data_section.excel_sheet_input.config(state="readonly")

            # in case there's something that goes wrong reading the file
            try:
                sheets = list(pd.ExcelFile(data_file).sheet_names)
            except (
                    OSError, IOError, KeyError,
                    TypeError, ValueError, XLRDError
            ):
                tk.messagebox.showwarning(
                    title="Excel Read Error",
                    message="Unable to read excel file. The file may be \
                        corrupted, or otherwise unable to be read."
                )
                return

            # update sheet options for dropdown and prompt user to select one
            self.data_section.excel_sheet_input["values"] = sheets
            tk.messagebox.showinfo(
                title="Excel Sheet Detected",
                message="You will have to specify the sheet to proceed."
            )

            # auto select the first
            # both in case there's only one (makes user's life easier)
            # and to make it more apparent what the user needs to do
            self.data_section.excel_sheet_input.set(sheets[0])
            self.sp.excel_sheet = sheets[0]
            self.get_sheet_cols(None)

        else:  # otherwise it's some sort of text file
            # disable sheet selection dropdown
            self.data_section.excel_sheet_input.config(
                state=tk.DISABLED)

            # in case there's a read or parse error of some kind
            try:
                column_names = pd.read_csv(
                    data_file,
                    na_values=utilities.NANS,
                    keep_default_na=True,
                    sep=None,
                    nrows=0,
                    engine="python"
                ).columns.values.tolist()
            except (
                    OSError, IOError, KeyError,
                    TypeError, ValueError
            ):
                tk.messagebox.showwarning(
                    title="File Read Error",
                    message="Unable to read specified file. The file may be \
                        corrupted, invalid or otherwise unable to be read."
                )
                return

            # update column selections for known class labels
            self.train_section.known_select["values"] = column_names

            # update the column select panel
            colnames_dict = {x: i for i, x in enumerate(column_names)}
            self.data_section.col_select_panel.update_contents(colnames_dict)

            # conditionally enable predicting
            self.check_enable_predictions()

    def get_selected_cols(self):
        # get selection
        selected_cols = list(
            self.data_section.col_select_panel.sel_contents.get(0, tk.END))

        # put in StandardProgram
        self.sp.feature_cols = selected_cols

        # return to be useful
        return selected_cols

    # TODO go through existing methods and make sure no unexpected class vars
    # are being defined
    # TODO implement getting sheet and columns from sheet
    # also send sheet selection to self.sp


def main():
    utilities.assure_path()

    root = tk.Tk()

    def graceful_exit():
        plt.close("all")
        root.quit()
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", graceful_exit)
    root.style = ttk.Style()
    root.style.theme_use("winnative")
    # print(root.style.theme_names())
    app = MainApp(root)
    root.update()
    root.minsize(root.winfo_width(), root.winfo_height())
    root.mainloop()


if __name__ == "__main__":
    main()
