import os
import logging
import threading

import numpy as np
import pandas as pd
import tkinter as tk

from joblib import dump
from tkinter import ttk
from tkinter import filedialog
from .elements import DataPanel, TrainPanel, PredictPanel, StatusBar


class MainApp(tk.Frame):
    def __init__(self, parent, *args, **kwargs):
        tk.Frame.__init__(self, parent, *args, **kwargs)

        # set log filename
        i = 0
        while os.path.exists("output/logs/rf-auto{}.log".format(i)):
            i += 1
        self.logname = "output/logs/rf-auto{}.log".format(i)

        # set up logging
        logging.basicConfig(
            level=logging.INFO,
            format="%(message)s",
            handlers=[
                logging.FileHandler(self.logname),
                logging.StreamHandler()
            ]
        )

        self.parent = parent
        self.core = None

        # pre-defined variables to store important stuff
        self.data_file = None
        self.sheets = None
        self.sheet = tk.StringVar()
        self.raw_data = None
        self.known_column = tk.StringVar()
        self.column_names = None
        self.data = None
        self.metadata = None
        self.data_norm = None
        self.classifier = None

        self.panels = tk.Frame(
            self.parent
        )
        self.panels.pack(fill=tk.BOTH, expand=True)

        parent.title("Random Forest Classifier Tool")

        self.data_section = DataPanel(
            self.panels,
            self,
            text="Data",
            labelanchor=tk.N)
        self.data_section.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.operate_section = tk.Frame(
            self.panels)
        self.operate_section.pack(side=tk.RIGHT, anchor=tk.N)

        self.train_section = TrainPanel(
            self.operate_section,
            self,
            text="Train",
            labelanchor=tk.N)
        self.train_section.pack(fill=tk.X)

        self.predict_section = PredictPanel(
            self.operate_section,
            self,
            text="Predict",
            labelanchor=tk.N)
        self.predict_section.pack(fill=tk.X)

        self.status_bar = StatusBar(
            self.parent,
            self
        )
        self.status_bar.pack(fill=tk.X)

    def get_data_file(self):
        # TODO add status update
        self.status_bar.progress.config(mode="indeterminate")
        self.status_bar.progress.start(10)
        # TODO learn how to thread or whatever so the progress bar moves
        self.core.assure_path()
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
        if data_file is None:
            self.status_bar.progress.stop()
            self.status_bar.progress.config(mode="determinate")
            return
        self.data_file = data_file
        self.parent.title("Random Forest Classifier Tool: {}".format(
            self.data_file.split("/")[-1]
        ))
        self.column_names = {}
        self.data_section.col_select_panel.update_contents()
        self.train_section.known_select["values"] = []
        self.known_column.set("")
        self.train_section.train_button.config(state=tk.DISABLED)
        self.sheet.set("")

        if self.data_file.split("/")[-1][-5:] == ".xlsx":
            self.data_section.excel_sheet_input.config(state="readonly")
            self.sheets = list(pd.read_excel(
                self.data_file, None).keys())
            self.data_section.excel_sheet_input["values"] = self.sheets
            tk.messagebox.showinfo(
                title="Excel Sheet Detected",
                message="You will have to specify the sheet to proceed."
            )
            self.data_section.excel_sheet_input.set(self.sheets[0])
        else:
            self.data_section.excel_sheet_input.config(
                state=tk.DISABLED)
            column_names = pd.read_csv(
                self.data_file,
                na_values=self.core.NANS,
                keep_default_na=True,
                sep=None,
                engine="python"
            ).columns.values.tolist()
            self.column_names = {x: i for i,
                                 x in enumerate(column_names)}
            self.train_section.known_select["values"] = column_names

            self.get_raw_data()

        self.data_section.col_select_panel.update_contents()
        self.status_bar.progress.stop()
        self.status_bar.progress.config(mode="determinate")

    def get_sheet_cols(self, event):
        # TODO add progress bar animation, status update
        self.known_column.set("")
        sheet = self.data_section.excel_sheet_input.get()
        self.sheet.set(sheet)
        column_names = pd.read_excel(
            self.data_file,
            sheet_name=sheet,
            na_values=self.parent.nans(),
            keep_default_na=True
        ).columns.values.tolist()
        self.column_names = {x: i for i,
                             x in enumerate(column_names)}
        self.train_section.known_select["values"] = column_names
        self.data_section.col_select_panel.update_contents()
        self.get_raw_data()

    def get_raw_data(self):
        # get raw data
        raw_data = self.core.get_data(self.data_file, self.sheet)

        # replace argument-added nans
        if self.core.NANS is not None:
            raw_data.replace({val: np.nan for val in self.core.NANS})

        self.raw_data = raw_data

    def get_split_data(self):
        # get currently selected columns
        data_cols = list(self.data_section.col_select_panel.sel_contents.get(
            0, tk.END))

        # split data into feature data and metadata
        self.data = self.raw_data[data_cols]
        self.metadata = self.raw_data.drop(data_cols, axis=1)

        # scale data
        self.data_norm = self.core.scale_data(self.data)

    def enable_train(self, event):
        self.train_section.train_button.config(state=tk.ACTIVE)

    def train_classifier(self):
        self.get_split_data()
        class_column = self.known_column.get()

        # convert class labels to lower if classes are in str format
        if not np.issubdtype(
                self.metadata[class_column].dtype, np.number):
            self.metadata[class_column] = \
                self.metadata[class_column].str.lower()

        # get only known data and metadata
        data_known, metadata_known = self.core.get_known(
            self.data_norm, self.metadata, class_column)

        # train classifier
        self.classifier = self.core.train_forest(
            data_known,
            metadata_known,
            class_column,
            int(self.train_section.n_input.get())
        )
        self.train_section.classifier_save.config(state=tk.ACTIVE)
        self.train_section.enable_outputs()

    def save_classifier(self):
        location = tk.filedialog.asksaveasfilename(
            title="Save Classifier",
            initialdir="output/classifiers/",
            defaultextension=".gz",
            filetypes=[("GNU zipped archive", ".gz")]
        )
        dump(self.classifier, location)

    def make_report(self):
        break_counter = 0
        capture = False
        captured_lines = []
        with open(self.logname, "r") as log:
            loglines = log.readlines()

        for line in reversed(loglines):
            if line == "---\n":
                captured_lines.append(line)
                capture = True
            elif capture:
                captured_lines.append(line)
                if "out-of-bag score:" in line:
                    break

        return "".join(reversed(captured_lines))

    def view_report(self):
        viewer = tk.Toplevel(self)
        viewer.minsize(500, 500)
        viewer_frame = tk.Frame(viewer)
        viewer_frame.pack(fill=tk.BOTH, expand=True)
        textbox = tk.Text(
            viewer_frame,
            wrap=tk.WORD
        )
        textbox.insert(tk.END, self.make_report())
        textbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        textscroll = tk.Scrollbar(
            viewer_frame,
            orient=tk.VERTICAL,
            command=textbox.yview()
        )
        textscroll.pack(side=tk.RIGHT, fill=tk.Y)
        textbox.config(yscrollcommand=textscroll.set)
        confirm = tk.Button(
            viewer,
            text="Ok",
            pady=5,
            padx=5,
            command=lambda: viewer.destroy())
        confirm.pack(anchor=tk.W)


def main():

    root = tk.Tk()
    root.style = ttk.Style()
    root.style.theme_use("winnative")
    # print(root.style.theme_names())
    MainApp(root)
    root.update()
    root.minsize(root.winfo_width(), root.winfo_height())
    root.mainloop()


if __name__ == "__main__":
    main()
