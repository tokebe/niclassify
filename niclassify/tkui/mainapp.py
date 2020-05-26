import os
import logging
import threading
import subprocess
import sys

import numpy as np
import pandas as pd
import tkinter as tk
import seaborn as sns

from tkinter import filedialog
from tkinter import ttk
from joblib import dump, load
from xlrd import XLRDError


from .elements import DataPanel, TrainPanel, PredictPanel, StatusBar


def view_open_file(filename):
    filename = os.path.realpath(filename)
    if sys.platform == "win32":
        os.startfile(filename)
    else:
        opener = "open" if sys.platform == "darwin" else "xdg-open"
        subprocess.call([opener, filename])


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
        # parameters for locating and operating on data
        self.data_file = None
        self.sheets = None
        self.sheet = None
        self.raw_data = None
        self.known_column = tk.StringVar()
        self.column_names = None
        # data storage
        self.data = None
        self.metadata = None
        self.data_norm = None
        self.data_imp = None
        self.data_known = None
        self.metadata_known = None
        # classifier & training
        self.classifier = None
        self.report = None
        self.cm = None
        self.cm_img = None
        # predictions
        self.predict = None
        self.predict_prob = None
        self.pairplot = None
        self.pairplot_img = None

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

    def reset_controls(self):
        # reset/disable things before opening new file
        # reset columns/sheets
        self.column_names = {}
        self.data_section.col_select_panel.update_contents()
        self.train_section.known_select["values"] = []
        self.known_column.set("")
        self.sheet = None
        # disable buttons
        self.train_section.reset_enabled()
        self.predict_section.reset_enabled()

    def reset_operations(self):
        self.known_column.set("")
        self.train_section.reset_enabled()
        self.predict_section.reset_enabled()

    def get_data_file(self):
        if self.classifier is not None and self.raw_data is not None:
            if (tk.messagebox.askokcancel(
                title="Existing Classifier",
                message="Opening a new data file will delete unsaved training and output results."
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
        if len(data_file) <= 0:  # if user cancels don't try to open nothing
            self.status_bar.progress.stop()
            self.status_bar.progress.config(mode="determinate")
            return
        # assuming user chooses a proper file:
        self.data_file = data_file
        # update window title to reflect chosen file
        self.parent.title("Random Forest Classifier Tool: {}".format(
            self.data_file.split("/")[-1]))
        # reset things
        self.reset_controls()
        self.raw_data = None

        # handle file import given whether file is excel or text
        if (self.data_file.split("/")[-1].split(".")[-1]
                in ["xlsx", "xlsm", "xlsb",
                    "xltx", "xltm", "xls",
                    "xlt", "xml"]):  # excel; user needs to specify sheet
            # enable sheet selection and get list of sheets for dropdown
            self.data_section.excel_sheet_input.config(state="readonly")
            try:
                self.sheets = list(pd.ExcelFile(self.data_file).sheet_names)
            except (
                    OSError, IOError, KeyError,
                    TypeError, ValueError, XLRDError
            ):  # in case there's something that goes wrong reading the file
                tk.messagebox.showwarning(
                    title="Excel Read Error",
                    message="Unable to read excel file. The file may be \
                        corrupted, or otherwise unable to be read."
                )
                return
            self.data_section.excel_sheet_input["values"] = self.sheets
            tk.messagebox.showinfo(
                title="Excel Sheet Detected",
                message="You will have to specify the sheet to proceed."
            )
            # auto select the first
            # both in case there's only one (makes user's life easier)
            # and to make it more apparent what the user needs to do
            self.data_section.excel_sheet_input.set(self.sheets[0])
            self.get_sheet_cols(None)
        else:  # otherwise it's some sort of text file
            self.data_section.excel_sheet_input.config(
                state=tk.DISABLED)
            try:
                column_names = pd.read_csv(
                    self.data_file,
                    na_values=self.core.NANS,
                    keep_default_na=True,
                    sep=None,
                    nrows=0,
                    engine="python"
                ).columns.values.tolist()
            except (
                    OSError, IOError, KeyError,
                    TypeError, ValueError
            ):  # in case there's something that goes wrong reading the file
                tk.messagebox.showwarning(
                    title="File Read Error",
                    message="Unable to read specified file. The file may be \
                        corrupted, invalid or otherwise unable to be read."
                )
                return
            self.column_names = {x: i for i,
                                 x in enumerate(column_names)}
            self.train_section.known_select["values"] = column_names
            self.data_section.col_select_panel.update_contents()
            # get the raw data and conditionally enable predicitons
            self.get_raw_data()
            self.check_enable_predictions()

    def get_sheet_cols(self, event):
        if self.classifier is not None and self.raw_data is not None:
            if (tk.messagebox.askokcancel(
                title="Existing Classifier",
                message="Selecting a new sheet will delete unsaved training and output results."
            )) is False:
                self.data_section.excel_sheet_input.set(self.sheet)
                return
        # reset controls because new data
        self.reset_controls()
        # get sheet name
        sheet = self.data_section.excel_sheet_input.get()
        # skip reloading if it's already selected
        if sheet == self.sheet:
            return
        # get sheet data
        self.sheet = sheet
        column_names = pd.read_excel(
            self.data_file,
            sheet_name=sheet,
            na_values=self.core.NANS,
            nrows=0,
            keep_default_na=True
        ).columns.values.tolist()
        # get column names and update appropriate
        self.column_names = {x: i for i,
                             x in enumerate(column_names)}
        self.train_section.known_select["values"] = column_names
        self.data_section.col_select_panel.update_contents()
        # prepare raw data and conditionally enable prediction controls
        self.get_raw_data()
        self.check_enable_predictions()

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
        # make sure that data columns are selected
        if len(self.data_section.col_select_panel.sel_contents.get(
                0, tk.END)) <= 0:
            tk.messagebox.showwarning(
                title="No Data Columns Selected",
                message="Please select at least one data column."
            )
            return

        # split data into data and metadata
        self.get_split_data()
        class_column = self.known_column.get()

        # convert class labels to lower if classes are in str format
        if not np.issubdtype(
                self.metadata[class_column].dtype, np.number):
            self.metadata[class_column] = \
                self.metadata[class_column].str.lower()

        # get only known data and metadata
        self.data_known, self.metadata_known = self.core.get_known(
            self.data_norm, self.metadata, class_column)

        # train classifier
        logging.info("training random forest...")
        self.classifier = self.core.train_forest(
            self.data_known,
            self.metadata_known,
            class_column,
            int(self.train_section.n_input.get())
        )
        # prepare training outputs and enable related buttons
        self.train_section.classifier_save.config(state=tk.ACTIVE)
        self.make_report()
        self.make_cm()
        self.save_temp_graph("cm")
        self.train_section.enable_outputs()
        self.check_enable_predictions()

    def save_classifier(self):
        location = tk.filedialog.asksaveasfilename(
            title="Save Classifier",
            initialdir="output/classifiers/",
            defaultextension=".gz",
            filetypes=[("GNU zipped archive", ".gz")]
        )
        dump(self.classifier, location)

    def make_report(self):
        # capture log output for report
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
                if "scaling data..." in line:
                    break

        self.report = "".join(reversed(captured_lines))

    def view_report(self):
        # new window with viewer
        viewer = tk.Toplevel(self)
        viewer.title("Training Report")
        viewer.minsize(500, 500)
        viewer_frame = tk.Frame(viewer)
        viewer_frame.pack(fill=tk.BOTH, expand=True)
        textbox = tk.Text(
            viewer_frame,
            wrap=tk.WORD
        )
        textbox.insert(tk.END, self.report)
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
            command=lambda: viewer.destroy()
        )
        confirm.pack(anchor=tk.E)

    def save_report(self):
        location = tk.filedialog.asksaveasfilename(
            title="Save Classifier Report",
            initialdir="output/",
            defaultextension=".txt",
            filetypes=[("Plain text file", ".txt")]
        )
        if len(location) == 0:
            return
        with open(location, "w") as output:
            output.write(self.report)

    def make_cm(self):
        self.cm = self.core.utilities.make_confm(
            self.classifier,
            self.data_known,
            self.metadata_known,
            self.known_column.get()
        )

    def save_temp_graph(self, graph):
        # save temporary image to open in viewer
        g = self.cm if graph == "cm" else self.pairplot
        g_img = self.cm_img if graph == "cm" else self.pairplot_img
        i = 0
        while os.path.exists("tkui/temp/temp{}.png".format(i)):
            i += 1
        if g_img is None:
            if graph == "cm":
                self.cm_img = "tkui/temp/temp{}.png".format(i)
            else:
                self.pairplot_img = "tkui/temp/temp{}.png".format(i)
            g.savefig(self.cm_img if graph == "cm" else self.pairplot_img)
            g_img = self.cm_img if graph == "cm" else self.pairplot_img

    def view_graph(self, graph):
        g_img = self.cm_img if graph == "cm" else self.pairplot_img
        view_open_file(g_img)

    def save_graph(self, graph):
        location = tk.filedialog.asksaveasfilename(
            title="Save {}".format(
                "Confusion Matrix" if graph == "cm" else "Pairplot"),
            initialdir="output/",
            defaultextension=".txt",
            filetypes=[("Portable Network Graphics image", ".png")]
        )
        if len(location) == 0:
            return
        if graph == "cm":
            self.cm.savefig(location)
        else:
            self.pairplot.savefig(location)

    def check_enable_predictions(self):
        conditions = [
            self.data_file is not None,
            self.raw_data is not None,
            self.classifier is not None
        ]
        print(conditions)
        if all(conditions):
            self.predict_section.prediction_make.config(state=tk.ACTIVE)
            return True
        else:
            return False

    def load_classifier(self):
        if self.classifier is not None and self.report is not None:
            if (tk.messagebox.askokcancel(
                title="Existing Classifier",
                message="Loading a saved classifier will overwrite the saved classifier, ensure it is saved if you wish to keep it."
            )) is False:
                return
        # disable train stuff in the event that a trained clf is overwritten
        self.reset_operations()

        clf_file = filedialog.askopenfilename(
            title="Open Saved Classifier",
            initialdir="output/classifiers/",
            filetypes=[
                ("Classifier archive file", ".gz .joblib .pickle")
            ]
        )
        if len(clf_file) > 0:
            self.classifier = load(clf_file)
            self.check_enable_predictions()

    def make_predictions(self):
        if len(self.data_section.col_select_panel.sel_contents.get(
                0, tk.END)) <= 0:
            tk.messagebox.showwarning(
                title="No Data Columns Selected",
                message="Please select at least one data column."
            )
            return
        self.get_split_data()
        # impute data
        logging.info("imputing data...")
        self.data_imp = self.core.impute_data(self.data_norm)
        # make predictions
        logging.info("predicting unknown class labels...")
        try:  # make sure that the right number of data columns are selected.
            predict = pd.DataFrame(self.classifier.predict(self.data_imp))
        except ValueError:
            tk.messagebox.showerror(
                title="Prediction Failure",
                message="The number of features selected does not match the number of features the classifier was trained on."
            )
            return
        # rename predict column
        predict.rename(columns={predict.columns[0]: "predict"}, inplace=True)
        # get predict probabilities
        predict_prob = pd.DataFrame(
            self.classifier.predict_proba(self.data_imp))
        # rename column
        predict_prob.rename(
            columns={
                predict_prob.columns[i]: "prob. {}".format(c)
                for i, c in enumerate(self.classifier.classes_)},
            inplace=True
        )
        self.predict = predict
        self.predict_prob = predict_prob
        self.make_pairplot()
        self.save_temp_graph("pairplot")
        self.predict_section.enable_outputs()
        self.predict_section.output_save.config(state=tk.ACTIVE)

    def make_pairplot(self):
        df = pd.concat([self.data_imp, self.predict], axis=1)
        self.pairplot = sns.pairplot(
            data=df,
            vars=df.columns[0:self.data.shape[1]],
            hue="predict",
            diag_kind='hist'
        )

    def save_output(self):
        location = tk.filedialog.asksaveasfilename(
            title="Save Output Predicitons",
            initialdir="output/",
            defaultextension=".csv",
            filetypes=[
                ("Comma separated values", ".csv .txt"),
                ("Tab separated values", ".tsv .txt"),
                ("Standard deliniated text file", ".txt"),
                ("All Files", ".*"),
            ]
        )
        if len(location) == 0:
            return
        # save output
        logging.info("saving new output...")
        df = pd.concat(
            [
                self.metadata,
                self.predict,
                self.predict_prob,
                self.data_norm
            ],
            axis=1
        )

        df.to_csv(location, index=False)

    def open_output_folder(self):
        view_open_file("output/")


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
