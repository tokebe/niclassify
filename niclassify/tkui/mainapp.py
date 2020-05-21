from .elements import DataPanel, TrainPanel, PredictPanel, StatusBar
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from tkinter import filedialog
from tkinter import ttk
from joblib import dump, load
import os
import logging
import threading

import numpy as np
import pandas as pd
import tkinter as tk
import matplotlib as plt
import seaborn as sns
plt.use("TkAgg")


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
        self.data_known = None
        self.metadata_known = None
        self.classifier = None
        self.report = None
        self.cm = None
        self.predict = None
        self.predict_prob = None
        self.pairplot = None

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
        if len(data_file) <= 0:
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
        self.predict_section.prediction_make.config(state=tk.DISABLED)
        self.train_section.classifier_save.config(state=tk.DISABLED)
        self.train_section.disable_outputs()
        self.predict_section.disable_outputs()

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
        self.data_known, self.metadata_known = self.core.get_known(
            self.data_norm, self.metadata, class_column)

        # train classifier
        self.classifier = self.core.train_forest(
            self.data_known,
            self.metadata_known,
            class_column,
            int(self.train_section.n_input.get())
        )
        self.train_section.classifier_save.config(state=tk.ACTIVE)
        self.make_report()
        self.make_cm()
        self.train_section.enable_outputs()
        self.predict_section.prediction_make.config(state=tk.ACTIVE)

    def save_classifier(self):
        location = tk.filedialog.asksaveasfilename(
            title="Save Classifier",
            initialdir="output/classifiers/",
            defaultextension=".gz",
            filetypes=[("GNU zipped archive", ".gz")]
        )
        dump(self.classifier, location)

    def make_report(self):
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

        self.report = "".join(reversed(captured_lines))

    def view_report(self):
        viewer = tk.Toplevel(self)
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
            command=lambda: viewer.destroy())
        confirm.pack(anchor=tk.E)

    def save_report(self):
        location = tk.filedialog.asksaveasfilename(
            title="Save Classifier Report",
            initialdir="output/",
            defaultextension=".txt",
            filetypes=[("Plain text file", ".txt")]
        )
        with open(location, "w") as output:
            output.write(self.report)

    def make_cm(self):
        self.cm = self.core.utilities.make_confm(
            self.classifier,
            self.data_known,
            self.metadata_known,
            self.known_column.get()
        )

    def view_graph(self, graph):
        viewer = tk.Toplevel(self)
        viewer.minsize(500, 500)
        canvas = FigureCanvasTkAgg(
            self.cm if graph == "cm" else self.pairplot,
            viewer
        )
        canvas.draw()
        canvas.get_tk_widget().pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True)

        toolbar = NavigationToolbar2Tk(canvas, viewer)
        toolbar.update()
        canvas._tkcanvas.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

    def save_graph(self, graph):
        location = tk.filedialog.asksaveasfilename(
            title="Save {}".format(
                "Confusion Matrix" if graph == "cm" else "Pairplot"),
            initialdir="output/",
            defaultextension=".txt",
            filetypes=[("Portable Network Graphics image", ".png")]
        )
        if graph == "cm":
            self.cm.savefig(location)
        else:
            self.pairplot.savefig(location)

    def check_enable_predictions(self):
        conditions = [
            self.data_file is not None
        ]
        if False not in conditions:
            self.predict_section.prediction_make.config(state=tk.ACTIVE)

    def load_classifier(self):
        # TODO add error checking for all file loading dialogs
        clf_file = filedialog.askopenfilename(
            title="Open Saved Classifier",
            initialdir="output/classifiers/",
            filetypes=[
                ("Classifier archive file", ".gz .joblib .pickle")
            ]
        )
        if len(clf_file) > 0:
            self.classifier = load(clf_file)
            self.check_enable_predictions

    def make_predictions(self):
        # TODO  make sure transformations are not done multiple times
        # Some sort of checking, etc. will need to happen in a few places
        # impute data
        logging.info("imputing data...")
        self.data_norm = self.core.impute_data(self.data_norm)
        # make predictions
        logging.info("predicting unknown class labels...")
        predict = pd.DataFrame(self.classifier.predict(self.data_norm))
        # rename predict column
        predict.rename(columns={predict.columns[0]: "predict"}, inplace=True)
        # get predict probabilities
        predict_prob = pd.DataFrame(
            self.classifier.predict_proba(self.data_norm))
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
        self.predict_section.enable_outputs()

    def make_pairplot(self):
        df = pd.concat([self.data_norm, self.predict], axis=1)
        self.pairplot = sns.pairplot(
            data=df,
            vars=df.columns[0:self.data.shape[1]],
            hue="predict",
            diag_kind='hist'
        )

    def save_output(self):
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

        location = tk.filedialog.asksaveasfilename(
            title="Save Output Predicitons",
            initialdir="output/",
            defaultextension=".csv",
            filetypes=[
                ("Comma separated values", ".csv .txt"),
                ("All Files", ".*"),
                ("Excel file", ".xlsx .xlsm .xlsb .xltx .xltm .xls .xlt .xml"),
                ("Tab separated values", ".tsv .txt"),
                ("Standard deliniated text file", ".txt")
            ]
        )

        df.to_csv(location, index=False)


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
