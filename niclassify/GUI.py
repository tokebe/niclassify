import csv
import logging
import os
import shutil
import subprocess
import sys
import threading
import tempfile

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tkinter as tk


from tkinter import filedialog
from joblib import dump, load
from tkinter import ttk
from xlrd import XLRDError


from core import utilities
from core.StandardProgram import StandardProgram
from core.classifiers import RandomForestAC
from tkui.elements import DataPanel, TrainPanel, PredictPanel, StatusBar


# TODO start extensive testing to see what isn't working as it should
# TODO once you have anything that needs fixing fixed, go through and clean
# TODO then go through everything in tkui and make sure it's docstring'd
# TODO then add the extra things, NaN editor and help button
# TODO once that's done, and nothing else has come up, I guess it's time to
# resume testing of new steps

# TODO an extra thing: add feature importances to report, if possible


class MainApp(tk.Frame):
    def __init__(self, parent, *args, **kwargs):
        tk.Frame.__init__(self, parent, *args, **kwargs)

        self.parent = parent

        # set up program utilities
        self.sp = StandardProgram(RandomForestAC())
        self.logname = self.sp.boilerplate()

        # stored files saved in a temporary directory, cleaned on exit
        self.tempdir = tempfile.TemporaryDirectory()
        # what follows are all to be tempfiles
        self.cm = None
        self.report = None
        self.pairplot = None
        self.output = None

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
        """
        Conditionally enable the predict button.

        If we have data and a classifier, enable predictions.

        Returns:
            Bool: True if the button was enabled else False.

        """
        conditions = [
            self.sp.data_file is not None,
            self.sp.clf.is_trained()
        ]
        print(conditions)
        if all(conditions):
            self.predict_section.prediction_make.config(state=tk.ACTIVE)
            return True
        else:
            return False

    def check_warn_overwrite(self):
        """
        Check if action would (should) overwrite current data or classifier.

        Returns:
            Bool: True if the action won't overwrite, or the result of the
                dialog if it will.

        """
        conditions = [
            self.sp.clf is not None,
            self.predict_section.prediction_make["state"] != tk.DISABLED
        ]
        if all(conditions):
            return tk.messagebox.askokcancel(
                title="Overwrite Warning",
                message="This will delete unsaved classifier and results."
            )
        else:
            return True

    def enable_train(self, event):
        """
        Enable the train button.

        Also updates the currently selected known label column.
        """
        # print(self.train_section.known_select.get())
        self.sp.class_column = self.train_section.known_select.get()
        self.train_section.train_button.config(state=tk.ACTIVE)

    def get_data_file(self):
        """
        Prompt the user for a file and update contents appropriately.

        Once the data file is selected, it is either recognized as excel or
        as text. If it's excel the sheet selector is populated and the user is
        prompted, otherwise the column selection panel is populated.
        """
        # check if user is overwriting and ask if they're ok with it
        if not self.check_warn_overwrite():
            return

        # prompt user for file
        data_file = filedialog.askopenfilename(
            title="Open Data File",
            initialdir=os.path.realpath(
                os.path.join(utilities.MAIN_PATH, "data/")),
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
        # reset things
        self.reset_controls()

        print("user chose file {}".format(data_file))
        self.sp.data_file = data_file

        # update window title to reflect chosen file
        self.parent.title(
            "Random Forest Classifier Tool: {}".format(
                os.path.basename(data_file))
        )

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
        """
        Get the currently selected feature columns.

        Returns:
            list: the currently selected feature columns.

        """
        # get selection
        selected_cols = list(
            self.data_section.col_select_panel.sel_contents.get(0, tk.END))

        # put in StandardProgram
        self.sp.feature_cols = selected_cols

        # return to be useful
        return selected_cols

    def get_sheet_cols(self, event):
        """
        Get the columns for a given excel sheet when it is selected.

        event isn't used so it doesn't really matter.
        """
        # check if user is overwriting and make sure they're ok with it
        if not self.check_warn_overwrite():
            # change selected sheet back to what it was
            self.data_section.excel_sheet_input.set(self.sp.excel_sheet)
            return

        # reset controls because new data
        self.reset_controls()

        # get sheet name
        sheet = self.data_section.excel_sheet_input.get()

        # skip reloading if it's already selected
        if sheet == self.sp.excel_sheet:
            return

        # get sheet column names
        self.sp.excel_sheet = sheet
        column_names = pd.read_excel(
            self.sp.data_file,
            sheet_name=sheet,
            na_values=utilities.NANS,
            nrows=0,
            keep_default_na=True
        ).columns.values.tolist()

        # update known class label dropdown
        self.train_section.known_select["values"] = column_names
        # update column selection panel
        self.data_section.col_select_panel.update_contents(
            {x: i for i, x in enumerate(column_names)})

        self.check_enable_predictions()

    def load_classifier(self):
        """
        Prompt the user to select a saved classifier and load it.

        Ensures that the classifier is at least the right object.
        """
        # check if user is overwriting and make sure they're ok with it
        if not self.check_warn_overwrite():
            return

        # prompt the user for the classifier file
        clf_file = filedialog.askopenfilename(
            title="Open Saved Classifier",
            initialdir=os.path.realpath(
                os.path.join(utilities.MAIN_PATH, "output/classifiers/")),
            filetypes=[
                ("Classifier archive file", ".gz .joblib .pickle")
            ]
        )
        # don't do anything if the user selected nothing
        if len(clf_file) <= 0:
            return

        # disable train stuff in the event that a trained clf is overwritten
        self.reset_controls(clf=True)
        self.sp.clf = utilities.load_classifier(clf_file)
        self.check_enable_predictions()

    def make_cm(self, features_known, metadata_known, class_column):
        """
        Generate a confusion matrix graph and save to tempfile.

        Args:
            features_known (DataFrame): Features data for which class labels
                are known.
            metadata_known (DataFrame):  Metadata for which class labels are
                known.
            class_column (DataFrame/Series): Class labels for given data.
        """
        # check if cm exists and make sure it's closed if it does
        if self.cm is not None:
            self.cm.close()
        # create the tempfile
        self.cm = tempfile.NamedTemporaryFile(
            mode="w+",
            suffix=".png",
            delete=False,
            dir=self.tempdir.name
        )
        self.cm.close()

        # make the plot and save to the tempfile.
        utilities.make_confm(
            self.sp.clf.clf,
            features_known,
            metadata_known,
            class_column
        ).savefig(self.cm.name)

    def make_report(self):
        """
        Grab a report of the classifier training from the log.

        Saves to a tempfile for easier copying when saving.
        """
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

        report = "".join(reversed(captured_lines))

        # check if report exists and make sure it's closed if it does
        if self.report is not None:
            self.report.close()
        # create the tempfile
        self.report = tempfile.NamedTemporaryFile(
            mode="w+",
            suffix=".txt",
            delete=False,
            dir=self.tempdir.name
        )
        # write the report
        self.report.write(report)
        # close the file so it's ready for open/copy
        self.report.close()

    def make_pairplot(self, data, predict):
        """Generate a pairplot and save to a tempfile.

        Args:
            data (DataFrame): The feature data predicted on.
            predict (DataFrame/Series): The class label predictions.
        """
        # check if pairplot exists and make sure it's closed if it does
        if self.pairplot is not None:
            self.pairplot.close()
        # create the tempfile
        self.pairplot = tempfile.NamedTemporaryFile(
            mode="w+",
            suffix=".png",
            delete=False,
            dir=self.tempdir.name
        )
        self.pairplot.close()

        # make the pairplot and save to the tempfile
        utilities.make_pairplot(
            data,
            predict
        ).savefig(self.pairplot.name)

    def make_predictions(self):
        """
        Make predictions and save to a tempfile.

        Set up to be agnostic of AutoClassifier support for predict_prob.
        """
        if len(
            self.data_section.col_select_panel.sel_contents.get(0, tk.END)
        ) <= 0:
            tk.messagebox.showwarning(
                title="No Data Columns Selected",
                message="Please select at least one feature column."
            )
            return

        # check if user is overwriting and make sure they're ok with it
        if self.predict_section.output_save["state"] != tk.DISABLED:
            if not tk.messagebox.askokcancel(
                title="Overwrite Warning",
                message="This will delete unsaved prediction results."
            ):
                return

        self.get_selected_cols()

        self.sp.print_vars()

        # get the data prepared
        raw_data, features, feature_norm, metadata = self.sp.prep_data()

        # impute the data
        feature_norm = self.sp.impute_data(feature_norm)

        # get predictions
        try:
            predict = self.sp.predict_AC(self.sp.clf, feature_norm)
        except (ValueError,  KeyError) as err:
            message = (
                "The classifier expects different number of features than \
selected."
                if isinstance(err, ValueError)
                else
                "Given feature names do not match those expected by the \
classfier"
            )
            tk.messagebox.showerror(
                title="Feature Data Error",
                message=message
            )
            return

        # Because predict_AC may return predict_prob we check if it's a tuple
        # and act accordingly
        if type(predict) == tuple:
            predict, predict_prob = predict
        else:
            predict_prob = None

        # TODO decide if keeping a tempfile is better or if it should be
        # kept in RAM
        # check if output exists and make sure it's closed if it does
        if self.output is not None:
            self.output.close()
        # create the tempfile
        self.output = tempfile.NamedTemporaryFile(
            mode="w+",
            suffix=".csv",
            delete=False,
            dir=self.tempdir.name
        )
        self.output.close()
        # save to output file
        utilities.save_predictions(
            metadata, predict, feature_norm, self.output.name, predict_prob)

        self.make_pairplot(feature_norm, predict)
        self.predict_section.enable_outputs()
        self.predict_section.output_save.config(state=tk.ACTIVE)

    def open_output_folder(self):
        """
        Open the output folder for classifiers, logs, etc.

        Doesn't open user-defined save locations.
        """
        utilities.view_open_file(os.path.join(utilities.MAIN_PATH, "output/"))

    def reset_controls(self, clf=False):
        """
        Reset controls and stored data.

        Set clf to True to not reset selected columns, etc, such as when
            loading a new classifier.

        Args:
            clf (bool, optional): Only reset for new clf. Defaults to False.
        """
        # reset stored StandardProgram information
        self.sp.clf = RandomForestAC()

        if not clf:
            self.sp.data_file = None
            self.sp.excel_sheet = None
            self.sp.feature_cols = None
            self.sp.class_column = None
            self.sp.multirun = None
            self.sp.nans = None

        # reset GUI controls
        if not clf:
            self.data_section.col_select_panel.update_contents({})
            self.train_section.known_select["values"] = []
            self.train_section.known_select.set(
                "")  # TODO this probably breaks it

        # disable buttons
        self.train_section.reset_enabled()
        if not clf:
            self.predict_section.reset_enabled()

        # don't have to handle stored tempfiles as their respective methods
        # handle that
        # TODO this whole method probably needs a refit

    def save_classifier(self):
        """
        Save the current classifier to a location the user chooses.

        Like all others, initialdir is calculated using utilities MAIN_PATH
        """
        # prompt user for location to save to
        location = tk.filedialog.asksaveasfilename(
            title="Save Classifier",
            initialdir=os.path.realpath(
                os.path.join(utilities.MAIN_PATH, "output/")),
            defaultextension=".gz",
            filetypes=[("GNU zipped archive", ".gz")]
        )

        # save the classifier
        dump(self.sp.clf, location)

    def save_item(self, item):
        """
        Save the chosen item (selected by str) to a user-designated place.

        Args:
            item (str): A tag identifying the item to be saved.
        """
        # set up some maps for easier automation
        tempfiles = {
            "cm": self.cm,
            "pairplot": self.pairplot,
            "report": self.report,
            "output": self.output
        }
        titles = {
            "cm": "Confusion Matrix",
            "pairplot": "Pairplot",
            "report": "Training Report",
            "output": "Prediction Output"
        }
        graphtypes = [("Portable Network Graphics image", ".png")]
        reportypes = [("Plain text file", ".txt")]
        outputtypes = [
            ("Comma separated values", ".csv"),
            ("Tab separated values", ".tsv"),
            ("All Files", ".*"),
        ]

        types = {
            "cm": graphtypes,
            "pairplot": graphtypes,
            "report": reportypes,
            "output": outputtypes
        }
        default_extensions = {
            "cm": ".png",
            "pairplot": ".png",
            "report": ".txt",
            "output": ".csv"
        }

        # prompt user for save location
        location = tk.filedialog.asksaveasfilename(
            title="Save {}".format(
                titles[item]),
            initialdir=os.path.join(utilities.MAIN_PATH, "output/"),
            defaultextension=default_extensions[item],
            filetypes=types[item]
        )
        # return if user cancels
        if len(location) == 0:
            return

        # if the item's the prediction output we need to convert it to whatever
        # type the user wants because apparently scientists hate CSV
        # If you're a bioinformatics person reading through the code to figure
        # out why the program is slow to save it's because nobody can just pick
        # a standard and stick to it. Sorry for the rant, love you all <3.
        extension = os.path.splitext(location)[1]
        if item == "output" and extension == ".tsv":
            with open(tempfiles[item].name) as csvin, open(location) as tsvout:
                csvin = csv.reader(csvin)
                tsvout = csv.writer(tsvout, delimiter='\t')

                for row in csvin:
                    tsvout.writerow(row)

        # if user chose some other extension, we're just giving them csv
        # because I can't be bothered to infer their preference

        # otherwise, copy the appropriate file
        shutil.copy(tempfiles[item].name, location)

    def train_classifier(self):
        """
        Train a classifier with the given data.

        Prepares outputs and enables buttons for next steps.
        """
        # make sure that appropriate feature columns are selected
        if len(
            self.data_section.col_select_panel.sel_contents.get(0, tk.END)
        ) <= 0:
            tk.messagebox.showwarning(
                title="No Data Columns Selected",
                message="Please select at least one feature column."
            )
            return
        if self.train_section.known_select.get() in self.get_selected_cols():
            tk.messagebox.showwarning(
                title="Known Class And Feature Columns Overlap",
                message="Class column must not be a selected feature column."
            )
            return

        self.sp.multirun = int(self.train_section.n_input.get())

        self.sp.print_vars()

        # check if user is overwriting and make sure they're ok with it
        if not self.check_warn_overwrite():
            return

        # get the data prepared
        raw_data, features, feature_norm, metadata = self.sp.prep_data()

        # get known for making cm
        features_known, metadata_known = utilities.get_known(
            feature_norm, metadata, self.sp.class_column)

        # train classifier
        self.sp.train_AC(feature_norm, metadata)

        # generate outputs and enable related buttons
        self.make_report()
        self.make_cm(features_known, metadata_known, self.sp.class_column)
        self.train_section.classifier_save.config(state=tk.ACTIVE)
        self.train_section.enable_outputs()
        self.check_enable_predictions()

    def view_item(self, item):
        """
        View an item using the system default.

        Args:
            item (str): A tag selecting which item to be viewed.
        """
        tempfiles = {
            "cm": self.cm,
            "pairplot": self.pairplot,
            "report": self.report,
            "output": self.output
        }
        utilities.view_open_file(tempfiles[item].name)


def main():
    """Run the GUI."""
    utilities.assure_path()

    root = tk.Tk()

    root.style = ttk.Style()
    root.style.theme_use("winnative")
    # print(root.style.theme_names())
    app = MainApp(root)
    root.update()
    root.minsize(root.winfo_width(), root.winfo_height())

    def graceful_exit():
        plt.close("all")
        root.quit()
        root.destroy()
        app.tempdir.cleanup()

    root.protocol("WM_DELETE_WINDOW", graceful_exit)

    root.mainloop()


if __name__ == "__main__":
    main()
