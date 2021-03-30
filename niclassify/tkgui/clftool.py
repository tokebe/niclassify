"""Classifier Tool/main window of GUI program."""
import csv
import json
import os
import shutil
import subprocess
import tempfile
import time
import traceback

import pandas as pd
import tkinter as tk

import importlib.resources as pkg_resources
from joblib import dump
from tkinter import filedialog
from tkinter import messagebox
from xlrd import XLRDError

from .clfpanels import DataPanel, PredictPanel, StatusBar, TrainPanel
from .smallwindows import NaNEditor, ProgressPopup
from .datatool import DataPreparationTool
from .wrappers import threaded, report_uncaught
from .dialogs.dialog import DialogLibrary


class ClassifierTool(tk.Frame):
    """
    The gui application class.

    Defines most of the buttons and functions to be used, with the rest in
    tkgui.
    """

    def __init__(
        self,
        parent,
        standard_program,
        classifier,
        utilities,
        *args,
        **kwargs
    ):
        """
        Instantiate the program.

        Args:
            parent (tk.Root): The root window.
        """
        tk.Frame.__init__(self, parent, *args, **kwargs)

        self.parent = parent

        # set up error handling for uncaught exceptions
        # self.parent.report_callback_exception = self.uncaught_exception
        # self.report_callback_exception = self.uncaught_exception

        # set up the inner-layer program and prep log, etc
        self.sp = standard_program(classifier())
        self.util = utilities
        self.logname = self.sp.boilerplate()
        self.data_win = None
        try:
            self.dlib = DialogLibrary()
        except json.JSONDecodeError:
            messagebox.showerror(
                title="Dialog Library Read Error",
                message="Unable to read dialog lib, likely due to a formatting\
 error.\nProgram will exit."
            )
            exit(-1)

        # set nans in sp
        self.sp.nans = self.util.NANS

        # stored files saved in a temporary directory, cleaned on exit
        self.tempdir = tempfile.TemporaryDirectory()
        # what follows are all to be tempfiles
        self.cm = None
        self.report = None
        self.pairplot = None
        self.output = None

        # set up elements of main window
        parent.title(
            "{} Classifier Tool".format(self.sp.clf.clf.__class__.__name__))
        # panel for most controls, so statusbar stays on bottom
        self.panels = tk.Frame(
            self.parent
        )
        self.panels.pack(fill=tk.BOTH, expand=True)
        # data import/column selection sec
        self.data_sec = DataPanel(
            self.panels,
            self,
            text="Data",
            labelanchor=tk.N
        )
        self.data_sec.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        # sec holds operational controls
        self.operate_sec = tk.Frame(
            self.panels
        )
        self.operate_sec.pack(side=tk.RIGHT, anchor=tk.N)
        # training controls
        self.train_sec = TrainPanel(
            self.operate_sec,
            self,
            text="Train",
            labelanchor=tk.N
        )
        self.train_sec.pack(fill=tk.X)
        # prediction controls
        self.predict_sec = PredictPanel(
            self.operate_sec,
            self,
            text="Predict",
            labelanchor=tk.N
        )
        self.predict_sec.pack(fill=tk.X)
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
        # print(conditions)
        if all(conditions):
            self.predict_sec.prediction_make.config(state=tk.ACTIVE)
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
            self.predict_sec.prediction_make["state"] != tk.DISABLED
        ]
        if all(conditions):
            return self.dlib.dialog(messagebox.askokcancel, "OVERWRITE_CLF")
        else:
            return True

    def enable_train(self, event):
        """
        Enable the train button.

        Also updates the currently selected known label column.
        """
        # print(self.train_sec.known_select.get())
        self.sp.class_column = self.train_sec.known_select.get()
        self.train_sec.train_button.config(state=tk.ACTIVE)

    def get_data_file(self, internal=None):
        """
        Prompt the user for a file and update contents appropriately.

        Once the data file is selected, it is either recognized as excel or
        as text. If it's excel the sheet selector is populated and the user is
        prompted, otherwise the column selection panel is populated.
        """
        # ----- threaded function -----
        @threaded
        def _get_data_file(data_file):
            """
            Get data file information in a thread.

            Args:
                data_file (str): The path to the data file.
            """
            # change status for user's sake
            self.status_bar.set_status("Reading file...")
            self.status_bar.progress["mode"] = "indeterminate"
            self.status_bar.progress.start()

            # handle file import given whether file is excel or text
            if (os.path.splitext(data_file)[1]  # some sort of excel file
                    in [
                        ".xlsx", ".xlsm", ".xlsb",
                        ".xltx", ".xltm", ".xls",
                        ".xlt", ".xml"
            ]):
                # enable sheet selection and get list of sheets for dropdown
                self.data_sec.excel_sheet_input.config(state="readonly")

                # in case there's something that goes wrong reading the file
                try:
                    sheets = list(pd.ExcelFile(data_file).sheet_names)
                except (
                        OSError, IOError, KeyError,
                        TypeError, ValueError, XLRDError
                ):
                    self.dlib.dialog(messagebox.showwarning, "EXCEL_READ_ERR")

                    # re-enable loading data
                    self.data_sec.load_data_button["state"] = tk.ACTIVE

                    # stop status
                    self.status_bar.set_status("Awaiting user input.")
                    self.status_bar.progress.stop()
                    self.status_bar.progress["mode"] = "determinate"
                    return

                # update sheet options for dropdown and prompt user to select
                self.data_sec.excel_sheet_input["values"] = sheets

                # auto select the first
                # both in case there's only one (makes user's life easier)
                # and to make it more apparent what the user needs to do
                self.data_sec.excel_sheet_input.set(sheets[0])
                self.sp.excel_sheet = sheets[0]
                # self.get_sheet_cols(None)

                # re-enable loading data
                self.data_sec.load_data_button["state"] = tk.ACTIVE

                # stop status
                self.status_bar.set_status("Awaiting user input.")
                self.status_bar.progress.stop()
                self.status_bar.progress["mode"] = "determinate"

                self.dlib.dialog(messagebox.showinfo, "EXCEL_DETECTED")

            else:  # otherwise it's some sort of text file
                # disable sheet selection dropdown
                self.data_sec.excel_sheet_input.config(
                    state=tk.DISABLED)

                # in case there's a read or parse error of some kind
                try:
                    column_names = pd.read_csv(
                        data_file,
                        na_values=self.util.NANS,
                        keep_default_na=True,
                        sep=None,
                        nrows=0,
                        engine="python"
                    ).columns.values.tolist()
                except (
                        OSError, IOError, KeyError,
                        TypeError, ValueError, csv.Error
                ):
                    self.dlib.dialog(messagebox.showwarning, "FILE_READ_ERR")
                    # re-enable loading data
                    self.data_sec.load_data_button["state"] = tk.ACTIVE

                    # stop status
                    self.status_bar.set_status("Awaiting user input.")
                    self.status_bar.progress.stop()
                    self.status_bar.progress["mode"] = "determinate"
                    return

                # update column selections for known class labels
                self.train_sec.known_select["values"] = column_names

                # update the column select panel
                colnames_dict = {x: i for i, x in enumerate(column_names)}
                self.data_sec.col_select_panel.update_contents(
                    colnames_dict)

                # conditionally enable predicting
                self.check_enable_predictions()

                # re-enable loading data
                self.data_sec.load_data_button["state"] = tk.ACTIVE

                # stop status
                self.status_bar.set_status("Awaiting user input.")
                self.status_bar.progress.stop()
                self.status_bar.progress["mode"] = "determinate"
        # ----- end threaded function -----

        # check if user is overwriting and ask if they're ok with it
        if not self.check_warn_overwrite():
            return

        self.status_bar.set_status("Awaiting user file selection...")

        if internal is None:
            # prompt user for file
            data_file = filedialog.askopenfilename(
                title="Open Data File",
                initialdir=os.path.join(self.util.USER_PATH, "data"),
                filetypes=[
                    ("All Files", ".*"),
                    ("Excel file",
                     ".xlsx .xlsm .xlsb .xltx .xltm .xls .xlt .xml"),
                    ("Comma-separated values", ".csv .txt"),
                    ("Tab-separated values", ".tsv .txt"),
                    ("Standard deliniated text file", ".txt")
                ]
            )
            # if user cancels don't try to open nothing
            if len(data_file) <= 0:
                # self.status_bar.progress.stop()
                # self.status_bar.progress.config(mode="determinate")
                self.status_bar.set_status("Awaiting user input.")
                return
        else:
            data_file = internal

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

        # disable load data button so we don't get multiple
        self.data_sec.load_data_button["state"] = tk.DISABLED

        # threaded portion of function
        _get_data_file(data_file)

    def get_selected_cols(self):
        """
        Get the currently selected feature columns.

        Returns:
            list: the currently selected feature columns.

        """
        # get selection
        selected_cols = list(
            self.data_sec.col_select_panel.sel_contents.get(0, tk.END))

        # put in StandardProgram
        self.sp.feature_cols = selected_cols

        # return to be useful
        return selected_cols

    def get_sheet_cols(self, event):
        """
        Get the columns for a given excel sheet when it is selected.

        event isn't used so it doesn't really matter.
        """
        # ---- threaded function -----
        @threaded
        def _get_sheet_cols(sheet):
            try:
                column_names = pd.read_excel(
                    self.sp.data_file,
                    sheet_name=sheet,
                    na_values=self.util.NANS,
                    nrows=0,
                    keep_default_na=True
                ).columns.values.tolist()
            except (
                    OSError, IOError, KeyError,
                    TypeError, ValueError, XLRDError
            ):
                self.dlib.dialog(
                    messagebox.showwarning, "FILE_READ_ERR_AFTER_SUCCESS")

                # reset status and re-enable sheet selection
                self.status_bar.set_status("Awaiting user input.")
                self.data_sec.excel_sheet_input["state"] = "readonly"
                self.status_bar.progress["mode"] = "determinate"
                self.status_bar.progress.stop()
                return

            # update known class label dropdown
            self.train_sec.known_select["values"] = column_names
            # update column selection panel
            self.data_sec.col_select_panel.update_contents(
                {x: i for i, x in enumerate(column_names)})

            self.check_enable_predictions()

            # reset status and re-enable sheet selection
            self.status_bar.set_status("Awaiting user input.")
            self.data_sec.excel_sheet_input["state"] = "readonly"
            self.status_bar.progress["mode"] = "determinate"
            self.status_bar.progress.stop()
        # ----- end threaded function -----

        # check if user is overwriting and make sure they're ok with it
        if not self.check_warn_overwrite():
            # change selected sheet back to what it was
            self.data_sec.excel_sheet_input.set(self.sp.excel_sheet)
            return

        # reset controls because new data
        self.reset_controls()

        # get sheet name
        sheet = self.data_sec.excel_sheet_input.get()

        # skip reloading if it's already selected
        if sheet == self.sp.excel_sheet:
            return

        # get sheet column names
        self.sp.excel_sheet = sheet

        # disable button and set status before launching thread
        self.status_bar.set_status("Reading sheet...")
        self.data_sec.excel_sheet_input["state"] = tk.DISABLED
        self.status_bar.progress["mode"] = "indeterminate"
        self.status_bar.progress.start()

        # launch thread to get columns
        _get_sheet_cols(sheet)

    def load_classifier(self):
        """
        Prompt the user to select a saved classifier and load it.

        Ensures that the classifier is at least the right object.
        """
        # ----- threaded function -----
        @threaded
        def _load_classifier(clf_file):
            """
            Load the chosen classifier in a thread.

            Args:
                clf_file (str): The path to the classifier.
            """
            # disable train in the event that a trained clf is overwritten
            try:
                self.sp.clf = self.util.load_classifier(clf_file)
            except (TypeError, KeyError):
                self.dlib.dialog(
                    messagebox.showwarning, "INCOMPATIBLE_NOT_CLF")

                self.status_bar.set_status("Awaiting user input.")
                self.predict_sec.classifier_load["state"] = tk.ACTIVE
                return
            except (OSError, IOError):
                self.dlib.dialog(messagebox.showerror, "FILE_READ_ERR")

            # reset controls and conditionally enable steps
            self.reset_controls(clf=True)
            self.check_enable_predictions()
            # reset status and disabled button
            self.status_bar.set_status("Awaiting user input.")
            self.predict_sec.classifier_load["state"] = tk.ACTIVE
        # ----- end threaded function -----

        self.status_bar.set_status("Awaiting user file selection...")

        # check if user is overwriting and make sure they're ok with it
        if not self.check_warn_overwrite():
            self.status_bar.set_status("Awaiting user input.")
            return

        # prompt the user for the classifier file
        clf_file = filedialog.askopenfilename(
            title="Open Saved Classifier",
            initialdir=os.path.realpath(
                os.path.join(self.util.USER_PATH, "output/classifiers/")),
            filetypes=[
                ("Classifier archive file", ".gz .joblib .pickle")
            ]
        )

        # don't do anything if the user selected nothing
        if len(clf_file) <= 0:
            self.status_bar.set_status("Awaiting user input.")
            return

        # disable button while launching thread
        self.predict_sec.classifier_load["state"] = tk.DISABLED
        # launch thread to load the classifier
        _load_classifier(clf_file)

    def make_cm(self, features_known, class_labels):
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
            prefix="confusion_matrix_",
            suffix=".png",
            delete=False,
            dir=self.tempdir.name
        )
        self.cm.close()

        # make the plot and save to the tempfile.
        self.util.make_confm(
            self.sp.clf.clf,
            features_known,
            class_labels
        ).savefig(self.cm.name)

    def make_report(self):
        """
        Grab a report of the classifier training from the log.

        Saves to a tempfile for easier copying when saving.
        """
        # capture log output for report
        capture = False
        captured_lines = []
        loglines = ""
        try:
            with open(self.logname, "r") as log:
                loglines = log.readlines()
        except IOError:
            self.dlib.dialog(messagebox.showerror, "REPORT_GENERATE_ERR")

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
            prefix="training_report_",
            suffix=".txt",
            delete=False,
            dir=self.tempdir.name
        )
        # write the report
        self.report.write(report)
        # close the file so it's ready for open/copy
        self.report.close()

    def make_pairplot(self, features, predict):
        """Generate a pairplot and save to a tempfile.

        Args:
            features (DataFrame): The feature data predicted on.
            predict (DataFrame/Series): The class label predictions.
        """
        # check if pairplot can/should be generated
        # pairplot generation fails beyond 92 variables
        if len(features.columns) > 25:
            if len(features.columns) > 92:
                self.dlib.dialog(
                    messagebox.showwarning, "NO_PAIRPLOT_OVERLOAD")

            else:
                self.dlib.dialog(messagebox.showwarning, "NO_PAIRPLOT_USELESS")
            self.predict_sec.pairplot_sec.disable_buttons()
            return
        # check if pairplot exists and make sure it's closed if it does
        if self.pairplot is not None:
            self.pairplot.close()
        # create the tempfile
        self.pairplot = tempfile.NamedTemporaryFile(
            mode="w+",
            prefix="pairplot_",
            suffix=".png",
            delete=False,
            dir=self.tempdir.name
        )
        self.pairplot.close()

        # make the pairplot and save to the tempfile
        self.util.make_pairplot(
            features,
            predict
        ).savefig(self.pairplot.name)

    def make_predictions(self):
        """
        Make predictions and save to a tempfile.

        Set up to be agnostic of AutoClassifier support for predict_prob.
        """
        # ----- threaded function -----
        @threaded
        def _make_predictions(on_finish, status_cb):
            """
            Make predictions on data in a thread.

            Args:
                on_finish (func): Function to call when complete. Primarily
                used for popup progressbar.
                status_cb (func): Function to call to update status. Primarily
                used for popup progressbar.
            """
            # update status and progress for user
            self.status_bar.progress["value"] = 0
            self.status_bar.progress.step(16.5)
            self.status_bar.set_status("Preparing data...")

            # get the data prepared
            features, metadata = self.sp.prep_data()

            self.status_bar.progress.step(16.5)
            self.status_bar.set_status("Splitting known data...")

            self.status_bar.progress.step(16.5)
            self.status_bar.set_status("Imputing data...")

            # impute the data
            features = self.sp.impute_data(features)

            self.status_bar.progress.step(16.5)
            self.status_bar.set_status("Making predictions...")
            status_cb("Making predictions...")

            # get predictions
            try:
                predict = self.sp.predict_AC(self.sp.clf, features, status_cb)
            except (ValueError,  KeyError) as err:
                problem = (
                    "FEATURE_ERR_NUM"
                    if isinstance(err, ValueError)
                    else
                    "FEATURE_ERR_NAMES"
                )
                self.dlib.dialog(messagebox.showerror, problem)
                # finish updating status
                self.status_bar.progress["value"] = 100
                time.sleep(0.2)
                self.status_bar.progress["value"] = 0
                self.status_bar.set_status("Awaiting user input.")

                # call finisher function
                on_finish()
                return

            # Because predict_AC may return predict_prob we check for a tuple
            # and act accordingly
            if type(predict) == tuple:
                predict, predict_prob = predict
            else:
                predict_prob = None

            self.status_bar.progress.step(16.5)
            self.status_bar.set_status("Generating reports...")
            status_cb("Generating reports...")

            # check if output exists and make sure it's closed if it does
            if self.output is not None:
                self.output.close()

            # create the tempfile
            self.output = tempfile.NamedTemporaryFile(
                mode="w+",
                prefix="prediction_ouput_",
                suffix=".csv",
                delete=False,
                dir=self.tempdir.name
            )
            self.output.close()

            # save to output file
            self.util.save_predictions(
                metadata, predict, features, self.output.name, predict_prob)

            # enable outputs
            self.predict_sec.enable_outputs()
            self.predict_sec.output_save.config(state=tk.ACTIVE)
            self.make_pairplot(features, predict)

            # finish updating status
            self.status_bar.progress["value"] = 100
            time.sleep(0.2)
            self.status_bar.progress["value"] = 0
            self.status_bar.set_status("Awaiting user input.")

            self.dlib.dialog(
                messagebox.showinfo,
                "PREDICT_COMPLETE"
            )

            # call finisher function
            on_finish()
        # ----- end threaded function -----

        # check that data columns are selected
        if len(
            self.data_sec.col_select_panel.sel_contents.get(0, tk.END)
        ) <= 0:
            self.dlib.dialog(messagebox.showwarning, "NO_COLUMNS_SELECTED")
            return

        # check if user is overwriting and make sure they're ok with it
        if self.predict_sec.output_save["state"] != tk.DISABLED:
            if not self.dlib.dialog(
                    messagebox.askokcancel, "PREDICTION_OVERWRITE"):
                return

        # check if file still exists
        if not os.path.exists(self.sp.data_file):
            self.dlib.dialog(
                messagebox.showwarning, "FILE_READ_ERR_AFTER_SUCCESS")
            return

        self.get_selected_cols()

        self.sp.print_vars()

        progress_popup = ProgressPopup(
            self,
            "Making Predictions",
            ""
        )

        _make_predictions(
            progress_popup.complete,
            progress_popup.set_status
        )

    def open_data_tool(self):
        """Open the data preparation tool."""
        # make sure multiple instances can't open
        self.data_sec.retrieve_data_button["state"] = tk.DISABLED
        # open new data tool
        self.data_win = DataPreparationTool(
            self, self, self.tempdir, self.util)

    def open_help(self):
        """Open User Manual document"""
        self.util.view_open_file(self.util.HELP_DOC)

    def open_nans(self):
        """Open a window to view and edit NaN values."""
        # test = tk.Toplevel(self.parent)
        self.data_sec.nan_check["state"] = tk.DISABLED
        NaNEditor(self, self)

    def open_output_folder(self):
        """
        Open the output folder for classifiers, logs, etc.

        Doesn't open user-defined save locations.
        """
        self.util.view_open_file(os.path.join(self.util.USER_PATH))

    def reset_controls(self, clf=False):
        """
        Reset controls and stored data.

        Set clf to True to not reset selected columns, etc, such as when
            loading a new classifier.

        Args:
            clf (bool, optional): Only reset for new clf. Defaults to False.
        """
        if not clf:
            # self.sp.data_file = None
            self.sp.excel_sheet = None
            self.sp.feature_cols = None
            self.sp.class_column = None
            self.sp.multirun = None
            self.sp.nans = None

        # reset GUI controls
        if not clf:
            self.data_sec.col_select_panel.update_contents({})
            self.train_sec.known_select["values"] = []
            self.train_sec.known_select.set(
                "")

        # disable buttons
        if not clf or self.train_sec.known_select.get() == "":
            self.train_sec.reset_enabled()
        if not clf:
            self.predict_sec.reset_enabled()

    def save_classifier(self):
        """
        Save the current classifier to a location the user chooses.

        Like all others, initialdir is calculated using self.util USER_PATH
        """
        # ----- threaded function -----
        @threaded
        def _save_classifier(location):
            """Save the classifier to a given location in a thread.

            Args:
                location (str): Path to the new file to save to.
            """
            # save the classifier
            dump(self.sp.clf, location)
            # reset buttons and status
            self.train_sec.classifier_save["state"] = tk.ACTIVE
            self.status_bar.progress["mode"] = "determinate"
            self.status_bar.progress.stop()
        # ----- end threaded function -----

        self.status_bar.set_status("Awaiting user save location...")
        # prompt user for location to save to
        location = tk.filedialog.asksaveasfilename(
            title="Save Classifier",
            initialdir=os.path.realpath(
                os.path.join(self.util.USER_PATH, "output/")),
            defaultextension=".gz",
            filetypes=[("GNU zipped archive", ".gz")]
        )
        # set status for user
        self.status_bar.set_status("Saving classifier...")
        self.status_bar.progress["mode"] = "indeterminate"
        self.status_bar.progress.start()

        # disable button while launching thread
        self.train_sec.classifier_save["state"] = tk.DISABLED
        # launch thread to save classifier
        _save_classifier(location)

    def save_item(self, item):
        """
        Save the chosen item (selected by str) to a user-designated place.

        Args:
            item (str): A tag identifying the item to be saved.
        """
        # set up some maps for easier automation
        titles = {
            "cm": "Confusion Matrix",
            "pairplot": "Pairplot",
            "report": "Training Report",
            "output": "Prediction Output",
            "bold_results": "BOLD Data",
            "merged_results": "Merged Data",
            "filtered_data": "Filtered Data",
            "raw_fasta": "Unaligned Fasta",
            "fasta_align": "Aligned Fasta",
            "finalized": "Prepared Sequence Data"
        }
        graphtypes = [("Portable Network Graphics image", ".png")]
        reportypes = [("Plain text file", ".txt")]
        outputtypes = [
            ("Comma-separated values", ".csv"),
            ("Tab-separated values", ".tsv"),
            ("All Files", ".*"),
        ]
        fastatypes = [("FASTA formatted sequence data", ".fasta")]
        types = {
            "cm": graphtypes,
            "pairplot": graphtypes,
            "report": reportypes,
            "output": outputtypes,
            "bold_results": outputtypes,
            "merged_results": outputtypes,
            "filtered_data": outputtypes,
            "raw_fasta": fastatypes,
            "fasta_align": fastatypes,
            "finalized": outputtypes
        }
        default_extensions = {
            "cm": ".png",
            "pairplot": ".png",
            "report": ".txt",
            "output": ".csv",
            "bold_results": ".csv",
            "merged_results": ".csv",
            "filtered_data": ".csv",
            "raw_fasta": ".fasta",
            "fasta_align": ".fasta",
            "finalized": ".csv"
        }
        buttons = {
            "cm": self.train_sec.cm_sec.button_save,
            "pairplot": self.predict_sec.pairplot_sec.button_save,
            "report": self.train_sec.report_sec.button_save,
            "output": self.predict_sec.output_save,
            "bold_results": (
                self.data_win.get_data_sec.save_bold_button
                if self.data_win is not None else None
            ),
            "merged_results": (
                self.data_win.get_data_sec.save_merge_button
                if self.data_win is not None else None
            ),
            "filtered_data": (
                self.data_win.data_sec.filtered_sec.button_save
                if self.data_win is not None else None
            ),
            "raw_fasta": (
                self.data_win.data_sec.fasta_sec.button_save
                if self.data_win is not None else None
            ),
            "fasta_align": (
                self.data_win.data_sec.align_save_button
                if self.data_win is not None else None
            ),
            "finalized": (
                self.data_win.data_sec.final_save_button
                if self.data_win is not None else None
            )
        }
        tempfiles = {
            "cm": self.cm,
            "pairplot": self.pairplot,
            "report": self.report,
            "output": self.output,
            "bold_results": (
                self.data_win.sequence_raw
                if self.data_win is not None else None
            ),
            "merged_results": (
                self.data_win.merged_raw
                if self.data_win is not None else None
            ),
            "filtered_data": (
                self.data_win.sequence_filtered
                if self.data_win is not None else None
            ),
            "raw_fasta": (
                self.data_win.fasta
                if self.data_win is not None else None
            ),
            "fasta_align": (
                self.data_win.fasta_align
                if self.data_win is not None else None
            ),
            "finalized": (
                self.data_win.finalized_data
                if self.data_win is not None else None
            )
        }

        # ----- threaded function -----
        @threaded
        def _save_item(item, location):
            """
            Save the given item to the given location, in a thread.

            Args:
                item (str): Identifier for which item to save.
                location (str): Path to the location to save to.
            """
            # check if user wants to translate between csv and tsv
            in_ext = os.path.splitext(tempfiles[item].name)[1]
            out_ext = os.path.splitext(location)[1]
            if (in_ext != out_ext) and (out_ext in (".csv", ".tsv")):
                try:
                    self.util.get_data(tempfiles[item].name).to_csv(
                        location,
                        sep=('\t' if out_ext == ".tsv" else ","),
                        index=False
                    )
                except (OSError, IOError):
                    self.dlib.dialog(messagebox.showerror, "FILE_WRITE_ERR")
            else:
                # if user chose some other extension, just give them csv
                # because I can't be bothered to infer atypical delimitation
                # standards

                # otherwise, copy the appropriate file
                try:
                    shutil.copy(tempfiles[item].name, location)
                except (OSError, IOError):
                    self.dlib.dialog(messagebox.showerror, "FILE_WRITE_ERR")

            # reset buttons and status
            buttons[item]["state"] = tk.ACTIVE
            self.status_bar.set_status("Awaiting user input.")
        # ----- end threaded function -----

        self.status_bar.set_status("Awaiting user save location...")

        # prompt user for save location
        location = tk.filedialog.asksaveasfilename(
            title="Save {}".format(
                titles[item]),
            initialdir=self.util.USER_PATH,
            defaultextension=default_extensions[item],
            filetypes=types[item]
        )
        # return if user cancels
        if len(location) == 0:
            self.status_bar.set_status("Awaiting user input.")
            return

        # set status and disable button while launching thread
        self.status_bar.set_status("Saving {}...".format(item))
        buttons[item]["state"] = tk.DISABLED

        # launch thread to save item
        _save_item(item, location)

    def save_nans(self):
        """Save nans value list to nans.json."""
        # ----- threaded function -----
        @threaded
        def _save_nans():
            """Save the NaN values to nans.json in a thread."""
            nans = self.sp.nans
            try:
                with open(
                    os.path.join(
                        self.util.USER_PATH,
                        "config/nans.json"
                    ),
                    "w"
                ) as nansfile:
                    json.dump(nans, nansfile)
            except IOError:
                self.dlib.dialog(messagebox.showerror, "NAN_DUMP_ERR")

            # reset buttons and status
            self.status_bar.set_status("Awaiting user input.")
            self.data_sec.nan_check["state"] = tk.ACTIVE
        # ----- end threaded function -----

        # set status and disable button while launching thread
        self.status_bar.set_status("Saving NaN values...")
        self.data_sec.nan_check["state"] = tk.DISABLED
        # launch thread to save NaN values
        _save_nans()

    def uncaught_exception(self, error_trace, logfile):
        """Report uncaught exceptions to the user."""
        self.dlib.dialog(
            messagebox.showerror,
            "UNHANDLED_EXCEPTION",
            form=(logfile,)
        )

    def train_classifier(self):
        """
        Train a classifier with the given data.

        Prepares outputs and enables buttons for next steps.
        """
        # ----- threaded function -----
        @threaded
        def _train_classifier(on_finish, status_cb):
            """
            Train the classifier in a thread.

            Args:
                on_finish (func): Function to call when complete. Primarily
                    used for popup progressbar.
                status_cb (func): Function to call to update status. Primarily
                    used for popup progressbar.
            """
            # set status and progress for user's sake
            self.status_bar.progress["value"] = 0
            self.status_bar.progress.step(20)
            self.status_bar.set_status("Preparing data...")

            # get the data prepared
            features, metadata = self.sp.prep_data()

            self.status_bar.progress.step(20)
            self.status_bar.set_status("Splitting known data...")

            # get known for making cm
            features, metadata = self.util.get_known(
                features, metadata, self.sp.class_column)

            # make sure there are at least 2 classes
            if not self.sp.check.check_enough_classes(
                metadata,
                lambda: self.dlib.dialog(
                    messagebox.showerror,
                    "CANNOT_TRAIN"
                )
            ):
                on_finish()
                return

            # check for extreme known class imbalance
            # using stdev as a very rough heuristic
            if not self.sp.check.check_inbalance(
                metadata,
                lambda: self.dlib.dialog(
                    messagebox.askokcancel,
                    "HIGH_IMBALANCE"
                )
            ):
                on_finish()
                return

            self.status_bar.progress.step(20)
            self.status_bar.set_status("Training classifier...")

            # train classifier
            self.sp.train_AC(features, metadata, status_cb)

            self.status_bar.progress.step(20)
            self.status_bar.set_status("Generating reports...")

            # generate outputs
            self.make_report()
            self.make_cm(features, metadata[self.sp.class_column])

            # finish up status updates
            self.status_bar.progress.step(20)
            time.sleep(0.2)
            self.status_bar.progress["value"] = 0
            self.status_bar.set_status("Awaiting user input.")

            # enable related output
            self.train_sec.classifier_save.config(state=tk.ACTIVE)
            self.train_sec.enable_outputs()
            self.check_enable_predictions()

            self.dlib.dialog(
                messagebox.showinfo,
                "TRAIN_COMPLETE"
            )

            # finish up
            on_finish()
        # ----- end threaded function -----

        # make sure that appropriate feature columns are selected
        if len(
            self.data_sec.col_select_panel.sel_contents.get(0, tk.END)
        ) <= 0:
            self.dlib.dialog(messagebox.showwarning, "NO_COLUMNS_SELECTED")
            return
        if self.train_sec.known_select.get() in self.get_selected_cols():
            self.dlib.dialog(
                messagebox.showwarning, "ERR_OVERLAP_FEATURES_CLASS")
            return

        # check if file still exists
        if not os.path.exists(self.sp.data_file):
            self.dlib.dialog(messagebox.showwarning, "FILE_READ_ERR")
            return

        # get multirun setting
        self.sp.multirun = int(self.train_sec.n_input.get())

        # for debug
        self.sp.print_vars()

        # check if user is overwriting and make sure they're ok with it
        if not self.check_warn_overwrite():
            return

        # disable save output and pairplot buttons
        self.predict_sec.output_save.config(state=tk.DISABLED)
        self.predict_sec.pairplot_sec.disable_buttons()

        # progress indicator so user doesn't think we're frozen
        progress_popup = ProgressPopup(
            self,
            "Training Classifier",
            ""
        )

        # start threaded portion
        _train_classifier(
            progress_popup.complete,
            progress_popup.set_status
        )

    def view_item(self, item):
        """
        View an item using the system default.

        Args:
            item (str): Path to a file to open.
        """
        try:
            self.util.view_open_file(item)
        except OSError:

            if self.dlib.dialog(
                messagebox.askokcancel,
                "NO_DEFAULT_PROGRAM",
                form=(os.path.splitext(item)[1],),
                icon="warning"
            ):
                subprocess.Popen('explorer /select,{}'.format(item))
