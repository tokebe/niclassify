"""The main GUI script controlling the GUI version of niclassify.

Technically extensible by subclassing and copying the main() function.
If you have to do that, I'm sorry. It probably won't be fun.
"""
import csv
import json
import matplotlib
import os
import shutil
import subprocess
import threading
import tempfile
import time

import matplotlib.pyplot as plt
import pandas as pd
import tkinter as tk

from joblib import dump
from tkinter import filedialog
from tkinter import ttk
from xlrd import XLRDError

from core import utilities
from core.StandardProgram import StandardProgram
from core.classifiers import RandomForestAC
from tkui.elements import DataPanel, TrainPanel, ProgressPopup
from tkui.elements import PredictPanel, StatusBar, NaNEditor
from tkui.elements import DataRetrievalWindow
from tkui.output import OutputConsole, TextRedirector

matplotlib.use('Agg')  # this makes threading not break


# NOW
# TODO add new delimitations back into filtered data
# will have to add support down the line for ignoring rows with no group name
# TODO implement native-nonnative lookup for known species

# LATER
# TODO implement backend for both GYMC and PTP and add to gui func

def threaded(func):
    def wrapper(*args, **kwargs):
        thread = threading.Thread(target=func, args=args, kwargs=kwargs)
        thread.start()
        return thread
    return wrapper


class MainApp(tk.Frame):
    """
    The gui application class.

    Defines most of the buttons and functions to be used, with the rest in
    tkui.
    """

    def __init__(self, parent, *args, **kwargs):
        """
        Instantiate the program.

        Args:
            parent (tk.Root): The root window.
        """
        tk.Frame.__init__(self, parent, *args, **kwargs)

        self.parent = parent

        # set up program utilities
        self.sp = StandardProgram(RandomForestAC())
        self.logname = self.sp.boilerplate()
        self.data_retrieval_window = None

        # set nans in sp
        self.sp.nans = utilities.NANS

        # stored files saved in a temporary directory, cleaned on exit
        self.tempdir = tempfile.TemporaryDirectory()
        # what follows are all to be tempfiles
        self.cm = None
        self.report = None
        self.pairplot = None
        self.output = None
        self.sequence_raw = None
        self.user_sequence_raw = None
        self.merged_raw = None
        self.sequence_filtered = None
        self.fasta = None
        self.fasta_align = None
        self.delim = None

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

    @threaded
    def _get_data_file(self, data_file):
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
                    message="Unable to read excel file. \nThe file may be \
                        corrupted, or otherwise unable to be read."
                )
                # re-enable loading data
                self.data_section.load_data_button["state"] = tk.ACTIVE

                # stop status
                self.status_bar.set_status("Awaiting user input.")
                self.status_bar.progress.stop()
                self.status_bar.progress["mode"] = "determinate"
                return

            # update sheet options for dropdown and prompt user to select one
            self.data_section.excel_sheet_input["values"] = sheets

            # auto select the first
            # both in case there's only one (makes user's life easier)
            # and to make it more apparent what the user needs to do
            self.data_section.excel_sheet_input.set(sheets[0])
            self.sp.excel_sheet = sheets[0]
            # self.get_sheet_cols(None)

            # re-enable loading data
            self.data_section.load_data_button["state"] = tk.ACTIVE

            # stop status
            self.status_bar.set_status("Awaiting user input.")
            self.status_bar.progress.stop()
            self.status_bar.progress["mode"] = "determinate"

            tk.messagebox.showinfo(
                title="Excel Sheet Detected",
                message="You will have to specify the sheet to proceed."
            )

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
                    TypeError, ValueError, csv.Error
            ):
                tk.messagebox.showwarning(
                    title="File Read Error",
                    message="Unable to read specified file. \nThe file may be \
                        corrupted, invalid or otherwise unable to be read."
                )
                # re-enable loading data
                self.data_section.load_data_button["state"] = tk.ACTIVE

                # stop status
                self.status_bar.set_status("Awaiting user input.")
                self.status_bar.progress.stop()
                self.status_bar.progress["mode"] = "determinate"
                return

            # update column selections for known class labels
            self.train_section.known_select["values"] = column_names

            # update the column select panel
            colnames_dict = {x: i for i, x in enumerate(column_names)}
            self.data_section.col_select_panel.update_contents(colnames_dict)

            # conditionally enable predicting
            self.check_enable_predictions()

            # re-enable loading data
            self.data_section.load_data_button["state"] = tk.ACTIVE

            # stop status
            self.status_bar.set_status("Awaiting user input.")
            self.status_bar.progress.stop()
            self.status_bar.progress["mode"] = "determinate"

    @threaded
    def _get_sheet_cols(self, sheet):
        try:
            column_names = pd.read_excel(
                self.sp.data_file,
                sheet_name=sheet,
                na_values=utilities.NANS,
                nrows=0,
                keep_default_na=True
            ).columns.values.tolist()
        except (
                OSError, IOError, KeyError,
                TypeError, ValueError, XLRDError
        ):
            tk.messagebox.showwarning(
                title="File Read Error",
                message="Unable to read file. \nThe file may have been \
corrupted, deleted, or renamed since being selected."
            )
            # reset status and re-enable sheet selection
            self.status_bar.set_status("Awaiting user input.")
            self.data_section.excel_sheet_input["state"] = "readonly"
            self.status_bar.progress["mode"] = "determinate"
            self.status_bar.progress.stop()
            return

        # update known class label dropdown
        self.train_section.known_select["values"] = column_names
        # update column selection panel
        self.data_section.col_select_panel.update_contents(
            {x: i for i, x in enumerate(column_names)})

        self.check_enable_predictions()

        # reset status and re-enable sheet selection
        self.status_bar.set_status("Awaiting user input.")
        self.data_section.excel_sheet_input["state"] = "readonly"
        self.status_bar.progress["mode"] = "determinate"
        self.status_bar.progress.stop()

    @threaded
    def _load_classifier(self, clf_file):
        """
        Load the chosen classifier in a thread.

        Args:
            clf_file (str): The path to the classifier.
        """
        # disable train stuff in the event that a trained clf is overwritten
        try:
            self.sp.clf = utilities.load_classifier(clf_file)
        except (TypeError, KeyError):
            tk.messagebox.showwarning(
                title="Incompatible File",
                message="The chosen file is not a compatible AutoClassifier."
            )
            self.status_bar.set_status("Awaiting user input.")
            self.predict_section.classifier_load["state"] = tk.ACTIVE
            return

        # reset controls and conditionally enable steps
        self.reset_controls(clf=True)
        self.check_enable_predictions()
        # reset status and disabled button
        self.status_bar.set_status("Awaiting user input.")
        self.predict_section.classifier_load["state"] = tk.ACTIVE

    @threaded
    def _make_predictions(self, on_finish, status_cb):
        """
        Make predictions on data in a thread.

        Args:
            on_finish (func): Function to call when complete. Primarily used
                for popup progressbar.
            status_cb (func): Function to call to update status. Primarily used
                for popup progressbar.
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
            # finish updating status
            self.status_bar.progress["value"] = 100
            time.sleep(0.2)
            self.status_bar.progress["value"] = 0
            self.status_bar.set_status("Awaiting user input.")

            # call finisher function
            on_finish()
            return

        # Because predict_AC may return predict_prob we check if it's a tuple
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
        utilities.save_predictions(
            metadata, predict, features, self.output.name, predict_prob)

        # enable outputs
        self.predict_section.enable_outputs()
        self.predict_section.output_save.config(state=tk.ACTIVE)
        self.make_pairplot(features, predict)

        # finish updating status
        self.status_bar.progress["value"] = 100
        time.sleep(0.2)
        self.status_bar.progress["value"] = 0
        self.status_bar.set_status("Awaiting user input.")

        # call finisher function
        on_finish()

    @threaded
    def _retrieve_seq_data(self, on_finish):
        """
        Pull data from BOLD in a thread.

        Args:
            on_finish (func): Function to call on completion.
        """
        # retrieve the data
        self.sp.retrieve_sequence_data()

        # conditionally enable merge data button
        if self.user_sequence_raw is not None:
            self.data_retrieval_window.merge_button.config(state=tk.ACTIVE)

        on_finish()

    @threaded
    def _save_classifier(self, location):
        """Save the classifier to a given location in a thread.

        Args:
            location (str): Path to the new file to save to.
        """
        # save the classifier
        dump(self.sp.clf, location)
        # reset buttons and status
        self.train_section.classifier_save["state"] = tk.ACTIVE
        self.status_bar.progress["mode"] = "determinate"
        self.status_bar.progress.stop()

    @threaded
    def _save_item(self, item, location):
        """
        Save the given item to the given location, in a thread.

        Args:
            item (str): Identifier for which item to save.
            location (str): Path to the location to save to.
        """
        # dictionaries for easier lookup
        tempfiles = {
            "cm": self.cm,
            "pairplot": self.pairplot,
            "report": self.report,
            "output": self.output,
            "fasta_align": self.fasta_align
        }
        buttons = {
            "cm": self.train_section.cm_section.button_save,
            "pairplot": self.predict_section.pairplot_section.button_save,
            "report": self.train_section.report_section.button_save,
            "output": self.predict_section.output_save,
            "fasta_align": self.data_retrieval_window.align_save_button
        }
        # if the item's the prediction output we need to convert it to whatever
        # type the user wants because apparently scientists hate CSV
        # If you're a bioinformatics person reading through the code to figure
        # out why the program is slow to save it's because nobody can just pick
        # a standard and stick to it. Sorry for the rant, love you all <3.
        extension = os.path.splitext(location)[1]
        if item == "output" and extension == ".tsv":
            with open(tempfiles[item].name, "r") as csvin, \
                    open(location, "w") as tsvout:
                csvin = csv.reader(csvin)
                tsvout = csv.writer(tsvout, delimiter='\t')

                for row in csvin:
                    tsvout.writerow(row)

        # if user chose some other extension, we're just giving them csv
        # because I can't be bothered to infer their preference

        # otherwise, copy the appropriate file
        shutil.copy(tempfiles[item].name, location)

        # reset buttons and status
        buttons[item]["state"] = tk.ACTIVE
        self.status_bar.set_status("Awaiting user input.")

    @threaded
    def _save_nans(self):
        """Save the NaN values to nans.json in a thread."""
        nans = {"nans": self.sp.nans}
        with open(
            os.path.join(
                utilities.MAIN_PATH, "niclassify/core/config/nans.json"),
            "w"
        ) as nansfile:
            json.dump(nans, nansfile)

        # reset buttons and status
        self.status_bar.set_status("Awaiting user input.")
        self.data_section.nan_check["state"] = tk.ACTIVE

    @threaded
    def _train_classifier(self, on_finish, status_cb):
        """
        Train the classifier in a thread.

        Args:
            on_finish (func): Function to call when complete. Primarily used
                for popup progressbar.
            status_cb (func): Function to call to update status. Primarily used
                for popup progressbar.
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
        features_known, metadata_known = utilities.get_known(
            features, metadata, self.sp.class_column)

        self.status_bar.progress.step(20)
        self.status_bar.set_status("Training classifier...")

        # train classifier
        self.sp.train_AC(features, metadata, status_cb)

        self.status_bar.progress.step(20)
        self.status_bar.set_status("Generating reports...")

        # generate outputs
        self.make_report()
        self.make_cm(features_known, metadata_known[self.sp.class_column])

        # finish up status updates
        self.status_bar.progress.step(20)
        time.sleep(0.2)
        self.status_bar.progress["value"] = 0
        self.status_bar.set_status("Awaiting user input.")

        # enable related output
        self.train_section.classifier_save.config(state=tk.ACTIVE)
        self.train_section.enable_outputs()
        self.check_enable_predictions()

        # finish up
        on_finish()

    def align_seq_data(self):
        # prepare tempfile for prepped data
        self.sequence_filtered = tempfile.NamedTemporaryFile(
            mode="w+",
            prefix="filtered_sequence_",
            suffix=".tsv",
            delete=False,
            dir=self.tempdir.name
        )
        self.sequence_filtered.close()

        # prepare tempfile for fasta
        self.fasta = tempfile.NamedTemporaryFile(
            mode="w+",
            prefix="unaligned_fasta_",
            suffix=".fasta",
            delete=False,
            dir=self.tempdir.name
        )
        self.fasta.close()

        # prepare tempfile for aligned fasta
        self.fasta_align = tempfile.NamedTemporaryFile(
            mode="w+",
            prefix="aligned_fasta_",
            suffix=".fasta",
            delete=False,
            dir=self.tempdir.name
        )
        self.fasta_align.close()

        # set filenames in StandardProgram
        self.sp.fasta_fname = self.fasta.name
        self.sp.fasta_align_fname = self.fasta_align.name

        # make a new output console to show results
        # console = OutputConsole(self, self)

        # get request result tsv and prep it (write fasta)
        if self.merged_raw is not None:
            self.sp.request_fname = self.merged_raw.name
        elif self.user_sequence_raw is not None:
            self.sp.request_fname = self.user_sequence_raw
        else:
            self.sp.request_fname = self.sequence_raw.name
        data = self.sp.prep_sequence_data(self.sp.get_sequence_data())

        # save filtered data for later use
        data.to_csv(self.sequence_filtered.name, sep="\t")

        # align the fasta file
        self.sp.align_fasta(external=True)

        # enable alignment save/load buttons
        self.data_retrieval_window.align_load_button["state"] = tk.ACTIVE
        self.data_retrieval_window.align_save_button["state"] = tk.ACTIVE

        # enable other saving buttons
        self.data_retrieval_window.raw_section.enable_buttons()
        self.data_retrieval_window.filtered_section.enable_buttons()
        self.data_retrieval_window.fasta_section.enable_buttons()

        # enable next step
        self.data_retrieval_window.data_prep["state"] = tk.ACTIVE

        # advise the user to check the alignment
        tk.messagebox.showinfo(
            title="Alignment Complete",
            message="The sequence alignment is complete. \nIt is recommended \
that you review the alignment file and edit it as necessary. \nPlease \
overwrite the file provided if you make any changes."
        )

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

        self.status_bar.set_status("Awaiting user file selection...")

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
            # self.status_bar.progress.stop()
            # self.status_bar.progress.config(mode="determinate")
            self.status_bar.set_status("Awaiting user input.")
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

        # disable load data button so we don't get multiple
        self.data_section.load_data_button["state"] = tk.DISABLED

        # threaded portion of function
        self._get_data_file(data_file)

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

        # disable button and set status before launching thread
        self.status_bar.set_status("Reading sheet...")
        self.data_section.excel_sheet_input["state"] = tk.DISABLED
        self.status_bar.progress["mode"] = "indeterminate"
        self.status_bar.progress.start()

        # launch thread to get columns
        self._get_sheet_cols(sheet)

    def load_alignment(self):
        self.status_bar.set_status("Awaiting user file selection...")

        # prompt the user for the classifier file
        file = filedialog.askopenfilename(
            title="Open Edited Alignment",
            initialdir=os.path.realpath(
                os.path.join(utilities.MAIN_PATH, "output/")),
            filetypes=[("FASTA formatted sequence data", ".fasta")]
        )

        # don't do anything if the user selected nothing
        if len(file) <= 0:
            self.status_bar.set_status("Awaiting user input.")
            return

        # overwrite the alignment file
        shutil.copy(file, self.fasta_align.name)

    def load_classifier(self):
        """
        Prompt the user to select a saved classifier and load it.

        Ensures that the classifier is at least the right object.
        """
        self.status_bar.set_status("Awaiting user file selection...")

        # check if user is overwriting and make sure they're ok with it
        if not self.check_warn_overwrite():
            self.status_bar.set_status("Awaiting user input.")
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
            self.status_bar.set_status("Awaiting user input.")
            return

        # disable button while launching thread
        self.predict_section.classifier_load["state"] = tk.DISABLED
        # launch thread to load the classifier
        self._load_classifier(clf_file)

    def load_sequence_data(self):
        """
        Get the location of custom user sequence data for later use.

        Also conditionally enables the 'merge data' button.
        """
        req_cols = [
            "processid",
            "nucleotides",
            "marker_codes",
            "species_name"
        ]

        self.status_bar.set_status("Awaiting user file selection...")

        # check if user is overwriting and make sure they're ok with it
        if self.user_sequence_raw is not None:
            if not tk.messagebox.askokcancel(
                title="Overwrite Warning",
                message="You've already loaded custom sequence data. Are you \
sure?"
            ):
                self.status_bar.set_status("Awaiting user input.")
                return

        # prompt the user for the classifier file
        file = filedialog.askopenfilename(
            title="Open Data File",
            initialdir=os.path.realpath(
                os.path.join(utilities.MAIN_PATH, "data/")),
            filetypes=[
                ("Standard deliniated text file", ".txt .tsv .csv"),
                ("Excel file", ".xlsx .xlsm .xlsb .xltx .xltm .xls .xlt .xml"),
                ("Comma separated values", ".csv .txt"),
                ("Tab separated values", ".tsv .txt"),
            ]
        )
        # don't do anything if the user selected nothing
        if len(file) <= 0:
            self.status_bar.set_status("Awaiting user input.")
            return

        # check that file has required column names
        self.status_bar.set_status("Checking user sequence file...")
        data_cols = utilities.get_data(file).columns.values.tolist()
        if not all(r in data_cols for r in req_cols):
            tk.messagebox.showwarning(
                title="Invalid data file",
                message="Selected data file does not contain required columns.\
\nPlease see help document for a list of required columns with exact names."
            )
            self.status_bar.set_status("Awaiting user input.")
            return

        # set file location
        self.user_sequence_raw = file

        self.status_bar.set_status("Awaiting user input.")

        # conditionally enable merge data button
        if self.sequence_raw is not None:
            self.data_retrieval_window.merge_button.config(state=tk.ACTIVE)

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
        utilities.make_confm(
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
        try:
            with open(self.logname, "r") as log:
                loglines = log.readlines()
        except IOError:
            tk.messagebox.showerror(
                title="Report Log Error",
                message="Unable to generate report: logfile inaccessible."
            )

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
                tk.messagebox.showwarning(
                    title="Pairplot Overload",
                    message="Due to software limitations, a pairplot will not \
be generated."
                )
            else:
                tk.messagebox.showwarning(
                    title="Pairplot Size",
                    message="Due to the number of variables, a pairplot will \
not be generated."
                )
            self.predict_section.pairplot_section.disable_buttons()
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
        utilities.make_pairplot(
            features,
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

        # check if file still exists
        if not os.path.exists(self.sp.data_file):
            tk.messagebox.showwarning(
                title="File Read Error",
                message="Unable to read file. \nThe file may have been \
corrupted, deleted, or renamed since being selected."
            )
            return

        self.get_selected_cols()

        self.sp.print_vars()

        progress_popup = ProgressPopup(
            self,
            "Making Predictions",
            ""
        )

        self._make_predictions(
            progress_popup.complete,
            progress_popup.set_status
        )

    def merge_sequence_data(self):
        # TODO thread this with a progress bar

        if self.merged_raw is not None:
            answer = tk.messagebox.askyesnocancel(
                title="Existing Merged Data",
                message="Merged data already exists. \nDo you wish to merge \
additional data? Selecting 'No' will Merge current user data with BOLD data, \
overwriting previous merge."
            )

            if answer is None:
                return
            elif answer is True:
                bold_data = utilities.get_data(self.merged_raw.name)
            else:
                bold_data = utilities.get_data(self.sequence_raw.name)

        else:
            bold_data = utilities.get_data(self.sequence_raw.name)

        user_data = utilities.get_data(self.user_sequence_raw)

        # merge the two sets
        merged = pd.concat(
            (bold_data, user_data),
            axis=0,
            ignore_index=True,
            sort=False
        )

        # create merged tempfile
        self.merged_raw = tempfile.NamedTemporaryFile(
            mode="w+",
            prefix="merged_seq_unfiltered_",
            suffix=".tsv",
            delete=False,
            dir=self.tempdir.name
        )
        self.merged_raw.close()

        merged.to_csv(self.merged_raw.name, sep="\t")

        tk.messagebox.showinfo(
            title="Merge Completed",
            message="Custom and BOLD data merged successfully."
        )

    def open_nans(self):
        """Open a window to view and edit NaN values."""
        # test = tk.Toplevel(self.parent)
        self.data_section.nan_check["state"] = tk.DISABLED
        NaNEditor(self, self)

    def open_output_folder(self):
        """
        Open the output folder for classifiers, logs, etc.

        Doesn't open user-defined save locations.
        """
        utilities.view_open_file(os.path.join(utilities.MAIN_PATH, "output/"))

    def prep_sequence_data(self):
        # create delim tempfile
        self.delim = tempfile.NamedTemporaryFile(
            mode="w+",
            prefix="species_delim_",
            suffix=".csv",
            delete=False,
            dir=self.tempdir.name
        )
        self.delim.close()
        self.sp.delim_fname = self.delim.name

        # delimit the species
        self.sp.delimit_species(external=True)

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
            self.data_section.col_select_panel.update_contents({})
            self.train_section.known_select["values"] = []
            self.train_section.known_select.set(
                "")

        # disable buttons
        if not clf or self.train_section.known_select.get() == "":
            self.train_section.reset_enabled()
        if not clf:
            self.predict_section.reset_enabled()

    def retrieve_seq_data(self):
        self.sp.geo = self.data_retrieval_window.geo_input.get()
        self.sp.taxon = self.data_retrieval_window.taxon_input.get()

        if (self.sp.geo is None or self.sp.taxon is None):
            tk.messagebox.showwarning(
                title="Missing Search Term(s)",
                message="Please fill in both search terms before requesting \
data."
            )
            return
        elif len(self.sp.geo) == 0 or len(self.sp.taxon) == 0:
            tk.messagebox.showwarning(
                title="Missing Search Term(s)",
                message="Please fill in both search terms before requesting \
data."
            )
            return

        if not tk.messagebox.askokcancel(
            title="Confirm Search Terms",
            message="Please confirm the search terms: \
\nGeography: {} \nTaxonomy: {}".format(self.sp.geo, self.sp.taxon)
        ):
            return

        progress_popup = ProgressPopup(
            self.data_retrieval_window,
            "BOLD Data Download",
            "Downloading from BOLD API..."
        )

        # set up tempfile for download
        # create the tempfile
        self.sequence_raw = tempfile.NamedTemporaryFile(
            mode="w+",
            prefix="unfiltered_sequence_",
            suffix=".tsv",
            delete=False,
            dir=self.tempdir.name
        )

        self.sequence_raw.close()
        self.sp.request_fname = self.sequence_raw.name

        self._retrieve_seq_data(progress_popup.complete)

    def retrieve_seq_data_win(self):
        self.data_retrieval_window = DataRetrievalWindow(self, self)

    def save_classifier(self):
        """
        Save the current classifier to a location the user chooses.

        Like all others, initialdir is calculated using utilities MAIN_PATH
        """
        self.status_bar.set_status("Awaiting user save location...")
        # prompt user for location to save to
        location = tk.filedialog.asksaveasfilename(
            title="Save Classifier",
            initialdir=os.path.realpath(
                os.path.join(utilities.MAIN_PATH, "output/")),
            defaultextension=".gz",
            filetypes=[("GNU zipped archive", ".gz")]
        )
        # set status for user
        self.status_bar.set_status("Saving classifier...")
        self.status_bar.progress["mode"] = "indeterminate"
        self.status_bar.progress.start()

        # disable button while launching thread
        self.train_section.classifier_save["state"] = tk.DISABLED
        # launch thread to save classifier
        self._save_classifier(location)

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
            "raw_data": "Raw Data",
            "filtered_data": "Filtered Data",
            "raw_fasta": "Unaligned Fasta",
            "fasta_align": "Aligned Fasta"
        }
        graphtypes = [("Portable Network Graphics image", ".png")]
        reportypes = [("Plain text file", ".txt")]
        outputtypes = [
            ("Comma separated values", ".csv"),
            ("Tab separated values", ".tsv"),
            ("All Files", ".*"),
        ]
        fastatypes = [("FASTA formatted sequence data", ".fasta")]

        types = {
            "cm": graphtypes,
            "pairplot": graphtypes,
            "report": reportypes,
            "output": outputtypes,
            "raw_data": outputtypes,
            "filtered_data": outputtypes,
            "raw_fasta": fastatypes,
            "fasta_align": fastatypes
        }
        default_extensions = {
            "cm": ".png",
            "pairplot": ".png",
            "report": ".txt",
            "output": ".csv",
            "raw_data": ".csv",
            "filtered_data": ".csv",
            "raw_fasta": ".fasta",
            "fasta_align": ".fasta"
        }
        buttons = {
            "cm": self.train_section.cm_section.button_save,
            "pairplot": self.predict_section.pairplot_section.button_save,
            "report": self.train_section.report_section.button_save,
            "output": self.predict_section.output_save,
            "raw_data": self.data_retrieval_window.raw_section.button_save,
            "filtered_data": self.data_retrieval_window.filtered_section.button_save,
            "raw_fasta": self.data_retrieval_window.fasta_section.button_save,
            "fasta_align": self.data_retrieval_window.align_save_button
        }

        self.status_bar.set_status("Awaiting user save location...")

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
            self.status_bar.set_status("Awaiting user input.")
            return

        # set status and disable button while launching thread
        self.status_bar.set_status("Saving {}...".format(item))
        buttons[item]["state"] = tk.DISABLED

        # launch thread to save item
        self._save_item(item, location)

    def save_nans(self):
        """Save nans value list to nans.json."""
        # set status and disable button while launching thread
        self.status_bar.set_status("Saving NaN values...")
        self.data_section.nan_check["state"] = tk.DISABLED
        # launch thread to save NaN values
        self._save_nans()

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

        # check if file still exists
        if not os.path.exists(self.sp.data_file):
            tk.messagebox.showwarning(
                title="File Read Error",
                message="Unable to read file. \nThe file may have been \
corrupted, deleted, or renamed since being selected."
            )
            return

        # get multirun setting
        self.sp.multirun = int(self.train_section.n_input.get())

        # for debug
        self.sp.print_vars()

        # check if user is overwriting and make sure they're ok with it
        if not self.check_warn_overwrite():
            return

        # disable save output and pairplot buttons
        self.predict_section.output_save.config(state=tk.DISABLED)
        self.predict_section.pairplot_section.disable_buttons()

        # progress indicator so user doesn't think we're frozen
        progress_popup = ProgressPopup(
            self,
            "Training Classifier",
            ""
        )

        # start threaded portion
        self._train_classifier(
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
            utilities.view_open_file(item)
        except OSError:
            if tk.messagebox.askokcancel(
                title="No Default Program",
                message="The file could not be opened becase there is no \
default program associated with the filetype <{}>. \nAn explorer window with \
the file selected will be opened so you may select a program.".format(
                    os.path.splitext(item)[1]),
                icon='warning'
            ):
                subprocess.Popen('explorer /select,{}'.format(item))


def main():
    """Run the GUI."""
    utilities.assure_path()

    root = tk.Tk()

    root.style = ttk.Style()
    app = MainApp(root)
    root.update()
    root.minsize(root.winfo_width(), root.winfo_height())

    def graceful_exit():
        """
        Exit the program gracefully.

        This includes cleaning tempfiles and closing any processes.
        """
        print(threading.active_count())
        plt.close("all")
        app.tempdir.cleanup()
        root.quit()
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", graceful_exit)

    root.mainloop()


if __name__ == "__main__":
    main()
