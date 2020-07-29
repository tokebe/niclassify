import csv
import json
import matplotlib
import os
import requests
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

from .datapanels import RetrievalPanel, PreparationPanel
from .elements import VS_Pair
from .smallwindows import ProgressPopup
from .threadwrap import threaded

# TODO add check and warning for unmerged data


class DataPreparationTool(tk.Toplevel):
    """A window for retrieving data from BOLD."""

    def __init__(self, parent, app, tempdir, utilities, *args, **kwargs):
        """
        Instantiate the window.

        Args:
            parent (TopLevel): The Parent window.
            app (MainApp): Generally the MainApp for easy method access.
        """
        super().__init__(parent, *args, **kwargs)
        self.parent = parent
        self.app = app
        self.util = utilities

        # tempdir for tempfiles
        self.tempdir = tempdir
        # tempfiles
        self.sequence_raw = None
        self.user_sequence_raw = None
        self.merged_raw = None
        self.sequence_filtered = None
        self.fasta = None
        self.fasta_align = None
        self.delim = None

        self.title("Sequence Data Tool")
        self.panels = tk.Frame(self)
        self.panels.pack(fill=tk.BOTH, expand=True)

        # stop main window interaction
        self.grab_set()

        self.get_data_sec = RetrievalPanel(
            self,
            self,
            text="Data Retrieval",
            labelanchor=tk.N
        )
        self.get_data_sec.pack(expand=True, fill=tk.X)

        self.data_sec = PreparationPanel(
            self,
            self,
            text="Data Preparation",
            labelanchor=tk.N
        )
        self.data_sec.pack(expand=True, fill=tk.X)

        self.minsize(300, self.winfo_height())
        self.resizable(False, False)

    def align_seq_data(self):
        """Filter and align sequences."""
        # ----- threaded function -----
        @threaded
        def _align_seq_data(on_finish, status_cb):
            """Filter and align sequences, in a thread.

            Args:
                on_finish (func): Function to call on completion.
                status_cb (func): Callback function for status updates.
            """
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
            self.app.sp.filtered_fname = self.sequence_filtered.name
            self.app.sp.fasta_fname = self.fasta.name
            self.app.sp.fasta_align_fname = self.fasta_align.name

            # get request result tsv and prep it (filter + write fasta)
            # check if using merged, user, or downloaded data
            if self.merged_raw is not None:
                self.app.sp.request_fname = self.merged_raw.name
            elif self.user_sequence_raw is not None:
                self.app.sp.request_fname = self.user_sequence_raw
            else:
                self.app.sp.request_fname = self.sequence_raw.name

            data = self.app.sp.prep_sequence_data(
                self.app.sp.get_sequence_data())

            # save filtered data for later use
            data.to_csv(self.sequence_filtered.name, sep="\t", index=False)

            status_cb("Aligning sequences...")

            # align the fasta file
            self.app.sp.align_fasta(external=True)

            # enable alignment save/load buttons
            self.data_sec.align_load_button["state"] = tk.ACTIVE
            self.data_sec.align_save_button["state"] = tk.ACTIVE

            # enable other saving buttons
            self.data_sec.filtered_sec.enable_buttons()
            self.data_sec.fasta_sec.enable_buttons()

            # enable next step
            self.data_sec.data_prep["state"] = tk.ACTIVE

            on_finish()

            # advise the user to check the alignment
            tk.messagebox.showinfo(
                parent=self,
                title="Alignment Complete",
                message="The sequence alignment is complete. \nIt is \
recommended that you review the alignment file and edit it as necessary. \
\nPlease load the new file after making any changes."
            )
        # ----- end threaded function -----

        # disable buttons by opening progress bar
        progress_popup = ProgressPopup(
            self,
            "Alignment",
            "Filtering data..."
        )

        # start threaded function
        _align_seq_data(
            progress_popup.complete,
            progress_popup.set_status
        )

    def get_geographies(self):
        return sorted(self.util.get_geographies())
        # return self.util.get_geographies()

    def load_item(self, item):
        self.app.status_bar.set_status("Awaiting user file selection...")

        filetypes = {
            "alignment": [("FASTA formatted sequence data", ".fasta")],
            "filtered": [
                ("All files", ".*"),
                ("Comma-separated values", ".csv"),
                ("Tab-separated values", ".tsv"),
            ]
        }
        tempfiles = {
            "alignment": self.fasta_align.name,
            "filtered": self.sequence_filtered.name
        }

        # prompt the user for the classifier file
        file = filedialog.askopenfilename(
            title="Open Edited Alignment",
            initialdir=os.path.realpath(
                os.path.join(self.util.MAIN_PATH, "output/")),
            filetypes=filetypes[item]
        )

        # don't do anything if the user selected nothing
        if len(file) <= 0:
            self.app.status_bar.set_status("Awaiting user input.")
            return

        # overwrite the alignment file
        shutil.copy(file, tempfiles[item])

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

        self.app.status_bar.set_status("Awaiting user file selection...")

        # check if user is overwriting and make sure they're ok with it
        if self.user_sequence_raw is not None:
            if not tk.messagebox.askokcancel(
                title="Overwrite Warning",
                message="You've already loaded custom sequence data. Are you \
sure?"
            ):
                self.app.status_bar.set_status("Awaiting user input.")
                return

        # prompt the user for the classifier file
        # TODO ask if user wants to keep current merged file
        file = filedialog.askopenfilename(
            title="Open Data File",
            initialdir=os.path.realpath(
                os.path.join(self.util.MAIN_PATH, "data/")),
            filetypes=[
                ("Standard deliniated text file", ".txt .tsv .csv"),
                ("Excel file", ".xlsx .xlsm .xlsb .xltx .xltm .xls .xlt .xml"),
                ("Comma separated values", ".csv .txt"),
                ("Tab separated values", ".tsv .txt"),
            ]
        )
        # don't do anything if the user selected nothing
        if len(file) <= 0:
            self.app.status_bar.set_status("Awaiting user input.")
            return

        # check that file has required column names
        self.app.status_bar.set_status("Checking user sequence file...")
        data_cols = self.util.get_data(file).columns.values.tolist()
        if not all(r in data_cols for r in req_cols):
            tk.messagebox.showwarning(
                title="Invalid data file",
                message="Selected data file does not contain required columns.\
\nPlease see help document for a list of required columns with exact names."
            )
            self.app.status_bar.set_status("Awaiting user input.")
            return

        # set file location
        self.user_sequence_raw = file

        self.app.status_bar.set_status("Awaiting user input.")

        # conditionally enable merge data button
        if self.sequence_raw is not None:
            self.get_data_sec.merge_button.config(state=tk.ACTIVE)

        # enable alignment button
        self.data_sec.align_button["state"] = tk.ACTIVE

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
                bold_data = self.util.get_data(self.merged_raw.name)
            else:
                bold_data = self.util.get_data(self.sequence_raw.name)

        else:
            bold_data = self.util.get_data(self.sequence_raw.name)

        user_data = self.util.get_data(self.user_sequence_raw)

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

        merged.to_csv(self.merged_raw.name, sep="\t", index=False)

        tk.messagebox.showinfo(
            parent=self,
            title="Merge Completed",
            message="Custom and BOLD data merged successfully."
        )

    def prep_sequence_data(self):
        # ----- threaded function -----
        @threaded
        def _prep_sequence_data(on_finish, status_cb):
            # create delim tempfile
            self.delim = tempfile.NamedTemporaryFile(
                mode="w+",
                prefix="species_delim_",
                suffix=".csv",
                delete=False,
                dir=self.tempdir.name
            )
            self.delim.close()
            self.app.sp.delim_fname = self.delim.name

            # delimit the species
            self.app.sp.delimit_species(external=True)

            status_cb(
                "Looking up known species statuses \
(this will take some time)...")

            # get statuses
            self.app.sp.ref_geo = self.data_sec.ref_geo_select.get()
            self.app.sp.lookup_status()

            n_classified = self.util.get_data(
                self.sequence_filtered.name).count()["final_status"]

            # next steps here

            self.data_sec.final_save_button["state"] = tk.ACTIVE
            self.data_sec.final_load_button["state"] = tk.ACTIVE

            # TODO change this threshold to something that makes sense
            if n_classified < 500:
                tk.messagebox.showwarning(
                    parent=self,
                    title="Low Classification Count",
                    message="Only {} observations were succesfully classified \
by online search. This may result in an inaccurate classifier if used for \
training.\nAdditional manual classification, if possible, is recommended.\
".format(n_classified)
                )
            tk.messagebox.showinfo(
                parent=self,
                title="Data Preparation Complete",
                message="{} observations were classified by online lookup.\
\nData is ready for use with classifier tool.".format(n_classified)
            )

            on_finish()
        # ----- end threaded function -----

        # make popup to keep user from pressing buttons and breaking it
        progress = ProgressPopup(
            self,
            "Data Preparation",
            "Delimiting species..."
        )

        # run time-consuming items in thread
        _prep_sequence_data(progress.complete, progress.set_status)

    def retrieve_seq_data(self):
        # ----- threaded function -----
        @threaded
        def _retrieve_seq_data(on_finish):
            """
            Pull data from BOLD in a thread.

            Args:
                on_finish (func): Function to call on completion.
            """
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
            self.app.sp.request_fname = self.sequence_raw.name
            try:
                # retrieve the data
                self.app.sp.retrieve_sequence_data()
            except requests.exceptions.RequestException:
                tk.messagebox.showerror(
                    parent=self,
                    title="Connection Error",
                    message="There was an error searching BOLD. \nPlease check \
    your connection and the status of the BOLD servers."
                )
                on_finish()
                return

            # check if file downloaded properly (parses successfully)
            try:
                nlines = self.util.get_data(self.sequence_raw.name).shape[0]
            except pd.errors.ParserError:
                tk.messagebox.showerror(
                    parent=self,
                    title="Download Error",
                    message="Download from BOLD failed, please try again."
                )
                on_finish()
                return

            # conditionally enable merge data button
            if self.user_sequence_raw is not None:
                self.get_data_sec.merge_button.config(state=tk.ACTIVE)

            self.data_sec.align_button["state"] = tk.ACTIVE

            on_finish()

            if nlines == 0:
                tk.messagebox.showwarning(
                    parent=self,
                    title="No Observations Found",
                    message="No observations were found. Please check your search \
    terms."
                )
                return

            # tell user it worked/how many lines downloaded
            tk.messagebox.showinfo(
                parent=self,
                title="BOLD Lookup Complete",
                message="BOLD lookup returned {} observations.".format(nlines)
            )
        # ----- end threaded function -----

        self.app.sp.geo = self.get_data_sec.geo_input.get()
        self.app.sp.taxon = self.get_data_sec.taxon_input.get()

        if (self.app.sp.geo is None or self.app.sp.taxon is None):
            tk.messagebox.showwarning(
                title="Missing Search Term(s)",
                message="Please fill in both search terms before requesting \
data."
            )
            return
        elif len(self.app.sp.geo) == 0 or len(self.app.sp.taxon) == 0:
            tk.messagebox.showwarning(
                title="Missing Search Term(s)",
                message="Please fill in both search terms before requesting \
data."
            )
            return

        if not tk.messagebox.askokcancel(
            parent=self,
            title="Confirm Search Terms",
            message="Please confirm the search terms: \
\nGeography: {} \nTaxonomy: {}".format(self.app.sp.geo, self.app.sp.taxon)
        ):
            return

        progress_popup = ProgressPopup(
            self,
            "BOLD Data Download",
            "Downloading from BOLD API..."
        )

        _retrieve_seq_data(progress_popup.complete)
