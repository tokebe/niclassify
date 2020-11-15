"""
Data preparation tool window.

Used exclusively in tandem with the classifer tool, which handles some key
functions such as access to the inner-layer StandardProgram.
"""
import csv
import os
from os.path import normcase
import re
import requests
import shutil

import tempfile

import numpy as np
import pandas as pd
import tkinter as tk

from tkinter import filedialog
from tkinter import messagebox
from pandas.errors import EmptyDataError, ParserError

from .datapanels import RetrievalPanel, PreparationPanel
from .smallwindows import ProgressPopup
from .threadwrap import threaded


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

        # filepath for most recent data for keeping selection up to date
        self.last_entered_data = None

        self.report_callback_exception = self.app.uncaught_exception

        self.taxon_level_name = "Order"
        self.taxon_level = "order_name"

        # tempdir for tempfiles
        self.tempdir = tempdir
        # tempfiles
        self.sequence_raw = None
        self.sequence_previous = None
        self.user_sequence_raw = None
        self.user_sequence_previous = None
        self.merged_raw = None
        self.sequence_filtered = None
        self.fasta = None
        self.fasta_align = None
        self.delim = None
        self.seq_features = None
        self.finalized_data = None

        self.title("Sequence Data Tool")

        # intialize UI panels
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

        def on_exit():
            self.app.data_sec.retrieve_data_button["state"] = tk.ACTIVE
            self.destroy()

        self.protocol("WM_DELETE_WINDOW", on_exit)

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
            # prepare tempfile for aligned fasta
            # self.fasta_align = tempfile.NamedTemporaryFile(
            #     mode="w+",
            #     prefix="aligned_fasta_",
            #     suffix=".fasta",
            #     delete=False,
            #     dir=self.tempdir.name
            # )
            # self.fasta_align.close()
            # self.app.sp.fasta_align_fname = self.fasta_align.name

            # align the fasta file
            try:
                self.app.sp.align_fasta(debug=False)
            except ChildProcessError:
                self.app.dlib.dialog(
                    messagebox.showerror,
                    "ALIGN_ERR",
                    parent=self
                )

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
            self.app.dlib.dialog(
                messagebox.showinfo, "ALIGNMENT_COMPLETE", parent=self)
        # ----- end threaded function -----

        # disable buttons by opening progress bar
        progress_popup = ProgressPopup(
            self,
            "Alignment",
            "Aligning Sequences..."
        )

        # start threaded function
        _align_seq_data(
            progress_popup.complete,
            progress_popup.set_status
        )

    def filter_seq_data(self):
        """Filter Sequence Data"""
        # ----- threaded function -----
        @threaded
        def _filter_seq_data(on_finish):
            # prepare tempfile for prepped data
            self.sequence_filtered = tempfile.NamedTemporaryFile(
                mode="w+",
                prefix="filtered_sequence_",
                suffix=".tsv",
                delete=False,
                dir=self.tempdir.name
            )
            self.sequence_filtered.close()
            self.app.sp.filtered_fname = self.sequence_filtered.name

            # prepare tempfile for fasta
            self.fasta = tempfile.NamedTemporaryFile(
                mode="w+",
                prefix="unaligned_fasta_",
                suffix=".fasta",
                delete=False,
                dir=self.tempdir.name
            )
            self.fasta.close()
            self.app.sp.fasta_fname = self.fasta.name

            # prepare tempfile for aligned fasta
            self.fasta_align = tempfile.NamedTemporaryFile(
                mode="w+",
                prefix="aligned_fasta_",
                suffix=".fasta",
                delete=False,
                dir=self.tempdir.name
            )
            self.fasta_align.close()
            self.app.sp.fasta_align_fname = self.fasta_align.name

            # get request result tsv and prep it (filter + write fasta)
            self.app.sp.request_fname = self.last_entered_data

            data = self.app.sp.prep_sequence_data(
                self.app.sp.get_sequence_data())

            # save filtered data for later use
            data.to_csv(self.sequence_filtered.name, sep="\t", index=False)

            # enable alignment buttons
            self.data_sec.align_button["state"] = tk.ACTIVE
            self.data_sec.align_load_button["state"] = tk.ACTIVE

            on_finish()
        # ----- end threaded function -----

        # disable buttons by opening progress bar
        progress_popup = ProgressPopup(
            self,
            "Data Preparation",
            "Filtering Sequences..."
        )

        _filter_seq_data(progress_popup.complete)

    def get_geographies(self):
        """
        Return a list of all geographies.

        Returns:
            list: All configured geographies.

        """
        return sorted(self.util.get_geographies())
        # return self.util.get_geographies()

    def load_item(self, item):
        """
        Load an item into the program.

        Args:
            item (str): key for which item to load.
        """
        filetypes = {
            "alignment": [("FASTA formatted sequence data", ".fasta")],
            "filtered": [
                ("All files", ".*"),
                ("Comma-separated values", ".csv"),
                ("Tab-separated values", ".tsv"),
            ],
            "finalized": [
                ("All files", ".*"),
                ("Comma-separated values", ".csv"),
                ("Tab-separated values", ".tsv"),
            ],
        }
        tempfiles = {
            "alignment": self.fasta_align.name,
            "filtered": self.sequence_filtered.name,
            "finalized": (self.finalized_data.name
                          if self.finalized_data is not None else None)
        }
        # ----- threaded function -----

        @threaded
        def _load_alignment(alignfname, on_finish, status_cb):
            # load raw data to check alignment against
            data = self.util.get_data(self.sequence_filtered.name)

            # check that the alignment matches the data
            with open(alignfname, "r") as file:
                names = []
                # multiple lines per sequence
                for line in file.readlines():
                    names.extend(re.findall("(?<=>).*(?=\n)", line))

                # print(names)

                names_not_found = []

                status_cb("Checking sample UPIDs...")

                for name in names:
                    if name not in data["UPID"].unique():
                        # print("'{}' not found".format(name))
                        names_not_found.append(name)

                if len(names_not_found) > 0:
                    self.app.dlib.dialog(
                        messagebox.showwarning,
                        "ALIGN_MISMATCH",
                        parent=self,
                        form="\n".join(names_not_found)
                    )

            # load file
            shutil.copy(alignfname, tempfiles[item])
            self.data_sec.align_load_button["state"] = tk.ACTIVE
            self.data_sec.data_prep["state"] = tk.ACTIVE

            on_finish()

        # ----- end threaded function -----
        self.app.status_bar.set_status("Awaiting user file selection...")

        # prompt the user for the classifier file
        file = filedialog.askopenfilename(
            title="Open Edited Alignment",
            initialdir=os.path.realpath(
                os.path.join(self.util.MAIN_PATH, "output/")),
            filetypes=filetypes[item],
            parent=self
        )

        # don't do anything if the user selected nothing
        if len(file) <= 0:
            self.app.status_bar.set_status("Awaiting user input.")
            return

        if item == "alignment":
            self.data_sec.align_load_button["state"] = tk.DISABLED

            progress = ProgressPopup(
                self,
                "Reading Alignment",
                "Reading file..."
            )

            # enable next step
            _load_alignment(file, progress.complete, progress.set_status)
        else:
            # overwrite the alignment file
            shutil.copy(file, tempfiles[item])

    def load_sequence_data(self):
        """
        Get the location of custom user sequence data for later use.

        Also conditionally enables the 'merge data' button.
        """
        req_cols = [
            ["processid", "UPID"],  # can have one or the other
            "nucleotides",
            "marker_codes",
            "species_name"
        ]

        self.app.status_bar.set_status("Awaiting user file selection...")

        # check if user is overwriting and make sure they're ok with it
        if self.user_sequence_raw is not None:
            if not self.app.dlib.dialog(
                    messagebox.askokcancel, "SEQUENCE_OVERWRITE", parent=self):
                self.app.status_bar.set_status("Awaiting user input.")
                return

            # keep old data in case user wants to merge
            self.user_sequence_previous = self.user_sequence_raw

        # prompt the user for the sequence file
        file = filedialog.askopenfilename(
            title="Open Data File",
            initialdir=os.path.realpath(
                os.path.join(self.util.MAIN_PATH, "data/")),
            filetypes=[
                ("Standard deliniated text file", ".txt .tsv .csv"),
                ("Excel file", ".xlsx .xlsm .xlsb .xltx .xltm .xls .xlt .xml"),
                ("Comma separated values", ".csv .txt"),
                ("Tab separated values", ".tsv .txt"),
            ],
            parent=self
        )
        # don't do anything if the user selected nothing
        if len(file) <= 0:
            self.app.status_bar.set_status("Awaiting user input.")
            return

        # check that file has required column names
        self.app.status_bar.set_status("Checking user sequence file...")
        try:
            data = self.util.get_data(file)
        except (ParserError, EmptyDataError, OSError, IOError, KeyError,
                TypeError, ValueError, csv.Error):
            self.app.dlib.dialog(
                messagebox.showwarning,
                "FILE_READ_ERR"
            )
            return

        if not all(
            r in data.columns.values.tolist()
                if not isinstance(r, list)
                else any(s in data.columns.values.tolist() for s in r)
            for r in req_cols
        ):
            self.app.dlib.dialog(
                messagebox.showwarning, "INVALID_SEQUENCE_DATA", parent=self)
            self.app.status_bar.set_status("Awaiting user input.")
            return
        # # check if processid is unique, warn user if not (fix in filtering)
        # elif not data["processid"].is_unique:
        #     self.app.dlib.dialog(
        #         messagebox.showwarning, "PID_NOT_UNIQUE", parent=self)

        # set file location
        self.user_sequence_raw = file
        self.last_entered_data = self.user_sequence_raw

        self.app.status_bar.set_status("Awaiting user input.")

        # conditionally enable merge data button
        if (self.sequence_raw is not None
                or self.user_sequence_previous is not None):
            self.get_data_sec.merge_button.config(state=tk.ACTIVE)

        # enable filter button
        self.data_sec.filter_button["state"] = tk.ACTIVE
        # disable buttons
        self.data_sec.align_button["state"] = tk.DISABLED
        self.data_sec.align_load_button["state"] = tk.DISABLED
        self.data_sec.align_save_button["state"] = tk.DISABLED
        self.data_sec.data_prep["state"] = tk.DISABLED
        self.data_sec.use_data_button["state"] = tk.DISABLED
        self.data_sec.final_load_button["state"] = tk.DISABLED
        self.data_sec.final_save_button["state"] = tk.DISABLED
        self.data_sec.use_data_button["state"] = tk.DISABLED

    def merge_sequence_data(self, bold=False):
        """
        Merge multiple sequence files.

        Args:
            bold (bool, optional): Denotes merging BOLD search results.
                Defaults to False.
        """
        # ----- threaded function -----
        @threaded
        def _merge_sequence_data(on_finish):
            if self.merged_raw is not None:
                if bold:
                    answer = self.app.dlib.dialog(
                        messagebox.askyesnocancel,
                        "MESSAGE_MERGE_BOLD",
                        parent=self
                    )
                else:
                    answer = self.app.dlib.dialog(
                        messagebox.askyesnocancel,
                        "EXISTING_MERGE_USER",
                        parent=self
                    )

                if answer is None:
                    return
                elif answer is True:
                    bold_data = self.util.get_data(self.merged_raw.name)
                    if bold:
                        user_data = self.util.get_data(self.sequence_raw.name)
                    else:
                        user_data = self.util.get_data(self.user_sequence_raw)
                else:
                    bold_data = self.util.get_data(self.sequence_raw.name)
                    if bold:
                        user_data = self.util.get_data(
                            self.sequence_previous.name)
                    else:
                        user_data = self.util.get_data(self.user_sequence_raw)

            else:
                if self.sequence_raw is None:
                    bold_data = self.util.get_data(self.user_sequence_previous)
                else:
                    bold_data = self.util.get_data(self.sequence_raw.name)
                if bold:
                    user_data = self.util.get_data(self.sequence_previous.name)
                else:
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

            self.last_entered_data = self.merged_raw.name

            self.get_data_sec.save_merge_button["state"] = tk.ACTIVE

            # re-enable buttons
            self.get_data_sec.merge_bold_button["state"] = tk.ACTIVE
            self.get_data_sec.merge_button["state"] = tk.ACTIVE

            # disable buttons
            self.data_sec.align_button["state"] = tk.DISABLED
            self.data_sec.align_load_button["state"] = tk.DISABLED
            self.data_sec.align_save_button["state"] = tk.DISABLED
            self.data_sec.data_prep["state"] = tk.DISABLED
            self.data_sec.use_data_button["state"] = tk.DISABLED
            self.data_sec.final_load_button["state"] = tk.DISABLED
            self.data_sec.final_save_button["state"] = tk.DISABLED
            self.data_sec.use_data_button["state"] = tk.DISABLED
            self.data_sec.fasta_sec.disable_buttons()
            self.data_sec.filtered_sec.disable_buttons()

            on_finish()

            self.app.dlib.dialog(
                messagebox.showinfo, "MERGE_COMPLETE", parent=self)

        # ----- end threaded function -----

        # disable buttons
        self.get_data_sec.merge_bold_button["state"] = tk.DISABLED
        self.get_data_sec.merge_button["state"] = tk.DISABLED

        # make popup to keep user from pressing buttons and breaking it
        progress = ProgressPopup(
            self,
            "Data Merge",
            "Merging data..."
        )

        _merge_sequence_data(progress.complete)

    def prep_sequence_data(self):
        """Prepare aligned sequence data."""
        # ----- threaded function -----
        @threaded
        def _prep_sequence_data(on_finish, status_cb):

            data = self.util.get_data(self.sequence_filtered.name)

            # TODO add actual cancelling functionality to these two
            if data[self.taxon_level].isna().any():
                if not self.app.dlib.dialog(
                    messagebox.askokcancel,
                    "NAN_TAXON",
                    form=(self.taxon_level_name,),
                    parent=self
                ):
                    on_finish()
                    return

            if 1 in data[self.taxon_level].value_counts(dropna=False).values:
                if not self.app.dlib.dialog(
                    messagebox.askokcancel,
                    "SINGLE_SPLIT",
                    form=(self.taxon_level_name,),
                    parent=self
                ):
                    on_finish()
                    return

            method = self.data_sec.method_select.get()

            # create delim tempfile
            self.delim = tempfile.NamedTemporaryFile(
                mode="w+",
                prefix="species_delim_",
                suffix=".tsv",
                delete=False,
                dir=self.tempdir.name
            )
            self.delim.close()
            self.app.sp.delim_fname = self.delim.name

            self.seq_features = tempfile.NamedTemporaryFile(
                mode="w+",
                prefix="features_",
                suffix=".tsv",
                delete=False,
                dir=self.tempdir.name
            )
            self.seq_features.close()
            self.app.sp.seq_features_fname = self.seq_features.name

            self.finalized_data = tempfile.NamedTemporaryFile(
                mode="w+",
                prefix="finalized_",
                suffix=".csv",
                delete=False,
                dir=self.tempdir.name
            )
            self.finalized_data.close()
            self.app.sp.finalized_fname = self.finalized_data.name

            # delimit the species
            print("DELIMITING SPECIES...")
            try:
                # TODO disable debug
                self.app.sp.delimit_species(
                    method, tax=self.taxon_level, debug=True)
            except (ChildProcessError, FileNotFoundError, IndexError) as err:
                self.app.dlib.dialog(
                    messagebox.showerror,
                    "DELIM_ERR",
                    form=(self.taxon_level_name, str(err)),
                    parent=self
                )
                on_finish()
                return

            status_cb("Generating species features \
(This will take some time)...")
            print("GENERATING FEATURES...")
            try:
                self.app.sp.generate_features(tax=self.taxon_level, debug=True)
            except (ChildProcessError, FileNotFoundError) as err:
                self.app.dlib.dialog(
                    messagebox.showerror,
                    "FEATURE_GEN_ERR",
                    form=(self.taxon_level_name, str(err)),
                    parent=self
                )
                on_finish()
                return

            status_cb(
                "Looking up known species statuses \
(this will take some time)...")
            print("EXECUTING STATUS LOOKUP...")
            # get statuses
            try:
                self.app.sp.ref_geo = self.data_sec.ref_geo_select.get()
                self.app.sp.lookup_status()
            except ChildProcessError:
                self.app.dlib.dialog(
                    messagebox.showerror,
                    "GEO_LOOKUP_ERR",
                    parent=self
                )
                on_finish()
                return

            final = self.util.get_data(self.finalized_data.name)
            n_classified = final.count()["final_status"]
            n_classes = final["final_status"].nunique()
            class_std = final["final_status"].value_counts(
                normalize=True).std()

            self.data_sec.final_save_button["state"] = tk.ACTIVE
            self.data_sec.final_load_button["state"] = tk.ACTIVE
            self.data_sec.use_data_button["state"] = tk.ACTIVE

            if n_classes < 2:
                self.app.dlib.dialog(
                    messagebox.showwarning,
                    "NOT_ENOUGH_CLASSES",
                    parent=self
                )
            else:
                if n_classified < 100:
                    self.app.dlib.dialog(
                        messagebox.showwarning,
                        "LOW_CLASS_COUNT",
                        form=(n_classified,),
                        parent=self
                    )
                # check for extreme known class imbalance
                # using stdev as a very rough heuristic
                if class_std > 0.35:
                    self.app.dlib.dialog(
                        messagebox.showwarning,
                        "HIGH_IMBALANCE"
                    )

                self.app.dlib.dialog(
                    messagebox.showinfo,
                    "DATA_PREP_COMPLETE",
                    form=(n_classified,),
                    parent=self
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
        """Search for sequence data from BOLD."""
        # ----- threaded function -----
        @threaded
        def _retrieve_seq_data(on_finish):
            """
            Pull data from BOLD in a thread.

            Args:
                on_finish (func): Function to call on completion.
            """
            # keep old search file if it exits
            if self.sequence_raw is not None:
                self.sequence_previous = self.sequence_raw
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
                self.app.dlib.dialog(
                    messagebox.showerror, "BOLD_SEARCH_ERR", parent=self)
                on_finish()
                return

            # check if file downloaded properly (parses successfully)
            try:
                nlines = self.util.get_data(self.sequence_raw.name).shape[0]
            except ParserError:
                self.app.dlib.dialog(
                    messagebox.showerror, "BOLD_FILE_ERR", parent=self)
                on_finish()
                return
            except EmptyDataError:
                self.app.dlib.dialog(
                    messagebox.showerror, "BOLD_NO_OBSERVATIONS", parent=self)
                on_finish()
                return

            self.last_entered_data = self.sequence_raw.name

            # conditionally enable merge data button
            if self.user_sequence_raw is not None:
                self.get_data_sec.merge_button.config(state=tk.ACTIVE)

            self.data_sec.filter_button["state"] = tk.ACTIVE
            self.get_data_sec.save_bold_button["state"] = tk.ACTIVE
            if (self.merged_raw is not None
                    or self.sequence_previous is not None):
                self.get_data_sec.merge_bold_button["state"] = tk.ACTIVE

            # disable buttons
            self.data_sec.align_button["state"] = tk.DISABLED
            self.data_sec.align_load_button["state"] = tk.DISABLED
            self.data_sec.align_save_button["state"] = tk.DISABLED
            self.data_sec.data_prep["state"] = tk.DISABLED
            self.data_sec.use_data_button["state"] = tk.DISABLED
            self.data_sec.final_load_button["state"] = tk.DISABLED
            self.data_sec.final_save_button["state"] = tk.DISABLED
            self.data_sec.use_data_button["state"] = tk.DISABLED
            self.data_sec.fasta_sec.disable_buttons()
            self.data_sec.filtered_sec.disable_buttons()

            on_finish()

            if nlines == 0:
                self.app.dlib.dialog(
                    messagebox.showwarning,
                    "BOLD_NO_OBSERVATIONS",
                    parent=self
                )

                return

            # tell user it worked/how many lines downloaded
            self.app.dlib.dialog(
                messagebox.showinfo,
                "BOLD_SEARCH_COMPLETE",
                parent=self,
                form=(nlines,)
            )
        # ----- end threaded function -----

        self.app.sp.geo = self.get_data_sec.geo_input.get()
        self.app.sp.taxon = self.get_data_sec.taxon_input.get()

        if (self.app.sp.geo is None or self.app.sp.taxon is None):
            self.app.dlib.dialog(
                messagebox.showwarning, "MISSING_SEARCH_TERMS", parent=self)
            return
        elif len(self.app.sp.geo) == 0 or len(self.app.sp.taxon) == 0:
            self.app.dlib.dialog(
                messagebox.showwarning, "MISSING_SEARCH_TERMS", parent=self)

            return

        if not self.app.dlib.dialog(
            messagebox.askokcancel,
            "CONFIRM_SEARCH_TERMS",
            form=(self.app.sp.geo, self.app.sp.taxon),
            parent=self
        ):
            return

        progress_popup = ProgressPopup(
            self,
            "BOLD Data Download",
            "Downloading from BOLD API..."
        )

        _retrieve_seq_data(progress_popup.complete)

    def set_taxon_level(self, event):
        levels = {
            "Phylum": "phylum_name",
            "Class": "class_name",
            "Order": "order_name",
            "Family": "family_name",
            "Subfamily": "subfamily_name",
            "Genus": "genus_name"
        }

        self.taxon_level_name = self.data_sec.taxon_split_selector.get()
        self.taxon_level = levels[self.taxon_level_name]

    def transfer_prepared_data(self):
        """Transfer prepared data to the classifier tool's data handling."""
        # drop extra columns to avoid confusion
        final_trimmed = tempfile.NamedTemporaryFile(
            mode="w+",
            prefix="final_trimmed_",
            suffix=".csv",
            delete=False,
            dir=self.tempdir.name
        )

        final_trimmed.close()

        data = self.util.get_data(self.finalized_data.name)
        cols = [
            "UPID",
            "processid",
            "phylum_name",
            "order_name",
            "family_name",
            "subfamily_name",
            "genus_name",
            "species_name",
            "subspecies_name",
            "species_group",
            "gbif_status",
            "itis_status",
            "final_status",
            "ksSim_mean",
            "ksSim_med",
            "ksSim_std",
            "ksSim_max",
            "kaSim_mean",
            "kaSim_med",
            "kaSim_std",
            "kaSim_max",
            "aaDist_mean",
            "aaDist_med",
            "aaDist_std",
            "aaDist_min",
            "aaDist_max",
            "aaSim_mean",
            "aaSim_med",
            "aaSim_std",
            "aaSim_max",
            "dnaDist_mean",
            "dnaDist_med",
            "dnaDist_std",
            "dnaDist_min",
            "dnaDist_max",
            "dnaSim_mean",
            "dnaSim_med",
            "dnaSim_std",
            "dnaSim_max"
        ]

        data = data.loc[:, data.columns.isin(cols)]
        data.to_csv(final_trimmed.name, index=False)

        self.app.get_data_file(internal=final_trimmed.name)
