"""
Data preparation tool window.

Used exclusively in tandem with the classifer tool, which handles some key
functions such as access to the inner-layer StandardProgram.
"""
import csv
import os
import re
import requests
import shutil
import tempfile
import traceback

import numpy as np
import pandas as pd
import tkinter as tk

from tkinter import filedialog
from tkinter import messagebox
from pandas.errors import EmptyDataError, ParserError

from .datapanels import RetrievalPanel, PreparationPanel
from .smallwindows import ProgressPopup
from .wrappers import threaded, report_uncaught

# TODO add any status updates that might be worthwhile


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

        self.dlib = self.app.dlib
        self.uncaught_exception = self.app.uncaught_exception

        # set window icon
        self.iconbitmap(self.util.PROGRAM_ICON)

        # filepath for most recent data for keeping selection up to date
        self.last_entered_data = None

        self.report_callback_exception = self.app.uncaught_exception

        self.taxon_level_name = "Order"
        self.taxon_level = "order_name"
        self.app.sp.taxon_split = "order_name"

        self.generated_alignment = False

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

    @report_uncaught
    def align_seq_data(self):
        """Filter and align sequences."""
        # ----- threaded function -----
        @threaded
        @report_uncaught
        def _align_seq_data(self, status_cb, on_finish=None):
            """
            Filter and align sequences, in a thread.

            Args:
                on_finish (func): Function to call on completion.
                status_cb (func): Callback function for status updates.
            """
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

            # align the fasta file
            try:
                self.app.sp.align_fasta(tax=self.taxon_level, debug=False)
            except ChildProcessError:
                self.dlib.dialog(
                    messagebox.showerror,
                    "ALIGN_ERR",
                    parent=self
                )
                if on_finish is not None:
                    on_finish()
                return
            except self.util.RNotFoundError:
                self.dlib.dialog(
                    messagebox.showerror,
                    "R_NOT_FOUND",
                    parent=self
                )
                if on_finish is not None:
                    on_finish()
                return
            except self.util.RScriptFailedError:
                self.dlib.dialog(
                    messagebox.showerror,
                    "R_SCRIPT_FAILED",
                    parent=self
                )
                if on_finish is not None:
                    on_finish()
                return

            # enable alignment save/load buttons
            self.data_sec.align_load_button["state"] = tk.ACTIVE
            self.data_sec.align_save_button["state"] = tk.ACTIVE

            # enable next step
            self.data_sec.delim_button["state"] = tk.ACTIVE
            self.data_sec.delim_load_button["state"] = tk.ACTIVE

            if on_finish is not None:
                on_finish()

            # advise the user to check the alignment
            self.dlib.dialog(
                messagebox.showinfo, "ALIGNMENT_COMPLETE", parent=self)

            # note that the alignment was generated (for warnings)
            self.generated_alignment = self.taxon_level_name
        # ----- end threaded function -----

        data = self.util.get_data(self.sequence_filtered.name)

        if not self.app.sp.check.check_taxon_exists(
            data,
            lambda: self.dlib.dialog(
                messagebox.showerror,
                "TAXON_NOT_PRESENT",
                form=(self.taxon_level_name,),
                parent=self
            )
        ):
            return

        if not self.app.sp.check.check_nan_taxon(
            data,
            lambda: self.dlib.dialog(
                messagebox.askokcancel,
                "NAN_TAXON",
                form=(self.taxon_level_name,),
                parent=self
            )
        ):
            return

        # check if subsets of one will occur and warn user
        if not self.app.sp.check.check_single_split(
            data,
            lambda: self.dlib.dialog(
                messagebox.askokcancel,
                "SINGLE_SPLIT",
                form=(self.taxon_level_name,),
                parent=self
            )
        ):
            return

        # disable buttons by opening progress bar
        progress_popup = ProgressPopup(
            self,
            "Alignment",
            "Aligning Sequences..."
        )

        def finish(self, on_finish):
            self.app.status_bar.set_status("Awaiting user input.")
            self.app.status_bar.progress.stop()
            self.app.status_bar.progress["mode"] = "determinate"
            on_finish()

        self.app.status_bar.set_status("Aligning sequences...")
        self.app.status_bar.progress["mode"] = "indeterminate"
        self.app.status_bar.progress.start()

        # start threaded function
        _align_seq_data(
            self,
            progress_popup.set_status,
            lambda: finish(self, on_finish=progress_popup.complete)
        )

    @report_uncaught
    def delim_species(self):
        """Delimit Species, splitting by taxonomic level."""
        # ----- threaded function -----

        @threaded
        @report_uncaught
        def _delim_species(self, status_cb, on_finish=None):
            data = self.util.get_data(self.sequence_filtered.name)

            method = self.data_sec.method_select.get()

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
            print("DELIMITING SPECIES...")
            try:
                self.app.sp.delimit_species(
                    method, tax=self.taxon_level, debug=False)
            except (ChildProcessError, FileNotFoundError, IndexError) as err:
                self.dlib.dialog(
                    messagebox.showerror,
                    "DELIM_ERR",
                    form=(self.taxon_level_name, str(err)),
                    parent=self
                )
                if on_finish is not None:
                    on_finish()
                return
            except self.util.RScriptFailedError:
                self.dlib.dialog(
                    messagebox.showerror,
                    "R_SCRIPT_FAILED",
                    parent=self
                )
                if on_finish is not None:
                    on_finish()
                return

            # note that alignment was loaded (skips alignment split warning)
            self.generated_delim = True

            # enable buttons
            self.data_sec.data_prep["state"] = tk.ACTIVE
            self.data_sec.delim_save_button["state"] = tk.ACTIVE

            # advise the user to check the delimitation
            self.dlib.dialog(
                messagebox.showinfo, "DELIM_COMPLETE", parent=self)

            if on_finish is not None:
                on_finish()
        # ----- end threaded function -----

        data = self.util.get_data(self.sequence_filtered.name)

        if not self.app.sp.check.check_taxon_exists(
            data,
            lambda: self.dlib.dialog(
                messagebox.showerror,
                "TAXON_NOT_PRESENT",
                form=(self.taxon_level_name,),
                parent=self
            )
        ):
            return

        if not self.app.sp.check.check_nan_taxon(
            data,
            lambda: self.dlib.dialog(
                messagebox.askokcancel,
                "NAN_TAXON",
                form=(self.taxon_level_name,),
                parent=self
            )
        ):
            return

        if not self.app.sp.check.check_single_split(
            data,
            lambda: self.dlib.dialog(
                messagebox.askokcancel,
                "SINGLE_SPLIT",
                form=(self.taxon_level_name,),
                parent=self
            )
        ):
            return

        # make popup to keep user from pressing buttons and breaking it
        progress = ProgressPopup(
            self,
            "Data Preparation",
            "Delimiting species..."
        )

        def finish(self, on_finish):
            self.app.status_bar.set_status("Awaiting user input.")
            self.app.status_bar.progress.stop()
            self.app.status_bar.progress["mode"] = "determinate"
            on_finish()

        self.app.status_bar.set_status("Delimiting species...")
        self.app.status_bar.progress["mode"] = "indeterminate"
        self.app.status_bar.progress.start()

        # run time-consuming items in thread
        _delim_species(
            self,
            progress.set_status,
            on_finish=lambda: finish(self, progress.complete)
        )

    @report_uncaught
    def filter_seq_data(self):
        """Filter Sequence Data"""
        # ----- threaded function -----
        @threaded
        @report_uncaught
        def _filter_seq_data(self, on_finish=None):

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

            # # prepare tempfile for aligned fasta
            # self.fasta_align = tempfile.NamedTemporaryFile(
            #     mode="w+",
            #     prefix="aligned_fasta_",
            #     suffix=".fasta",
            #     delete=False,
            #     dir=self.tempdir.name
            # )
            # self.fasta_align.close()
            # self.app.sp.fasta_align_fname = self.fasta_align.name

            # get request result tsv and prep it (filter + write fasta)
            self.app.sp.request_fname = self.last_entered_data

            bp = self.data_sec.filter_size.get()

            bp = int(bp) if bp is not None else 0

            data = self.app.sp.filter_sequence_data(
                self.app.sp.get_sequence_data(), bp)

            # save filtered data for later use
            data.to_csv(self.sequence_filtered.name, sep="\t", index=False)

            # TODO set to "no split" if no appropriate column
            # TODO also set it to order or the next lowest level if there is

            levels = [  # ordered by preference
                "order_name",
                "family_name",
                "subfamily_name",
                "genus_name",
                "phylum_name",
            ]

            if "order_name" not in data.columns:
                for level in levels:
                    if level in data.columns:
                        self.data_sec.taxon_split_selector.set(
                            level.split("_")[0].capitalize()
                        )
                        break
                else:  # a level was not set
                    self.data_sec.taxon_split_selector.set("No Split")
            else:  # first preference, set it
                self.data_sec.taxon_split_selector.set("Order")

            # enable alignment buttons
            self.data_sec.align_button["state"] = tk.ACTIVE
            self.data_sec.align_load_button["state"] = tk.ACTIVE

            # enable other saving buttons
            self.data_sec.filtered_sec.enable_buttons()
            self.data_sec.fasta_sec.enable_buttons()

            if on_finish is not None:
                on_finish()
        # ----- end threaded function -----

        # disable buttons by opening progress bar
        progress_popup = ProgressPopup(
            self,
            "Data Preparation",
            "Filtering Sequences..."
        )

        def finish(self, on_finish):
            self.app.status_bar.set_status("Awaiting user input.")
            self.app.status_bar.progress.stop()
            self.app.status_bar.progress["mode"] = "determinate"
            on_finish()

        self.app.status_bar.set_status("Filtering sequences...")
        self.app.status_bar.progress["mode"] = "indeterminate"
        self.app.status_bar.progress.start()

        _filter_seq_data(
            self,
            on_finish=lambda: finish(self, progress_popup.complete)
        )

    @report_uncaught
    def get_geographies(self):
        """
        Return a list of all geographies.

        Returns:
            list: All configured geographies.

        """
        return sorted(self.util.get_geographies())
        # return self.util.get_geographies()

    @report_uncaught
    def load_item(self, item):
        """
        Load an item into the program.

        Args:
            item (str): key for which item to load.
        """

        table = [
            ("All files", ".*"),
            ("Comma-separated values", ".csv"),
            ("Tab-separated values", ".tsv"),
        ]

        filetypes = {
            "alignment": [("FASTA formatted sequence data", ".fasta")],
            "filtered": table,
            "finalized": table,
            "delimitation": table,
        }

        if self.fasta_align is None:
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

        if self.delim is None and item == "delimitation":
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

        tempfiles = {
            "alignment": self.fasta_align.name,
            "filtered": self.sequence_filtered.name,
            "finalized": (
                self.finalized_data.name
                if self.finalized_data is not None else None
            ),
            "delimitation": (
                self.delim.name if self.delim is not None else None
            )
        }
        # ----- threaded function -----

        @threaded
        @report_uncaught
        def _load_alignment(self, alignfname, status_cb, on_finish=None):
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

            missing = os.path.join(
                self.util.USER_PATH,
                "logs/missing_PIDs.log"
            )

            if os.path.exists(missing):
                os.remove(missing)

            if len(names_not_found) > 0:
                open(missing, "w").close()

                with open(missing, "w") as error_log:
                    error_log.write("\n".join(names_not_found))

                self.dlib.dialog(
                    messagebox.showwarning,
                    "ALIGN_MISMATCH",
                    parent=self,
                    form=(missing.replace("/", "\\"),)
                )

            # load file
            shutil.copy(alignfname, tempfiles[item])

            # note that alignment was loaded (skips alignment split warning)
            self.generated_alignment = False

            if on_finish is not None:
                on_finish()

        # ----- end threaded function -----

        # ----- threaded function -----

        @threaded
        @report_uncaught
        def _load_delim(self, delimfname, status_cb, on_finish=None):
            # load raw data to check alignment against
            data = self.util.get_data(self.sequence_filtered.name)

            # check chosen file is not empty
            if os.stat(delimfname).st_size == 0:
                self.dlib.dialog(
                    messagebox.showwarning,
                    "EMPTY_FILE",
                    parent=self
                )
                on_finish()
                return

            # check file actually reads
            try:
                delim = self.util.get_data(delimfname)
            except (TypeError, ValueError):
                self.dlib.dialog(
                    messagebox.showerror,
                    "INCOMPATIBLE_GENERIC",
                    parent=self
                )
                on_finish()
                return

            missing_upids = []
            extra_upids = []

            # check file has valid columns
            req_cols = ("Delim_spec", "sample_name")
            for col in req_cols:
                if col not in delim.columns:
                    self.dlib.dialog(
                        messagebox.showerror,
                        "MISSING_REQUIRED_COLUMNS",
                        parent=self
                    )
                    on_finish()
                    return

            # Get a list of pids which are ignored due to single split
            # These don't need to be in the delimitation
            ignored = (
                data
                .groupby(self.taxon_level)
                .filter(lambda g: len(g) == 1)["UPID"]
                .tolist()
            )
            # check if each upid from data is in user delim
            for upid in data["UPID"].values:
                if upid not in delim["sample_name"].values:
                    if upid not in ignored:
                        missing_upids.append(upid)
            # check if each upid in user delim is in data
            for name in delim["sample_name"].values:
                if name not in data["UPID"].values:
                    extra_upids.append(name)
            # make sure nothing appears more than once
            counts = delim["sample_name"].value_counts()
            if sum(counts) != len(counts):
                self.dlib.dialog(
                    messagebox.showwarning,
                    "INVALID_DELIM",
                    parent=self
                )
                on_finish()
                return
            # ensure no empty entries
            if delim.isnull().values.any():
                self.dlib.dialog(
                    messagebox.showwarning,
                    "DELIM_MISSING_ENTRIES",
                    parent=self
                )
                on_finish()
                return

            missing = os.path.join(
                self.util.USER_PATH,
                "logs/missing_PIDs.log"
            )

            if os.path.exists(missing):
                os.remove(missing)

            extra = os.path.join(
                self.util.USER_PATH,
                "logs/extra_PIDs.log"
            )

            if os.path.exists(extra):
                os.remove(extra)

            # if any pids are missing warn user and cancel load
            if len(missing_upids) > 0:
                open(missing, "w").close()

                with open(missing, "w") as error_log:
                    error_log.write("\n".join(missing_upids))

                self.dlib.dialog(
                    messagebox.showwarning,
                    "MISSING_PIDS",
                    parent=self,
                    form=(missing.replace("/", "\\"),)
                )
                # on_finish()
                # return

            # if any pids are not in data warn user and cancel load
            if len(extra_upids) > 0:
                open(extra, "w").close()

                with open(extra, "w") as error_log:
                    error_log.write("\n".join(extra_upids))

                self.dlib.dialog(
                    messagebox.showwarning,
                    "EXTRA_PIDS",
                    parent=self,
                    form=(extra.replace("/", "\\"),)
                )
                # on_finish()
                # return

            # load file
            shutil.copy(delimfname, tempfiles[item])

            # note that delim was loaded (skips delim split warning)
            self.generated_delim = False

            if on_finish is not None:
                on_finish()

        # ----- end threaded function -----

        self.app.status_bar.set_status("Awaiting user file selection...")

        # prompt the user for the classifier file
        file = filedialog.askopenfilename(
            title="Open Edited Alignment",
            initialdir=os.path.realpath(
                os.path.join(self.util.USER_PATH, "data/")),
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

            def finish(self, on_finish):  # specify self for report_uncaught
                self.data_sec.align_load_button["state"] = tk.ACTIVE
                # enable next step
                self.data_sec.delim_button["state"] = tk.ACTIVE
                self.data_sec.delim_load_button["state"] = tk.ACTIVE
                self.app.status_bar.set_status("Awaiting user input.")
                on_finish()

            _load_alignment(
                self,
                file,
                progress.set_status,
                on_finish=lambda: finish(self, progress.complete)
            )
        elif item == "delimitation":
            self.data_sec.delim_load_button["state"] = tk.DISABLED

            progress = ProgressPopup(
                self,
                "Reading Delimitation",
                "Reading file..."
            )

            def finish(self, on_finish):  # specify self for report_uncaught
                self.data_sec.delim_load_button["state"] = tk.ACTIVE
                # enable next step
                self.data_sec.data_prep["state"] = tk.ACTIVE
                self.app.status_bar.set_status("Awaiting user input.")
                on_finish()

            _load_delim(
                self,
                file,
                progress.set_status,
                on_finish=lambda: finish(self, progress.complete)
            )

        else:
            # overwrite the alignment file
            shutil.copy(file, tempfiles[item])
            self.app.status_bar.set_status("Awaiting user input.")

    @report_uncaught
    def load_sequence_data(self):
        """
        Get the location of custom user sequence data for later use.

        Also conditionally enables the 'merge data' button.
        """
        self.app.status_bar.set_status("Awaiting user file selection...")

        # check if user is overwriting and make sure they're ok with it
        if self.user_sequence_raw is not None:
            if not self.dlib.dialog(
                    messagebox.askokcancel, "SEQUENCE_OVERWRITE", parent=self):
                self.app.status_bar.set_status("Awaiting user input.")
                return

            # keep old data in case user wants to merge
            self.user_sequence_previous = self.user_sequence_raw

        # prompt the user for the sequence file
        file = filedialog.askopenfilename(
            title="Open Data File",
            initialdir=os.path.realpath(
                os.path.join(self.util.USER_PATH, "data/")),
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

        # read in data for other checks and make sure it reads
        self.app.status_bar.set_status("Checking user sequence file...")
        try:
            data = self.util.get_data(file)
        except (ParserError, EmptyDataError, OSError, IOError, KeyError,
                TypeError, ValueError, csv.Error):
            self.dlib.dialog(
                messagebox.showwarning,
                "FILE_READ_ERR"
            )
            return

        # check if sequence data has required columns
        if not self.app.sp.check.check_required_columns(
            data,
            lambda: self.dlib.dialog(
                messagebox.showwarning, "MISSING_REQUIRED_COLUMNS", parent=self)
        ):
            self.app.status_bar.set_status("Awaiting user input.")
            return

        # check that UPID is actually unique, if provided
        if "UPID" in data.columns:
            if not self.app.sp.check.check_UPID_unique(
                data,
                lambda: self.dlib.dialog(
                    messagebox.askokcancel, "UPID_NOT_UNIQUE", parent=self)
            ):
                self.app.status_bar.set_status("Awaiting user input.")
                return

        # check if user data has any reserved columns
        if not self.app.sp.check.check_reserved_columns(
            data,
            lambda: self.dlib.dialog(
                messagebox.askokcancel, "RESERVED_COLUMNS", parent=self)
        ):
            return

        # check if species_name is provided
        self.app.sp.check.check_has_species_name(
            data,
            lambda: self.dlib.dialog(
                messagebox.showinfo, "NO_SPECIES_NAME", parent=self)
        )

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
        self.data_sec.delim_button["state"] = tk.DISABLED
        self.data_sec.data_prep["state"] = tk.DISABLED
        self.data_sec.use_data_button["state"] = tk.DISABLED
        self.data_sec.final_load_button["state"] = tk.DISABLED
        self.data_sec.final_save_button["state"] = tk.DISABLED
        self.data_sec.use_data_button["state"] = tk.DISABLED

    @report_uncaught
    def merge_sequence_data(self, bold=False):
        """
        Merge multiple sequence files.

        Args:
            bold (bool, optional): Denotes merging BOLD search results.
                Defaults to False.
        """
        # ----- threaded function -----
        @threaded
        @report_uncaught
        def _merge_sequence_data(self, on_finish=None):
            if self.merged_raw is not None:
                if bold:
                    answer = self.dlib.dialog(
                        messagebox.askyesnocancel,
                        "MESSAGE_MERGE_BOLD",
                        parent=self
                    )
                else:
                    answer = self.dlib.dialog(
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

            # disable buttons on successful merge
            self.data_sec.align_button["state"] = tk.DISABLED
            self.data_sec.align_load_button["state"] = tk.DISABLED
            self.data_sec.align_save_button["state"] = tk.DISABLED
            self.data_sec.delim_button["state"] = tk.DISABLED
            self.data_sec.data_prep["state"] = tk.DISABLED
            self.data_sec.use_data_button["state"] = tk.DISABLED
            self.data_sec.final_load_button["state"] = tk.DISABLED
            self.data_sec.final_save_button["state"] = tk.DISABLED
            self.data_sec.use_data_button["state"] = tk.DISABLED
            self.data_sec.fasta_sec.disable_buttons()
            self.data_sec.filtered_sec.disable_buttons()

            if on_finish is not None:
                on_finish()

            self.dlib.dialog(
                messagebox.showinfo, "MERGE_COMPLETE", parent=self)

        # ----- end threaded function -----

        # disable buttons
        self.get_data_sec.merge_bold_button["state"] = tk.DISABLED
        self.get_data_sec.merge_button["state"] = tk.DISABLED

        def finish(self, on_finish):
            # re-enable buttons
            self.get_data_sec.merge_bold_button["state"] = tk.ACTIVE
            self.get_data_sec.merge_button["state"] = tk.ACTIVE
            on_finish()

        # make popup to keep user from pressing buttons and breaking it
        progress = ProgressPopup(
            self,
            "Data Merge",
            "Merging data..."
        )

        _merge_sequence_data(
            self,
            on_finish=lambda: finish(self, progress.complete)
        )

    @report_uncaught
    def prep_sequence_data(self):
        """Prepare aligned sequence data."""
        # ----- threaded function -----
        @threaded
        @report_uncaught
        def _prep_sequence_data(self, status_cb, on_finish=None):
            # data = self.util.get_data(self.sequence_filtered.name)

            # method = self.data_sec.method_select.get()

            # # create delim tempfile
            # self.delim = tempfile.NamedTemporaryFile(
            #     mode="w+",
            #     prefix="species_delim_",
            #     suffix=".tsv",
            #     delete=False,
            #     dir=self.tempdir.name
            # )
            # self.delim.close()
            # self.app.sp.delim_fname = self.delim.name

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

            # # delimit the species
            # print("DELIMITING SPECIES...")
            # try:
            #     self.app.sp.delimit_species(
            #         method, tax=self.taxon_level, debug=False)
            # except (ChildProcessError, FileNotFoundError, IndexError) as err:
            #     self.dlib.dialog(
            #         messagebox.showerror,
            #         "DELIM_ERR",
            #         form=(self.taxon_level_name, str(err)),
            #         parent=self
            #     )
            #     if on_finish is not None:
            #         on_finish()
            #     return
            # except self.util.RScriptFailedError:
            #     self.dlib.dialog(
            #         messagebox.showerror,
            #         "R_SCRIPT_FAILED",
            #         parent=self
            #     )
            #     if on_finish is not None:
            #         on_finish()
            #     return

            status_cb("Generating species features (This will take some time)...")
            print("GENERATING FEATURES...")
            try:
                self.app.sp.generate_features(
                    tax=self.taxon_level, debug=False)
            except (ChildProcessError, FileNotFoundError) as err:
                self.dlib.dialog(
                    messagebox.showerror,
                    "FEATURE_GEN_ERR",
                    form=(self.taxon_level_name, str(err)),
                    parent=self
                )
                if on_finish is not None:
                    on_finish()
                return
            except self.util.RNotFoundError:
                self.dlib.dialog(
                    messagebox.showerror,
                    "R_NOT_FOUND",
                    parent=self
                )
                if on_finish is not None:
                    on_finish()
                return
            except self.util.RScriptFailedError:
                self.dlib.dialog(
                    messagebox.showerror,
                    "R_SCRIPT_FAILED",
                    parent=self
                )
                if on_finish is not None:
                    on_finish()
                return

            final = self.util.get_data(self.finalized_data.name)
            if "species_name" in final.columns:
                status_cb(
                    "Looking up known species statuses (this will take some time)...")
                print("EXECUTING STATUS LOOKUP...")
                self.app.status_bar.set_status("Looking up statuses...")
                # get statuses
                try:
                    self.app.sp.ref_geo = self.data_sec.ref_geo_select.get()
                    self.app.sp.lookup_status()
                except ChildProcessError:
                    self.dlib.dialog(
                        messagebox.showerror,
                        "GEO_LOOKUP_ERR",
                        parent=self
                    )
                    if on_finish is not None:
                        on_finish()
                    return

            final = self.util.get_data(self.finalized_data.name)
            if "final_status" in final:
                n_classified = final.count()["final_status"]

            self.data_sec.final_save_button["state"] = tk.ACTIVE
            self.data_sec.final_load_button["state"] = tk.ACTIVE
            self.data_sec.use_data_button["state"] = tk.ACTIVE

            if "final_status" in final:
                # check if there are at least 2 classes
                if self.app.sp.check.check_enough_classes(
                    final["final_status"],
                    lambda: self.dlib.dialog(
                        messagebox.showwarning,
                        "NOT_ENOUGH_CLASSES",
                        parent=self
                    )
                ):
                    # check that enough samples were classified
                    self.app.sp.check.check_enough_classified(
                        final["final_status"],
                        lambda: self.dlib.dialog(
                            messagebox.showwarning,
                            "LOW_CLASS_COUNT",
                            form=(n_classified,),
                            parent=self
                        )
                    )

                    # check for extreme inbalance using stdev as a heuristic
                    self.app.sp.check.check_inbalance(
                        final["final_status"],
                        lambda: self.dlib.dialog(
                            messagebox.showwarning,
                            "HIGH_IMBALANCE"
                        )
                    )

                # notify of completion
                self.dlib.dialog(
                    messagebox.showinfo,
                    "DATA_PREP_COMPLETE",
                    form=(n_classified,),
                    parent=self
                )

            if on_finish is not None:
                on_finish()
        # ----- end threaded function -----

        data = self.util.get_data(self.sequence_filtered.name)

        if not self.app.sp.check.check_taxon_exists(
            data,
            lambda: self.dlib.dialog(
                messagebox.showerror,
                "TAXON_NOT_PRESENT",
                form=(self.taxon_level_name,),
                parent=self
            )
        ):
            return

        if not self.app.sp.check.check_nan_taxon(
            data,
            lambda: self.dlib.dialog(
                messagebox.askokcancel,
                "NAN_TAXON",
                form=(self.taxon_level_name,),
                parent=self
            )
        ):
            return

        if not self.app.sp.check.check_single_split(
            data,
            lambda: self.dlib.dialog(
                messagebox.askokcancel,
                "SINGLE_SPLIT",
                form=(self.taxon_level_name,),
                parent=self
            )
        ):
            return

        # make popup to keep user from pressing buttons and breaking it
        progress = ProgressPopup(
            self,
            "Data Preparation",
            "Preparing sequence data..."
        )

        def finish(self, on_finish):
            self.app.status_bar.set_status("Awaiting user input.")
            self.app.status_bar.progress.stop()
            self.app.status_bar.progress["mode"] = "determinate"
            on_finish()

        self.app.status_bar.set_status("Generating features...")
        self.app.status_bar.progress["mode"] = "indeterminate"
        self.app.status_bar.progress.start()

        # run time-consuming items in thread
        _prep_sequence_data(
            self,
            progress.set_status,
            on_finish=lambda: finish(self, progress.complete)
        )

    @report_uncaught
    def retrieve_seq_data(self):
        """Search for sequence data from BOLD."""
        # ----- threaded function -----
        @threaded
        @report_uncaught
        def _retrieve_seq_data(self, on_finish=None):
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
                self.dlib.dialog(
                    messagebox.showerror, "BOLD_SEARCH_ERR", parent=self)
                if on_finish is not None:
                    on_finish()
                return
            except UnicodeDecodeError:
                self.dlib.dialog(
                    messagebox.showerror, "RESPONSE_DECODE_ERR", parent=self)
                if on_finish is not None:
                    on_finish()
                return

            # check if file downloaded properly (parses successfully)
            try:
                print(self.sequence_raw.name)
                nlines = self.util.get_data(self.sequence_raw.name).shape[0]
            except ParserError:
                self.dlib.dialog(
                    messagebox.showerror, "BOLD_FILE_ERR", parent=self)
                if on_finish is not None:
                    on_finish()
                return
            except EmptyDataError:
                self.dlib.dialog(
                    messagebox.showerror, "BOLD_NO_OBSERVATIONS", parent=self)
                if on_finish is not None:
                    on_finish()
                return
            except UnicodeDecodeError:
                traceback.print_exc()
                self.dlib.dialog(
                    messagebox.showerror, "RESPONSE_DECODE_ERR", parent=self)
                if on_finish is not None:
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

            if on_finish is not None:
                on_finish()

            if nlines == 0:
                self.dlib.dialog(
                    messagebox.showwarning,
                    "BOLD_NO_OBSERVATIONS",
                    parent=self
                )

                return

            # tell user it worked/how many lines downloaded
            self.dlib.dialog(
                messagebox.showinfo,
                "BOLD_SEARCH_COMPLETE",
                parent=self,
                form=(nlines,)
            )
        # ----- end threaded function -----

        self.app.sp.geo = self.get_data_sec.geo_input.get()
        self.app.sp.taxon = self.get_data_sec.taxon_input.get()

        if (self.app.sp.geo is None or self.app.sp.taxon is None):
            self.dlib.dialog(
                messagebox.showwarning, "MISSING_SEARCH_TERMS", parent=self)
            return
        elif len(self.app.sp.geo) == 0 or len(self.app.sp.taxon) == 0:
            self.dlib.dialog(
                messagebox.showwarning, "MISSING_SEARCH_TERMS", parent=self)

            return

        if not self.dlib.dialog(
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

        _retrieve_seq_data(self, on_finish=progress_popup.complete)

    @report_uncaught
    def set_taxon_level(self, event):
        """
        Set the taxonomic level to the user selection.

        event is kept as argument to avoid too many arguments error.
        """
        levels = {
            "No Split": 0,
            "Phylum": "phylum_name",
            "Class": "class_name",
            "Order": "order_name",
            "Family": "family_name",
            "Subfamily": "subfamily_name",
            "Genus": "genus_name"
        }

        prev_tln = self.taxon_level_name
        new_tln = self.data_sec.taxon_split_selector.get()

        # if an alignment was already generated on this split level
        if all([
            self.fasta_align is not None,
            self.generated_alignment is not False,
            self.generated_alignment != new_tln
        ]):
            if not self.dlib.dialog(
                    messagebox.askokcancel,
                    "TAXON_CHANGE",
                    form=(self.generated_alignment,)
            ):
                self.data_sec.taxon_split_selector.set(prev_tln)
                return

        self.taxon_level_name = new_tln
        self.taxon_level = levels[self.taxon_level_name]
        self.app.sp.taxon_split = self.taxon_level

    @report_uncaught
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
