"""Panel structures used in the data tool window."""
import tkinter as tk

from tkinter import ttk

from .elements import VS_Pair


class RetrievalPanel(ttk.LabelFrame):
    """Panel containing data retrieval interface."""

    def __init__(self, parent, app, *args, **kwargs):
        """
        Initialize the panel.

        Note parent and app should be the same in the normal use case.

        Args:
            parent (Frame): The container for this panel.
            app (TopLevel): The program containing the button functions.
        """
        super().__init__(parent, *args, **kwargs)

        self.app = app

        self.search_sec = ttk.LabelFrame(
            self,
            text="BOLD Lookup",
            labelanchor=tk.N
        )
        self.search_sec.pack(expand=True, fill=tk.X, side=tk.LEFT)

        self.taxon_label = tk.Label(
            self.search_sec,
            text="Taxonomy"
        )
        self.taxon_label.pack(anchor=tk.W)

        self.taxon_input = tk.Entry(
            self.search_sec
        )
        self.taxon_input.pack(fill=tk.X)

        self.geo_label = tk.Label(
            self.search_sec,
            text="Geography"
        )
        self.geo_label.pack(anchor=tk.W)

        self.geo_input = tk.Entry(
            self.search_sec
        )
        self.geo_input.pack(fill=tk.X)

        self.data_lookup = tk.Button(
            self.search_sec,
            text="Search Data From BOLD",
            command=self.app.retrieve_seq_data
        )
        self.data_lookup.pack(expand=True, fill=tk.X, padx=1, pady=1)

        self.util_button_sec = tk.Frame(
            self.search_sec
        )
        self.util_button_sec.pack(fill=tk.X, expand=True)

        self.merge_bold_button = tk.Button(
            self.util_button_sec,
            text="Add To Merge",
            command=lambda: self.app.merge_sequence_data(bold=True),
            state=tk.DISABLED
        )
        self.merge_bold_button.pack(
            expand=True, fill=tk.BOTH, side=tk.RIGHT, padx=1, pady=1)

        self.save_bold_button = tk.Button(
            self.util_button_sec,
            text="Save Results",
            command=lambda: self.app.app.save_item("bold_results"),
            state=tk.DISABLED
        )
        self.save_bold_button.pack(
            expand=True, fill=tk.BOTH, side=tk.RIGHT, padx=1, pady=1)

        self.load_sec = ttk.LabelFrame(
            self,
            text="Custom Data",
            labelanchor=tk.N
        )
        self.load_sec.pack(expand=True, fill=tk.BOTH, side=tk.RIGHT)

        self.load_button = tk.Button(
            self.load_sec,
            text="Load Custom Data",
            command=self.app.load_sequence_data
        )
        self.load_button.pack(expand=True, fill=tk.BOTH, padx=1, pady=1)

        self.merge_button = tk.Button(
            self.load_sec,
            text="Add to Merge",
            command=self.app.merge_sequence_data,
            state=tk.DISABLED
        )
        self.merge_button.pack(expand=True, fill=tk.BOTH, padx=1, pady=1)

        self.save_merge_button = tk.Button(
            self.load_sec,
            text="Save Merged File",
            command=lambda: self.app.app.save_item("merged_results"),
            state=tk.DISABLED
        )
        self.save_merge_button.pack(expand=True, fill=tk.BOTH, padx=1, pady=1)


class PreparationPanel(ttk.LabelFrame):
    """Panel containing the data preparation interface."""

    def __init__(self, parent, app, *args, **kwargs):
        """Initialize the panel.

        Note that parent and app should be the same in the normal use case.

        Args:
            parent (Frame): The container for this panel.
            app (TopLevel): The program containing functions for the buttons.
        """
        super().__init__(parent, *args, **kwargs)

        self.app = app

        self.filter_button = tk.Button(
            self,
            text="Filter Sequences",
            command=self.app.filter_seq_data,
            pady=5,
            state=tk.DISABLED
        )
        self.filter_button.pack(expand=True, fill=tk.X, padx=1, pady=1)

        self.align_button = tk.Button(
            self,
            text="Align Sequences",
            command=self.app.align_seq_data,
            pady=5,
            state=tk.DISABLED
        )
        self.align_button.pack(expand=True, fill=tk.X, padx=1, pady=1)

        self.align_box = tk.Frame(
            self
        )
        self.align_box.pack(expand=True, fill=tk.X)

        self.align_load_button = tk.Button(
            self.align_box,
            text="Load Edited Alignment",
            command=lambda: self.app.load_item("alignment"),
            state=tk.DISABLED
        )
        self.align_load_button.pack(
            expand=True, fill=tk.X, side=tk.RIGHT, padx=1, pady=1)

        self.align_save_button = tk.Button(
            self.align_box,
            text="Save Alignment For Editing",
            command=lambda: self.app.app.save_item("fasta_align"),
            state=tk.DISABLED
        )
        self.align_save_button.pack(
            expand=True, fill=tk.X, side=tk.RIGHT, padx=1, pady=1)

        self.row1 = tk.Frame(self)
        self.row1.pack(expand=True, fill=tk.X)

        self.filtered_sec = VS_Pair(
            self.row1,
            self.app,
            lambda: self.app.app.view_item(self.app.sequence_filtered.name),
            lambda: self.app.app.save_item("filtered_data"),
            text="Filtered Data",
            labelanchor=tk.N
        )
        self.filtered_sec.pack(side=tk.RIGHT, expand=True, fill=tk.X)

        self.fasta_sec = VS_Pair(
            self.row1,
            self.app,
            lambda: self.app.app.view_item(self.app.fasta.name),
            lambda: self.app.app.save_item("raw_fasta"),
            text="Raw .fasta",
            labelanchor=tk.N
        )
        self.fasta_sec.pack(side=tk.RIGHT, expand=True, fill=tk.X)

        self.method_select_label = tk.Label(
            self,
            text="Species Delimitation Method:"

        )
        self.method_select_label.pack(anchor=tk.W)

        self.method_select = ttk.Combobox(
            self,
            height=10,
            state="readonly",
            # textvariable=self.app.known_column
        )
        self.method_select["values"] = ("GMYC", "bPTP")
        self.method_select.set("bPTP")
        self.method_select.pack(fill=tk.X)

        self.reference_geo_label = tk.Label(
            self,
            text="Reference Geography:"
        )
        self.reference_geo_label.pack(anchor=tk.W)

        self.ref_geo_select = ttk.Combobox(
            self,
            height=10,
            state="readonly",
            # textvariable=self.app.known_column
        )
        self.ref_geo_select["values"] = self.app.get_geographies()
        self.ref_geo_select.set("Continental US")
        self.ref_geo_select.pack(fill=tk.X)

        self.data_prep = tk.Button(
            self,
            text="Prepare Data",
            command=self.app.prep_sequence_data,
            pady=5,
            state=tk.DISABLED
        )
        self.data_prep.pack(expand=True, fill=tk.X, padx=1, pady=1)

        self.finalized_edit_sec = tk.Frame(self)
        self.finalized_edit_sec.pack(expand=True, fill=tk.X)

        self.final_save_button = tk.Button(
            self.finalized_edit_sec,
            text="Save Prepared Data",
            command=lambda: self.app.app.save_item("finalized"),
            pady=5,
            state=tk.DISABLED
        )
        self.final_save_button.pack(
            expand=True, fill=tk.X, padx=1, pady=1, side=tk.LEFT)

        self.final_load_button = tk.Button(
            self.finalized_edit_sec,
            text="Load Edited Data",
            command=lambda: self.app.load_item("finalized"),
            pady=5,
            state=tk.DISABLED
        )
        self.final_load_button.pack(
            expand=True, fill=tk.X, padx=1, pady=1, side=tk.RIGHT)

        self.row2 = tk.Frame(self)
        self.row2.pack(expand=True, fill=tk.X)

        self.use_data_button = tk.Button(
            self,
            text="Use Prepared Data",
            command=self.app.transfer_prepared_data,
            pady=5,
            state=tk.DISABLED
        )
        self.use_data_button.pack(expand=True, fill=tk.X, padx=1, pady=1)
