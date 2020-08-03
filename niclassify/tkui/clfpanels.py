"""A collection of panels used in the Classifier Tool GUI."""
import ast

import tkinter as tk

from tkinter import ttk
from .elements import VS_Pair


class TwoColumnSelect(tk.Frame):
    """
    A two column selection interface.

    Allows users to swap items between the selected and deselected columns.
    Hopefully not hard for the user to follow.

    Also maintains order of items because I thought that'd be important.
    """

    def __init__(self, parent, *args, **kwargs):
        """
        Initialize the interface.

        Args:
            parent (Frame): Whatever's holding this interface.
        """
        tk.Frame.__init__(self, parent, *args, **kwargs)
        self.parent = parent

        self.column_names = {}

        # variables for contents
        self.desel = tk.StringVar()
        self.sel = tk.StringVar()

        self.desel.set(value=[])
        self.sel.set(value=[])

        # define box for deselected items
        self.desel_frame = tk.LabelFrame(
            self,
            text="Not Selected",
            labelanchor=tk.N,
            borderwidth=0)
        self.desel_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.desel_contents = tk.Listbox(
            self.desel_frame,
            selectmode=tk.EXTENDED,
            listvariable=self.desel,
            width=50,
            height=20)
        self.desel_contents.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.desel_cont_sb = tk.Scrollbar(
            self.desel_frame,
            orient=tk.VERTICAL)
        self.desel_cont_sb.config(command=self.desel_contents.yview)
        self.desel_cont_sb.pack(side=tk.RIGHT, fill=tk.Y)
        self.desel_contents.config(yscrollcommand=self.desel_cont_sb.set)

        # define box for selection controls
        self.selection_buttons_frame = tk.Frame(self)
        self.selection_buttons_frame.pack(side=tk.LEFT)
        self.all_right = tk.Button(
            self.selection_buttons_frame,
            text=">>",
            width=5,
            height=2,
            command=lambda: self.to_right(True))
        self.all_right.pack(
            padx=1, pady=1)
        self.sel_right = tk.Button(
            self.selection_buttons_frame,
            text=">",
            width=5,
            height=2,
            command=self.to_right)
        self.sel_right.pack(padx=1, pady=1)
        self.spacer = tk.Frame(self.selection_buttons_frame, height=10)
        self.spacer.pack()
        self.sel_left = tk.Button(
            self.selection_buttons_frame,
            text="<",
            width=5,
            height=2,
            command=self.to_left)
        self.sel_left.pack(padx=1, pady=1)
        self.all_left = tk.Button(
            self.selection_buttons_frame,
            text="<<",
            width=5,
            height=2,
            command=lambda: self.to_left(True))
        self.all_left.pack(padx=1, pady=1)

        # define box for selected items
        self.sel_frame = tk.LabelFrame(
            self,
            text="Selected",
            labelanchor=tk.N,
            borderwidth=0)
        self.sel_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.sel_contents = tk.Listbox(
            self.sel_frame,
            selectmode=tk.EXTENDED,
            listvariable=self.sel,
            width=50,
            height=20)
        self.sel_contents.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.sel_cont_sb = tk.Scrollbar(
            self.sel_frame,
            orient=tk.VERTICAL)
        self.sel_cont_sb.config(command=self.sel_contents.yview)
        self.sel_cont_sb.pack(side=tk.RIGHT, fill=tk.Y)
        self.sel_contents.config(yscrollcommand=self.sel_cont_sb.set)

    def to_right(self, allitems=False):
        """
        Move selected (or all) items from the left to the right.

        Args:
            allitems (bool, optional): Move all items. Defaults to False.
        """
        desel_items = (list(ast.literal_eval(self.desel.get()))
                       if len(self.desel.get()) > 0 else [])
        sel_items = (list(ast.literal_eval(self.sel.get()))
                     if len(self.sel.get()) > 0 else [])
        if allitems:
            sel_items.extend(desel_items)
            sel_items.sort(key=lambda x: self.column_names[x])

            self.desel.set(value=[])
            self.sel.set(value=sel_items)

        elif self.desel_contents.curselection() == ():
            return

        else:
            sel_items.extend(desel_items[i]
                             for i
                             in self.desel_contents.curselection())
            sel_items.sort(key=lambda x: self.column_names[x])
            desel_items = [x
                           for i, x in enumerate(desel_items)
                           if i not in self.desel_contents.curselection()]
            desel_items.sort(key=lambda x: self.column_names[x])

            self.desel.set(value=desel_items)
            self.sel.set(value=sel_items)

    def to_left(self, allitems=False):
        """
        Move selected (or all) items from the right to the left.

        Args:
            allitems (bool, optional): Move all items. Defaults to False.
        """
        desel_items = (list(ast.literal_eval(self.desel.get()))
                       if len(self.desel.get()) > 0 else [])
        sel_items = (list(ast.literal_eval(self.sel.get()))
                     if len(self.sel.get()) > 0 else [])
        if allitems:
            desel_items.extend(sel_items)
            desel_items.sort(key=lambda x: self.column_names[x])

            self.desel.set(value=desel_items)
            self.sel.set(value=[])

        elif self.sel_contents.curselection() == ():
            return

        else:
            desel_items.extend(sel_items[i]
                               for i
                               in self.sel_contents.curselection())
            desel_items.sort(key=lambda x: self.column_names[x])
            sel_items = [x
                         for i, x in enumerate(sel_items)
                         if i not in self.sel_contents.curselection()]
            sel_items.sort(key=lambda x: self.column_names[x])

            self.desel.set(value=desel_items)
            self.sel.set(value=sel_items)

    def update_contents(self, colnames_dict):
        """
        Replace the contents with a new set of contents.

        Uses a dictionary so order may be maintained when moving contents.

        Args:
            colnames_dict (dict): a dictionary of item: index.
        """
        self.column_names = colnames_dict
        colnames = list(colnames_dict.keys())
        colnames.sort(key=lambda x: colnames_dict[x])
        self.desel.set(value=colnames)
        self.sel.set(value=[])


class DataPanel(tk.LabelFrame):
    """
    One of the panels in MainApp, for opening and interacting with data.

    Mostly just pre-defined GUI objects.
    """

    def __init__(self, parent, app, *args, **kwargs):
        """
        Initialize the DataPanel.

        Args:
            parent (Frame): Whatever's holding the DataPanel.
            app (MainApp): Generally the MainApp for easy method access.
        """
        tk.LabelFrame.__init__(self, parent, *args, **kwargs)
        self.parent = parent
        self.app = app

        self.data_button_frame = tk.Frame(self)
        self.data_button_frame.pack(fill=tk.X)

        # button to load data
        self.load_data_button = tk.Button(
            self.data_button_frame,
            text="Load Data",
            pady=5,
            command=self.app.get_data_file
        )
        self.load_data_button.pack(
            side=tk.RIGHT, fill=tk.X, padx=1, pady=1, expand=True)

        # button to load data
        self.retrieve_data_button = tk.Button(
            self.data_button_frame,
            text="Prepare Sequence Data",
            pady=5,
            command=self.app.open_data_tool
        )
        self.retrieve_data_button.pack(
            side=tk.LEFT, fill=tk.X, padx=1, pady=1, expand=True)

        # excel sheet selection
        self.excel_label = tk.Label(
            self,
            text="Specify Excel Sheet:")
        self.excel_label.pack(anchor=tk.W)
        self.excel_sheet_input = ttk.Combobox(
            self,
            height=10,
            state=tk.DISABLED,
            # textvariable=self.app.sheet
        )
        self.excel_sheet_input.bind(
            "<<ComboboxSelected>>", self.app.get_sheet_cols)
        self.excel_sheet_input.pack(fill=tk.X)

        # column selection sec
        self.col_select_hint = tk.Label(
            self,
            text="Select Feature Columns:")
        self.col_select_hint.pack(anchor=tk.W)
        # selection panel, depends on twocolumnselect.py
        self.col_select_panel = TwoColumnSelect(self)
        self.col_select_panel.pack(fill=tk.BOTH, expand=True)

        self.output_open = tk.Button(
            self,
            text="Open Output Folder",
            command=self.app.open_output_folder
        )
        self.output_open.pack(side=tk.LEFT, anchor=tk.NW, padx=1, pady=1)

        # button to open window allowing NaN values to be edited
        self.nan_check = tk.Button(
            self,
            text="Check Recognized NaN values",
            command=self.app.open_nans
        )
        self.nan_check.pack(side=tk.LEFT, anchor=tk.NW, padx=1, pady=1)

        # button to open helpfile
        self.help_button = tk.Button(
            self,
            text="Help")
        self.help_button.pack(side=tk.LEFT, anchor=tk.NW, padx=1, pady=1)


class OperationsPanel(tk.LabelFrame):
    """
    The panel holding train and predict panels.

    Has useful methods for controlling both.
    """

    def __init__(self, parent,  *args, **kwargs):
        """
        Initialize the panel.

        Args:
            parent (Frame): Whatever's holding the panel.
        """
        super().__init__(parent, *args, **kwargs)
        self.parent = parent

    def enable_outputs(self):
        """Enable output control buttons for viewing and saving."""
        try:
            self.classifier_save.config(state=tk.ACTIVE)
            self.report_sec.enable_buttons()
            self.cm_sec.enable_buttons()
        except AttributeError:
            self.pairplot_sec.enable_buttons()

    def disable_outputs(self):
        """Disable output control buttons for viewing and saving."""
        try:
            self.classifier_save.config(state=tk.DISABLED)
            self.report_sec.disable_buttons()
            self.cm_sec.disable_buttons()
        except AttributeError:
            self.pairplot_sec.disable_buttons()


class TrainPanel(OperationsPanel):
    """
    The training controls panel.

    Basically just the predefined GUI parts.
    """

    def __init__(self, parent, app, *args, **kwargs):
        """Initialize the panel.

        Args:
            parent (Frame): Whatever's holding the panel.
            app (MainApp): Generally the MainApp for easy method access.
        """
        super().__init__(parent, *args, **kwargs)
        self.parent = parent
        self.app = app

        # select known class column
        self.known_select_label = tk.Label(
            self,
            text="Select Known Class Column:")
        self.known_select_label.pack(anchor=tk.W)
        # combobox (see top todo)
        self.known_select = ttk.Combobox(
            self,
            height=10,
            state="readonly",
            # textvariable=self.app.known_column
        )
        self.known_select.bind(
            "<<ComboboxSelected>>", self.app.enable_train)
        self.known_select.pack(fill=tk.X)

        # select N multirun
        self.n_label = tk.Label(
            self,
            text="N Classifiers to Compare:")
        self.n_label.pack(anchor=tk.W)
        # combobox (see above todo)
        validate_input = (self.app.parent.register(
            self.validate_n_input), '%P')
        self.n_input = ttk.Spinbox(
            self,
            from_=1,
            to=float('inf'),
            validate="all",
            validatecommand=validate_input)
        self.n_input.set(1000)
        self.n_input.pack(fill=tk.X)

        # button to train the classifier
        self.train_button = tk.Button(
            self,
            text="Train Classifier",
            pady=5,
            state=tk.DISABLED,
            command=self.app.train_classifier)
        self.train_button.pack(fill=tk.X, expand=True, padx=1, pady=1)

        # button to save the classifier
        self.classifier_save = tk.Button(
            self,
            text="Save Classifier",
            pady=5,
            width=5,
            state=tk.DISABLED,
            command=self.app.save_classifier)
        self.classifier_save.pack(fill=tk.X, expand=True, padx=1, pady=1)

        # buttons for viewing and saving report
        self.report_sec = VS_Pair(
            self,
            self.app,
            lambda: self.app.view_item(self.app.report.name),
            lambda: self.app.save_item("report"),
            text="Report",
            labelanchor=tk.N)
        self.report_sec.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # buttons for viewing and saving cm
        self.cm_sec = VS_Pair(
            self,
            self.app,
            lambda: self.app.view_item(self.app.cm.name),
            lambda: self.app.save_item("cm"),
            text="Conf. Matrix",
            labelanchor=tk.N)
        self.cm_sec.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

    def validate_n_input(self, value):
        """
        Validate that the given input is a number.

        Args:
            value (str): Input value.

        Returns:
            bool: True if value is number or blank else False.

        """
        if value == "":
            return True
        elif not value.isdigit():
            return False
        elif int(value) < 1:
            return False
        else:
            return True

    def reset_enabled(self):
        """Disable certain buttons for resetting between data/clf/etc."""
        self.disable_outputs()
        self.train_button.config(state=tk.DISABLED)
        self.classifier_save.config(state=tk.DISABLED)


class PredictPanel(OperationsPanel):
    """
    Panel holding controls for predicting on a dataset with a classifier.

    Mostly prefab GUI and not much else.
    """

    def __init__(self, parent, app, *args, **kwargs):
        """
        Initialize the Panel.

        Args:
            parent (Frame): Whatever's holding the Panel.
            app (MainApp): Generally the MainApp for easy method access.
        """
        super().__init__(parent, *args, **kwargs)
        self.parent = parent
        self.app = app

        self.classifier_load = tk.Button(
            self,
            text="Load Classifier",
            pady=5,
            command=self.app.load_classifier
        )
        self.classifier_load.pack(padx=1, pady=1, fill=tk.X, anchor=tk.S)

        self.prediction_make = tk.Button(
            self,
            text="Make Predictions",
            pady=5,
            state=tk.DISABLED,
            command=self.app.make_predictions
        )
        self.prediction_make.pack(padx=1, pady=1, fill=tk.X)

        self.pairplot_sec = VS_Pair(
            self,
            self.app,
            lambda: self.app.view_item(self.app.pairplot.name),
            lambda: self.app.save_item("pairplot"),
            text="Pairplot",
            labelanchor=tk.N
        )
        self.pairplot_sec.pack(padx=5, anchor=tk.N, fill=tk.X)

        self.output_save = tk.Button(
            self,
            text="Save Output",
            pady=5,
            width=5,
            state=tk.DISABLED,
            command=lambda: self.app.save_item("output")
        )
        self.output_save.pack(fill=tk.X, padx=1, pady=1)

    def reset_enabled(self):
        """Disable certain buttons for resetting between data/clf/etc."""
        self.prediction_make.config(state=tk.DISABLED)
        self.disable_outputs()
        self.output_save.config(state=tk.DISABLED)


class StatusBar(tk.Frame):
    """A statusbar to show the user that something is currently happening.

    Progress bars should also be used in their own windows when doing something
    intensive such as training or predicting.
    """

    def __init__(self, parent, app, *args, **kwargs):
        """Initialize the statusbar.

        Args:
            parent (Frame): Whatever's holding the statusbar.
            app (MainApp): Generally the MainApp for easy method access.
        """
        tk.LabelFrame.__init__(self, parent, *args, **kwargs)
        self.parent = parent
        self.app = app

        self.status = tk.Label(
            self,
            text="Status: Awaiting user input."
        )
        self.status.pack(side=tk.LEFT)

        self.progress = ttk.Progressbar(
            self,
            orient=tk.HORIZONTAL,
            length=100,
            mode="determinate",
            value=0
        )
        self.progress.pack(side=tk.RIGHT)

    def set_status(self, text):
        """
        Set the current status.

        Args:
            text (str): A new status.
        """
        self.status["text"] = "Status: {}".format(text)
