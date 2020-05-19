import tkinter as tk
from tkinter import ttk
from .twocolumnselect import TwoColumnSelect

# TODO make a combobox class which resizes dropdown and validates input


class VS_Pair(tk.LabelFrame):
    def __init__(
            self,
            parent,
            app,
            # view_callback,
            # save_callback,
            *args,
            **kwargs):
        tk.LabelFrame.__init__(
            self,
            parent,
            # view_callback,
            # save_callback,
            *args,
            **kwargs)
        self.parent = parent
        self.app = app

        self.report_view = tk.Button(
            self,
            text="View",
            width=5,
            # command=view_callback
        )
        self.report_view.pack(padx=1, pady=1)

        self.report_save = tk.Button(
            self,
            text="Save",
            width=5,
            # command=save_callback
        )
        self.report_save.pack(padx=1, pady=1)


class DataPanel(tk.LabelFrame):
    def __init__(self, parent, app, *args, **kwargs):
        tk.LabelFrame.__init__(self, parent, *args, **kwargs)
        self.parent = parent
        self.app = app

        self.column_names = "Aute in ea ipsum eu minim Lorem enim adipisicing eiusmod laboris magna dolor deserunt ANOTHERHORRIBLYLONGLINEFORTESTINGPURPOSESBYYOURSTRULYJC".split(
            " ")

        # button to load data
        self.load_data_button = tk.Button(
            self,
            text="Load Data",
            pady=5,
            command=self.app.get_data
        )
        self.load_data_button.pack(fill=tk.X, padx=1, pady=1)

        # excel sheet selection
        # TODO grey out unless opened file is excel
        self.excel_label = tk.Label(
            self,
            text="Specify Excel Sheet:")
        # self.excel_label.bind('<Configure>') # write a function to set size
        # depends on having access to list of items
        # see https://stackoverflow.com/questions/39915275/change-width-of-dropdown-listbox-of-a-ttk-combobox
        self.excel_label.pack(anchor=tk.W)
        # TODO highlight text red if invalid
        self.excel_sheet_input = ttk.Combobox(
            self,
            height=10,
            state=tk.DISABLED
        )
        self.excel_sheet_input.pack(fill=tk.X)

        # column selection section
        self.col_select_hint = tk.Label(
            self,
            text="Select Data Columns:")
        self.col_select_hint.pack(anchor=tk.W)
        # selection panel, depends on twocolumnselect.py
        self.col_select_panel = TwoColumnSelect(self)
        self.col_select_panel.pack(fill=tk.BOTH, expand=True)

        # button to open window allowing NaN values to be edited
        # TODO implement the window, make it save to config (edit core, easier)
        self.nan_check = tk.Button(
            self,
            text="Check Recognized NaN values")
        self.nan_check.pack(anchor=tk.NW, padx=1, pady=1)


class TrainPanel(tk.LabelFrame):
    def __init__(self, parent, app, *args, **kwargs):
        tk.LabelFrame.__init__(self, parent, *args, **kwargs)
        self.parent = parent
        self.app = app

        # select known class column
        self.known_select_label = tk.Label(
            self,
            text="Select Known Class Column:")
        self.known_select_label.pack(anchor=tk.W)
        # combobox (see top todo)
        # TODO highlight text red if invalid
        self.known_select = ttk.Combobox(
            self,
            height=10
        )
        self.known_select.pack(fill=tk.X)

        # select N multirun
        self.n_label = tk.Label(
            self,
            text="N Classifiers to Compare:")
        self.n_label.pack(anchor=tk.W)
        # combobox (see above todo)
        self.n_input = ttk.Combobox(
            self,
            values=[1, 10, 50, 100, 500, 1000],
            height=6)
        self.n_input.pack(fill=tk.X)

        # button to train the classifier
        self.train_button = tk.Button(
            self,
            text="Train Classifier",
            pady=5)
        self.train_button.pack(fill=tk.X, expand=True, padx=1, pady=1)

        # button to save the classifier
        self.classifier_save = tk.Button(
            self,
            text="Save Classifier",
            pady=5,
            width=5)
        self.classifier_save.pack(fill=tk.X, expand=True, padx=1, pady=1)

        # buttons for viewing and saving report
        self.report_section = VS_Pair(
            self,
            self.app,
            text="Report",
            labelanchor=tk.N)
        self.report_section.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # buttons for viewing and saving cm
        self.cm_section = VS_Pair(
            self,
            self.app,
            text="Conf. Matrix",
            labelanchor=tk.N)
        self.cm_section.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)


class PredictPanel(tk.LabelFrame):
    def __init__(self, parent, app, *args, **kwargs):
        tk.LabelFrame.__init__(self, parent, *args, **kwargs)
        self.parent = parent
        self.app = app

        self.classifier_load = tk.Button(
            self,
            text="Load Classifier",
            pady=5)
        self.classifier_load.pack(padx=1, pady=1, fill=tk.X, anchor=tk.S)

        self.prediction_make = tk.Button(
            self,
            text="Make Predictions",
            pady=5)
        self.prediction_make.pack(padx=1, pady=1, fill=tk.X)

        self.pairplot_section = VS_Pair(
            self,
            self.app,
            text="Pairplot",
            labelanchor=tk.N)
        self.pairplot_section.pack(padx=5, anchor=tk.N, fill=tk.X)

        self.output_save = tk.Button(
            self,
            text="Save Output",
            pady=5,
            width=5)
        self.output_save.pack(fill=tk.X, padx=1, pady=1)
