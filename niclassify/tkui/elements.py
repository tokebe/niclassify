import tkinter as tk
from tkinter import ttk
from .twocolumnselect import TwoColumnSelect
import threading
from PIL import Image, ImageTk


class VS_Pair(tk.LabelFrame):
    def __init__(
            self,
            parent,
            app,
            view_callback,
            save_callback,
            *args,
            **kwargs):
        tk.LabelFrame.__init__(
            self,
            parent,
            *args,
            **kwargs)
        self.parent = parent
        self.app = app

        self.button_view = tk.Button(
            self,
            text="View",
            width=5,
            state=tk.DISABLED,
            command=view_callback
        )
        self.button_view.pack(padx=1, pady=1)

        self.button_save = tk.Button(
            self,
            text="Save",
            width=5,
            state=tk.DISABLED,
            command=save_callback
        )
        self.button_save.pack(padx=1, pady=1)

    def enable_buttons(self):
        self.button_view.config(state=tk.ACTIVE)
        self.button_save.config(state=tk.ACTIVE)

    def disable_buttons(self):
        self.button_view.config(state=tk.DISABLED)
        self.button_save.config(state=tk.DISABLED)


class DataPanel(tk.LabelFrame):
    def __init__(self, parent, app, *args, **kwargs):
        tk.LabelFrame.__init__(self, parent, *args, **kwargs)
        self.parent = parent
        self.app = app

        # button to load data
        self.load_data_button = tk.Button(
            self,
            text="Load Data",
            pady=5,
            command=self.app.get_data_file
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
        self.excel_sheet_input = ttk.Combobox(
            self,
            height=10,
            state=tk.DISABLED,
            # textvariable=self.app.sheet
        )
        self.excel_sheet_input.bind(
            "<<ComboboxSelected>>", self.app.get_sheet_cols)
        self.excel_sheet_input.pack(fill=tk.X)

        # column selection section
        self.col_select_hint = tk.Label(
            self,
            text="Select Data Columns:")
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
        # TODO implement the window, make it save to config (edit core, easier)
        self.nan_check = tk.Button(
            self,
            text="Check Recognized NaN values"
        )
        self.nan_check.pack(side=tk.LEFT, anchor=tk.NW, padx=1, pady=1)

        # button to open helpfile
        self.help_button = tk.Button(
            self,
            text="Help")
        self.help_button.pack(side=tk.LEFT, anchor=tk.NW, padx=1, pady=1)

    def sheet_validate(self, name):
        return name in self.app.sheets
    # TODO implement more robust validation, and function for invalidcommand
    # see https://stackoverflow.com/questions/4140437/interactively-validating-entry-widget-content-in-tkinter


class OperationsPanel(tk.LabelFrame):
    def __init__(self, parent,  *args, **kwargs):
        super().__init__(parent, *args, **kwargs)
        self.parent = parent

    def enable_outputs(self):
        try:
            self.classifier_save.config(state=tk.ACTIVE)
            self.report_section.enable_buttons()
            self.cm_section.enable_buttons()
        except AttributeError:
            self.pairplot_section.enable_buttons()

    def disable_outputs(self):
        try:
            self.classifier_save.config(state=tk.DISABLED)
            self.report_section.disable_buttons()
            self.cm_section.disable_buttons()
        except AttributeError:
            self.pairplot_section.disable_buttons()


class TrainPanel(OperationsPanel):
    def __init__(self, parent, app, *args, **kwargs):
        super().__init__(parent, *args, **kwargs)
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
            height=10,
            state="readonly",
            textvariable=self.app.known_column
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
        self.report_section = VS_Pair(
            self,
            self.app,
            self.app.view_report,
            self.app.save_report,
            text="Report",
            labelanchor=tk.N)
        self.report_section.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # buttons for viewing and saving cm
        self.cm_section = VS_Pair(
            self,
            self.app,
            lambda: self.app.view_graph("cm"),
            lambda: self.app.save_graph("cm"),
            text="Conf. Matrix",
            labelanchor=tk.N)
        self.cm_section.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

    def validate_n_input(self, value):
        if value == "":
            return True
        elif not value.isdigit():
            return False
        elif int(value) < 1:
            return False
        else:
            return True

    def reset_enabled(self):
        self.disable_outputs()
        self.train_button.config(state=tk.DISABLED)
        self.classifier_save.config(state=tk.DISABLED)


class PredictPanel(OperationsPanel):
    def __init__(self, parent, app, *args, **kwargs):
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

        self.pairplot_section = VS_Pair(
            self,
            self.app,
            lambda: self.app.view_graph("pairplot"),
            lambda: self.app.save_graph("pairplot"),
            text="Pairplot",
            labelanchor=tk.N
        )
        self.pairplot_section.pack(padx=5, anchor=tk.N, fill=tk.X)

        self.output_save = tk.Button(
            self,
            text="Save Output",
            pady=5,
            width=5,
            state=tk.DISABLED,
            command=self.app.save_output
        )
        self.output_save.pack(fill=tk.X, padx=1, pady=1)

    def reset_enabled(self):
        self.prediction_make.config(state=tk.DISABLED)
        self.disable_outputs()
        self.output_save.config(state=tk.DISABLED)


class StatusBar(tk.Frame):
    def __init__(self, parent, app, *args, **kwargs):
        tk.LabelFrame.__init__(self, parent, *args, **kwargs)
        self.parent = parent
        self.app = app

        self.label = tk.Label(
            self,
            text="Status:"
        )
        self.label.pack(side=tk.LEFT)

        self.status = tk.Label(
            self,
            text=""
        )
        self.label.pack(side=tk.LEFT)

        self.progress = ttk.Progressbar(
            self,
            orient=tk.HORIZONTAL,
            length=100,
            mode="determinate",
            value=0
        )
        self.progress.pack(side=tk.RIGHT)

    def set_status(text):
        self.status.config(text="Status: {}".format(text))


class ImageFrame(tk.Frame):
    def __init__(self, parent, img, *args, **kwargs):
        tk.LabelFrame.__init__(self, parent, *args, **kwargs)
        self.parent = parent

        self.image = Image.open(img)
        self.img_copy = self.image.copy()

        self.background_image = ImageTk.PhotoImage(self.image)

        self.background = tk.Label(self, image=self.background_image)
        self.background.pack(fill=tk.BOTH, expand=tk.YES)
        self.background.bind('<Configure>', self._resize_image)

    def _resize_image(self, event):

        new_width = event.width
        new_height = event.height

        self.image = self.img_copy.resize((new_width, new_height))

        self.background_image = ImageTk.PhotoImage(self.image)
        self.background.configure(image=self.background_image)
