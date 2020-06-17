import ast
import tkinter as tk
from tkinter import ttk
from .twocolumnselect import TwoColumnSelect
import threading


class VS_Pair(tk.LabelFrame):
    """A pair of buttons for viewing and saving a file.

    Contains a couple of useful methods, not much else.
    """

    def __init__(
            self,
            parent,
            app,
            view_callback,
            save_callback,
            *args,
            **kwargs):
        """
        Instantiate the VS_pair.

        Args:
            parent (Frame): Whatever tk object holds this pair.
            app (MainApp): Generally the MainApp for easy method access.
            view_callback (func): A function to call when view is pressed.
            save_callback (func): A function to call when save is pressed.
        """
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
        """Enable the buttons."""
        self.button_view.config(state=tk.ACTIVE)
        self.button_save.config(state=tk.ACTIVE)

    def disable_buttons(self):
        """Disable the buttons."""
        self.button_view.config(state=tk.DISABLED)
        self.button_save.config(state=tk.DISABLED)


class DataPanel(tk.LabelFrame):
    """
    One of the panels in MainApp, for opening and interacting with data.

    Mostly just pre-defined GUI objects.
    """

    def __init__(self, parent, app, *args, **kwargs):
        """
        Instantiate the DataPanel.

        Args:
            parent (Frame): Whatever's holding the DataPanel.
            app (MainApp): Generally the MainApp for easy method access.
        """
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
        # TODO implement the window, make it save to config (edit core, easier)
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
        Instantiate the panel.

        Args:
            parent (Frame): Whatever's holding the panel.
        """
        super().__init__(parent, *args, **kwargs)
        self.parent = parent

    def enable_outputs(self):
        """Enable output control buttons for viewing and saving."""
        try:
            self.classifier_save.config(state=tk.ACTIVE)
            self.report_section.enable_buttons()
            self.cm_section.enable_buttons()
        except AttributeError:
            self.pairplot_section.enable_buttons()

    def disable_outputs(self):
        """Disable output control buttons for viewing and saving."""
        try:
            self.classifier_save.config(state=tk.DISABLED)
            self.report_section.disable_buttons()
            self.cm_section.disable_buttons()
        except AttributeError:
            self.pairplot_section.disable_buttons()


class TrainPanel(OperationsPanel):
    """
    The training controls panel.

    Basically just the predefined GUI parts.
    """

    def __init__(self, parent, app, *args, **kwargs):
        """Instantiate the panel.

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
        # TODO highlight text red if invalid
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
        self.report_section = VS_Pair(
            self,
            self.app,
            lambda: self.app.view_item("report"),
            lambda: self.app.save_item("report"),
            text="Report",
            labelanchor=tk.N)
        self.report_section.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # buttons for viewing and saving cm
        self.cm_section = VS_Pair(
            self,
            self.app,
            lambda: self.app.view_item("cm"),
            lambda: self.app.save_item("cm"),
            text="Conf. Matrix",
            labelanchor=tk.N)
        self.cm_section.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

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
        Instantiate the Panel.

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

        self.pairplot_section = VS_Pair(
            self,
            self.app,
            lambda: self.app.view_item("pairplot"),
            lambda: self.app.save_item("pairplot"),
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
        """Instantiate the statusbar.

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


class NaNEditor(tk.Toplevel):
    """
    A small window for checking and editing recognized nan values.

    Saves to nans.json when closed.
    """

    def __init__(self, parent, app, *args, **kwargs):
        """Instantiate the editor window.

        Args:
            parent (Frame): Whatever's holding the panel.
            app (MainApp): Generally the MainApp for easy method access.
        """
        super().__init__(parent, *args, **kwargs)
        self.parent = parent
        self.app = app

        self.protocol("WM_DELETE_WINDOW", self.wm_exit)

        self.title("NaN Value Editor")
        self.minsize(width=300, height=400)

        self.item_var = tk.StringVar()
        self.item_var.set(value=[])

        self.itemframe = tk.Frame(self)
        self.itemframe.pack(side=tk.LEFT, expand=True, fill=tk.BOTH)

        self.items = tk.Listbox(
            self.itemframe,
            selectmode=tk.EXTENDED,
            listvariable=self.item_var
        )
        self.items.pack(side=tk.LEFT, expand=True, fill=tk.BOTH)

        self.item_scroll = tk.Scrollbar(
            self.itemframe,
            orient=tk.VERTICAL)
        self.item_scroll.config(command=self.items.yview)
        self.item_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.items.config(yscrollcommand=self.item_scroll.set)

        self.buttonframe = tk.Frame(self)
        self.buttonframe.pack(side=tk.RIGHT, fill=tk.Y)

        self.addbutton = tk.Button(
            self.buttonframe,
            text="Add",
            command=self.additem
        )
        self.addbutton.pack(fill=tk.X)

        self.removebutton = tk.Button(
            self.buttonframe,
            text="Remove",
            command=self.removeitem
        )
        self.removebutton.pack(fill=tk.X)

        self.item_var.set(value=self.app.sp.nans)

    def additem(self):
        """Open a string dialog to add a recognized value."""
        val = tk.simpledialog.askstring(
            parent=self,
            title="New NaN value",
            prompt="Please input a string to be recognized as NaN",
        )
        self.items.insert(tk.END, val)

    def removeitem(self):
        """Remove the currently selected items."""
        items = [
            j
            for i, j
            in enumerate(list(
                ast.literal_eval(self.item_var.get()))
                if len(self.item_var.get()) > 0
                else [])
            if i not in self.items.curselection()
        ]
        self.item_var.set(value=items)

    def wm_exit(self):
        """Close the window while saving the updated nans list."""
        self.app.sp.nans = self.items.get(0, tk.END)
        self.app.save_nans()
        self.destroy()


class ProgressPopup(tk.Toplevel):
    def __init__(self, parent, title, status, *args, **kwargs):
        """Instantiate the editor window.

        Args:
            parent (Frame): Whatever's holding the panel.
            app (MainApp): Generally the MainApp for easy method access.
        """
        super().__init__(parent, *args, **kwargs)
        self.parent = parent

        # stop main window interaction
        self.grab_set()

        # make window unclosable
        self.protocol("WM_DELETE_WINDOW", lambda: None)

        # set up window
        self.title(title)

        # frame for aligning progress and text
        self.align = tk.Frame(
            self,
            padx=10,
            pady=10
        )
        self.align.pack()

        # status label
        self.status = tk.Label(
            self.align,
            text=status,
            pady=5
        )
        self.status.pack(anchor=tk.W)

        # progress bar
        self.progress = ttk.Progressbar(
            self.align,
            orient=tk.HORIZONTAL,
            mode="indeterminate",
            length=250
        )
        self.progress.pack()

        self.progress.start(interval=10)

        self.resizable(False, False)
        # self.minsize(width=350, height=150)

    def complete(self):
        print("progress popup complete!")
        self.grab_release()
        self.destroy()

    def set_status(self, status):
        self.status["text"] = status
