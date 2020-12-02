"""Some smaller, less complicated windows that serve single functions."""
import ast

import tkinter as tk
import tkinter.simpledialog

from tkinter import ttk


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
        val = tkinter.simpledialog.askstring(
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
    """A popup which displays a progressbar and status message."""

    def __init__(self, parent, title, status, *args, **kwargs):
        """
        Instantiate the progress window.

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

        self.focus_force()

    def complete(self):
        """Complete progress, returning control to main window."""
        # print("progress popup complete!")
        self.grab_release()
        self.destroy()

    def set_status(self, status):
        """Change the current status message.

        Args:
            status (str): A new status message.
        """
        self.status["text"] = status
