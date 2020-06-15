import tkinter as tk
from tkinter import ttk
import ast


class TwoColumnSelect(tk.Frame):
    """
    A two column selection interface.

    Allows users to swap items between the selected and deselected columns.
    Hopefully not hard for the user to follow.

    Also maintains order of items because I thought that'd be important.
    """

    def __init__(self, parent, *args, **kwargs):
        """
        Instantiate the interface.

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
