import tkinter as tk
from tkinter import ttk
import ast


class TwoColumnSelect(tk.Frame):
    def __init__(self, parent, *args, **kwargs):
        tk.Frame.__init__(self, parent, *args, **kwargs)
        self.parent = parent

        # variables for contents
        self.desel = tk.StringVar()
        self.sel = tk.StringVar()

        self.desel.set(value=parent.column_names)
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
        desel_items = (list(ast.literal_eval(self.desel.get()))
                       if len(self.desel.get()) > 0 else [])
        sel_items = (list(ast.literal_eval(self.sel.get()))
                     if len(self.sel.get()) > 0 else [])
        if allitems:
            sel_items.extend(desel_items)

            self.desel.set(value=[])
            self.sel.set(value=sel_items)

        elif self.desel_contents.curselection() == ():
            return

        else:
            # TODO modify to preserve column order, if possible
            sel_items.extend(desel_items[i]
                             for i
                             in self.desel_contents.curselection())
            desel_items = [x
                           for i, x in enumerate(desel_items)
                           if i not in self.desel_contents.curselection()]

            self.desel.set(value=desel_items)
            self.sel.set(value=sel_items)

    def to_left(self, allitems=False):
        desel_items = (list(ast.literal_eval(self.desel.get()))
                       if len(self.desel.get()) > 0 else [])
        sel_items = (list(ast.literal_eval(self.sel.get()))
                     if len(self.sel.get()) > 0 else [])
        if allitems:
            desel_items.extend(sel_items)

            self.desel.set(value=desel_items)
            self.sel.set(value=[])

        elif self.sel_contents.curselection() == ():
            return

        else:
            # TODO modify to preserve column order, if possible
            # probably use dict
            desel_items.extend(sel_items[i]
                               for i
                               in self.sel_contents.curselection())
            sel_items = [x
                         for i, x in enumerate(sel_items)
                         if i not in self.sel_contents.curselection()]

            self.desel.set(value=desel_items)
            self.sel.set(value=sel_items)

    def update_contents(self, values):
        None



if __name__ == "__main__":
    root = tk.Tk()
    root.style = ttk.Style()
    root.style.theme_use("winnative")
    # print(root.style.theme_names())
    TwoColumnSelect(root)
    root.update()
    root.minsize(root.winfo_width(), root.winfo_height())
    root.mainloop()
