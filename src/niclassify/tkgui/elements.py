import tkinter as tk


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
