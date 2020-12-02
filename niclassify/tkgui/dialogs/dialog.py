"""Handler for generating dialogs from json files."""
import json
import os
import sys


import importlib.resources as pkg_resources

from . import messages


class DialogLibrary:
    """
    A library of dialog contents.

    Contains function for generating dialog with given contents.
    """

    def __init__(self):
        """
        Initialize the library.

        Args:
            dialogs_folder (str): Path to the folder containing dialog jsons.
        """
        self.items = {}
        files = ["message.json", "warning.json", "error.json"]
        try:
            for file in files:
                with pkg_resources.open_text(messages, file) as msg_file:
                    self.items[os.path.splitext(file)[0]] = json.load(
                        msg_file)
        except FileNotFoundError:
            for file in files:
                with open(
                    os.path.join(sys._MEIPASS, "dialogs/{}".format(file))
                ) as msg_file:
                    self.items[os.path.splitext(file)[0]] = json.load(
                        msg_file)
        # for file in [
        #     f
        #     for f in os.listdir(dialogs_folder)
        #     if os.path.isfile(os.path.join(dialogs_folder, f))
        # ]:
        #     print(os.path.splitext(file))
        #     if os.path.splitext(file)[1] == ".json":
        #         print("json file found")
        #         with open(os.path.join(dialogs_folder, file)) as dialog_file:
        #             self.items[os.path.splitext(file)[0]] = json.load(
        #                 dialog_file)

    def __str__(self):
        """
        Return a string representation of the library contents.

        Returns:
            str: str output of self.items dict

        """
        return json.dumps(self.items)

    def get(self, desc):
        """
        Find and return the dialog contents for a given type and desc.

        Args:
            diag_type (str): Type of dialog, used as key for search.
            desc (str): Short description of dialog.

        Returns:
            dict: A dictionary containing the title and message of the dialog.

        """
        for name, lib in self.items.items():
            for description, contents in lib.items():
                if description == desc:
                    return (contents, name)
        return None

    def dialog(self, diag_type, desc, form=(None,), **kwargs):
        """
        Get user response to a dialog window with content lookup from lib.

        Args:
            diag_type (func): A tkinter messagebox method.
            desc (str): The dialog descriptor for content search.
            form (object, optional): A tuple of formatting arguments.

        **kwargs:
            <See **kwargs for messagebox dialogs>

        Returns:
            bool: Return value of the dialog generated.

        """
        if type(form) != tuple:
            form = (form,)
        diag_info = self.get(desc)

        if diag_info is None:
            raise KeyError("dialog description not found")

        contents = diag_info[0]

        dialog = diag_type(
            title=contents["title"],
            message="{}\n{}".format(
                contents["message"].format(*form),
                "code: {}".format(desc) if diag_info[1] == "error" else ""
            ),
            **kwargs
        )
        return dialog
