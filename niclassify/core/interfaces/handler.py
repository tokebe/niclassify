from rich import print
import typer
import json
from pathlib import Path
from typing import Union, List
from threading import Lock


# class dotdict(dict):
#     """dot.notation access to dictionary attributes"""

#     __getattr__ = dict.get
#     __setattr__ = dict.__setitem__
#     __delattr__ = dict.__delitem__


# prefab_messages = {}
# with open(Path(__file__).parent / "../messages.json", "r") as messages:
#     prefab_messages = json.load(messages)
#     reformatted = {
#         **prefab_messages["error"],
#         **prefab_messages["warning"],
#         **prefab_messages["message"],
#     }

#     prefab_messages = dotdict(
#         {
#             code: f"{info['title']}: {info['message']}"
#             for (code, info) in reformatted.items()
#         }
#     )


class Handler:

    # prefab = prefab_messages
    debug_lock: Lock = None

    def log(*message: str):
        pass

    def message(*message: str):
        pass

    def warning(*message: str):
        pass

    def error(*error: str, abort=False):
        pass

    def confirm(*message, abort=False):
        pass

    def debug():
        pass

    def abort(self) -> None:
        pass

    def select(self, *prompt: str, options: List[str], abort=False) -> Union[str, None]:
        pass

    def select_multiple(
        self, *prompt: str, options: List[str], abort=False
    ) -> Union[str, None]:
        pass
