from rich import print
import typer
import json
from pathlib import Path
from ..core.utils import dotdict


prefab_messages = {}
with open(Path(__file__).parent / "messages.json", "r") as messages:
    prefab_messages = json.load(messages)
    reformatted = {**prefab_messages["error"], **prefab_messages["warning"], **prefab_messages["message"]}

    prefab_messages = dotdict({code: f"{info['title']}: {info['message']}" for (code, info) in reformatted.items()})


# TODO figure out if you want special formatting/etc


class Handler:

    prefab = prefab_messages

    def log(message: str):
        print(message)

    def message(message: str):
        print(message)

    def warning(message: str):
        print(message)

    def error(error: str, should_exit=False):
        print(error)
        if should_exit:
            typer.Exit(code=1)

    def confirm(message, abort=False):
        return typer.confirm(message, abort=abort)

