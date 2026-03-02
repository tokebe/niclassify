from rich.markup import render
from rich.text import Text


def strip_markup(string: str | Text) -> str:
    """Strip Rich markup from the given text."""
    if isinstance(string, Text):
        string = string.plain
    return render(string).plain
