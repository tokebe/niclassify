import os
import math
from rich import print
from rich.table import Table
from typing import List


def columnize(values: List[str], dry_run: bool = False, number: bool = False) -> Table:

    if number:
        values = [f"{i + 1}) {v}" for i, v in enumerate(values)]

    console_size = os.get_terminal_size()
    height = console_size.lines - 4
    width = console_size.columns
    max_width_item = len(max(values, key=len))
    max_columns = math.floor(width / (max_width_item + 4))

    if len(values) <= height:
        table = Table(
            show_edge=False,
            show_header=False,
            box=None,
            padding=(0, 2),
        )
        table.add_column()
        for v in values:
            table.add_row(v)
        if not dry_run:
            print(table)
        return table

    table = Table(
        show_edge=False,
        show_header=False,
        box=None,
        padding=(0, 2),
    )

    if math.ceil(len(values) / max_columns) >= height:
        columns = max_columns
    else:
        columns = math.ceil(len(values) / height)

    for i in range(columns):
        table.add_column()
    for row in (values[i : i + len(values) : height] for i in range(0, height)):
        table.add_row(*row)

    if not dry_run:
        print(table)
    return table
