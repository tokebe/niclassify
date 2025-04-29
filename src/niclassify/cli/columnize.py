from itertools import zip_longest
import os
import math
from rich import print
from rich.table import Table
from typing import List

def split(a, n):
    k, m = divmod(len(a), n)
    return (a[i*k+min(i, m):(i+1)*k+min(i+1, m)] for i in range(n))


def columnize(values: List[str], dry_run: bool = False, number: bool = False) -> Table:

    number_len = 0

    if number:
        number_len = len(str(len(values) + 1))
        values = [f"{(i + 1):{number_len}}) {v}" for i, v in enumerate(values)]

    console_size = os.get_terminal_size()
    width = console_size.columns
    max_width_item = len(max(values, key=len))
    columns = math.ceil(width / (max_width_item + 4 + number_len))

    table = Table(
        show_edge=False,
        show_header=False,
        box=None,
        padding=(0, 1),
    )

    splits = list(split(values, columns))

    for i in range(columns):
        table.add_column()
    for row in zip_longest(*splits):
        table.add_row(*row)

    if not dry_run:
        print(table)
    return table
