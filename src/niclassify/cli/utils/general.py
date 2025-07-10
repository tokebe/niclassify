# Order commands in order defined
from typing import override

from typer.core import TyperGroup


class NaturalOrderGroup(TyperGroup):
    @override
    def list_commands(self, ctx):
        return list(self.commands)
