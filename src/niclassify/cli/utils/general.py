# Order commands in order defined

from typer.core import TyperGroup


class NaturalOrderGroup(TyperGroup):
    def list_commands(self, ctx):
        return list(self.commands)
