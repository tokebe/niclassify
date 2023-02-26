from ...core.lookup import get_geographies
import typer

GEOGRAPHIES = get_geographies()


def validate_geography(value: str) -> str:
    try:
        if int(value) < 1 or int(value) > len(GEOGRAPHIES):
            raise typer.BadParameter(
                f"{value} is not a valid geography selection. See use --list or use prompt to list geographies."
            )
    except (ValueError, TypeError):
        if value is not None and value not in GEOGRAPHIES:
            raise typer.BadParameter(
                f"{value} is not a known geography. See use --list or use prompt to list geographies."
            )
    return value
