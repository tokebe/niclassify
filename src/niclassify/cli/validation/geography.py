import typer

from niclassify.config.regions import REGIONS_FLAT


GEOGRAPHIES = list(REGIONS_FLAT.keys())


def validate_geography(value: str) -> str:
    """Validate that geography is known."""
    try:
        if int(value) < 1 or int(value) > len(GEOGRAPHIES):
            raise typer.BadParameter(
                f"{value} is not a valid geography selection. See use --list or use prompt to list geographies."
            )
    except (ValueError, TypeError):
        if value not in GEOGRAPHIES:
            raise typer.BadParameter(
                f"{value} is not a known geography. See use --list or use prompt to list geographies."
            ) from None
    return value
