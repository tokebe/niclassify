from niclassify.config.regions import REGIONS_FLAT
from niclassify.core.utils.fuzzy_match import score

GEOGRAPHIES = list(REGIONS_FLAT.keys())


def complete_geography(incomplete: str) -> list[str]:
    """Complete a partial string into a known geography."""
    if len(incomplete) == 0:
        return GEOGRAPHIES
    scores = {string: score(incomplete, string) for string in GEOGRAPHIES}
    return sorted(
        filter(lambda s: scores[s] > 0, GEOGRAPHIES),
        key=lambda s: scores[s],
    )
