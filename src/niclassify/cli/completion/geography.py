
from niclassify.core.lookup.get_geographies import get_geographies
from niclassify.core.utils.fuzzy_match import score

GEOGRAPHIES = get_geographies()


def complete_geography(incomplete: str) -> list[str]:
    if len(incomplete) == 0:
        return GEOGRAPHIES
    scores = {string: score(incomplete, string) for string in GEOGRAPHIES}
    return sorted(
        filter(lambda s: scores(s) > 0, GEOGRAPHIES),
        key=lambda s: scores[s],
    )
