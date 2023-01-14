from typing import List
from ...core.lookup import get_geographies
from ...core.utils.fuzzy_match import score

GEOGRAPHIES = get_geographies()


def complete_geography(incomplete: str) -> List[str]:
    if len(incomplete) == 0:
        return GEOGRAPHIES
    scores = {string: score(incomplete, string) for string in GEOGRAPHIES}
    return sorted(
        filter(lambda s: scores(s) > 0, GEOGRAPHIES),
        key=lambda s: scores[s],
    )
