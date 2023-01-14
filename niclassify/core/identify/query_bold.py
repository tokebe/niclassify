from ..interfaces.handler import Handler
from typing import Optional
import requests
from collections import Counter

from ratelimiter import RateLimiter
from xml.etree import ElementTree


@RateLimiter(max_calls=10, period=1)
def query_bold(
    sequence: str, min_similarity: float, min_agreement: float, handler: Handler
) -> Optional[str]:
    """Return the maximum-confidence species name for a given sequence."""

    api_url = (
        "https://www.boldsystems.org/index.php/Ids_xml?db=COX1_SPECIES_PUBLIC&sequence="
    )

    request = f"{api_url}{sequence}"

    response = requests.get(request)
    response.raise_for_status()

    # get xml tree from response
    tree = ElementTree.fromstring(response.content)

    matches = [
        (
            match.findall("taxonomicidentification")[0].text,
            float(match.findall("similarity")[0].text),
        )
        for match in tree.iter("match")
    ]
    if len(matches) == 0:
        handler.debug("No matches found.")
        return None

    max_similarity = max((similarity for tax, similarity in matches))

    if max_similarity < min_similarity:
        handler.debug("No matches met minimum similarity.")
        return None

    matches = [
        (tax, similarity) for tax, similarity in matches if similarity >= max_similarity
    ]

    counts = Counter((tax for tax, similarity in matches))

    match_proportions = sorted(
        [
            (name, counts[name] / len(matches))
            for name in counts
            if counts[name] / len(matches) >= min_agreement
        ],
        key=lambda element: element[1],
    )

    if len(match_proportions) == 0:
        handler.debug("No matches met minimum agreement.")
        return None

    return match_proportions[0][0]
