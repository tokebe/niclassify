from ..interfaces.handler import Handler
from typing import Optional
import requests
from collections import Counter

from xml.etree import ElementTree
from typing import Set, Dict
import json
from ratelimit import RateLimitException, limits
from backoff import on_exception, expo

# TODO get order, family, subfamily, genus from top match if identify success
# TODO state warnings if identify gives


@on_exception(expo, RateLimitException)
@limits(calls=10, period=1)
def query_bold(
    sequence: str,
    min_similarity: float,
    min_agreement: float,
    orders: Set[str],
    handler: Handler,
) -> Optional[Dict[str, str]]:
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
            match.findall("ID"),
        )
        for match in tree.iter("match")
    ]
    if len(matches) == 0:
        handler.debug("  No matches found.")
        return None

    max_similarity = max((similarity for tax, similarity, pid in matches))

    if max_similarity < min_similarity:
        handler.debug("  No matches met minimum similarity.")
        return None

    best_matches = [
        (tax, similarity, pid)
        for tax, similarity, pid in matches
        if similarity >= max_similarity
    ]

    counts = Counter((tax for tax, similarity, pid in best_matches))

    match_proportions = sorted(
        [
            (name, counts[name] / len(best_matches))
            for name in counts
            if counts[name] / len(best_matches) >= min_agreement
        ],
        key=lambda element: element[1],
    )

    if len(match_proportions) == 0:
        handler.debug("  No matches met minimum agreement.")
        return None

    species = match_proportions[0][0]

    # get taxonID to use to get hierarchy
    request = f"https://boldsystems.org/index.php/API_Tax/TaxonSearch?taxName={species}"
    response = requests.get(request)
    response.raise_for_status()
    # TODO some error checking here
    taxID = json.loads(response.text)["top_matched_names"][0]["taxid"]
    request = f"https://boldsystems.org/index.php/API_Tax/TaxonData?taxId={taxID}&includeTree=true&dataTypes=basic"
    response = requests.get(request)
    response.raise_for_status()
    info = {
        f"{info['tax_rank']}_name": info["taxon"]
        for taxID, info in json.loads(response.text).items()
    }
    handler.debug(f"  Successfully identified: {species}")
    if info.get("order_name", None) is not None and info["order_name"] not in orders:
        if len(orders) > 0:
            handler.warning(
                f"  Identified species {species} is of order {info['order_name']}, which is not present in original data. Please check output for potential misidentification."
            )
    return info
