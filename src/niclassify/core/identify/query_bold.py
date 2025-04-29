from niclassify.core.interfaces.handler import Handler
from typing import Optional, cast
from collections import Counter

from xml.etree import ElementTree
from typing import Set, Dict
import json
from ratelimit import RateLimitException, limits
from backoff import on_exception, expo
import httpx

# TODO get order, family, subfamily, genus from top match if identify success
# TODO state warnings if identify gives

client = httpx.Client(
    transport=httpx.HTTPTransport(retries=3), timeout=600, follow_redirects=True
)


@on_exception(expo, RateLimitException)
@limits(calls=500, period=10)
def query_bold(
    uid: str,
    sequence: str,
    min_similarity: float,
    min_agreement: float,
    orders: Set[str],
    handler: Handler,
) -> dict[str, str | None]:
    """Return the maximum-confidence species name for a given sequence."""

    normalized_info: dict[str, str | None] = {
        "subspecies_name": None,
        "species_name": None,
        "subgenus_name": None,
        "genus_name": None,
        "tribe_name": None,
        "subfamily_name": None,
        "family_name": None,
        "class_name": None,
        "phylum_name": None
    }

    api_url = (
        "https://www.boldsystems.org/index.php/Ids_xml?db=COX1_SPECIES_PUBLIC&sequence="
    )

    try:
        request = f"{api_url}{sequence}"
        handler.debug(f"  {uid}: Making query...")
        response = client.get(request)
        response.raise_for_status()
    except httpx.HTTPError as error:
        handler.error(str(error))
        handler.error("BOLD query failed. See error above.")
        return normalized_info

    # get xml tree from response
    tree = ElementTree.fromstring(response.content)

    matches = [
        (
            match.findall("taxonomicidentification")[0].text,
            float(cast(str, match.findall("similarity")[0].text)),
            match.findall("ID"),
        )
        for match in tree.iter("match")
    ]
    if len(matches) == 0:
        handler.debug(f"  {uid}: No matches found.")
        return normalized_info

    max_similarity = max((similarity for tax, similarity, pid in matches))

    if max_similarity < min_similarity:
        handler.debug(f"  {uid}: No matches met minimum similarity.")
        return normalized_info

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
        handler.debug(f"  {uid}: No matches met minimum agreement.")
        return normalized_info

    species = match_proportions[0][0]

    # get taxonID to use to get hierarchy
    try:
        request = (
            f"https://boldsystems.org/index.php/API_Tax/TaxonSearch?taxName={species}"
        )
        response = client.get(request)
        response.raise_for_status()
        taxID = json.loads(response.text)["top_matched_names"][0]["taxid"]
        request = f"https://boldsystems.org/index.php/API_Tax/TaxonData?taxId={taxID}&includeTree=true&dataTypes=basic"
        response = client.get(request)
        response.raise_for_status()
    except httpx.HTTPError as error:
        handler.error(str(error))
        handler.error("BOLD query failed. See error above.")
        return normalized_info

    for taxID, info in json.loads(response.text).items():
        normalized_info[f"{info['tax_rank']}_name"] = info["taxon"]

    handler.debug(f"  {uid}: Successfully identified: {species}")
    if (
        normalized_info.get("order_name", None) is not None
        and normalized_info["order_name"] not in orders
    ):
        if len(orders) > 0:
            handler.warning(
                f"  {uid}: Identified species {species} is of order {normalized_info['order_name']}, which is not present in original data. Please check output for potential misidentification."
            )
    return normalized_info
