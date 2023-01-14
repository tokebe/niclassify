from ..interfaces.handler import Handler
import requests
from ratelimiter import RateLimiter
from .get_ref_hierarchy import get_ref_hierarchy
from .geo_contains import geo_contains
import re
from typing import Optional


@RateLimiter(max_calls=10, period=1)
def query_gbif(species_name: str, geography: str, handler: Handler) -> Optional[str]:
    code_url = "http://api.gbif.org/v1/species?name="
    records_url = "http://api.gbif.org/v1/species/"

    # get taxonKey
    request = f"{code_url}{species_name.lower().replace(' ', '%20')}"

    response = requests.get(request)
    response.raise_for_status()

    # search for "taxonID":"gbif:" with some numbers, getting the numbers
    taxonKey = re.search('(?<="taxonID":"gbif:)\\d+', response.text)

    if taxonKey is None:
        handler.debug(f"  (Unknown) GBIF: {species_name}: No data")
        return None
    taxonKey = taxonKey.group()

    # get native range
    request = f"{records_url}{taxonKey}/descriptions"

    response = requests.get(request)
    response.raise_for_status()

    try:
        results = response.json()
    except UnicodeDecodeError:
        handler.debug(f"  (Unknown) GBIF: {species_name}: No data")
        return None

    lookup = [
        res["description"]
        for res in results["results"]
        if res["type"] == "native range"
    ]

    if len(lookup) == 0:
        handler.debug(f"  (Unknown) GBIF: {species_name}: No data")
        return None
    nranges = lookup

    cryptogenic = False

    check = 1

    for nrange in nranges:
        if nrange == "Cosmopolitan, Cryptogenic":
            handler.debug(f"  (Unknown) GBIF: {species_name}: cryptogenic")
            cryptogenic = True
            continue
        if nrange == "Pantropical, Circumtropical":
            ref_hierarchy = get_ref_hierarchy(geography)
            if ref_hierarchy["Pantropical"] is True:
                handler.debug(f"  (Native) GBIF: {species_name}: Pantropical")
                return "Native"
            handler.debug(f"  (Introduced) GBIF: {species_name}: Pantropical")
            continue
        if nrange == "Subtropics":
            ref_hierarchy = get_ref_hierarchy(geography)
            if ref_hierarchy["Subtropics"] is True:
                handler.debug(f"  (Native) GBIF: {species_name}: Subtropical")
                return "Native"
            handler.debug(f"  (Introduced) GBIF: {species_name}: Subtropical")
            continue

        if nrange == geography:
            handler.debug(
                f"  (Native) GBIF: {species_name}: directly native to reference geography {nrange}"
            )
            return "Native"
        if geo_contains(geography, nrange) or geo_contains(nrange, geography):
            handler.debug(
                f"  (Native) GBIF: {species_name}: reference geography {geography} <=> native range {nrange}"
            )
            return "Native"
        handler.debug(
            f"  (Mismatch) GBIF: {species_name} (attempt {check}): reference geography {geography} <!=> native range {nrange}"
        )
        check += 1

    # if it hasn't found a reason to call it native
    handler.debug(
        f"  {'(Unknown)' if cryptogenic else '(Introduced)'} GBIF: {species_name}: species is {'cryptogenic' if cryptogenic else f'not native to reference geography {geography}'}"
    )
    return "Introduced" if not cryptogenic else None
