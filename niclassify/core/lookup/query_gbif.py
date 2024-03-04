from ..interfaces.handler import Handler
import requests
from throttler import throttle
from .get_ref_hierarchy import get_ref_hierarchy
from .geo_contains import geo_contains
import re
from typing import Optional
from time import sleep
from backoff import on_exception, expo
from ratelimit import limits, RateLimitException


@on_exception(expo, RateLimitException)
@limits(calls=60, period=60)
def query_gbif(species_name: str, ref_geo: str, handler: Handler) -> Optional[str]:
    """Query GBIF to determine if a species is native or introduced to a reference geography."""
    taxon_key_url = "http://api.gbif.org/v1/species?name="
    records_url = "http://api.gbif.org/v1/species/"

    tries = 3

    while tries > 0:
        try:
            # get taxonKey
            request = f"{taxon_key_url}{species_name.lower().replace(' ', '%20')}"
            response = requests.get(request)
            response.raise_for_status()

            # search for "taxonID":"gbif:" with some numbers, getting the numbers
            taxon_key = re.search('(?<="taxonID":"gbif:)\\d+', response.text)

            if taxon_key is None:
                handler.debug(f"  (Unknown) GBIF: {species_name}: No data")
                return None
            taxon_key = taxon_key.group()

            # get native range
            request = f"{records_url}{taxon_key}/descriptions"
            response = requests.get(request)
            response.raise_for_status()

            break

        except (
            ConnectionError,
            ConnectionRefusedError,
            ConnectionAbortedError,
            ConnectionResetError,
        ) as error:
            handler.debug(f"Failed on attempt {4 - tries}. See error below.")
            handler.debug(error)
            tries -= 1
            sleep(3)

    if tries < 1:
        handler.error(
            "Connection Error, GBIF may be having issues. Please try again later.",
            abort=True,
        )

    try:
        results = response.json()
    except UnicodeDecodeError:
        handler.debug(f"  (Unknown) GBIF: {species_name}: No data")
        return None

    native_ranges = [
        res["description"]
        for res in results["results"]
        if res["type"] == "native range"
    ]

    if len(native_ranges) == 0:
        handler.debug(f"  (Unknown) GBIF: {species_name}: No data")
        return None

    cryptogenic = False

    check = 1

    for native_range in native_ranges:
        if native_range == "Cosmopolitan, Cryptogenic":
            handler.debug(f"  (Unknown) GBIF: {species_name}: cryptogenic")
            cryptogenic = True
            continue
        if native_range == "Pantropical, Circumtropical":
            reference_hierarchy = get_ref_hierarchy(ref_geo)
            if reference_hierarchy["Pantropical"] is True:
                handler.debug(f"  (Native) GBIF: {species_name}: Pantropical")
                return "Native"
            handler.debug(f"  (Introduced) GBIF: {species_name}: Pantropical")
            continue
        if native_range == "Subtropics":
            reference_hierarchy = get_ref_hierarchy(ref_geo)
            if reference_hierarchy["Subtropics"] is True:
                handler.debug(f"  (Native) GBIF: {species_name}: Subtropical")
                return "Native"
            handler.debug(f"  (Introduced) GBIF: {species_name}: Subtropical")
            continue

        if native_range == ref_geo:
            handler.debug(
                f"  (Native) GBIF: {species_name}:",
                f"directly native to reference geography {native_range}",
            )
            return "Native"
        if geo_contains(ref_geo, native_range, handler) or geo_contains(native_range, ref_geo, handler):
            handler.debug(
                f"  (Native) GBIF: {species_name}:",
                f"reference geography {ref_geo} <=> native range {native_range}",
            )
            return "Native"
        handler.debug(
            f"  (Mismatch) GBIF: {species_name} (attempt {check}):",
            f"reference geography {ref_geo} <!=> native range {native_range}",
        )
        check += 1

    # if it hasn't found a reason to call it native
    species_status = "(Unknown)" if cryptogenic else "(Introduced)"
    if cryptogenic:
        status_description = "cryptogenic"
    else:
        status_description = f"not native to reference geography {ref_geo}"

    handler.debug(
        f"  {species_status} GBIF: {species_name}: species is {status_description}"
    )
    return "Introduced" if not cryptogenic else None
