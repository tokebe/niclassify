import re

import httpx
from backoff import expo, on_exception
from ratelimit import RateLimitException, limits

from niclassify.config.general import CONFIG
from niclassify.core.interfaces.handler import Handler
from niclassify.core.lookup.geo_contains import geo_contains

client = httpx.Client(
    transport=httpx.HTTPTransport(retries=3), timeout=60, follow_redirects=True
)


@on_exception(expo, RateLimitException)
@limits(calls=CONFIG.apis.gbif.rate_limit, period=60)
def query_gbif(species_name: str, ref_geo: str, handler: Handler) -> str | None:
    """Query GBIF to determine if a species is native or introduced to a reference geography."""
    taxon_key_url = f"{CONFIG.apis.gbif.host}/species?name="
    records_url = f"{CONFIG.apis.gbif.host}/species/"

    try:
        # get taxonKey
        request = f"{taxon_key_url}{species_name.lower().replace(' ', '%20')}"
        response = client.get(request)
        response.raise_for_status()

        # search for "taxonID":"gbif:" with some numbers, getting the numbers
        taxon_key = re.search('(?<="taxonID":"gbif:)\\d+', response.text)

        if taxon_key is None:
            handler.debug(f"  {species_name}: GBIF: (Unknown)  No data")
            return None
        taxon_key = taxon_key.group()

        # get native range
        request = f"{records_url}{taxon_key}/descriptions"
        response = client.get(request)
        response.raise_for_status()

    except httpx.HTTPError as error:
        handler.debug(str(error))
        handler.debug("  GBIF lookup failed. See error above.")
        return None

    try:
        results = response.json()
    except UnicodeDecodeError:
        handler.debug(f"  {species_name}: GBIF: (Unknown)  No data")
        return None

    native_ranges = [
        res["description"]
        for res in results["results"]
        if res["type"] == "native range"
    ]

    return determine_status(species_name, ref_geo, native_ranges, handler)


def determine_status(
    species_name: str, ref_geo: str, native_ranges: list[str], handler: Handler
) -> str | None:
    """Given a reference geography and set of known native ranges, return whether the species' status."""
    if len(native_ranges) == 0:
        handler.debug(f"  {species_name}: GBIF: (Unknown) No data")
        return None

    cryptogenic = False

    check = 1

    for native_range in native_ranges:
        if native_range == "Cosmopolitan, Cryptogenic":
            handler.log(f"  {species_name}: GBIF: (Unknown) Cryptogenic")
            cryptogenic = True
            continue
        if native_range == "Pantropical, Circumtropical":
            if geo_contains("Pantropics", ref_geo, handler):
                handler.log(f"  {species_name}: GBIF: (Native) Pantropical")
                return "Native"
            handler.log(f"  {species_name}: GBIF: (Introduced) Pantropical")
            continue
        if native_range == "Subtropics":
            if geo_contains("Subtropics", ref_geo, handler):
                handler.log(f"  {species_name}: GBIF: (Native) Subtropical")
                return "Native"
            handler.log(f"  {species_name}: GBIF: (Introduced) Subtropical")
            continue

        if native_range == ref_geo:
            handler.log(
                f"  {species_name}: GBIF: (Native) ",
                f"directly native to reference geography {native_range}",
            )
            return "Native"
        if geo_contains(ref_geo, native_range, handler) or geo_contains(
            native_range, ref_geo, handler
        ):
            handler.log(
                f"  {species_name}: GBIF: (Native) ",
                f"reference geography {ref_geo} <=> native range {native_range}",
            )
            return "Native"
        handler.log(
            f"  {species_name}: GBIF: (Mismatch)  (attempt {check}):",
            f"reference geography {ref_geo} <!=> native range {native_range}",
        )
        check += 1

    # if it hasn't found a reason to call it native
    species_status = "(Unknown)" if cryptogenic else "(Introduced)"
    if cryptogenic:
        status_description = "cryptogenic"
    else:
        status_description = f"introduced to reference geography {ref_geo}"

    handler.log(
        f"  {species_name} GBIF: {species_status}: species is {status_description}"
    )
    return "Introduced" if not cryptogenic else None
