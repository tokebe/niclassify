from typing import cast
from xml.etree import ElementTree

import httpx
from backoff import expo, on_exception
from ratelimit import RateLimitException, limits

from niclassify.config.general import CONFIG
from niclassify.core.interfaces.handler import Handler
from niclassify.core.lookup.geo_contains import geo_contains

client = httpx.Client(
    transport=httpx.HTTPTransport(retries=3), timeout=60, follow_redirects=True
)

# TODO: handle `cosmopolitan`

@on_exception(expo, RateLimitException)
@limits(calls=CONFIG.apis.itis.rate_limit, period=60)
def query_itis(species_name: str, geography: str, handler: Handler) -> str | None:
    """Query ITIS to determine if a species is native or introduced to a given reference geography."""
    tsn_url = f"{CONFIG.apis.itis.host}/getITISTermsFromScientificName?srchKey="
    jurisdiction_url = f"{CONFIG.apis.itis.host}/getJurisdictionalOriginFromTSN?tsn="

    try:
        # get TSN
        request = f"{tsn_url}{species_name.replace(' ', '%20')}"

        response = client.get(request)
        response.raise_for_status()
        # get xml tree from response
        result_tree = ElementTree.fromstring(response.content)
        # get any TSN's
        matched_tsns = [
            i.text
            for i in result_tree.iter("{http://data.itis_service.itis.usgs.gov/xsd}tsn")
        ]

        if len(matched_tsns) != 1:  # skip if there's no tsn to be found
            handler.debug(f"  {species_name}: ITIS: (Unknown) no data")
            return None

        tsn = matched_tsns[0]  # tsn captured

        # get jurisdiction
        request = f"{jurisdiction_url}{tsn}"
        response = client.get(request)
        response.raise_for_status()

    except httpx.HTTPError as error:
        handler.debug(str(error))
        handler.debug("  ITIS query failed. See error above.")
        return

    try:
        result_tree = ElementTree.fromstring(response.content)
    except UnicodeDecodeError:
        return None

    jurisdictions = cast(
        dict[str, str],
        {
            j.text: n.text
            for j, n in zip(
                result_tree.iter(
                    "{http://data.itis_service.itis.usgs.gov/xsd}jurisdictionValue"
                ),
                result_tree.iter("{http://data.itis_service.itis.usgs.gov/xsd}origin"),
                strict=False,
            )
        },
    )

    return determine_status(species_name, geography, jurisdictions, handler)


def determine_status(
    species_name: str, geography: str, jurisdictions: dict[str, str], handler: Handler
) -> str | None:
    """Given a reference geography and set of known jurisdictions, return whether the species' status."""
    if len(jurisdictions) == 0:  # or if it's somehow returned empty
        handler.debug(f"  {species_name}: ITIS: (Unknown) no data")
        return None

    for jurisdiction, status in jurisdictions.items():
        # If the jurisdiction matches the reference, the status is relevant
        if geo_contains(geography, jurisdiction, handler) or geo_contains(
            jurisdiction, geography, handler
        ):
            handler.log(
                f"  {species_name} ITIS: ({status}):",
                "reference geography",
                f"{geography} <=> {status.lower()} range {jurisdiction}",
            )
            return status if status != "Native&Introduced" else None

    # if it hasn't found a reason to call it native
    return None
