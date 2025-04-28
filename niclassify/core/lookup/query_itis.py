from niclassify.core.interfaces.handler import Handler
from typing import Optional, cast
from xml.etree import ElementTree
from throttler import throttle
from niclassify.core.lookup.get_ref_hierarchy import get_ref_hierarchy
from niclassify.core.lookup.geo_contains import geo_contains
from time import sleep
from backoff import on_exception, expo
from ratelimit import limits, RateLimitException
import httpx

client = httpx.Client(transport=httpx.HTTPTransport(retries=3), timeout=60, follow_redirects=True)

@on_exception(expo, RateLimitException)
@limits(calls=60, period=60)
def query_itis(species_name: str, geography: str, handler: Handler) -> Optional[str]:
    tsn_url = "http://www.itis.gov/ITISWebService/services/ITISService/\
getITISTermsFromScientificName?srchKey="
    jurisdiction_url = "http://www.itis.gov/ITISWebService/services/ITISService/\
getJurisdictionalOriginFromTSN?tsn="

    try:
        # get TSN
        request = f"{tsn_url}{species_name.replace(' ', '%20')}"

        response = client.get(request)
        response.raise_for_status()
        # get xml tree from response
        result_tree = ElementTree.fromstring(response.content)
        # get any TSN's
        matched_TSNs = [
            i.text
            for i in result_tree.iter(
                "{http://data.itis_service.itis.usgs.gov/xsd}tsn"
            )
        ]

        if matched_TSNs is None:  # skip if there's no tsn to be found
            handler.debug(f"  (Unknown) ITIS: {species_name}: no data")
            return None

        elif (
            len(matched_TSNs) != 1
        ):  # skip if tsn is empty or there are more than one
            handler.debug(f"  (Unknown) ITIS: {species_name}: no data")
            return None

        tsn = matched_TSNs[0]  # tsn captured

        # get jurisdiction
        request = f"{jurisdiction_url}{tsn}"
        response = client.get(request)
        response.raise_for_status()

    except httpx.HTTPError as error:
        handler.debug(str(error))
        handler.debug(f"ITIS query failed. See error above.")
        return


    try:
        result_tree = ElementTree.fromstring(response.content)
    except UnicodeDecodeError:
        return None

    jurisdictions = cast(dict[str, str], {
        j.text: n.text
        for j, n in zip(
            result_tree.iter(
                "{http://data.itis_service.itis.usgs.gov/xsd}jurisdictionValue"
            ),
            result_tree.iter("{http://data.itis_service.itis.usgs.gov/xsd}origin"),
        )
    })

    if len(jurisdictions) == 0:  # or if it's somehow returned empty
        handler.debug(f"  (Unknown) ITIS: {species_name}: no data")
        return None

    for jurisdiction, status in jurisdictions.items():
        # simple first: check if jurisdiction is just current reference geo
        if jurisdiction == geography:
            handler.debug(
                f"  ({status}) ITIS: {species_name}:",
                f"directly {status.lower()} to reference geography {geography}",
            )
            return status if status != "Native&Introduced" else None
        # otherwise if one contains the other it's native
        if geo_contains(geography, jurisdiction, handler) or geo_contains(
            jurisdiction, geography, handler
        ):
            handler.debug(
                f"  ({status}) ITIS: {species_name}:",
                "reference geography",
                f"{geography} <=> {status.lower()} range {jurisdiction}",
            )
            return status if status != "Native&Introduced" else None

    # if it hasn't found a reason to call it native
    return None
