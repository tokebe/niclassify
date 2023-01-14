from ..interfaces.handler import Handler
from typing import Optional
import requests
from xml.etree import ElementTree
from ratelimiter import RateLimiter
from .get_ref_hierarchy import get_ref_hierarchy
from .geo_contains import geo_contains


@RateLimiter(max_calls=10, period=1)
def query_itis(species_name: str, geography: str, handler: Handler) -> Optional[str]:

    tsn_url = "http://www.itis.gov/ITISWebService/services/ITISService/\
getITISTermsFromScientificName?srchKey="
    jurisdiction_url = "http://www.itis.gov/ITISWebService/services/ITISService/\
getJurisdictionalOriginFromTSN?tsn="

    # get TSN
    req = f"{tsn_url}{species_name.replace(' ', '%20')}"

    response = requests.get(req)
    response.raise_for_status()
    # get xml tree from response
    tree = ElementTree.fromstring(response.content)
    # get any TSN's
    vals = [
        i.text for i in tree.iter("{http://data.itis_service.itis.usgs.gov/xsd}tsn")
    ]

    if vals is None:  # skip if there's no tsn to be found
        handler.debug(f"  (Unknown) ITIS: {species_name}: no data")
        return None

    elif len(vals) != 1:  # skip if tsn is empty or there are more than one
        handler.debug(f"  (Unknown) ITIS: {species_name}: no data")
        return None

    tsn = vals[0]  # tsn captured

    # get jurisdiction
    req = f"{jurisdiction_url}{tsn}"

    response = requests.get(req)

    response.raise_for_status()

    try:
        tree = ElementTree.fromstring(response.content)
    except UnicodeDecodeError:
        return None

    jurisdictions = {
        j.text: n.text
        for j, n in zip(
            tree.iter("{http://data.itis_service.itis.usgs.gov/xsd}jurisdictionValue"),
            tree.iter("{http://data.itis_service.itis.usgs.gov/xsd}origin"),
        )
    }

    if len(jurisdictions) == 0:  # or if it's somehow returned empty
        handler.debug(f"  (Unknown) ITIS: {species_name}: no data")
        return None

    for jurisdiction, status in jurisdictions.items():
        # simple first: check if jurisdiction is just current reference geo
        if jurisdiction == geography:
            handler.debug(
                f"  (Native) ITIS: {species_name}: directly native to reference geography {geography}"
            )
            return status if status != "Native&Introduced" else None
        # otherwise if one contains the other it's native
        if geo_contains(geography, jurisdiction) or geo_contains(
            jurisdiction, geography
        ):
            handler.debug(
                f"  (Native) ITIS: {species_name}: reference geography {geography} <=> native range {jurisdiction}"
            )
            return status if status != "Native&Introduced" else None
        handler.debug(
            f"  (Introduced) ITIS: {species_name}: reference geography {geography} <!=> native range {jurisdiction}"
        )

    # if it hasn't found a reason to call it native
    return None
