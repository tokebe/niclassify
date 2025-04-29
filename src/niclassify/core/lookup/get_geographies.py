from pathlib import Path
import json
from natsort import natsorted, ns
from typing import List

REGIONS = None

with open(Path(__file__).parent / "../../config/regions.json") as regions_file:
    REGIONS = json.load(regions_file)


def get_geographies() -> List[str]:
    """Return a list of all geographies in regions config file."""

    def getlist(section):
        items = []
        for name, sub in section.items():
            items.append(name)
            if sub["Contains"] is not None:
                items.extend(getlist(sub["Contains"]))
        return items

    return natsorted(getlist(REGIONS), alg=ns.IGNORECASE)
