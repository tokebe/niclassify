from pathlib import Path
from typing import NamedTuple

import yaml
from pydantic import TypeAdapter

with (Path(__file__).parent / "../../../config/regions.yaml").open() as regions_file:
    REGIONS_FLAT = TypeAdapter(dict[str, list[str]]).validate_python(
        yaml.safe_load(regions_file)
    )


class Region(NamedTuple):
    """A region which provides parents and children for easy traversal."""

    parents: set[str]
    children: set[str]


REGIONS = dict[str, Region]()

for region, children in REGIONS_FLAT.items():
    if region not in REGIONS:
        REGIONS[region] = Region(parents=set(), children=set(children))
    for child in children:
        if child not in REGIONS:
            REGIONS[child] = Region(parents=set(), children=set())
        REGIONS[child].parents.add(region)
