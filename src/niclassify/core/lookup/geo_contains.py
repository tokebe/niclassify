from niclassify.config.regions import REGIONS, REGIONS_FLAT, Region
from niclassify.core.interfaces.handler import Handler


def geo_contains(ref_geo: str, geo: str, handler: Handler) -> bool:
    """Check if a given reference geography contains another geography."""
    if geo not in REGIONS_FLAT:
        handler.warning(
            f"geographic region <{geo}> not recognized.",
            "Please register an issue regarding this region name",
            "at https://github.com/tokebe/niclassify",
        )

    def traverse(region: Region) -> bool:
        return geo in region.children or any(
            traverse(REGIONS[child]) for child in region.children
        )

    return ref_geo == geo or traverse(REGIONS[ref_geo])
