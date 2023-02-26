from .get_ref_hierarchy import get_ref_hierarchy
from ..interfaces.handler import Handler


def geo_contains(ref_geo: str, geo: str, handler: Handler) -> bool:
    """Check if a given reference geography contains another geography."""
    # get the actual hierarchy
    hierarchy = get_ref_hierarchy(ref_geo)
    if hierarchy is None:
        # raise TypeError(
        handler.warning(
            f"geographic region <{ref_geo}> not recognized.",
            "Please register an issue regarding this region name",
            "at https://github.com/tokebe/niclassify",
        )
        return False
    if hierarchy["Contains"] is None:
        return False

    def match_geo(level, ref):
        if level is None:
            return False
        result = False

        for name, sub in level.items():
            if name == ref:
                result = True
                break
            elif not result and sub["Contains"] is not None:
                result = match_geo(sub["Contains"], ref)

        return result

    return match_geo(hierarchy["Contains"], geo)
