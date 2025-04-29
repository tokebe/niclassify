from pathlib import Path
import json

with open(Path(__file__).parent / "../../config/regions.json") as regions_file:
    REGIONS = json.load(regions_file)

def get_ref_hierarchy(ref_geo: str) -> dict:
    """Get a hierarchy contained in a given reference geography."""
    def find_geo(level, ref):
        if level is None:
            return None
        result = None

        for name, sub in level.items():
            if name == ref:
                result = sub
                break
            elif result is None:
                result = find_geo(sub["Contains"], ref)

        return result

    return find_geo(REGIONS, ref_geo)
