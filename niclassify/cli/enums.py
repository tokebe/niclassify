from enum import Enum


class TaxonomicHierarchy(str, Enum):
    none = "none"
    phylum = "phylum"
    _class = "class"
    order = "order"
    family = "family"
    subfamily = "subfamily"
    genus = "genus"

class Methods(str, Enum):
    bPTP = "bPTP"
    GMYC = "GMYC"
