from enum import Enum


class TaxonomicHierarchy(str, Enum):
    none = "none"
    phylum = "phylum"
    _class = "class"
    order = "order"
    family = "family"
    subfamily = "subfamily"
    genus = "genus"

    def __str__(self):
        return self.name


class Methods(str, Enum):
    bPTP = "bPTP"
    GMYC = "GMYC"

    def __str__(self):
        return self.name
