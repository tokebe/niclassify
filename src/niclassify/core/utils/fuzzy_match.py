from typing import List

def indexes(sample: str, target: str) -> List[int]:
    """Get a list of indexes of target that sample matches in discontinuous order."""
    matches = []
    start = 0
    for s in sample:
        try:
            start = target[start:].index(s)
            matches.append(start)
        except ValueError:
            break
    return matches

def score(sample: str, target: str) -> float:
    """Return a score based on discontinuous ordered matches from 0 to 1."""
    matches = indexes(sample, target)
    return len(matches) / len(target)
