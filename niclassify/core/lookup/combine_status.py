from ..interfaces.handler import Handler


def combine_status(status_gbif: str, status_itis: str) -> str:
    """Combine given GBIF and ITIS statuses."""
    if status_itis == status_gbif:
        return status_itis
    if status_itis is None and status_gbif is not None:
        return status_gbif
    if status_gbif is None and status_itis is not None:
        return status_itis
    if (status_itis != status_gbif) and (
        status_itis is not None and status_gbif is not None
    ):
        return "Unknown"  # unknown strictly means conflicting answers
    return None
