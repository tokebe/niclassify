from typing import ClassVar

from pydantic import BaseModel
from pydantic_settings import BaseSettings, SettingsConfigDict


class BOLDAPISettings(BaseModel):
    """Settings concerning interface with the BOLD API."""

    host: str = "https://v4.boldsystems.org/index.php"
    taxon_levels: list[str] = [
        "subspecies_name",
        "species_name",
        "subgenus_name",
        "genus_name",
        "tribe_name",
        "subfamily_name",
        "family_name",
        "order_name",
        "class_name",
        "phylum_name",
    ]
    rate_limit: int = 500


class GBIFAPISettings(BaseModel):
    """Settings concerning interface with the GBIF API."""

    host: str = "http://api.gbif.org/v1"
    rate_limit: int = 60


class ITISAPISettings(BaseModel):
    """Settings concerning interface with the ITIS API."""

    host: str = "http://www.itis.gov/ITISWebService/services/ITISService"
    rate_limit: int = 60


class APISettings(BaseModel):
    """Settings concerning various APIs."""

    bold: BOLDAPISettings = BOLDAPISettings()
    gbif: GBIFAPISettings = GBIFAPISettings()
    itis: ITISAPISettings = ITISAPISettings()


class GeneralConfig(BaseSettings):
    """General configuration for NIClassify behavior."""

    apis: APISettings = APISettings()

    model_config: ClassVar[SettingsConfigDict] = SettingsConfigDict(
        case_sensitive=False,
        env_nested_delimiter="__",
        env_file=".env",
        env_file_encoding="utf-8",
        yaml_file="config/config.yaml",
        yaml_file_encoding="utf-8",
    )


CONFIG = GeneralConfig()
