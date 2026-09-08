from .base import Source
from .homerun import HomeRun
from .ikea_in import IkeaIndia

SOURCES: dict[str, type[Source]] = {
    HomeRun.name: HomeRun,
    IkeaIndia.name: IkeaIndia,
}

__all__ = ["Source", "HomeRun", "IkeaIndia", "SOURCES"]
