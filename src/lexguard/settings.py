import os
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import Self


class ModelLocation(StrEnum):
    LOCAL = "local"
    PRIVATE_CLOUD = "private_cloud"
    EXTERNAL = "external"


class Boundary(StrEnum):
    """How far data may travel. The strictest mode is the default."""

    LOCAL_ONLY = "local_only"
    PRIVATE_CLOUD = "private_cloud"
    DEIDENTIFIED_EGRESS = "deidentified_egress"

    def admits(self, location: ModelLocation) -> bool:
        return location in _ADMITTED[self]


_ADMITTED = {
    Boundary.LOCAL_ONLY: {ModelLocation.LOCAL},
    Boundary.PRIVATE_CLOUD: {ModelLocation.LOCAL, ModelLocation.PRIVATE_CLOUD},
    Boundary.DEIDENTIFIED_EGRESS: set(ModelLocation),
}


@dataclass(frozen=True, slots=True)
class Settings:
    boundary: Boundary = Boundary.LOCAL_ONLY
    min_group_size: int = 5
    epsilon: float = 1.0

    @classmethod
    def from_env(cls, env: Mapping[str, str] = os.environ) -> Self:
        defaults = cls()
        return cls(
            boundary=Boundary(env.get("LEXGUARD_BOUNDARY", defaults.boundary)),
            min_group_size=int(env.get("LEXGUARD_MIN_GROUP_SIZE", defaults.min_group_size)),
            epsilon=float(env.get("LEXGUARD_EPSILON", defaults.epsilon)),
        )
