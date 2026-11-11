from collections import Counter
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from enum import StrEnum

import numpy as np

from lexguard.analytics.budget import PrivacyLedger
from lexguard.documents import DocumentStore
from lexguard.domain import (
    AccessRequest,
    Action,
    Document,
    Matter,
    Principal,
    Purpose,
    Resource,
    Sensitivity,
    Zone,
)
from lexguard.errors import AccessDenied
from lexguard.guard import Guard
from lexguard.settings import Settings
from lexguard.storage import ZonedStore

COUNT_SENSITIVITY = 1.0


class Dimension(StrEnum):
    KIND = "kind"
    MATTER = "matter"
    MONTH = "month"


_GROUP: dict[Dimension, Callable[[Document], str]] = {
    Dimension.KIND: lambda document: document.kind.value,
    Dimension.MATTER: lambda document: document.matter_id,
    Dimension.MONTH: lambda document: f"{document.created_at:%Y-%m}",
}


@dataclass(frozen=True, slots=True)
class Release:
    dimension: Dimension
    counts: Mapping[str, int]
    suppressed: int
    epsilon: float
    noise_scale: float


@dataclass(slots=True)
class Aggregator:
    """Noisy document counts over the metadata of the de-identified zone.

    The privacy unit is the document: each one falls in exactly one group, so the L1
    sensitivity of a count query is 1 and the Laplace scale is 1 / epsilon. Groups are
    suppressed on their true size before noise is added, so epsilon covers the released
    counts, not the fact that a group reached the minimum size.
    """

    store: ZonedStore
    guard: Guard
    settings: Settings = field(default_factory=Settings)
    rng: np.random.Generator = field(default_factory=np.random.default_rng)
    ledger: PrivacyLedger = field(init=False)

    def __post_init__(self) -> None:
        self.ledger = PrivacyLedger(self.settings.epsilon)

    def count_by(
        self, principal: Principal, matters: Iterable[Matter], dimension: Dimension, epsilon: float
    ) -> Release:
        compartments: set[tuple[str, Sensitivity]] = set()
        for matter in matters:
            if Purpose.ANALYTICS not in matter.purposes:
                continue
            for sensitivity in Sensitivity:
                request = _request(principal, matter.id, sensitivity)
                if self.guard.evaluate(request, dimension=dimension).allowed:
                    compartments.add((matter.id, sensitivity))
        if not compartments:
            raise AccessDenied("no compartment admits this aggregate")
        self.ledger.charge(principal.id, f"count by {dimension}", epsilon)

        documents = DocumentStore(self.store)
        group = _GROUP[dimension]
        sizes = Counter(
            group(document)
            for matter_id in sorted({matter_id for matter_id, _ in compartments})
            for document in documents.scan(Zone.DEIDENTIFIED, matter_id)
            if (matter_id, document.sensitivity) in compartments
        )
        scale = COUNT_SENSITIVITY / epsilon
        counts = {
            key: max(0, round(size + float(self.rng.laplace(0.0, scale))))
            for key, size in sorted(sizes.items())
            if size >= self.settings.min_group_size
        }
        return Release(dimension, counts, len(sizes) - len(counts), epsilon, scale)


def _request(principal: Principal, matter_id: str, sensitivity: Sensitivity) -> AccessRequest:
    resource = Resource(Zone.DEIDENTIFIED, matter_id, sensitivity)
    return AccessRequest(principal, Action.AGGREGATE, Purpose.ANALYTICS, resource)
