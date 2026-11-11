from dataclasses import dataclass, field

import pytest

from lexguard.analytics.aggregates import Aggregator
from lexguard.crypto import KeyRing
from lexguard.documents import DocumentStore
from lexguard.domain import Zone
from lexguard.guard import Guard
from lexguard.settings import Settings
from lexguard.storage import ZonedStore
from lexguard.synthetic import generate
from lexguard.testing import AllowAll, LabelDeidentifier, MemoryAuditLog


@dataclass
class FixedNoise:
    """Stands in for the random generator: always draws `value` and records the scales asked."""

    value: float = 0.0
    scales: list[float] = field(default_factory=list)

    def laplace(self, loc: float, scale: float) -> float:
        self.scales.append(scale)
        return loc + self.value


@pytest.fixture(scope="session")
def corpus():
    return generate()


@pytest.fixture
def store(tmp_path, corpus):
    """A lake whose de-identified zone holds the pseudonymized corpus."""
    keys = KeyRing.generate()
    zoned = ZonedStore(tmp_path, keys)
    documents = DocumentStore(zoned)
    deidentifier = LabelDeidentifier(corpus)
    for labeled in corpus.documents:
        keys.enroll(labeled.document.matter_id)
        documents.save(Zone.DEIDENTIFIED, deidentifier.pseudonymize(labeled.document))
    return zoned


@pytest.fixture
def audit():
    return MemoryAuditLog()


@pytest.fixture
def noise():
    return FixedNoise()


@pytest.fixture
def aggregator(store, audit, noise):
    def _aggregator(policy=None, **settings):
        return Aggregator(store, Guard(policy or AllowAll(), audit), Settings(**settings), noise)

    return _aggregator
