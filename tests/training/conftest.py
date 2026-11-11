import re
from dataclasses import dataclass, field

import pytest

from lexguard.crypto import KeyRing
from lexguard.documents import DocumentStore
from lexguard.domain import AccessRequest, Decision, PiiKind, PiiSpan, Sensitivity, Zone
from lexguard.guard import Guard
from lexguard.storage import ZonedStore
from lexguard.synthetic import generate
from lexguard.testing import AllowAll, LabelDeidentifier, MemoryAuditLog
from lexguard.training.dataset import TrainingSets
from lexguard.training.manifest import ManifestRegistry

_RUT = re.compile(r"\b\d{1,2}\.\d{3}\.\d{3}-[\dkK]")


@dataclass
class Scripted:
    """Policy that denies the listed (matter, sensitivity) compartments and records every ask."""

    denied: frozenset[tuple[str, Sensitivity]] = frozenset()
    asked: list[AccessRequest] = field(default_factory=list)

    def decide(self, request: AccessRequest) -> Decision:
        self.asked.append(request)
        compartment = (request.resource.matter_id, request.resource.sensitivity)
        return Decision(compartment not in self.denied, "scripted")


@pytest.fixture
def scripted():
    return Scripted


@pytest.fixture(scope="session")
def corpus():
    return generate()


@pytest.fixture(scope="session")
def pseudonymized(corpus):
    deidentifier = LabelDeidentifier(corpus)
    return [deidentifier.pseudonymize(labeled.document) for labeled in corpus.documents]


@pytest.fixture
def keys(corpus):
    ring = KeyRing.generate()
    for matter in corpus.matters:
        ring.enroll(matter.id)
    return ring


@pytest.fixture
def store(tmp_path, keys, pseudonymized):
    """A lake whose de-identified zone holds the pseudonymized corpus."""
    zoned = ZonedStore(tmp_path / "lake", keys)
    documents = DocumentStore(zoned)
    for document in pseudonymized:
        documents.save(Zone.DEIDENTIFIED, document)
    return zoned


@pytest.fixture
def audit():
    return MemoryAuditLog()


@pytest.fixture
def manifests(tmp_path):
    return ManifestRegistry(tmp_path / "manifests")


@pytest.fixture
def scan():
    """Residual-PII scanner that recognizes a RUT left in the clear."""

    def _scan(text):
        return [PiiSpan(m.start(), m.end(), PiiKind.RUT, m.group()) for m in _RUT.finditer(text)]

    return _scan


@pytest.fixture
def training_sets(store, manifests, audit, scan):
    def _training_sets(policy=None, **options):
        return TrainingSets(store, manifests, Guard(policy or AllowAll(), audit), scan, **options)

    return _training_sets
