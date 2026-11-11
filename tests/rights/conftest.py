from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from operator import attrgetter
from types import SimpleNamespace

import pytest

from lexguard import tokens
from lexguard.crypto import KeyRing
from lexguard.documents import DocumentStore
from lexguard.domain import (
    AccessRequest,
    Decision,
    Document,
    DocumentKind,
    PiiKind,
    Sensitivity,
    Zone,
)
from lexguard.guard import Guard
from lexguard.rights.subjects import Identifier, SubjectRequests
from lexguard.storage import ZonedStore
from lexguard.synthetic import LabeledDocument, Pii, generate, render
from lexguard.testing import AllowAll, MemoryAuditLog
from lexguard.training.dataset import TrainingSets
from lexguard.training.manifest import ManifestRegistry

PEOPLE = SimpleNamespace(rut="12.345.678-5", name="Ana Fuentes Lagos", bystander="Pedro Vidal")


class MemoryVault:
    """In-memory `TokenVault` keyed per matter, so its tokens die with the matter's key.

    Forgetting drops the mapping and keeps it from coming back.
    """

    def __init__(self, keys: KeyRing) -> None:
        self._keys = keys
        self._values: dict[tuple[str, str], str] = {}
        self._forgotten: set[tuple[str, str]] = set()

    def token_for(self, kind: PiiKind, value: str, matter_id: str) -> str:
        token = tokens.pseudonym(kind, value, self._keys.key_for(matter_id, "pseudonym"))
        if (matter_id, token) not in self._forgotten:
            self._values[matter_id, token] = value
        return token

    def reveal(self, token: str, matter_id: str) -> str:
        return self._values[matter_id, token]

    def forget(self, token: str, matter_id: str) -> None:
        self._values.pop((matter_id, token), None)
        self._forgotten.add((matter_id, token))


@dataclass
class Scripted:
    """Policy that denies the listed (matter, sensitivity) compartments and records every ask."""

    denied: frozenset[tuple[str, Sensitivity]] = frozenset()
    asked: list[AccessRequest] = field(default_factory=list)

    def decide(self, request: AccessRequest) -> Decision:
        self.asked.append(request)
        compartment = (request.resource.matter_id, request.resource.sensitivity)
        return Decision(compartment not in self.denied, "scripted")


def _letter(matter_id: str) -> LabeledDocument:
    """A document about the same person, so that one subject appears in two matters."""
    text, spans = render(
        "Carta de {subject}, RUT {rut}, dirigida a {bystander}. {subject} pide una reunión.",
        {
            "subject": Pii(PiiKind.PERSON, PEOPLE.name),
            "rut": Pii(PiiKind.RUT, PEOPLE.rut),
            "bystander": Pii(PiiKind.PERSON, PEOPLE.bystander),
        },
    )
    document = Document(
        id=f"{matter_id}-L001",
        matter_id=matter_id,
        kind=DocumentKind.EMAIL,
        sensitivity=Sensitivity.CONFIDENTIAL,
        text=text,
        created_at=datetime(2025, 3, 1, tzinfo=UTC),
    )
    return LabeledDocument(document, spans)


def _pseudonymize(labeled: LabeledDocument, vault: MemoryVault) -> Document:
    document, text = labeled.document, labeled.document.text
    for span in sorted(labeled.spans, key=attrgetter("start"), reverse=True):
        token = vault.token_for(span.kind, span.text, document.matter_id)
        text = text[: span.start] + token + text[span.end :]
    return replace(document, text=text)


@pytest.fixture(scope="session")
def corpus():
    return generate()


@pytest.fixture
def people():
    return PEOPLE


@pytest.fixture
def subject():
    return [Identifier(PiiKind.RUT, PEOPLE.rut)]


@pytest.fixture
def keys(corpus):
    ring = KeyRing.generate()
    for matter in corpus.matters:
        ring.enroll(matter.id)
    return ring


@pytest.fixture
def vault(keys):
    return MemoryVault(keys)


@pytest.fixture
def store(tmp_path, keys, corpus, vault):
    """A lake with the corpus and one letter about the subject in M-001 and in M-002."""
    zoned = ZonedStore(tmp_path / "lake", keys)
    documents = DocumentStore(zoned)
    for labeled in (*corpus.documents, _letter("M-001"), _letter("M-002")):
        documents.save(Zone.RAW, labeled.document)
        documents.save(Zone.DEIDENTIFIED, _pseudonymize(labeled, vault))
    return zoned


@pytest.fixture
def manifests(tmp_path):
    return ManifestRegistry(tmp_path / "manifests")


@pytest.fixture
def audit():
    return MemoryAuditLog()


@pytest.fixture
def scripted():
    return Scripted


@pytest.fixture
def requests(store, vault, manifests, audit):
    def _requests(policy=None):
        return SubjectRequests(store, vault, Guard(policy or AllowAll(), audit), manifests)

    return _requests


@pytest.fixture
def training_sets(store, manifests):
    return TrainingSets(store, manifests, Guard(AllowAll(), MemoryAuditLog()), lambda text: ())
