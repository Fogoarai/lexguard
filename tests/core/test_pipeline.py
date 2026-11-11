import pytest

from lexguard import tokens
from lexguard.crypto import KeyRing
from lexguard.documents import DocumentStore
from lexguard.domain import Action, Zone
from lexguard.errors import AccessDenied
from lexguard.guard import Guard
from lexguard.pipeline import Pipeline
from lexguard.storage import ZonedStore
from lexguard.synthetic import generate
from lexguard.testing import AllowAll, DenyAll, LabelDeidentifier, MembershipPolicy, MemoryAuditLog


@pytest.fixture(scope="module")
def corpus():
    return generate(matters=2, documents_per_matter=4)


@pytest.fixture
def audit():
    return MemoryAuditLog()


@pytest.fixture
def build(tmp_path, corpus, audit):
    def _build(policy):
        keys = KeyRing.generate()
        documents = DocumentStore(ZonedStore(tmp_path, keys))
        return Pipeline(Guard(policy, audit), keys, documents, LabelDeidentifier(corpus))

    return _build


def test_land_stores_admitted_documents_in_the_raw_zone(build, corpus, audit):
    pipeline = build(AllowAll())
    batch = [labeled.document for labeled in corpus.documents]

    landing = pipeline.land(corpus.principal("svc-pipeline"), batch)

    assert landing.accepted == tuple(document.id for document in batch)
    assert not landing.refused
    assert len(list(pipeline.documents.scan(Zone.RAW, "M-001"))) == 4
    assert {event.action for event in audit.events} == {Action.INGEST}


def test_land_leaves_refused_documents_in_the_lake(build, corpus):
    pipeline = build(MembershipPolicy())
    batch = [labeled.document for labeled in corpus.documents]

    landing = pipeline.land(corpus.principal("lawyer-2"), batch)

    assert all(document_id.startswith("M-002") for document_id in landing.accepted)
    assert set(landing.refused.values()) == {"not assigned"}
    assert "M-001" not in pipeline.keys


def test_promote_pseudonymizes_and_empties_the_raw_zone(build, corpus):
    pipeline = build(AllowAll())
    service = corpus.principal("svc-pipeline")
    pipeline.land(service, [labeled.document for labeled in corpus.documents])

    promoted = pipeline.promote(service, "M-001")

    assert len(promoted) == 4
    assert list(pipeline.documents.scan(Zone.RAW, "M-001")) == []
    labels = {labeled.document.id: labeled.spans for labeled in corpus.documents}
    for document in pipeline.documents.scan(Zone.DEIDENTIFIED, "M-001"):
        assert list(tokens.find(document.text))
        assert not any(span.text in document.text for span in labels[document.id])


def test_promote_can_retain_raw(build, corpus):
    pipeline = build(AllowAll())
    service = corpus.principal("svc-pipeline")
    pipeline.land(service, [labeled.document for labeled in corpus.documents])

    pipeline.promote(service, "M-001", retain_raw=True)

    assert len(list(pipeline.documents.scan(Zone.RAW, "M-001"))) == 4


def test_promote_requires_permission_to_read_raw(build, corpus):
    allowed = build(AllowAll())
    allowed.land(corpus.principal("svc-pipeline"), [corpus.documents[0].document])
    denied = Pipeline(
        Guard(DenyAll(), MemoryAuditLog()), allowed.keys, allowed.documents, allowed.deidentifier
    )

    with pytest.raises(AccessDenied):
        denied.promote(corpus.principal("scientist-1"), "M-001")
    assert list(allowed.documents.scan(Zone.DEIDENTIFIED, "M-001")) == []
