import pytest

from lexguard.crypto import KeyRing
from lexguard.documents import DocumentStore, decode, encode
from lexguard.domain import Zone
from lexguard.storage import ZonedStore
from lexguard.synthetic import generate


@pytest.fixture(scope="module")
def corpus():
    return generate(matters=2, documents_per_matter=4)


@pytest.fixture
def store(tmp_path, corpus):
    keys = KeyRing.generate()
    for matter in corpus.matters:
        keys.enroll(matter.id)
    return ZonedStore(tmp_path, keys)


def test_codec_round_trips_every_document(corpus):
    for labeled in corpus.documents:
        assert decode(encode(labeled.document)) == labeled.document


def test_scan_returns_only_documents_of_that_zone_and_matter(store, corpus):
    documents = DocumentStore(store)
    for labeled in corpus.documents:
        documents.save(Zone.RAW, labeled.document)
    store.put(Zone.RAW, "M-001", "other.bin", b"not a document")

    expected = [d.document for d in corpus.documents if d.document.matter_id == "M-001"]
    assert list(documents.scan(Zone.RAW, "M-001")) == expected
    assert list(documents.scan(Zone.DEIDENTIFIED, "M-001")) == []


def test_load_and_remove(store, corpus):
    documents = DocumentStore(store)
    document = corpus.documents[0].document
    documents.save(Zone.RAW, document)
    assert documents.load(Zone.RAW, document.matter_id, document.id) == document

    documents.remove(Zone.RAW, document.matter_id, document.id)
    assert list(documents.scan(Zone.RAW, document.matter_id)) == []
