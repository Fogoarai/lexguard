from datetime import UTC, datetime

import pytest

from lexguard import tokens
from lexguard.domain import Document, DocumentKind, Sensitivity
from lexguard.rag.chunking import split

TOKEN = "[PERSON_3fa9c2d1e0b4]"


def document(text: str) -> Document:
    return Document(
        "D-1",
        "M-001",
        DocumentKind.MEMO,
        Sensitivity.PRIVILEGED,
        text,
        datetime(2025, 1, 1, tzinfo=UTC),
    )


@pytest.mark.parametrize("max_chars", [1, 15, 80, 500])
def test_no_pseudonym_is_ever_split(documents, max_chars):
    for doc in documents:
        chunks = split(doc, max_chars)
        found = [match.group() for chunk in chunks for match in tokens.find(chunk.text)]
        assert found == [match.group() for match in tokens.find(doc.text)]


@pytest.mark.parametrize("max_chars", [15, 80, 500])
def test_chunks_cover_the_document_in_order(documents, max_chars):
    for doc in documents:
        chunks = split(doc, max_chars)
        assert " ".join(chunk.text for chunk in chunks).split() == doc.text.split()
        assert [chunk.ordinal for chunk in chunks] == list(range(len(chunks)))
        assert len({chunk.id for chunk in chunks}) == len(chunks)


def test_chunks_keep_document_matter_and_sensitivity(documents):
    for doc in documents:
        for chunk in split(doc, 200):
            assert (chunk.document_id, chunk.matter_id, chunk.sensitivity) == (
                doc.id,
                doc.matter_id,
                doc.sensitivity,
            )


def test_chunks_stay_within_the_limit(documents):
    for doc in documents:
        chunks = split(doc, 300)
        assert len(chunks) > 1
        assert all(len(chunk.text) <= 300 for chunk in chunks)


def test_an_unbreakable_run_longer_than_the_limit_stays_whole():
    chunks = split(document(f"uno dos {TOKEN},{TOKEN} tres"), max_chars=8)
    assert [chunk.text for chunk in chunks] == ["uno dos", f"{TOKEN},{TOKEN}", "tres"]


def test_cuts_prefer_a_paragraph_break():
    first = "PRIMERO: El arrendador entrega el inmueble."
    second = "SEGUNDO: La renta se paga cada mes."
    chunks = split(document(f"{first}\n\n{second}"), max_chars=60)
    assert [chunk.text for chunk in chunks] == [first, second]


def test_a_blank_document_has_no_chunks():
    assert split(document(" \n\n ")) == []
