from collections import Counter

import pytest

from lexguard import rut
from lexguard.domain import DocumentKind, PiiKind, Purpose, Role, Sensitivity
from lexguard.synthetic import Pii, generate, render


@pytest.fixture(scope="module")
def corpus():
    return generate()


def test_generation_is_deterministic(corpus):
    assert generate() == corpus
    assert generate(seed=8) != corpus


def test_shape(corpus):
    assert len({m.client for m in corpus.matters}) == len(corpus.matters) == 4
    assert Counter(d.document.matter_id for d in corpus.documents) == {
        m.id: 6 for m in corpus.matters
    }
    assert {d.document.kind for d in corpus.documents} == set(DocumentKind)
    assert {d.document.sensitivity for d in corpus.documents} == set(Sensitivity)


def test_spans_match_the_text(corpus):
    for labeled in corpus.documents:
        text = labeled.document.text
        assert labeled.spans
        assert all(text[span.start : span.end] == span.text for span in labeled.spans)


def test_every_pii_kind_is_covered(corpus):
    assert {span.kind for doc in corpus.documents for span in doc.spans} == set(PiiKind)


def test_generated_ruts_are_valid(corpus):
    ruts = [s.text for d in corpus.documents for s in d.spans if s.kind is PiiKind.RUT]
    assert ruts
    assert all(rut.is_valid(text) for text in ruts)


def test_purposes_and_walls(corpus):
    matters = corpus.matter_index
    assert Purpose.MODEL_DEVELOPMENT in matters["M-001"].purposes
    assert matters["M-002"].purposes == {Purpose.LEGAL_ADVICE}
    assert "lawyer-1" in matters["M-002"].walled_off
    assert "M-002" in corpus.principal("lawyer-1").matters


def test_cast_of_principals(corpus):
    assert {p.role for p in corpus.principals} == set(Role)
    assert not corpus.principal("scientist-1").matters


def test_render_tracks_only_pii_fields():
    text, spans = render(
        "Don {name} debe ${amount} a {name}.",
        {"name": Pii(PiiKind.PERSON, "José Díaz"), "amount": "1.000"},
    )
    assert text == "Don José Díaz debe $1.000 a José Díaz."
    assert [(s.start, s.end) for s in spans] == [(4, 13), (28, 37)]
