from dataclasses import replace

import pytest

from lexguard.deid import Score, detect, evaluate
from lexguard.domain import PiiKind
from lexguard.synthetic import generate

HANDWRITTEN_FLOORS = {
    PiiKind.RUT: (0.95, 0.95),
    PiiKind.EMAIL: (0.95, 0.95),
    PiiKind.CASE_NUMBER: (0.95, 0.95),
    PiiKind.BANK_ACCOUNT: (0.95, 0.95),
    PiiKind.ADDRESS: (0.95, 0.95),
    PiiKind.PHONE: (0.95, 0.85),
    PiiKind.ORGANIZATION: (0.95, 0.55),
    PiiKind.PERSON: (0.70, 0.85),
}


@pytest.mark.parametrize("seed", range(10))
def test_synthetic_corpus_with_exact_matching(seed):
    corpus = generate(seed=seed, documents_per_matter=8)
    for kind, score in evaluate(detect, corpus.documents).items():
        assert score.labeled > 0, kind
        assert score.precision >= 0.98, kind
        assert score.recall >= 0.98, kind


def test_handwritten_texts_with_overlap_matching(handwritten):
    scores = evaluate(detect, handwritten.values(), exact=False)
    for kind, (precision, recall) in HANDWRITTEN_FLOORS.items():
        assert scores[kind].precision >= precision, kind
        assert scores[kind].recall >= recall, kind


def test_exact_matching_needs_the_same_boundaries_and_overlap_does_not(handwritten):
    document = handwritten["dev/finiquito"]
    people = sum(span.kind is PiiKind.PERSON for span in document.spans)

    def clipped(text):
        return [replace(span, end=span.end - 1, text=span.text[:-1]) for span in document.spans]

    exact = evaluate(clipped, [document])
    assert exact[PiiKind.PERSON] == Score(predicted=people, correct=0, labeled=people, found=0)
    overlap = evaluate(clipped, [document], exact=False)
    assert overlap[PiiKind.PERSON] == Score(people, people, people, people)


def test_a_span_of_the_wrong_kind_counts_against_both_kinds(handwritten):
    document = handwritten["dev/finiquito"]
    ruts = sum(span.kind is PiiKind.RUT for span in document.spans)

    def confused(text):
        return [replace(span, kind=PiiKind.PHONE) for span in document.spans]

    scores = evaluate(confused, [document], exact=False)
    assert scores[PiiKind.RUT] == Score(labeled=ruts)
    assert scores[PiiKind.PHONE] == Score(predicted=len(document.spans))


def test_score_ratios():
    score = Score(predicted=4, correct=3, labeled=6, found=3)
    assert (score.precision, score.recall) == (0.75, 0.5)
    assert (Score().precision, Score().recall) == (1.0, 1.0)
    assert Score(labeled=2).recall == 0.0
