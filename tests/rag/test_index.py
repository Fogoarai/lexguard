from dataclasses import replace

import numpy as np
import pytest

from lexguard.domain import Action, Decision, Obligation, Purpose, Sensitivity, Zone
from lexguard.rag import HashingEmbedder, SecureIndex
from lexguard.testing import AllowAll, MembershipPolicy

RENT = "¿Cuál es la renta mensual del arrendamiento?"
NOTICES = "correo para notificaciones"


def build(guard, documents, **options) -> SecureIndex:
    index = SecureIndex(HashingEmbedder(), guard, **options)
    index.add(documents)
    return index


def ranking(retrieval):
    return [(hit.chunk.id, pytest.approx(hit.score)) for hit in retrieval.hits]


def no_privileged(request):
    return Decision(request.resource.sensitivity is not Sensitivity.PRIVILEGED, "need to know")


def test_search_ranks_the_matching_chunks_first(guard_with, documents, corpus):
    index = build(guard_with(MembershipPolicy()), documents)

    retrieval = index.search(RENT, corpus.principal("lawyer-3"), Purpose.LEGAL_ADVICE)

    assert retrieval.hits
    assert "renta mensual" in " ".join(hit.chunk.text for hit in retrieval.hits[:3])
    scores = [hit.score for hit in retrieval.hits]
    assert scores == sorted(scores, reverse=True)


def test_denied_matters_change_neither_scores_nor_counts(guard_with, documents, corpus):
    lawyer = corpus.principal("lawyer-3")
    guard = guard_with(MembershipPolicy())
    whole_lake = build(guard, documents)
    own_matter = build(guard, [doc for doc in documents if doc.matter_id == "M-003"])

    for question in (RENT, NOTICES):
        trimmed = whole_lake.search(question, lawyer, Purpose.LEGAL_ADVICE, k=4)
        assert len(trimmed.hits) == 4
        assert {hit.chunk.matter_id for hit in trimmed.hits} == {"M-003"}
        assert ranking(trimmed) == ranking(
            own_matter.search(question, lawyer, Purpose.LEGAL_ADVICE, k=4)
        )


def test_a_denied_chunk_that_matches_best_takes_no_slot(guard_with, documents, corpus):
    lawyer = corpus.principal("lawyer-3")
    index = build(guard_with(MembershipPolicy()), documents)
    before = index.search(RENT, lawyer, Purpose.LEGAL_ADVICE, k=3)

    bait = replace(documents[0], id="M-001-BAIT", text=RENT)
    index.add([bait])

    assert ranking(index.search(RENT, lawyer, Purpose.LEGAL_ADVICE, k=3)) == ranking(before)
    everyone = build(guard_with(AllowAll()), [*documents, bait])
    assert everyone.search(RENT, lawyer, Purpose.LEGAL_ADVICE, k=1).hits[0].chunk.document_id == (
        "M-001-BAIT"
    )


def test_a_denied_sensitivity_is_trimmed_inside_a_matter(guard_with, documents, corpus):
    paralegal = corpus.principal("paralegal-1")
    index = build(guard_with(no_privileged), documents)
    without = build(
        guard_with(AllowAll()),
        [doc for doc in documents if doc.sensitivity is not Sensitivity.PRIVILEGED],
    )
    question = "estrategia recomendada para la audiencia y secreto profesional"

    trimmed = index.search(question, paralegal, Purpose.LEGAL_ADVICE, k=50)

    assert Sensitivity.PRIVILEGED not in {hit.chunk.sensitivity for hit in trimmed.hits}
    assert ranking(trimmed) == ranking(
        without.search(question, paralegal, Purpose.LEGAL_ADVICE, k=50)
    )


def test_the_guard_is_asked_once_per_compartment(guard_with, audit, documents, corpus):
    index = build(guard_with(MembershipPolicy()), documents)
    lawyer = corpus.principal("lawyer-1")

    index.search(RENT, lawyer, Purpose.LEGAL_ADVICE)

    asked = [(event.matter_id, event.detail["sensitivity"]) for event in audit.events]
    assert sorted(asked) == sorted({(doc.matter_id, doc.sensitivity) for doc in documents})
    assert len(asked) == 12
    for event in audit.events:
        assert (event.principal_id, event.action, event.zone, event.purpose) == (
            "lawyer-1",
            Action.RETRIEVE,
            Zone.SERVING,
            Purpose.LEGAL_ADVICE,
        )
        assert event.allowed is (event.matter_id in lawyer.matters)
        assert RENT not in repr(event)


def test_a_query_can_be_restricted_to_one_matter(guard_with, audit, documents, corpus):
    index = build(guard_with(MembershipPolicy()), documents)
    lawyer = corpus.principal("lawyer-1")

    retrieval = index.search(RENT, lawyer, Purpose.LEGAL_ADVICE, matter_id="M-002")

    assert {hit.chunk.matter_id for hit in retrieval.hits} == {"M-002"}
    assert {event.matter_id for event in audit.events} == {"M-002"}
    assert len(audit.events) == 3


def test_a_principal_without_access_gets_nothing(guard_with, documents, corpus):
    index = build(guard_with(MembershipPolicy()), documents)
    scientist = corpus.principal("scientist-1")

    assert index.search(RENT, scientist, Purpose.MODEL_DEVELOPMENT) == index.search(
        RENT, corpus.principal("lawyer-1"), Purpose.LEGAL_ADVICE, matter_id="M-999"
    )
    assert not index.search(RENT, scientist, Purpose.MODEL_DEVELOPMENT).hits


def test_obligations_come_only_from_the_compartments_of_the_hits(guard_with, documents, corpus):
    def decide(request):
        if request.resource.matter_id == "M-004":
            return Decision(False, "walled off", frozenset({Obligation.NO_EGRESS}))
        if request.resource.sensitivity is Sensitivity.CONFIDENTIAL:
            return Decision(True, "pseudonymized only", frozenset({Obligation.KEEP_PSEUDONYMS}))
        return Decision(True, "assigned")

    index = build(guard_with(decide), documents)
    paralegal = corpus.principal("paralegal-1")

    contracts = index.search(RENT, paralegal, Purpose.LEGAL_ADVICE, k=2)
    filings = index.search("despido injustificado", paralegal, Purpose.LEGAL_ADVICE, k=2)

    assert {hit.chunk.sensitivity for hit in contracts.hits} == {Sensitivity.CONFIDENTIAL}
    assert contracts.obligations == {Obligation.KEEP_PSEUDONYMS}
    assert {hit.chunk.sensitivity for hit in filings.hits} == {Sensitivity.INTERNAL}
    assert filings.obligations == frozenset()


def test_weak_matches_are_not_released(guard_with, documents, corpus):
    lawyer = corpus.principal("lawyer-1")
    question = "receta de cocina con tomates"

    strict = build(guard_with(AllowAll()), documents)
    lenient = build(guard_with(AllowAll()), documents, min_score=0.0)

    assert not strict.search(question, lawyer, Purpose.LEGAL_ADVICE).hits
    assert lenient.search(question, lawyer, Purpose.LEGAL_ADVICE).hits


def test_documents_can_be_replaced_and_forgotten(guard_with, documents, corpus):
    index = build(guard_with(AllowAll()), documents)
    lawyer = corpus.principal("lawyer-1")
    size = len(index)

    index.add(documents[:3])
    assert len(index) == size

    removed = index.remove("M-001", documents[0].id)
    assert removed > 0
    assert len(index) == size - removed

    index.remove("M-001")
    hits = index.search(RENT, lawyer, Purpose.LEGAL_ADVICE, k=100).hits
    assert hits
    assert "M-001" not in {hit.chunk.matter_id for hit in hits}


def test_any_embedder_can_be_plugged_in(guard_with, documents, corpus):
    class KeywordEmbedder:
        """Unnormalized on purpose: the index must not rely on unit vectors."""

        def embed(self, texts):
            return np.array(
                [
                    [10.0 * ("renta" in text.lower()), 3.0 * ("despido" in text.lower()), 0.5]
                    for text in texts
                ]
            )

    index = SecureIndex(KeywordEmbedder(), guard_with(AllowAll()))
    index.add(documents)

    hits = index.search("renta", corpus.principal("lawyer-1"), Purpose.LEGAL_ADVICE, k=3).hits

    assert all("renta" in hit.chunk.text.lower() for hit in hits)
    assert all(hit.score == pytest.approx(hits[0].score) for hit in hits)
    assert hits[0].score <= 1.0 + 1e-9
