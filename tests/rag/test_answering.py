import re
from collections.abc import Callable
from dataclasses import dataclass, field

import pytest

from lexguard import tokens
from lexguard.domain import Action, Decision, Obligation, PiiKind, Purpose, Sensitivity
from lexguard.egress import EgressGateway
from lexguard.errors import EgressBlocked
from lexguard.rag import NO_ANSWER, Answerer, ExtractiveModel, HashingEmbedder, SecureIndex
from lexguard.rag.prompt import parse
from lexguard.settings import Boundary, ModelLocation, Settings
from lexguard.testing import MembershipPolicy

RENT = "¿Cuál es la renta mensual del arrendamiento?"
NOTICES = "correo para notificaciones"
ADVICE = Purpose.LEGAL_ADVICE


@dataclass
class FakeModel:
    reply: Callable[[str], str]
    location: ModelLocation = ModelLocation.LOCAL
    name: str = "fake-model"
    prompts: list[str] = field(default_factory=list)

    def complete(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return self.reply(prompt)


def echo(prompt: str) -> str:
    """Repeats every passage and cites it, like a model that uses all of its context."""
    return "\n".join(f"{text} [{n}]" for n, text in enumerate(parse(prompt)[1], start=1))


def membership(oblige=lambda request: frozenset(), deny=lambda request: False):
    def decide(request):
        if request.resource.matter_id not in request.principal.matters or deny(request):
            return Decision(False, "denied")
        return Decision(True, "assigned", oblige(request))

    return decide


def keep_pseudonyms(request):
    return frozenset({Obligation.KEEP_PSEUDONYMS})


def has_tokens(text: str) -> bool:
    return any(tokens.find(text))


@pytest.fixture
def raw(corpus):
    return {labeled.document.id: labeled.document.text for labeled in corpus.documents}


@pytest.fixture
def build(guard_with, documents, deidentifier, scan):
    def _build(policy=None, model=None, boundary=Boundary.LOCAL_ONLY) -> Answerer:
        guard = guard_with(policy or MembershipPolicy())
        index = SecureIndex(HashingEmbedder(), guard)
        index.add(documents)
        gateway = EgressGateway(guard, Settings(boundary), scan)
        return Answerer(index, gateway, model or ExtractiveModel(), deidentifier, guard)

    return _build


def test_an_assigned_lawyer_gets_a_reidentified_answer_with_citations(build, corpus, raw):
    answer = build().answer(RENT, corpus.principal("lawyer-1"), ADVICE, matter_id="M-001")

    assert answer.reidentified
    assert "renta mensual" in answer.text
    assert not has_tokens(answer.text)
    assert answer.citations
    for citation in answer.citations:
        assert citation.matter_id == "M-001"
        assert citation.passage in raw[citation.document_id]
    quotes = re.findall(r"(.+?) \[(\d+)\]\s*", answer.text)
    assert {int(number) for _, number in quotes} == set(range(1, len(answer.citations) + 1))
    for quote, number in quotes:
        assert quote in answer.citations[int(number) - 1].passage


def test_one_question_leaves_a_complete_trail_without_content(build, audit, corpus):
    stages = [Action.RETRIEVE, Action.EGRESS, Action.REIDENTIFY]

    build().answer(RENT, corpus.principal("lawyer-1"), ADVICE, matter_id="M-001")

    actions = [event.action for event in audit.events]
    assert set(actions) == set(stages)
    assert actions == sorted(actions, key=stages.index)
    assert {(event.principal_id, event.matter_id, event.allowed) for event in audit.events} == {
        ("lawyer-1", "M-001", True)
    }
    trail = repr(audit.events)
    assert RENT not in trail
    assert not any(tokens.find(trail))
    assert not any(span.text in trail for doc in corpus.documents for span in doc.spans)


def test_citations_name_the_document_and_the_chunk(build, corpus, documents):
    answerer = build(membership(keep_pseudonyms))
    lawyer = corpus.principal("lawyer-1")

    answer = answerer.answer(RENT, lawyer, ADVICE, matter_id="M-001")

    hits = answerer.index.search(RENT, lawyer, ADVICE, matter_id="M-001").hits
    chunks = {(hit.chunk.document_id, hit.chunk.ordinal): hit.chunk for hit in hits}
    for citation in answer.citations:
        assert chunks[citation.document_id, citation.ordinal].text == citation.passage


def test_keep_pseudonyms_leaves_the_answer_pseudonymized(build, audit, corpus):
    answerer = build(membership(keep_pseudonyms))

    answer = answerer.answer(RENT, corpus.principal("lawyer-1"), ADVICE, matter_id="M-001")

    assert not answer.reidentified
    assert has_tokens(answer.text)
    assert all(has_tokens(citation.passage) for citation in answer.citations)
    released = answer.text + "".join(citation.passage for citation in answer.citations)
    assert not any(span.text in released for doc in corpus.documents for span in doc.spans)
    assert Action.REIDENTIFY not in {event.action for event in audit.events}


def test_reidentification_needs_every_compartment_of_the_context(build, audit, corpus):
    def no_internal_reidentification(request):
        return (
            request.action is Action.REIDENTIFY
            and request.resource.sensitivity is Sensitivity.INTERNAL
        )

    answerer = build(membership(deny=no_internal_reidentification), FakeModel(echo))
    lawyer = corpus.principal("lawyer-1")

    mixed = answerer.answer(NOTICES, lawyer, ADVICE, matter_id="M-001")
    asked = {
        event.detail["sensitivity"]: event.allowed
        for event in audit.events
        if event.action is Action.REIDENTIFY
    }
    contracts_only = answerer.answer(RENT, lawyer, ADVICE, matter_id="M-001", k=2)

    assert asked == {"confidential": True, "internal": False}
    assert not mixed.reidentified
    assert has_tokens(mixed.text)
    assert contracts_only.reidentified
    assert not has_tokens(contracts_only.text)


def test_each_passage_is_reidentified_with_its_own_matter(build, corpus, raw, deidentifier):
    answerer = build(model=FakeModel(echo))

    answer = answerer.answer(RENT, corpus.principal("lawyer-1"), ADVICE, k=6)

    assert {citation.matter_id for citation in answer.citations} == {"M-001", "M-002"}
    assert answer.reidentified
    assert not has_tokens(answer.text)
    for citation in answer.citations:
        assert citation.passage in raw[citation.document_id]
        assert citation.passage in answer.text
    assert has_tokens(deidentifier.reidentify(echo(answerer.model.prompts[0]), "M-001"))


def test_a_token_from_outside_the_context_stays_masked(build, corpus, documents, deidentifier):
    lawyer = corpus.principal("lawyer-1")
    probe = build()
    (hit,) = probe.index.search(RENT, lawyer, ADVICE, matter_id="M-001", k=1).hits
    inside = next(tokens.find(hit.chunk.text)).group()
    outside = next(
        match.group()
        for doc in documents
        if doc.matter_id == "M-001" and doc.id != hit.chunk.document_id
        for match in tokens.find(doc.text)
        if match.group() not in hit.chunk.text
    )
    answerer = build(model=FakeModel(lambda prompt: f"Firmó {inside} y no {outside} [1]"))

    answer = answerer.answer(RENT, lawyer, ADVICE, matter_id="M-001", k=1)

    assert answer.reidentified
    assert deidentifier.reidentify(outside, "M-001") != outside
    assert answer.text == f"Firmó {deidentifier.reidentify(inside, 'M-001')} y no {outside} [1]"


def test_denied_compartments_reach_neither_the_model_nor_the_citations(build, corpus, documents):
    model = FakeModel(echo)
    answerer = build(model=model)

    answer = answerer.answer(RENT, corpus.principal("lawyer-3"), ADVICE, k=20)

    (prompt,) = model.prompts
    foreign = {
        match.group()
        for doc in documents
        if doc.matter_id != "M-003"
        for match in tokens.find(doc.text)
    }
    assert not foreign & {match.group() for match in tokens.find(prompt)}
    assert {citation.matter_id for citation in answer.citations} == {"M-003"}
    assert len(answer.citations) == len(parse(prompt)[1])


def test_nothing_retrievable_means_the_model_is_never_called(build, audit, corpus):
    model = FakeModel(echo)

    answer = build(model=model).answer(RENT, corpus.principal("scientist-1"), ADVICE)

    assert (answer.text, answer.citations, answer.reidentified) == (NO_ANSWER, (), False)
    assert model.prompts == []
    assert {event.action for event in audit.events} == {Action.RETRIEVE}
    assert not any(event.allowed for event in audit.events)


def test_a_model_that_finds_no_answer_cites_nothing(build, corpus):
    answerer = build(model=FakeModel(lambda prompt: NO_ANSWER))

    answer = answerer.answer(RENT, corpus.principal("lawyer-1"), ADVICE)

    assert (answer.text, answer.citations, answer.reidentified) == (NO_ANSWER, (), False)


def test_markers_are_renumbered_to_index_the_citations(build, corpus):
    keep = membership(keep_pseudonyms)
    reply = "Según [3] y [1], pero no [9]; de nuevo [3]."
    answerer = build(keep, FakeModel(lambda prompt: reply))
    lawyer = corpus.principal("lawyer-1")

    answer = answerer.answer(RENT, lawyer, ADVICE, matter_id="M-001")

    context = [
        hit.chunk for hit in answerer.index.search(RENT, lawyer, ADVICE, matter_id="M-001").hits
    ]
    assert answer.text == "Según [1] y [2], pero no [9]; de nuevo [1]."
    assert [(c.document_id, c.ordinal) for c in answer.citations] == [
        (context[2].document_id, context[2].ordinal),
        (context[0].document_id, context[0].ordinal),
    ]


def test_a_reply_without_markers_is_attributed_to_the_whole_context(build, corpus):
    model = FakeModel(lambda prompt: "La renta se paga mensualmente.")
    answerer = build(model=model)

    answer = answerer.answer(RENT, corpus.principal("lawyer-1"), ADVICE, matter_id="M-001", k=3)

    assert answer.text == "La renta se paga mensualmente."
    assert len(answer.citations) == len(parse(model.prompts[0])[1]) == 3


def test_blocked_egress_propagates_and_sends_nothing(build, audit, corpus):
    model = FakeModel(echo, ModelLocation.EXTERNAL)

    with pytest.raises(EgressBlocked, match="does not admit external"):
        build(model=model).answer(RENT, corpus.principal("lawyer-1"), ADVICE, matter_id="M-001")

    assert model.prompts == []
    assert Action.REIDENTIFY not in {event.action for event in audit.events}


def test_an_external_model_sees_only_pseudonymized_text(build, corpus):
    model = FakeModel(echo, ModelLocation.EXTERNAL)
    answerer = build(model=model, boundary=Boundary.DEIDENTIFIED_EGRESS)
    lawyer = corpus.principal("lawyer-1")

    answer = answerer.answer(RENT, lawyer, ADVICE, matter_id="M-001")

    (prompt,) = model.prompts
    assert has_tokens(prompt)
    assert not any(span.text in prompt for doc in corpus.documents for span in doc.spans)
    assert answer.reidentified
    assert not has_tokens(answer.text)


def test_a_question_that_names_a_person_does_not_leave_the_firm(build, corpus):
    model = FakeModel(echo, ModelLocation.EXTERNAL)
    answerer = build(model=model, boundary=Boundary.DEIDENTIFIED_EGRESS)
    person = next(span.text for span in corpus.documents[0].spans if span.kind is PiiKind.PERSON)

    with pytest.raises(EgressBlocked, match=r"residual PII in outbound prompt: \d person$"):
        answerer.answer(
            f"¿Qué renta paga {person}?", corpus.principal("lawyer-1"), ADVICE, matter_id="M-001"
        )

    assert model.prompts == []


@pytest.mark.parametrize("location", [ModelLocation.LOCAL, ModelLocation.EXTERNAL])
def test_a_no_egress_obligation_on_retrieval_travels_to_the_gateway(build, corpus, location):
    def on_retrieve(request):
        return frozenset({Obligation.NO_EGRESS} if request.action is Action.RETRIEVE else ())

    model = FakeModel(echo, location)
    answerer = build(membership(on_retrieve), model, Boundary.DEIDENTIFIED_EGRESS)
    lawyer = corpus.principal("lawyer-1")

    if location is ModelLocation.LOCAL:
        assert answerer.answer(RENT, lawyer, ADVICE, matter_id="M-001").citations
    else:
        with pytest.raises(EgressBlocked, match="no_egress"):
            answerer.answer(RENT, lawyer, ADVICE, matter_id="M-001")
        assert model.prompts == []
