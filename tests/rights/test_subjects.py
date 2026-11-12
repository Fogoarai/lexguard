from dataclasses import replace
from types import SimpleNamespace

import pytest

from lexguard import tokens
from lexguard.documents import DocumentStore
from lexguard.domain import (
    Action,
    Decision,
    DocumentKind,
    PiiKind,
    Principal,
    Purpose,
    Role,
    Sensitivity,
    Zone,
)
from lexguard.rights.subjects import Hit, Identifier
from lexguard.testing import DenyAll
from lexguard.training.dataset import DatasetTainted, shard_name

OFFICER = Principal("dpo-1", Role.PRIVACY_OFFICER)
SCIENTIST = Principal("scientist-1", Role.DATA_SCIENTIST)
MATTERS = ["M-001", "M-002", "M-003", "M-004"]


def shape(request):
    return (request.action, request.resource.zone, request.purpose, request.resource.sensitivity)


def test_locate_reports_the_documents_that_mention_the_subject(requests, subject):
    location = requests().locate(OFFICER, subject, MATTERS)

    assert location.hits == (
        Hit("M-001", "M-001-L001", DocumentKind.EMAIL, Sensitivity.CONFIDENTIAL, 1),
        Hit("M-002", "M-002-L001", DocumentKind.EMAIL, Sensitivity.CONFIDENTIAL, 1),
    )
    assert location.unsearched == ()


def test_every_identifier_of_the_subject_is_searched(requests, people):
    subject = [Identifier(PiiKind.RUT, people.rut), Identifier(PiiKind.PERSON, people.name)]

    (hit,) = requests().locate(OFFICER, subject, ["M-001"]).hits

    assert hit.mentions == 3


def test_a_person_is_found_where_only_part_of_the_name_is_written(
    requests, people, store, vault, corpus
):
    surname_only = replace(
        corpus.documents[0].document,
        id="M-001-L002",
        text=f"La señora {vault.token_for(PiiKind.PERSON, 'Fuentes', 'M-001')} no aceptará.",
    )
    DocumentStore(store).save(Zone.DEIDENTIFIED, surname_only)
    subject = [Identifier(PiiKind.PERSON, people.name)]

    location = requests().locate(OFFICER, subject, ["M-001"])
    erasure = requests().erase(OFFICER, subject, ["M-001"], remove_documents=True)

    assert [(hit.document_id, hit.partial) for hit in location.hits] == [
        ("M-001-L001", False),
        ("M-001-L002", True),
    ]
    assert erasure.hits == location.hits
    assert erasure.removed == location.hits[:1]
    left = store.names(Zone.DEIDENTIFIED, "M-001")
    assert "M-001-L001.doc.json" not in left and "M-001-L002.doc.json" in left
    with pytest.raises(KeyError):
        vault.reveal(vault.token_for(PiiKind.PERSON, "Fuentes", "M-001"), "M-001")
    assert vault.reveal(vault.token_for(PiiKind.PERSON, people.bystander, "M-001"), "M-001")


def test_a_partial_hit_is_removed_once_a_person_confirms_it(requests, people, store, vault, corpus):
    surname_only = replace(
        corpus.documents[0].document,
        id="M-001-L002",
        text=f"La señora {vault.token_for(PiiKind.PERSON, 'Fuentes', 'M-001')} no aceptará.",
    )
    DocumentStore(store).save(Zone.DEIDENTIFIED, surname_only)
    subject = [Identifier(PiiKind.PERSON, people.name)]
    reviewed = [hit for hit in requests().locate(OFFICER, subject, ["M-001"]).hits if hit.partial]

    erasure = requests().erase(
        OFFICER, subject, ["M-001"], remove_documents=True, confirmed=reviewed
    )

    assert [hit.document_id for hit in erasure.removed] == ["M-001-L001", "M-001-L002"]
    left = store.names(Zone.DEIDENTIFIED, "M-001")
    assert not {"M-001-L001.doc.json", "M-001-L002.doc.json"} & set(left)


def test_a_shorter_form_says_nothing_where_no_full_identifier_appears(
    requests, people, store, vault, corpus
):
    namesake = replace(
        corpus.documents[0].document,
        id="M-003-L002",
        matter_id="M-003",
        text=f"La señora {vault.token_for(PiiKind.PERSON, 'Fuentes', 'M-003')} no aceptará.",
    )
    DocumentStore(store).save(Zone.DEIDENTIFIED, namesake)
    subject = [Identifier(PiiKind.PERSON, people.name)]

    erasure = requests().erase(OFFICER, subject, ["M-003"])

    assert erasure.hits == ()
    assert erasure.tokens_forgotten == 1
    assert vault.reveal(vault.token_for(PiiKind.PERSON, "Fuentes", "M-003"), "M-003") == "Fuentes"


def test_locate_discloses_no_text_and_no_token(requests, subject, people):
    disclosed = repr(requests().locate(OFFICER, subject, MATTERS))

    assert not list(tokens.find(disclosed))
    assert people.bystander not in disclosed
    assert "Carta" not in disclosed


def test_unknown_subject_is_found_nowhere(requests):
    stranger = [Identifier(PiiKind.RUT, "9.999.999-3")]

    assert requests().locate(OFFICER, stranger, MATTERS).hits == ()


def test_locate_reads_only_the_compartments_the_guard_allows(requests, subject, scripted):
    policy = scripted(denied=frozenset({("M-001", Sensitivity.CONFIDENTIAL)}))

    location = requests(policy).locate(OFFICER, subject, MATTERS)

    assert [hit.matter_id for hit in location.hits] == ["M-002"]
    assert location.unsearched == (("M-001", Sensitivity.CONFIDENTIAL),)


def test_locate_denied_everywhere_finds_nothing(requests, subject):
    location = requests(DenyAll()).locate(OFFICER, subject, MATTERS)

    assert location.hits == ()
    assert len(location.unsearched) == len(MATTERS) * len(Sensitivity)


def test_shredded_matter_is_passed_over(requests, subject, store):
    store.purge("M-001")

    location = requests().locate(OFFICER, subject, MATTERS)
    erasure = requests().erase(OFFICER, subject, MATTERS)

    assert [hit.matter_id for hit in location.hits] == ["M-002"]
    assert [hit.matter_id for hit in erasure.hits] == ["M-002"]
    assert erasure.tokens_forgotten == len(MATTERS) - 1


def test_locate_asks_with_the_data_subject_shape(requests, subject, scripted):
    policy = scripted()
    requests(policy).locate(OFFICER, subject, ["M-001"])

    assert {request.resource.matter_id for request in policy.asked} == {"M-001"}
    assert [shape(request) for request in policy.asked] == [
        (Action.READ_DEIDENTIFIED, Zone.DEIDENTIFIED, Purpose.DATA_SUBJECT_REQUEST, sensitivity)
        for sensitivity in Sensitivity
    ]


def test_erase_forgets_the_subject_and_nobody_else(requests, subject, people, vault):
    own = vault.token_for(PiiKind.RUT, people.rut, "M-001")
    bystander = vault.token_for(PiiKind.PERSON, people.bystander, "M-001")

    erasure = requests().erase(OFFICER, subject, MATTERS)

    assert [hit.document_id for hit in erasure.hits] == ["M-001-L001", "M-002-L001"]
    assert erasure.tokens_forgotten == len(MATTERS)
    assert erasure.refused == ()
    with pytest.raises(KeyError):
        vault.reveal(own, "M-001")
    assert vault.reveal(bystander, "M-001") == people.bystander


def test_erase_keeps_the_documents_unless_asked_to_remove_them(requests, subject, store):
    erasure = requests().erase(OFFICER, subject, MATTERS)

    assert erasure.removed == ()
    assert DocumentStore(store).load(Zone.DEIDENTIFIED, "M-001", "M-001-L001")


def test_erase_can_remove_the_documents_and_their_training_copies(
    requests, subject, store, training_sets, corpus
):
    training_sets.build(SCIENTIST, corpus.matters, "ds-1")
    before = store.names(Zone.DEIDENTIFIED, "M-001")

    erasure = requests().erase(OFFICER, subject, MATTERS, remove_documents=True)

    assert [hit.document_id for hit in erasure.removed] == ["M-001-L001", "M-002-L001"]
    assert set(before) - set(store.names(Zone.DEIDENTIFIED, "M-001")) == {"M-001-L001.doc.json"}
    assert set(before) - set(store.names(Zone.RAW, "M-001")) == {"M-001-L001.doc.json"}
    assert store.names(Zone.TRAINING, "M-001") == []
    assert store.names(Zone.TRAINING, "M-003") == [shard_name("ds-1")]
    assert requests().locate(OFFICER, subject, MATTERS).hits == ()


def test_erase_taints_every_dataset_that_used_an_affected_document(
    requests, subject, training_sets, corpus, manifests
):
    training_sets.build(SCIENTIST, corpus.matters, "ds-all")
    training_sets.build(SCIENTIST, [corpus.matter_index["M-003"]], "ds-other")

    erasure = requests().erase(OFFICER, subject, MATTERS)

    assert erasure.tainted_datasets == ("ds-all",)
    (taint,) = manifests.load("ds-all").taints
    assert (taint.reason, taint.document_ids) == ("data subject erasure", ("M-001-L001",))
    assert not manifests.load("ds-other").tainted
    with pytest.raises(DatasetTainted):
        training_sets.load(SCIENTIST, "ds-all")
    assert training_sets.load(SCIENTIST, "ds-other").examples


def test_erase_is_all_or_nothing_within_a_matter(requests, subject, people, vault, scripted):
    kept = vault.token_for(PiiKind.RUT, people.rut, "M-002")
    policy = scripted(denied=frozenset({("M-002", Sensitivity.PRIVILEGED)}))

    erasure = requests(policy).erase(OFFICER, subject, MATTERS)

    assert erasure.refused == ("M-002",)
    assert [hit.matter_id for hit in erasure.hits] == ["M-001"]
    assert vault.reveal(kept, "M-002") == people.rut


def test_erase_asks_with_the_data_subject_shape(requests, subject, scripted):
    policy = scripted()
    requests(policy).erase(OFFICER, subject, ["M-001"])

    assert {request.resource.matter_id for request in policy.asked} == {"M-001"}
    assert [shape(request) for request in policy.asked] == [
        (Action.ERASE, Zone.DEIDENTIFIED, Purpose.DATA_SUBJECT_REQUEST, sensitivity)
        for sensitivity in Sensitivity
    ]


def test_removing_documents_also_asks_to_erase_the_raw_zone(requests, subject, scripted):
    policy = scripted()
    requests(policy).erase(OFFICER, subject, ["M-001"], remove_documents=True)

    assert [shape(request) for request in policy.asked] == [
        (Action.ERASE, zone, Purpose.DATA_SUBJECT_REQUEST, sensitivity)
        for zone in (Zone.DEIDENTIFIED, Zone.RAW)
        for sensitivity in Sensitivity
    ]


def test_a_matter_whose_raw_zone_is_closed_to_the_officer_keeps_its_documents(
    requests, subject, store
):
    def no_raw(request):
        return Decision(request.resource.zone is not Zone.RAW, "scripted")

    erasure = requests(SimpleNamespace(decide=no_raw)).erase(
        OFFICER, subject, ["M-001"], remove_documents=True
    )

    assert erasure.refused == ("M-001",)
    assert DocumentStore(store).load(Zone.DEIDENTIFIED, "M-001", "M-001-L001")
    assert DocumentStore(store).load(Zone.RAW, "M-001", "M-001-L001")
