import hashlib
import json
from dataclasses import replace

import pytest

from lexguard import tokens
from lexguard.documents import DocumentStore
from lexguard.domain import Action, DocumentKind, Principal, Purpose, Role, Sensitivity, Zone
from lexguard.errors import AccessDenied, IntegrityError, KeyUnavailable
from lexguard.testing import DenyAll, MembershipPolicy
from lexguard.training.dataset import DatasetTainted, shard_name
from lexguard.training.manifest import Exclusion, Reason

SCIENTIST = Principal("scientist-1", Role.DATA_SCIENTIST)
EXPORTABLE = {f"{matter}-D00{n}" for matter in ("M-001", "M-003") for n in (1, 2, 5, 6)}


def test_exports_permitted_matters_without_privileged_documents(training_sets, corpus):
    manifest = training_sets().build(SCIENTIST, corpus.matters, "ds-1")

    assert {entry.document_id for entry in manifest.entries} == EXPORTABLE
    assert manifest.purpose is Purpose.MODEL_DEVELOPMENT
    assert Exclusion("M-002", Reason.PURPOSE_NOT_PERMITTED) in manifest.exclusions
    assert Exclusion("M-001", Reason.PRIVILEGED, Sensitivity.PRIVILEGED) in manifest.exclusions


def test_purpose_limit_holds_under_a_policy_that_allows_everything(training_sets, corpus, audit):
    manifest = training_sets().build(SCIENTIST, corpus.matters, "ds-1")

    assert {entry.matter_id for entry in manifest.entries} == {"M-001", "M-003"}
    assert {event.matter_id for event in audit.events} == {"M-001", "M-003"}


def test_guard_is_asked_per_compartment_with_the_export_shape(training_sets, corpus, scripted):
    policy = scripted()
    training_sets(policy).build(SCIENTIST, corpus.matters, "ds-1")

    assert {(r.action, r.resource.zone, r.purpose) for r in policy.asked} == {
        (Action.EXPORT_TRAINING, Zone.DEIDENTIFIED, Purpose.MODEL_DEVELOPMENT)
    }
    assert [(r.resource.matter_id, r.resource.sensitivity) for r in policy.asked] == [
        ("M-001", Sensitivity.INTERNAL),
        ("M-001", Sensitivity.CONFIDENTIAL),
        ("M-003", Sensitivity.INTERNAL),
        ("M-003", Sensitivity.CONFIDENTIAL),
    ]


def test_denied_matter_is_skipped_and_recorded(training_sets, corpus, store):
    member = Principal("scientist-2", Role.DATA_SCIENTIST, frozenset({"M-001"}))
    manifest = training_sets(MembershipPolicy()).build(member, corpus.matters, "ds-1")

    assert {entry.matter_id for entry in manifest.entries} == {"M-001"}
    denied = Exclusion("M-003", Reason.POLICY_DENIED, Sensitivity.INTERNAL, detail="not assigned")
    assert denied in manifest.exclusions
    assert store.names(Zone.TRAINING, "M-003") == []


def test_denied_sensitivity_is_left_out_of_an_allowed_matter(training_sets, corpus, scripted):
    policy = scripted(denied=frozenset({("M-001", Sensitivity.CONFIDENTIAL)}))
    manifest = training_sets(policy).build(SCIENTIST, corpus.matters, "ds-1")

    from_first = {entry.document_id for entry in manifest.entries if entry.matter_id == "M-001"}
    assert from_first == {"M-001-D002", "M-001-D006"}


def test_privileged_documents_are_opt_in(training_sets, corpus):
    manifest = training_sets(include_privileged=True).build(SCIENTIST, corpus.matters, "ds-1")

    assert len(manifest.entries) == 12
    assert Sensitivity.PRIVILEGED in {entry.sensitivity for entry in manifest.entries}
    assert Reason.PRIVILEGED not in {exclusion.reason for exclusion in manifest.exclusions}


def test_document_with_residual_pii_is_refused(training_sets, corpus, store, pseudonymized):
    leaky = replace(pseudonymized[0], text=pseudonymized[0].text + "RUT 12.345.678-5\n")
    DocumentStore(store).save(Zone.DEIDENTIFIED, leaky)
    sets = training_sets()

    manifest = sets.build(SCIENTIST, corpus.matters, "ds-1")

    assert {entry.document_id for entry in manifest.entries} == EXPORTABLE - {leaky.id}
    refusal = Exclusion("M-001", Reason.RESIDUAL_PII, Sensitivity.CONFIDENTIAL, leaky.id, "rut")
    assert refusal in manifest.exclusions
    assert "12.345.678-5" not in "".join(sets.load(SCIENTIST, "ds-1").fine_tuning_records())


def test_manifest_records_hashes_and_never_text(
    training_sets, corpus, manifests, pseudonymized, tmp_path
):
    manifest = training_sets().build(SCIENTIST, corpus.matters, "ds-1")
    on_disk = (tmp_path / "manifests" / "ds-1.manifest.json").read_text()

    assert manifests.load("ds-1") == manifest
    assert set(json.loads(on_disk)) == {
        "dataset_id",
        "created_at",
        "purpose",
        "entries",
        "exclusions",
        "taints",
    }
    texts = {document.id: document.text for document in pseudonymized}
    for entry in manifest.entries:
        assert entry.sha256 == hashlib.sha256(texts[entry.document_id].encode()).hexdigest()
    assert not list(tokens.find(on_disk))
    assert not any(line in on_disk for text in texts.values() for line in text.splitlines() if line)


def test_dataset_serves_both_tasks(training_sets, corpus, pseudonymized):
    sets = training_sets()
    sets.build(SCIENTIST, corpus.matters, "ds-1")
    exported = [document for document in pseudonymized if document.id in EXPORTABLE]

    dataset = sets.load(SCIENTIST, "ds-1")

    assert dataset.fine_tuning_records() == [document.text for document in exported]
    assert dataset.classification_pairs() == [(d.text, d.kind) for d in exported]
    assert {kind for _, kind in dataset.classification_pairs()} == {
        DocumentKind.CONTRACT,
        DocumentKind.FILING,
    }


def test_shredding_a_matter_makes_its_shard_unreadable(training_sets, corpus, store, keys):
    sets = training_sets()
    sets.build(SCIENTIST, corpus.matters, "ds-1")
    assert store.names(Zone.TRAINING, "M-001") == [shard_name("ds-1")]

    keys.shred("M-001")

    assert store.names(Zone.TRAINING, "M-001") == [shard_name("ds-1")]
    with pytest.raises(KeyUnavailable):
        store.get(Zone.TRAINING, "M-001", shard_name("ds-1"))
    with pytest.raises(KeyUnavailable):
        sets.load(SCIENTIST, "ds-1")
    assert store.get(Zone.TRAINING, "M-003", shard_name("ds-1"))


def test_tainted_dataset_cannot_be_loaded(training_sets, corpus, manifests):
    sets = training_sets()
    sets.build(SCIENTIST, corpus.matters, "ds-1")
    sets.build(SCIENTIST, [corpus.matter_index["M-003"]], "ds-2")

    marked = manifests.taint(lambda entry: entry.document_id == "M-001-D001", "erasure request")

    assert [manifest.dataset_id for manifest in marked] == ["ds-1"]
    assert manifests.load("ds-1").taints[0].document_ids == ("M-001-D001",)
    with pytest.raises(DatasetTainted, match="erasure request"):
        sets.load(SCIENTIST, "ds-1")
    assert not manifests.load("ds-2").tainted
    assert sets.load(SCIENTIST, "ds-2").examples


def test_loading_needs_the_export_to_still_be_allowed(training_sets, corpus, audit):
    training_sets().build(SCIENTIST, corpus.matters, "ds-1")

    with pytest.raises(AccessDenied):
        training_sets(DenyAll()).load(SCIENTIST, "ds-1")
    assert audit.events[-1].detail == {"dataset": "ds-1"}


def test_load_rejects_a_shard_that_does_not_match_its_manifest(training_sets, corpus, manifests):
    sets = training_sets()
    manifest = sets.build(SCIENTIST, corpus.matters, "ds-1")
    first, *rest = manifest.entries
    manifests.save(replace(manifest, entries=(replace(first, sha256="0" * 64), *rest)))

    with pytest.raises(IntegrityError, match="M-001"):
        sets.load(SCIENTIST, "ds-1")


def test_dataset_id_is_generated_and_never_reused(training_sets, corpus, manifests):
    sets = training_sets()
    generated = sets.build(SCIENTIST, corpus.matters).dataset_id

    assert generated in manifests
    with pytest.raises(ValueError, match="already exists"):
        sets.build(SCIENTIST, corpus.matters, generated)


def test_matter_listed_twice_is_exported_once(training_sets, corpus):
    manifest = training_sets().build(SCIENTIST, [*corpus.matters, *corpus.matters], "ds-1")

    assert sorted(entry.document_id for entry in manifest.entries) == sorted(EXPORTABLE)


def test_registry_rejects_unsafe_ids(manifests):
    with pytest.raises(ValueError, match="unsafe"):
        manifests.load("../outside")
