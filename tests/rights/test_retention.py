import shutil

import pytest

from lexguard.crypto import KeyRing
from lexguard.documents import DocumentStore
from lexguard.domain import Action, Principal, Purpose, Role, Zone
from lexguard.errors import AccessDenied, KeyUnavailable
from lexguard.guard import Guard
from lexguard.rights.retention import ClosureIncomplete, close_matter
from lexguard.storage import ZonedStore
from lexguard.testing import AllowAll, DenyAll
from lexguard.training.dataset import DatasetTainted, shard_name

OFFICER = Principal("dpo-1", Role.PRIVACY_OFFICER)
SCIENTIST = Principal("scientist-1", Role.DATA_SCIENTIST)


@pytest.fixture
def guard(audit):
    return Guard(AllowAll(), audit)


def test_closed_matter_is_unreadable_even_from_a_backup(
    store, guard, manifests, training_sets, corpus, tmp_path
):
    training_sets.build(SCIENTIST, corpus.matters, "ds-1")
    lake, backup = tmp_path / "lake", tmp_path / "backup"
    shutil.copytree(lake, backup)
    documents = DocumentStore(store)

    closure = close_matter(store, guard, training_sets, OFFICER, "M-001")

    assert closure.destroyed == {
        Zone.RAW: 7,
        Zone.DEIDENTIFIED: 7,
        Zone.SERVING: 0,
        Zone.TRAINING: 1,
    }
    assert all(store.names(zone, "M-001") == [] for zone in Zone)
    assert not any(list(documents.scan(zone, "M-001")) for zone in Zone)

    shutil.copytree(backup, lake, dirs_exist_ok=True)
    for zone in (Zone.RAW, Zone.DEIDENTIFIED):
        with pytest.raises(KeyUnavailable):
            documents.load(zone, "M-001", "M-001-D001")
    with pytest.raises(KeyUnavailable):
        store.get(Zone.TRAINING, "M-001", shard_name("ds-1"))
    assert documents.load(Zone.RAW, "M-002", "M-002-D001")
    assert store.get(Zone.TRAINING, "M-003", shard_name("ds-1"))


def test_closing_a_matter_taints_the_datasets_that_used_it(
    store, guard, manifests, training_sets, corpus
):
    training_sets.build(SCIENTIST, corpus.matters, "ds-all")
    training_sets.build(SCIENTIST, [corpus.matter_index["M-003"]], "ds-other")

    closure = close_matter(store, guard, training_sets, OFFICER, "M-001")

    assert closure.tainted_datasets == ("ds-all",)
    assert manifests.load("ds-all").taints[0].reason == "matter closed"
    assert not manifests.load("ds-other").tainted
    with pytest.raises(DatasetTainted):
        training_sets.load(SCIENTIST, "ds-all")


def test_closure_is_authorized_as_a_retention_erase_of_the_raw_zone(
    store, guard, training_sets, audit
):
    close_matter(store, guard, training_sets, OFFICER, "M-004")

    (event,) = audit.events
    assert (event.action, event.zone, event.purpose, event.matter_id) == (
        Action.ERASE,
        Zone.RAW,
        Purpose.RETENTION,
        "M-004",
    )


def test_denied_closure_destroys_nothing(store, training_sets, audit):
    with pytest.raises(AccessDenied):
        close_matter(store, Guard(DenyAll(), audit), training_sets, OFFICER, "M-001")

    assert len(store.names(Zone.RAW, "M-001")) == 7
    assert DocumentStore(store).load(Zone.RAW, "M-001", "M-001-D001")


def test_closure_that_leaves_the_key_alive_is_reported(tmp_path, guard, training_sets):
    class KeepsKey(ZonedStore):
        def purge(self, matter_id):
            for zone in Zone:
                for name in self.names(zone, matter_id):
                    self.delete(zone, matter_id, name)

    keys = KeyRing.generate()
    keys.enroll("M-001")
    store = KeepsKey(tmp_path / "sloppy", keys)
    store.put(Zone.RAW, "M-001", "doc.txt", b"contenido")

    with pytest.raises(ClosureIncomplete, match="raw"):
        close_matter(store, guard, training_sets, OFFICER, "M-001")
