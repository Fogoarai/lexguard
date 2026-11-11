import pytest

from lexguard.crypto import KeyRing
from lexguard.domain import Zone
from lexguard.errors import IntegrityError, KeyUnavailable
from lexguard.storage import ZonedStore


@pytest.fixture
def store(tmp_path):
    keys = KeyRing.generate()
    keys.enroll("M-001")
    keys.enroll("M-002")
    return ZonedStore(tmp_path, keys)


def test_round_trip(store):
    store.put(Zone.RAW, "M-001", "doc.txt", b"contenido")
    assert store.get(Zone.RAW, "M-001", "doc.txt") == b"contenido"
    assert store.names(Zone.RAW, "M-001") == ["doc.txt"]


def test_nothing_is_stored_in_the_clear(store, tmp_path):
    store.put(Zone.RAW, "M-001", "doc.txt", b"contenido muy reconocible")
    assert b"reconocible" not in (tmp_path / "raw" / "M-001" / "doc.txt").read_bytes()


@pytest.mark.parametrize(
    "target",
    [("deidentified", "M-001", "doc.txt"), ("raw", "M-002", "doc.txt"), ("raw", "M-001", "b.txt")],
)
def test_blob_cannot_be_moved(store, tmp_path, target):
    store.put(Zone.RAW, "M-001", "doc.txt", b"contenido")
    zone, matter_id, name = target
    destination = tmp_path / zone / matter_id / name
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes((tmp_path / "raw" / "M-001" / "doc.txt").read_bytes())
    with pytest.raises(IntegrityError):
        store.get(Zone(zone), matter_id, name)


@pytest.mark.parametrize("name", ["..", "../x", "a/b", "", ".hidden"])
def test_rejects_unsafe_names(store, name):
    with pytest.raises(ValueError, match="unsafe"):
        store.put(Zone.RAW, "M-001", name, b"x")


def test_empty_listing_for_untouched_matter(store):
    assert store.names(Zone.SERVING, "M-002") == []


def test_delete_is_idempotent(store):
    store.put(Zone.RAW, "M-001", "doc.txt", b"x")
    store.delete(Zone.RAW, "M-001", "doc.txt")
    store.delete(Zone.RAW, "M-001", "doc.txt")
    assert store.names(Zone.RAW, "M-001") == []


def test_purge_shreds_every_zone_and_spares_other_matters(store):
    for zone in Zone:
        store.put(zone, "M-001", "doc.txt", b"x")
    store.put(Zone.RAW, "M-002", "doc.txt", b"y")

    store.purge("M-001")

    assert all(store.names(zone, "M-001") == [] for zone in Zone)
    with pytest.raises(KeyUnavailable):
        store.put(Zone.RAW, "M-001", "doc.txt", b"x")
    assert store.get(Zone.RAW, "M-002", "doc.txt") == b"y"
