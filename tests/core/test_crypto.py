import pytest

from lexguard.crypto import KeyRing, seal, unseal
from lexguard.errors import IntegrityError, KeyUnavailable


@pytest.fixture
def keys():
    ring = KeyRing.generate()
    ring.enroll("M-001")
    ring.enroll("M-002")
    return ring


def test_seal_round_trip(keys):
    key = keys.key_for("M-001", "raw")
    assert unseal(key, seal(key, b"secreto", b"ctx"), b"ctx") == b"secreto"


def test_unseal_rejects_wrong_context(keys):
    key = keys.key_for("M-001", "raw")
    with pytest.raises(IntegrityError):
        unseal(key, seal(key, b"secreto", b"ctx"), b"other")


def test_unseal_rejects_tampering(keys):
    key = keys.key_for("M-001", "raw")
    blob = bytearray(seal(key, b"secreto"))
    blob[-1] ^= 1
    with pytest.raises(IntegrityError):
        unseal(key, bytes(blob))


def test_keys_are_separated_by_matter_and_scope(keys):
    derived = {
        keys.key_for("M-001", "raw"),
        keys.key_for("M-001", "deidentified"),
        keys.key_for("M-002", "raw"),
    }
    assert len(derived) == 3


def test_key_derivation_is_stable(keys):
    assert keys.key_for("M-001", "raw") == keys.key_for("M-001", "raw")


def test_unknown_matter_has_no_key(keys):
    with pytest.raises(KeyUnavailable):
        keys.key_for("M-999", "raw")


def test_shredded_matter_cannot_be_read_or_re_enrolled(keys):
    keys.shred("M-001")
    assert "M-001" not in keys
    with pytest.raises(KeyUnavailable):
        keys.key_for("M-001", "raw")
    with pytest.raises(KeyUnavailable):
        keys.enroll("M-001")


def test_service_key_belongs_to_the_ring_and_to_no_matter(keys):
    audit = keys.service_key("audit")

    assert audit == keys.service_key("audit")
    assert audit != keys.service_key("ledger")
    assert audit != KeyRing.generate().service_key("audit")
    keys.shred("M-001")
    assert keys.service_key("audit") == audit
