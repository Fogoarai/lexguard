import os
from typing import Self

from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.hkdf import HKDF

from lexguard.errors import IntegrityError, KeyUnavailable

KEY_SIZE = 32
NONCE_SIZE = 12


def seal(key: bytes, plaintext: bytes, context: bytes = b"") -> bytes:
    nonce = os.urandom(NONCE_SIZE)
    return nonce + AESGCM(key).encrypt(nonce, plaintext, context)


def unseal(key: bytes, blob: bytes, context: bytes = b"") -> bytes:
    nonce, ciphertext = blob[:NONCE_SIZE], blob[NONCE_SIZE:]
    try:
        return AESGCM(key).decrypt(nonce, ciphertext, context)
    except InvalidTag as exc:
        raise IntegrityError("ciphertext failed authentication") from exc


def derive(key: bytes, scope: str) -> bytes:
    return HKDF(hashes.SHA256(), KEY_SIZE, salt=None, info=scope.encode()).derive(key)


class KeyRing:
    """Envelope encryption with one root key per matter.

    Stands in for a KMS or HSM. Every use of a matter's data goes through a key derived
    from its root, so shredding the root makes all of that matter's ciphertext unreadable.
    """

    def __init__(self, master_key: bytes) -> None:
        self._master = master_key
        self._wrapped: dict[str, bytes] = {}
        self._shredded: set[str] = set()

    @classmethod
    def generate(cls) -> Self:
        return cls(AESGCM.generate_key(bit_length=256))

    def __contains__(self, matter_id: str) -> bool:
        return matter_id in self._wrapped

    def enroll(self, matter_id: str) -> None:
        if matter_id in self._shredded:
            raise KeyUnavailable(f"matter {matter_id} was shredded")
        if matter_id not in self._wrapped:
            root = os.urandom(KEY_SIZE)
            self._wrapped[matter_id] = seal(self._master, root, matter_id.encode())

    def key_for(self, matter_id: str, scope: str) -> bytes:
        try:
            wrapped = self._wrapped[matter_id]
        except KeyError:
            raise KeyUnavailable(f"no key for matter {matter_id}") from None
        return derive(unseal(self._master, wrapped, matter_id.encode()), scope)

    def service_key(self, scope: str) -> bytes:
        """A key no matter owns, for what must outlive every matter, such as the audit chain."""
        return derive(self._master, f"service/{scope}")

    def shred(self, matter_id: str) -> None:
        self._wrapped.pop(matter_id, None)
        self._shredded.add(matter_id)
