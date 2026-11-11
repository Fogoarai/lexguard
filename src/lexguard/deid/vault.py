import re
import shutil
from collections.abc import Iterator
from pathlib import Path

from lexguard import tokens
from lexguard.crypto import KeyRing, seal, unseal
from lexguard.deid.canonical import NAME_PARTICLES, canonicalize
from lexguard.domain import PiiKind
from lexguard.errors import LexguardError

_SAFE_NAME = re.compile(r"\w[\w.-]*")
_FORGOTTEN = ""


class TokenNotFound(LexguardError, KeyError):
    """The token was never issued in this matter, or has been forgotten."""


class SealedTokenVault:
    """Token to original value, one sealed file per token under `<root>/<matter>/`.

    Entries are sealed with the matter's "vault" key, so they die with the matter's root key.
    A forgotten token keeps an empty entry, which stops its value from ever being recorded
    again. Forgetting a person also forgets every shorter form of the name, because partial
    mentions of that person carry those tokens.
    """

    def __init__(self, keys: KeyRing, root: Path) -> None:
        self._keys = keys
        self._root = root

    def token_for(self, kind: PiiKind, value: str, matter_id: str) -> str:
        """The token this value has in the matter. Computes only; nothing is stored."""
        key = self._keys.key_for(matter_id, "pseudonym")
        return tokens.pseudonym(kind, canonicalize(kind, value), key)

    def remember(self, kind: PiiKind, value: str, matter_id: str) -> str:
        token = self.token_for(kind, value, matter_id)
        if self._read(token, matter_id) is None:
            self._write(token, matter_id, value)
        return token

    def reveal(self, token: str, matter_id: str) -> str:
        value = self._read(token, matter_id)
        if not value:
            raise TokenNotFound(token)
        return value

    def forget(self, token: str, matter_id: str) -> None:
        value = self._read(token, matter_id)
        self._write(token, matter_id, _FORGOTTEN)
        if value and tokens.kind_of(token) is PiiKind.PERSON:
            for form in _shorter_forms(value):
                self._write(self.token_for(PiiKind.PERSON, form, matter_id), matter_id, _FORGOTTEN)

    def discard(self, matter_id: str) -> None:
        """Delete a matter's entries. Once its key is shredded they are unreadable anyway."""
        shutil.rmtree(self._directory(matter_id), ignore_errors=True)

    def _read(self, token: str, matter_id: str) -> str | None:
        key = self._keys.key_for(matter_id, "vault")
        path = self._path(token, matter_id)
        if not path.exists():
            return None
        return unseal(key, path.read_bytes(), _context(token, matter_id)).decode()

    def _write(self, token: str, matter_id: str, value: str) -> None:
        key = self._keys.key_for(matter_id, "vault")
        path = self._path(token, matter_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(seal(key, value.encode(), _context(token, matter_id)))

    def _path(self, token: str, matter_id: str) -> Path:
        if not tokens.TOKEN.fullmatch(token):
            raise ValueError(f"not a pseudonym token: {token!r}")
        return self._directory(matter_id) / f"{token[1:-1]}.token"

    def _directory(self, matter_id: str) -> Path:
        if not _SAFE_NAME.fullmatch(matter_id):
            raise ValueError(f"unsafe matter id: {matter_id!r}")
        return self._root / matter_id


def _context(token: str, matter_id: str) -> bytes:
    return f"vault/{matter_id}/{token}".encode()


def _shorter_forms(name: str) -> Iterator[str]:
    words = name.split()
    for size in range(1, len(words)):
        for start in range(len(words) - size + 1):
            form = words[start : start + size]
            if NAME_PARTICLES.isdisjoint({form[0].casefold(), form[-1].casefold()}):
                yield " ".join(form)
