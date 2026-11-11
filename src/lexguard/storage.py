import re
import shutil
from pathlib import Path

from lexguard.crypto import KeyRing, seal, unseal
from lexguard.domain import Zone

_SAFE_NAME = re.compile(r"\w[\w.-]*")


class ZonedStore:
    """Encrypted object store partitioned by zone and matter.

    Each zone uses its own key derived from the matter root, and the ciphertext is bound
    to its location, so a blob cannot be read from, or replayed into, another zone or matter.
    """

    def __init__(self, root: Path, keys: KeyRing) -> None:
        self._root = root
        self._keys = keys

    def put(self, zone: Zone, matter_id: str, name: str, data: bytes) -> None:
        path = self._path(zone, matter_id, name)
        key = self._keys.key_for(matter_id, zone)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(seal(key, data, _context(zone, matter_id, name)))

    def get(self, zone: Zone, matter_id: str, name: str) -> bytes:
        path = self._path(zone, matter_id, name)
        key = self._keys.key_for(matter_id, zone)
        return unseal(key, path.read_bytes(), _context(zone, matter_id, name))

    def names(self, zone: Zone, matter_id: str) -> list[str]:
        directory = self._path(zone, matter_id)
        return sorted(p.name for p in directory.iterdir()) if directory.is_dir() else []

    def delete(self, zone: Zone, matter_id: str, name: str) -> None:
        self._path(zone, matter_id, name).unlink(missing_ok=True)

    def purge(self, matter_id: str) -> None:
        """Crypto-shred the matter, then remove its ciphertext from every zone."""
        self._keys.shred(matter_id)
        for zone in Zone:
            shutil.rmtree(self._path(zone, matter_id), ignore_errors=True)

    def _path(self, zone: Zone, *parts: str) -> Path:
        for part in parts:
            if not _SAFE_NAME.fullmatch(part):
                raise ValueError(f"unsafe path component: {part!r}")
        return self._root.joinpath(zone, *parts)


def _context(zone: Zone, matter_id: str, name: str) -> bytes:
    return f"{zone}/{matter_id}/{name}".encode()
