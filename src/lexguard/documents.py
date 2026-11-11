import json
from collections.abc import Iterator
from dataclasses import asdict
from datetime import datetime

from lexguard.domain import Document, DocumentKind, Sensitivity, Zone
from lexguard.storage import ZonedStore

_SUFFIX = ".doc.json"


def encode(document: Document) -> bytes:
    fields = asdict(document) | {"created_at": document.created_at.isoformat()}
    return json.dumps(fields, ensure_ascii=False).encode()


def decode(data: bytes) -> Document:
    fields = json.loads(data)
    return Document(
        id=fields["id"],
        matter_id=fields["matter_id"],
        kind=DocumentKind(fields["kind"]),
        sensitivity=Sensitivity(fields["sensitivity"]),
        text=fields["text"],
        created_at=datetime.fromisoformat(fields["created_at"]),
    )


class DocumentStore:
    """Documents at rest in a zone. Other artifacts in the same zone use their own suffix."""

    def __init__(self, store: ZonedStore) -> None:
        self._store = store

    def save(self, zone: Zone, document: Document) -> None:
        self._store.put(zone, document.matter_id, document.id + _SUFFIX, encode(document))

    def load(self, zone: Zone, matter_id: str, document_id: str) -> Document:
        return decode(self._store.get(zone, matter_id, document_id + _SUFFIX))

    def scan(self, zone: Zone, matter_id: str) -> Iterator[Document]:
        for name in self._store.names(zone, matter_id):
            if name.endswith(_SUFFIX):
                yield decode(self._store.get(zone, matter_id, name))

    def remove(self, zone: Zone, matter_id: str, document_id: str) -> None:
        self._store.delete(zone, matter_id, document_id + _SUFFIX)
