from collections.abc import Iterable, Mapping
from dataclasses import dataclass

from lexguard.crypto import KeyRing
from lexguard.documents import DocumentStore
from lexguard.domain import AccessRequest, Action, Document, Principal, Purpose, Resource, Zone
from lexguard.guard import Guard
from lexguard.ports import Deidentifier


@dataclass(frozen=True, slots=True)
class Landing:
    accepted: tuple[str, ...]
    refused: Mapping[str, str]


@dataclass(frozen=True, slots=True)
class Pipeline:
    """Moves documents from the lake into the raw zone and on to the de-identified zone.

    The lake stays the system of record: the raw zone is a landing strip, emptied once a
    document has been pseudonymized.
    """

    guard: Guard
    keys: KeyRing
    documents: DocumentStore
    deidentifier: Deidentifier

    def land(
        self,
        principal: Principal,
        batch: Iterable[Document],
        purpose: Purpose = Purpose.LEGAL_ADVICE,
    ) -> Landing:
        """Copy only what the policy admits; everything else never leaves the lake."""
        accepted: list[str] = []
        refused: dict[str, str] = {}
        for document in batch:
            request = _request(principal, Action.INGEST, purpose, document)
            decision = self.guard.evaluate(request, document=document.id)
            if not decision.allowed:
                refused[document.id] = decision.reason
                continue
            self.keys.enroll(document.matter_id)
            self.documents.save(Zone.RAW, document)
            accepted.append(document.id)
        return Landing(tuple(accepted), refused)

    def promote(
        self,
        principal: Principal,
        matter_id: str,
        purpose: Purpose = Purpose.LEGAL_ADVICE,
        retain_raw: bool = False,
    ) -> tuple[str, ...]:
        promoted: list[str] = []
        for document in list(self.documents.scan(Zone.RAW, matter_id)):
            request = _request(principal, Action.READ_RAW, purpose, document)
            self.guard.authorize(request, document=document.id)
            self.documents.save(Zone.DEIDENTIFIED, self.deidentifier.pseudonymize(document))
            if not retain_raw:
                self.documents.remove(Zone.RAW, matter_id, document.id)
            promoted.append(document.id)
        return tuple(promoted)


def _request(
    principal: Principal, action: Action, purpose: Purpose, document: Document
) -> AccessRequest:
    resource = Resource(Zone.RAW, document.matter_id, document.sensitivity)
    return AccessRequest(principal, action, purpose, resource)
