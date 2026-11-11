from collections.abc import Collection, Iterable, Iterator
from contextlib import suppress
from dataclasses import dataclass
from datetime import UTC, datetime

from lexguard import tokens
from lexguard.documents import DocumentStore
from lexguard.domain import (
    AccessRequest,
    Action,
    DocumentKind,
    PiiKind,
    Principal,
    Purpose,
    Resource,
    Sensitivity,
    Zone,
)
from lexguard.errors import KeyUnavailable
from lexguard.guard import Guard
from lexguard.ports import TokenVault
from lexguard.storage import ZonedStore
from lexguard.training.dataset import shard_name
from lexguard.training.manifest import Entry, ManifestRegistry


@dataclass(frozen=True, slots=True)
class Identifier:
    kind: PiiKind
    value: str


@dataclass(frozen=True, slots=True)
class Hit:
    """A document that mentions the subject, described without any of its text."""

    matter_id: str
    document_id: str
    kind: DocumentKind
    sensitivity: Sensitivity
    mentions: int


@dataclass(frozen=True, slots=True)
class Location:
    hits: tuple[Hit, ...]
    unsearched: tuple[tuple[str, Sensitivity], ...]


@dataclass(frozen=True, slots=True)
class Erasure:
    at: datetime
    hits: tuple[Hit, ...]
    tokens_forgotten: int
    documents_removed: bool
    tainted_datasets: tuple[str, ...]
    refused: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class SubjectRequests:
    """Access and erasure requests of a data subject, answered from the de-identified zone.

    A subject is the set of identifiers they supply. Each one maps to a different token in
    every matter, so the search runs matter by matter and matches nothing but their own
    tokens. A matter whose key has been shredded holds nothing to find and is passed over.
    """

    store: ZonedStore
    vault: TokenVault
    guard: Guard
    manifests: ManifestRegistry

    def locate(
        self, principal: Principal, subject: Collection[Identifier], matter_ids: Iterable[str]
    ) -> Location:
        hits: list[Hit] = []
        unsearched: list[tuple[str, Sensitivity]] = []
        for matter_id in dict.fromkeys(matter_ids):
            allowed = set()
            for sensitivity in Sensitivity:
                request = _request(principal, Action.READ_DEIDENTIFIED, matter_id, sensitivity)
                if self.guard.evaluate(request).allowed:
                    allowed.add(sensitivity)
                else:
                    unsearched.append((matter_id, sensitivity))
            if allowed:
                with suppress(KeyUnavailable):
                    hits += self._search(matter_id, self._tokens(subject, matter_id), allowed)
        return Location(tuple(hits), tuple(unsearched))

    def erase(
        self,
        principal: Principal,
        subject: Collection[Identifier],
        matter_ids: Iterable[str],
        *,
        remove_documents: bool = False,
    ) -> Erasure:
        """Forget the subject's tokens and taint every dataset that used their documents.

        A matter is erased only if every sensitivity compartment admits it: a token cannot
        be forgotten for part of a matter. With `remove_documents`, the documents leave the
        de-identified zone and the shards that copied them leave the training zone.
        """
        hits: list[Hit] = []
        forgotten: list[tuple[str, str]] = []
        refused: list[str] = []
        for matter_id in dict.fromkeys(matter_ids):
            decisions = [
                self.guard.evaluate(_request(principal, Action.ERASE, matter_id, sensitivity))
                for sensitivity in Sensitivity
            ]
            if not all(decision.allowed for decision in decisions):
                refused.append(matter_id)
                continue
            with suppress(KeyUnavailable):
                subject_tokens = self._tokens(subject, matter_id)
                hits += self._search(matter_id, subject_tokens, set(Sensitivity))
                forgotten += [(token, matter_id) for token in subject_tokens]

        located = {(hit.matter_id, hit.document_id) for hit in hits}

        def affected(entry: Entry) -> bool:
            return (entry.matter_id, entry.document_id) in located

        tainted = self.manifests.taint(affected, "data subject erasure")
        for token, matter_id in forgotten:
            self.vault.forget(token, matter_id)
        if remove_documents:
            documents = DocumentStore(self.store)
            for hit in hits:
                documents.remove(Zone.DEIDENTIFIED, hit.matter_id, hit.document_id)
            for manifest in tainted:
                for matter_id in {entry.matter_id for entry in manifest.entries if affected(entry)}:
                    self.store.delete(Zone.TRAINING, matter_id, shard_name(manifest.dataset_id))

        return Erasure(
            at=datetime.now(UTC),
            hits=tuple(hits),
            tokens_forgotten=len(forgotten),
            documents_removed=remove_documents,
            tainted_datasets=tuple(manifest.dataset_id for manifest in tainted),
            refused=tuple(refused),
        )

    def _tokens(self, subject: Collection[Identifier], matter_id: str) -> set[str]:
        return {self.vault.token_for(i.kind, i.value, matter_id) for i in subject}

    def _search(
        self, matter_id: str, subject_tokens: set[str], allowed: set[Sensitivity]
    ) -> Iterator[Hit]:
        for document in DocumentStore(self.store).scan(Zone.DEIDENTIFIED, matter_id):
            if document.sensitivity not in allowed:
                continue
            mentions = sum(match.group() in subject_tokens for match in tokens.find(document.text))
            if mentions:
                yield Hit(matter_id, document.id, document.kind, document.sensitivity, mentions)


def _request(
    principal: Principal, action: Action, matter_id: str, sensitivity: Sensitivity
) -> AccessRequest:
    resource = Resource(Zone.DEIDENTIFIED, matter_id, sensitivity)
    return AccessRequest(principal, action, Purpose.DATA_SUBJECT_REQUEST, resource)
