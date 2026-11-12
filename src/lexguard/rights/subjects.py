from collections import Counter
from collections.abc import Collection, Iterable
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
from lexguard.ports import DatasetRegistry, TokenVault
from lexguard.storage import ZonedStore


@dataclass(frozen=True, slots=True)
class Identifier:
    kind: PiiKind
    value: str


@dataclass(frozen=True, slots=True)
class Hit:
    """A document that mentions the subject, described without any of its text.

    `partial` marks a match on nothing but a shorter form of a name ("la señora Fuentes"),
    which a namesake in the same matter would share: a person has to confirm it.
    """

    matter_id: str
    document_id: str
    kind: DocumentKind
    sensitivity: Sensitivity
    mentions: int
    partial: bool = False


@dataclass(frozen=True, slots=True)
class Location:
    hits: tuple[Hit, ...]
    unsearched: tuple[tuple[str, Sensitivity], ...]


@dataclass(frozen=True, slots=True)
class Erasure:
    at: datetime
    hits: tuple[Hit, ...]
    tokens_forgotten: int
    removed: tuple[Hit, ...]
    tainted_datasets: tuple[str, ...]
    refused: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class SubjectRequests:
    """Access and erasure requests of a data subject, answered from the de-identified zone.

    A subject is the set of identifiers they supply. Each one maps to a different token in
    every matter, so the search runs matter by matter and matches nothing but their own
    tokens. In a matter that holds one of those, the shorter forms of a name count too,
    as partial hits. A matter whose key has been shredded holds nothing to find and is
    passed over.
    """

    store: ZonedStore
    vault: TokenVault
    guard: Guard
    datasets: DatasetRegistry

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
                    hits += self._search(matter_id, *self._tokens(subject, matter_id), allowed)
        return Location(tuple(hits), tuple(unsearched))

    def erase(
        self,
        principal: Principal,
        subject: Collection[Identifier],
        matter_ids: Iterable[str],
        *,
        remove_documents: bool = False,
        confirmed: Collection[Hit] = (),
    ) -> Erasure:
        """Forget the subject's tokens and taint every dataset that used their documents.

        A matter is erased only if every sensitivity compartment admits it: a token cannot
        be forgotten for part of a matter. With `remove_documents`, the documents leave the
        de-identified zone, a copy retained in the raw zone goes with them and the shards
        that copied them leave the training zone. A partial hit loses its tokens like any
        other but keeps its document, which may be a namesake's, unless a person who
        reviewed it lists it in `confirmed`. A document still waiting in the raw zone is
        not searched; by the time it is promoted the subject's tokens are forgotten.
        """
        zones = (Zone.DEIDENTIFIED, Zone.RAW) if remove_documents else (Zone.DEIDENTIFIED,)
        hits: list[Hit] = []
        forgotten: list[tuple[str, str]] = []
        refused: list[str] = []
        for matter_id in dict.fromkeys(matter_ids):
            decisions = [
                self.guard.evaluate(_request(principal, Action.ERASE, matter_id, sensitivity, zone))
                for zone in zones
                for sensitivity in Sensitivity
            ]
            if not all(decision.allowed for decision in decisions):
                refused.append(matter_id)
                continue
            with suppress(KeyUnavailable):
                own, shorter = self._tokens(subject, matter_id)
                found = self._search(matter_id, own, shorter, set(Sensitivity))
                hits += found
                forgotten += [(token, matter_id) for token in (own | shorter if found else own)]

        located = {(hit.matter_id, hit.document_id) for hit in hits}
        tainted = self.datasets.retract(
            lambda matter_id, document_id: (matter_id, document_id) in located,
            "data subject erasure",
            drop_copies=remove_documents,
        )
        for token, matter_id in forgotten:
            self.vault.forget(token, matter_id)
        removed: list[Hit] = []
        if remove_documents:
            documents = DocumentStore(self.store)
            reviewed = {(hit.matter_id, hit.document_id) for hit in confirmed}
            removed = [
                hit
                for hit in hits
                if not hit.partial or (hit.matter_id, hit.document_id) in reviewed
            ]
            for hit in removed:
                for zone in zones:
                    documents.remove(zone, hit.matter_id, hit.document_id)

        return Erasure(
            at=datetime.now(UTC),
            hits=tuple(hits),
            tokens_forgotten=len(forgotten),
            removed=tuple(removed),
            tainted_datasets=tuple(tainted),
            refused=tuple(refused),
        )

    def _tokens(self, subject: Collection[Identifier], matter_id: str) -> tuple[set[str], set[str]]:
        """The tokens of the identifiers as given, and those of their shorter forms."""
        own = {self.vault.token_for(i.kind, i.value, matter_id) for i in subject}
        mentions = {
            token
            for i in subject
            for token in self.vault.mention_tokens(i.kind, i.value, matter_id)
        }
        return own, mentions - own

    def _search(
        self, matter_id: str, own: set[str], shorter: set[str], allowed: set[Sensitivity]
    ) -> list[Hit]:
        """A shorter form says nothing in a matter where no full identifier appears."""
        found = []
        for document in DocumentStore(self.store).scan(Zone.DEIDENTIFIED, matter_id):
            if document.sensitivity not in allowed:
                continue
            seen = Counter(match.group() for match in tokens.find(document.text))
            exact = sum(seen[token] for token in own)
            partial = sum(seen[token] for token in shorter)
            if exact or partial:
                found.append(
                    Hit(
                        matter_id,
                        document.id,
                        document.kind,
                        document.sensitivity,
                        mentions=exact + partial,
                        partial=not exact,
                    )
                )
        return found if any(not hit.partial for hit in found) else []


def _request(
    principal: Principal,
    action: Action,
    matter_id: str,
    sensitivity: Sensitivity,
    zone: Zone = Zone.DEIDENTIFIED,
) -> AccessRequest:
    resource = Resource(zone, matter_id, sensitivity)
    return AccessRequest(principal, action, Purpose.DATA_SUBJECT_REQUEST, resource)
