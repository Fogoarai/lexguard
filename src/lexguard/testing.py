"""In-memory stand-ins for the ports, so each module can be built and tested on its own."""

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from operator import attrgetter

from lexguard import tokens
from lexguard.domain import AccessRequest, AuditEvent, Decision, Document, PiiSpan
from lexguard.synthetic import Corpus


class AllowAll:
    def decide(self, request: AccessRequest) -> Decision:
        return Decision(allowed=True, reason="allow-all")


class DenyAll:
    def decide(self, request: AccessRequest) -> Decision:
        return Decision(allowed=False, reason="deny-all")


class MembershipPolicy:
    """Allows a request only when the principal is assigned to the resource's matter."""

    def decide(self, request: AccessRequest) -> Decision:
        assigned = request.resource.matter_id in request.principal.matters
        return Decision(allowed=assigned, reason="assigned" if assigned else "not assigned")


@dataclass(slots=True)
class MemoryAuditLog:
    events: list[AuditEvent] = field(default_factory=list)

    def record(self, event: AuditEvent) -> None:
        self.events.append(event)


class LabelDeidentifier:
    """Pseudonymizes with the ground-truth spans of a synthetic corpus instead of detecting."""

    def __init__(self, corpus: Corpus) -> None:
        self._spans = {labeled.document.id: labeled.spans for labeled in corpus.documents}
        self._originals: dict[tuple[str, str], str] = {}

    def detect(self, text: str) -> Sequence[PiiSpan]:
        return ()

    def pseudonymize(self, document: Document) -> Document:
        key = hashlib.sha256(document.matter_id.encode()).digest()
        text = document.text
        for span in sorted(self._spans[document.id], key=attrgetter("start"), reverse=True):
            token = tokens.pseudonym(span.kind, span.text, key)
            self._originals[document.matter_id, token] = span.text
            text = text[: span.start] + token + text[span.end :]
        return replace(document, text=text)

    def reidentify(self, text: str, matter_id: str) -> str:
        return tokens.TOKEN.sub(
            lambda match: self._originals.get((matter_id, match.group()), match.group()), text
        )
