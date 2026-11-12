from collections.abc import Callable, Collection, Sequence
from typing import Protocol

import numpy as np

from lexguard.domain import AccessRequest, AuditEvent, Decision, Document, PiiKind, PiiSpan
from lexguard.settings import ModelLocation


class PolicyDecisionPoint(Protocol):
    def decide(self, request: AccessRequest) -> Decision: ...


class AuditLog(Protocol):
    def record(self, event: AuditEvent) -> None: ...


class Deidentifier(Protocol):
    def detect(self, text: str) -> Sequence[PiiSpan]: ...

    def pseudonymize(self, document: Document) -> Document: ...

    def tokenize(self, text: str, matter_id: str) -> str:
        """Swap in the tokens the matter can still reveal; nothing is recorded.

        A value the matter never saw or has forgotten stays as written, so the two cannot
        be told apart.
        """

    def reidentify(self, text: str, matter_id: str) -> str: ...


class TokenVault(Protocol):
    def token_for(self, kind: PiiKind, value: str, matter_id: str) -> str: ...

    def mention_tokens(self, kind: PiiKind, value: str, matter_id: str) -> Collection[str]:
        """The value's token and those of the shorter forms a mention of it may carry."""

    def reveal(self, token: str, matter_id: str) -> str: ...

    def forget(self, token: str, matter_id: str) -> None: ...


class Embedder(Protocol):
    name: str
    location: ModelLocation

    def embed(self, texts: Sequence[str]) -> np.ndarray: ...


class LanguageModel(Protocol):
    name: str
    location: ModelLocation

    def complete(self, prompt: str) -> str: ...


class DatasetRegistry(Protocol):
    def retract(
        self, used: Callable[[str, str], bool], reason: str, *, drop_copies: bool = False
    ) -> Sequence[str]:
        """Taint every dataset holding a document for which `used(matter_id, document_id)`.

        Returns their ids. With `drop_copies`, the stored copies of those documents go too.
        """
