from collections.abc import Sequence
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

    def reidentify(self, text: str, matter_id: str) -> str: ...


class TokenVault(Protocol):
    def token_for(self, kind: PiiKind, value: str, matter_id: str) -> str: ...

    def reveal(self, token: str, matter_id: str) -> str: ...

    def forget(self, token: str, matter_id: str) -> None: ...


class Embedder(Protocol):
    def embed(self, texts: Sequence[str]) -> np.ndarray: ...


class LanguageModel(Protocol):
    name: str
    location: ModelLocation

    def complete(self, prompt: str) -> str: ...
