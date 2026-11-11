from collections.abc import Sequence
from typing import Protocol

import numpy as np

from lexguard.domain import AccessRequest, AuditEvent, Decision, Document, PiiSpan


class PolicyDecisionPoint(Protocol):
    def decide(self, request: AccessRequest) -> Decision: ...


class AuditLog(Protocol):
    def record(self, event: AuditEvent) -> None: ...


class Deidentifier(Protocol):
    def detect(self, text: str) -> Sequence[PiiSpan]: ...

    def pseudonymize(self, document: Document) -> Document: ...

    def reidentify(self, text: str, matter_id: str) -> str: ...


class Embedder(Protocol):
    def embed(self, texts: Sequence[str]) -> np.ndarray: ...


class LanguageModel(Protocol):
    name: str
    external: bool

    def complete(self, prompt: str) -> str: ...
