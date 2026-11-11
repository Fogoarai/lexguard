import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Protocol

from lexguard import tokens
from lexguard.domain import (
    AccessRequest,
    Action,
    Chunk,
    Obligation,
    Principal,
    Purpose,
    Sensitivity,
)
from lexguard.guard import Guard
from lexguard.ports import Deidentifier, LanguageModel
from lexguard.rag.index import Retrieval, SecureIndex
from lexguard.rag.prompt import NO_ANSWER, build

_MARKER = re.compile(r"\[(\d+)\]")


class ModelGateway(Protocol):
    """What answering needs from the egress gateway (`lexguard.egress.EgressGateway`)."""

    def complete(
        self,
        model: LanguageModel,
        prompt: str,
        *,
        principal: Principal,
        purpose: Purpose,
        compartments: Iterable[tuple[str, Sensitivity]],
        obligations: Iterable[Obligation] = (),
    ) -> str: ...


@dataclass(frozen=True, slots=True)
class Citation:
    document_id: str
    ordinal: int
    matter_id: str
    passage: str


@dataclass(frozen=True, slots=True)
class Answer:
    """A marker `[n]` in `text` refers to `citations[n - 1]`."""

    text: str
    citations: tuple[Citation, ...]
    reidentified: bool


@dataclass(frozen=True, slots=True)
class Answerer:
    """Answers from retrieved passages, pseudonymized unless re-identification is cleared.

    The context is every passage sent to the model, cited or not. Its identities are restored
    only if no retrieve decision behind it carried `KEEP_PSEUDONYMS` and the guard allows
    `REIDENTIFY` on each of its compartments; otherwise the whole answer keeps its tokens.
    """

    index: SecureIndex
    gateway: ModelGateway
    model: LanguageModel
    deidentifier: Deidentifier
    guard: Guard

    def answer(
        self,
        question: str,
        principal: Principal,
        purpose: Purpose,
        *,
        matter_id: str | None = None,
        k: int = 5,
    ) -> Answer:
        """Answer from the chunks the principal may retrieve; `EgressBlocked` propagates."""
        retrieval = self.index.search(question, principal, purpose, matter_id=matter_id, k=k)
        if not retrieval.hits:
            return Answer(NO_ANSWER, (), reidentified=False)

        context = [hit.chunk for hit in retrieval.hits]
        reply = self.gateway.complete(
            self.model,
            build(question, [chunk.text for chunk in context]),
            principal=principal,
            purpose=purpose,
            compartments=retrieval.compartments,
            obligations=retrieval.obligations,
        )
        if reply.strip() == NO_ANSWER:
            return Answer(NO_ANSWER, (), reidentified=False)

        text, cited = _cite(reply, context)
        reveal = self._may_reidentify(principal, purpose, retrieval)
        if reveal:
            text = self._reveal(text, context)
        citations = tuple(
            Citation(
                chunk.document_id,
                chunk.ordinal,
                chunk.matter_id,
                self.deidentifier.reidentify(chunk.text, chunk.matter_id) if reveal else chunk.text,
            )
            for chunk in cited
        )
        return Answer(text, citations, reidentified=reveal)

    def _may_reidentify(self, principal: Principal, purpose: Purpose, retrieval: Retrieval) -> bool:
        if Obligation.KEEP_PSEUDONYMS in retrieval.obligations:
            return False
        at = datetime.now(UTC)
        decisions = [
            self.guard.evaluate(
                AccessRequest(principal, Action.REIDENTIFY, purpose, compartment.resource, at),
                sensitivity=compartment.sensitivity.value,
            )
            for compartment in retrieval.compartments
        ]
        return all(decision.allowed for decision in decisions)

    def _reveal(self, text: str, context: Sequence[Chunk]) -> str:
        """Resolve each token with the matter of the passage it came from.

        A token found in no passage of the context is left alone, so a model cannot be used
        to unmask something the principal did not retrieve.
        """
        owner = {
            match.group(): chunk.matter_id for chunk in context for match in tokens.find(chunk.text)
        }

        def original(match: re.Match[str]) -> str:
            token = match.group()
            return self.deidentifier.reidentify(token, owner[token]) if token in owner else token

        return tokens.TOKEN.sub(original, text)


def _cite(reply: str, context: Sequence[Chunk]) -> tuple[str, list[Chunk]]:
    """Renumber the passage markers the model used by first mention and list those passages.

    A reply without a valid marker is attributed to the whole context.
    """
    order: list[int] = []

    def renumber(match: re.Match[str]) -> str:
        number = int(match[1])
        if not 1 <= number <= len(context):
            return match.group()
        if number not in order:
            order.append(number)
        return f"[{order.index(number) + 1}]"

    text = _MARKER.sub(renumber, reply)
    return text, [context[number - 1] for number in order] or list(context)
