from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import NamedTuple, Self

import numpy as np

from lexguard.domain import (
    AccessRequest,
    Action,
    Chunk,
    Decision,
    Document,
    Obligation,
    Principal,
    Purpose,
    Resource,
    Sensitivity,
    Zone,
)
from lexguard.errors import EgressBlocked
from lexguard.guard import Guard
from lexguard.ports import Deidentifier, Embedder
from lexguard.rag.chunking import split
from lexguard.settings import ModelLocation


class Compartment(NamedTuple):
    matter_id: str
    sensitivity: Sensitivity

    @classmethod
    def of(cls, chunk: Chunk) -> Self:
        return cls(chunk.matter_id, chunk.sensitivity)

    @property
    def resource(self) -> Resource:
        return Resource(Zone.SERVING, self.matter_id, self.sensitivity)


@dataclass(frozen=True, slots=True)
class Hit:
    chunk: Chunk
    score: float


@dataclass(frozen=True, slots=True)
class Retrieval:
    """`obligations` are those of the retrieve decisions that admitted the hits.

    `wording` is the question as each matter of the hits was asked it.
    """

    hits: tuple[Hit, ...]
    obligations: frozenset[Obligation]
    wording: Mapping[str, str] = field(default_factory=dict)

    @property
    def compartments(self) -> list[Compartment]:
        return sorted({Compartment.of(hit.chunk) for hit in self.hits})


class SecureIndex:
    """In-memory vector index that trims to the caller's compartments before it ranks.

    A score is the cosine between the query and one chunk, with no corpus statistics, so a
    compartment the caller is denied cannot change the scores, order or number of results.
    Chunks scoring below `min_score` are not released at all; the floor depends on the embedder.

    The embedder reads every chunk, privileged ones included, and every question as asked,
    with no policy decision in between, so only a local one is accepted.

    With a `deidentifier`, a matter whose identities the caller may read in every
    compartment is asked the question in its own tokens, so naming a person finds their
    passages. Anyone else's question is matched as asked: a rewrite would let them test
    which token a name has, or whether a compartment closed to them knows the name.
    """

    def __init__(
        self,
        embedder: Embedder,
        guard: Guard,
        deidentifier: Deidentifier | None = None,
        max_chars: int = 500,
        min_score: float = 0.05,
    ) -> None:
        if embedder.location is not ModelLocation.LOCAL:
            raise EgressBlocked(
                f"embedder {embedder.name} is {embedder.location}: it would read every chunk"
            )
        self._embedder = embedder
        self._guard = guard
        self._deidentifier = deidentifier
        self._max_chars = max_chars
        self._min_score = min_score
        self._chunks: list[Chunk] = []
        self._vectors = np.empty((0, 0), dtype=np.float32)

    def __len__(self) -> int:
        return len(self._chunks)

    def add(self, documents: Iterable[Document]) -> None:
        """Index pseudonymized documents, replacing any that were indexed before.

        The last version of a document in the batch wins. Nothing changes if embedding fails.
        """
        latest = {(document.matter_id, document.id): document for document in documents}
        chunks = [
            chunk for document in latest.values() for chunk in split(document, self._max_chars)
        ]
        vectors = _unit(self._embedder.embed([chunk.text for chunk in chunks])) if chunks else None
        for matter_id, document_id in latest:
            self.remove(matter_id, document_id)
        if vectors is not None:
            self._vectors = np.vstack((self._vectors, vectors)) if self._chunks else vectors
            self._chunks += chunks

    def remove(self, matter_id: str, document_id: str | None = None) -> int:
        """Forget one document, or the whole matter when no document is named."""
        keep = np.array(
            [
                chunk.matter_id != matter_id or document_id not in (None, chunk.document_id)
                for chunk in self._chunks
            ],
            dtype=bool,
        )
        self._chunks = [chunk for chunk, kept in zip(self._chunks, keep, strict=True) if kept]
        self._vectors = self._vectors[keep]
        return int((~keep).sum())

    def search(
        self,
        question: str,
        principal: Principal,
        purpose: Purpose,
        *,
        matter_id: str | None = None,
        k: int = 5,
    ) -> Retrieval:
        at = datetime.now(UTC)
        candidates = [
            row for row, chunk in enumerate(self._chunks) if matter_id in (None, chunk.matter_id)
        ]
        compartments = sorted({Compartment.of(self._chunks[row]) for row in candidates})
        decisions = {
            compartment: self._guard.evaluate(
                AccessRequest(principal, Action.RETRIEVE, purpose, compartment.resource, at),
                sensitivity=compartment.sensitivity.value,
            )
            for compartment in compartments
        }
        rows = [row for row in candidates if decisions[Compartment.of(self._chunks[row])].allowed]
        if not rows or k < 1:
            return Retrieval((), frozenset())

        wording = self._wording(question, decisions)
        wordings = list(dict.fromkeys(wording.values()))
        queries = _unit(self._embedder.embed(wordings))
        asked = [wordings.index(wording[self._chunks[row].matter_id]) for row in rows]
        scores = np.einsum("ij,ij->i", self._vectors[rows], queries[asked])
        best = np.argsort(-scores, kind="stable")[:k]
        hits = tuple(
            Hit(self._chunks[rows[i]], float(scores[i]))
            for i in best
            if scores[i] >= self._min_score
        )
        obligations = frozenset().union(
            *(decisions[Compartment.of(hit.chunk)].obligations for hit in hits)
        )
        matters = {hit.chunk.matter_id for hit in hits}
        return Retrieval(hits, obligations, {matter: wording[matter] for matter in matters})

    def _wording(self, question: str, decisions: Mapping[Compartment, Decision]) -> dict[str, str]:
        closed = {
            compartment.matter_id
            for compartment, decision in decisions.items()
            if not decision.allowed or Obligation.KEEP_PSEUDONYMS in decision.obligations
        }
        return {
            matter_id: question
            if self._deidentifier is None or matter_id in closed
            else self._deidentifier.tokenize(question, matter_id)
            for matter_id in sorted({compartment.matter_id for compartment in decisions})
        }


def _unit(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.where(norms == 0, 1, norms)
