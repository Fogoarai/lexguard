"""Private retrieval: chunking, local embedding, security-trimmed search and cited answers."""

from lexguard.rag.answering import Answer, Answerer, Citation, ModelGateway
from lexguard.rag.chunking import split
from lexguard.rag.embedding import HashingEmbedder
from lexguard.rag.extractive import ExtractiveModel
from lexguard.rag.index import Compartment, Hit, Retrieval, SecureIndex
from lexguard.rag.prompt import NO_ANSWER

__all__ = [
    "NO_ANSWER",
    "Answer",
    "Answerer",
    "Citation",
    "Compartment",
    "ExtractiveModel",
    "HashingEmbedder",
    "Hit",
    "ModelGateway",
    "Retrieval",
    "SecureIndex",
    "split",
]
