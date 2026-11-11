import re
from collections.abc import Iterator

from lexguard.domain import Chunk, Document
from lexguard.tokens import TOKEN

_ATOM = re.compile(rf"(?:{TOKEN.pattern}|\S)+\s*")


def split(document: Document, max_chars: int = 500) -> list[Chunk]:
    """Cut a pseudonymized document into chunks of at most `max_chars`.

    A cut falls only on whitespace outside a pseudonym token, at a paragraph break when one
    is near, so no token is ever split. A single unbreakable run longer than the limit
    becomes a chunk of its own.
    """
    pieces = (document.text[start:end].strip() for start, end in _cuts(document.text, max_chars))
    return [
        Chunk(
            id=f"{document.id}#{ordinal}",
            document_id=document.id,
            matter_id=document.matter_id,
            sensitivity=document.sensitivity,
            ordinal=ordinal,
            text=piece,
        )
        for ordinal, piece in enumerate(piece for piece in pieces if piece)
    ]


def _cuts(text: str, max_chars: int) -> Iterator[tuple[int, int]]:
    start = paragraph = 0
    for atom in _ATOM.finditer(text):
        content_end = atom.start() + len(atom.group().rstrip())
        while content_end - start > max_chars and atom.start() > start:
            cut = paragraph if paragraph > start + max_chars // 2 else atom.start()
            yield start, cut
            start = cut
        if atom.group().count("\n") > 1:
            paragraph = atom.end()
    yield start, len(text)
