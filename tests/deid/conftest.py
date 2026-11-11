import re
from datetime import UTC, datetime
from pathlib import Path

import pytest

from lexguard.crypto import KeyRing
from lexguard.deid import Pseudonymizer, SealedTokenVault
from lexguard.domain import Document, DocumentKind, PiiKind, PiiSpan, Sensitivity
from lexguard.synthetic import LabeledDocument

_LABEL = re.compile(r"\[\[(?P<kind>[A-Z_]+)\|(?P<text>.+?)\]\]", re.DOTALL)


def _labeled(name: str, marked: str) -> LabeledDocument:
    """Turn `[[KIND|text]]` markup into a document and its labeled spans."""
    pieces: list[str] = []
    spans: list[PiiSpan] = []
    cursor = length = 0
    for match in _LABEL.finditer(marked):
        pieces += [marked[cursor : match.start()], match["text"]]
        length += match.start() - cursor
        end = length + len(match["text"])
        spans.append(PiiSpan(length, end, PiiKind[match["kind"]], match["text"]))
        cursor, length = match.end(), end
    pieces.append(marked[cursor:])
    document = Document(
        id=name,
        matter_id="M-001",
        kind=DocumentKind.MEMO,
        sensitivity=Sensitivity.CONFIDENTIAL,
        text="".join(pieces),
        created_at=datetime(2024, 1, 1, tzinfo=UTC),
    )
    return LabeledDocument(document, tuple(spans))


@pytest.fixture(scope="session")
def handwritten() -> dict[str, LabeledDocument]:
    """Texts written by hand, outside the synthetic templates.

    The `dev/` ones were used while writing the recognizers; the `heldout/` ones were
    written afterwards and scored once.
    """
    source = Path(__file__).with_name("handwritten.txt").read_text(encoding="utf-8")
    sections = re.split(r"^=== (\S+)\n", source, flags=re.MULTILINE)[1:]
    return {
        name: _labeled(name.replace("/", "-"), body)
        for name, body in zip(sections[::2], sections[1::2], strict=True)
    }


@pytest.fixture
def keys() -> KeyRing:
    ring = KeyRing.generate()
    for matter_id in ("M-001", "M-002"):
        ring.enroll(matter_id)
    return ring


@pytest.fixture
def vault(keys, tmp_path) -> SealedTokenVault:
    return SealedTokenVault(keys, tmp_path / "vault")


@pytest.fixture
def pseudonymizer(vault) -> Pseudonymizer:
    return Pseudonymizer(vault)
