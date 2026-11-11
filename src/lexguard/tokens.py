import hashlib
import hmac
import re
from collections.abc import Iterator

from lexguard.domain import PiiKind

TOKEN = re.compile(r"\[(?P<kind>[A-Z_]+)_(?P<digest>[0-9a-f]{12})\]")


def pseudonym(kind: PiiKind, value: str, key: bytes) -> str:
    """Deterministic under one key, so the same entity keeps one token within a matter only."""
    digest = hmac.new(key, f"{kind}\x1f{value}".encode(), hashlib.sha256).hexdigest()[:12]
    return f"[{kind.name}_{digest}]"


def find(text: str) -> Iterator[re.Match[str]]:
    return TOKEN.finditer(text)


def kind_of(token: str) -> PiiKind:
    match = TOKEN.fullmatch(token)
    if match is None:
        raise ValueError(f"not a pseudonym token: {token!r}")
    return PiiKind[match["kind"]]
