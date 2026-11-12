import re
import unicodedata

from lexguard import rut
from lexguard.domain import PiiKind

NAME_PARTICLES = frozenset({"de", "del", "la", "las", "los", "von", "van", "der"})

_SPACES = (0x00A0, 0x1680, *range(0x2000, 0x200B), 0x202F, 0x205F, 0x3000)
_HYPHENS = (0x2010, 0x2011, 0x2012, 0x2013, 0x2212)
_APOSTROPHES = (0x2018, 0x2019, 0x02BC)
_INVISIBLE = (0x00AD, 0x200B, 0x200C, 0x200D, 0x2060, 0xFEFF)
_LIGATURES = {0xFB00: "ff", 0xFB01: "fi", 0xFB02: "fl", 0xFB03: "ffi", 0xFB04: "ffl"}
_TILDE = "\u0303"

PLAIN = {
    **dict.fromkeys(_SPACES, " "),
    **dict.fromkeys(_HYPHENS, "-"),
    **dict.fromkeys(_APOSTROPHES, "'"),
    **dict.fromkeys(_INVISIBLE, ""),
    **_LIGATURES,
    ord("\r"): " ",
}
RESIZED = re.compile("[" + "".join(chr(c) for c in (*_INVISIBLE, *_LIGATURES)) + "]")


def plain(text: str) -> str:
    """Composed characters, ordinary spaces, hyphens and apostrophes, nothing invisible.

    What word processors and PDF extraction do to a text without changing what it says.
    """
    return unicodedata.normalize("NFC", text).translate(PLAIN)


def without_accents(text: str) -> str:
    """Drops accents and keeps the tilde: capitals often lose the first, never the second."""
    decomposed = unicodedata.normalize("NFD", text)
    kept = (char for char in decomposed if not unicodedata.combining(char) or char == _TILDE)
    return unicodedata.normalize("NFC", "".join(kept))


def canonicalize(kind: PiiKind, value: str) -> str:
    """The form that gets tokenized, so every spelling of one entity shares a token."""
    collapsed = " ".join(plain(value).split())
    match kind:
        case PiiKind.RUT:
            return rut.canonical(collapsed) if rut.parse(collapsed) else collapsed.upper()
        case PiiKind.EMAIL:
            return collapsed.lower()
        case PiiKind.PHONE:
            digits = re.sub(r"\D", "", collapsed)
            return digits.removeprefix("56") if len(digits) == 11 else digits
        case PiiKind.BANK_ACCOUNT:
            return re.sub(r"\D", "", collapsed)
        case PiiKind.CASE_NUMBER:
            return re.sub(r"[ .]", "", collapsed).upper()
        case PiiKind.ORGANIZATION:
            return without_accents(re.sub(r"[.,]", "", collapsed).casefold())
        case _:
            return without_accents(collapsed.casefold())


def name_parts(name: str) -> list[str]:
    """Given names and surnames of a full name, each of which may later stand in for it."""
    return [word for word in name.split() if word.casefold() not in NAME_PARTICLES]
