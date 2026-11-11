import re

from lexguard import rut
from lexguard.domain import PiiKind

NAME_PARTICLES = frozenset({"de", "del", "la", "las", "los", "von", "van", "der"})


def canonicalize(kind: PiiKind, value: str) -> str:
    """The form that gets tokenized, so every spelling of one entity shares a token."""
    collapsed = " ".join(value.split())
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
            return re.sub(r"[.,]", "", collapsed).casefold()
        case _:
            return collapsed.casefold()


def name_parts(name: str) -> list[str]:
    """Given names and surnames of a full name, each of which may later stand in for it."""
    return [word for word in name.split() if word.casefold() not in NAME_PARTICLES]
