import re
from collections.abc import Iterable

from lexguard import tokens
from lexguard.deid.canonical import name_parts
from lexguard.deid.recognizers import RECOGNIZERS, Recognizer
from lexguard.domain import PiiKind, PiiSpan

_MASK = "\x00"
_MOST_SPECIFIC_FIRST = (
    PiiKind.EMAIL,
    PiiKind.BANK_ACCOUNT,
    PiiKind.CASE_NUMBER,
    PiiKind.PHONE,
    PiiKind.RUT,
    PiiKind.ORGANIZATION,
    PiiKind.ADDRESS,
    PiiKind.PERSON,
)


def detect(text: str, recognizers: Iterable[Recognizer] = RECOGNIZERS) -> list[PiiSpan]:
    """Non-overlapping spans in text order. Pseudonym tokens already in the text are skipped."""
    masked = tokens.TOKEN.sub(lambda match: _MASK * len(match.group()), text)
    spans = resolve(span for recognizer in recognizers for span in recognizer.find(masked))
    return resolve([*spans, *_mentions(masked, spans)])


def resolve(spans: Iterable[PiiSpan]) -> list[PiiSpan]:
    """The longest span wins an overlap; ties go to the more specific kind, then the earlier."""

    def preference(span: PiiSpan) -> tuple[int, int, int]:
        return span.start - span.end, _MOST_SPECIFIC_FIRST.index(span.kind), span.start

    kept: list[PiiSpan] = []
    for span in sorted(set(spans), key=preference):
        if all(span.end <= other.start or other.end <= span.start for other in kept):
            kept.append(span)
    return sorted(kept, key=lambda s: s.start)


def _mentions(text: str, spans: Iterable[PiiSpan]) -> list[PiiSpan]:
    """A given name or surname alone, or the full name in capitals, of a person found here."""
    names = {span.text for span in spans if span.kind is PiiKind.PERSON}
    parts = {part for name in names for part in name_parts(name) if len(part) > 2}
    forms = parts | {part.title() for part in parts} | {name.upper() for name in names}
    if not forms:
        return []
    choices = "|".join(re.escape(form) for form in sorted(forms, key=lambda f: (-len(f), f)))
    return [
        PiiSpan(match.start(), match.end(), PiiKind.PERSON, match.group())
        for match in re.finditer(rf"(?<!\w)(?:{choices})(?!\w)", text)
    ]
