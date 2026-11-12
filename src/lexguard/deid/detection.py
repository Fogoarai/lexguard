import re
import unicodedata
from bisect import bisect_right
from collections.abc import Iterable, Iterator, Sequence
from operator import attrgetter

from lexguard import tokens
from lexguard.deid.canonical import PLAIN, RESIZED, name_parts, plain, without_accents
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
    """Non-overlapping spans in text order. Pseudonym tokens already in the text are skipped.

    Recognizers read a plain view of the text (see `canonical.plain`), so a name is found
    whether its accents are composed or not and whatever kind of space or hyphen it holds.
    Offsets and span text always refer to `text` as given.
    """
    masked = tokens.TOKEN.sub(lambda match: _MASK * len(match.group()), text)
    view, starts, ends = _plain_view(masked)
    found = resolve(
        span
        for recognizer in recognizers
        for span in recognizer.find(view)
        if span.start < span.end
    )
    spans = []
    for span in resolve([*found, *_mentions(view, found)]):
        start, end = starts[span.start], ends[span.end - 1]
        spans.append(PiiSpan(start, end, span.kind, text[start:end]))
    return spans


def resolve(spans: Iterable[PiiSpan]) -> list[PiiSpan]:
    """The longest span wins an overlap; ties go to the more specific kind, then the earlier."""

    def preference(span: PiiSpan) -> tuple[int, int, int]:
        return span.start - span.end, _MOST_SPECIFIC_FIRST.index(span.kind), span.start

    kept: list[PiiSpan] = []
    for span in sorted(set(spans), key=preference):
        at = bisect_right(kept, span.start, key=attrgetter("start"))
        clear_before = at == 0 or kept[at - 1].end <= span.start
        clear_after = at == len(kept) or span.end <= kept[at].start
        if clear_before and clear_after:
            kept.insert(at, span)
    return kept


def _plain_view(text: str) -> tuple[str, Sequence[int], Sequence[int]]:
    """The plain form of `text` and, for each of its characters, where it starts and ends."""
    if unicodedata.is_normalized("NFC", text) and not RESIZED.search(text):
        return text.translate(PLAIN), range(len(text)), range(1, len(text) + 1)
    pieces: list[str] = []
    starts: list[int] = []
    ends: list[int] = []
    for start, cluster in _clusters(text):
        piece = plain(cluster)
        pieces.append(piece)
        starts += [start] * len(piece)
        ends += [start + len(cluster)] * len(piece)
    return "".join(pieces), starts, ends


def _clusters(text: str) -> Iterator[tuple[int, str]]:
    """Each character together with the combining marks that follow it."""
    start = 0
    for index in range(1, len(text) + 1):
        if index == len(text) or not unicodedata.combining(text[index]):
            yield start, text[start:index]
            start = index


def _mentions(text: str, spans: Iterable[PiiSpan]) -> list[PiiSpan]:
    """A given name or surname alone, or the full name in capitals, of a person found here."""
    names = {span.text for span in spans if span.kind is PiiKind.PERSON}
    parts = {part for name in names for part in name_parts(name) if len(part) > 2}
    forms = parts | {part.title() for part in parts} | {name.upper() for name in names}
    forms |= {without_accents(form) for form in forms}
    if not forms:
        return []
    choices = "|".join(re.escape(form) for form in sorted(forms, key=lambda f: (-len(f), f)))
    return [
        PiiSpan(match.start(), match.end(), PiiKind.PERSON, match.group())
        for match in re.finditer(rf"(?<!\w)(?:{choices})(?!\w)", text)
    ]
