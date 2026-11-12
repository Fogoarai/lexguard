import re
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass
from typing import Protocol

from lexguard import rut
from lexguard.deid import patterns
from lexguard.deid.canonical import NAME_PARTICLES
from lexguard.domain import PiiKind, PiiSpan


class Recognizer(Protocol):
    """One source of spans. A statistical NER model plugs in by implementing this.

    `text` arrives in plain form (`canonical.plain`) with pseudonym tokens blanked out,
    and the offsets returned refer to it.
    """

    def find(self, text: str) -> Iterable[PiiSpan]: ...


@dataclass(frozen=True, slots=True)
class PatternRecognizer:
    """Matches of one pattern; a group named `pii` narrows the span to leave out its context."""

    kind: PiiKind
    pattern: re.Pattern[str]
    accept: Callable[[str], bool] | None = None

    def find(self, text: str) -> Iterator[PiiSpan]:
        group = "pii" if "pii" in self.pattern.groupindex else 0
        for match in self.pattern.finditer(text):
            if self.accept is None or self.accept(match[group]):
                yield PiiSpan(match.start(group), match.end(group), self.kind, match[group])


class PersonRecognizer:
    """Runs of capitalized words, or a single one after an honorific, party label or sign-off.

    With no name list behind it, a run counts as a name unless it starts with a function
    word or holds an institutional noun. Other capitalized phrases (place names, defined
    terms) are therefore masked rather than missed.
    """

    def find(self, text: str) -> Iterator[PiiSpan]:
        for pattern in (patterns.NAME, patterns.NAME_AFTER_CUE, patterns.NAME_IN_CAPITALS):
            for match in pattern.finditer(text):
                if span := _person(match, cued=pattern is patterns.NAME_AFTER_CUE):
                    yield span


def _person(match: re.Match[str], cued: bool) -> PiiSpan | None:
    words = list(re.finditer(r"\S+", match["name"]))
    lowered = [word.group().casefold() for word in words]
    if patterns.NOT_PERSONS.intersection(lowered):
        return None
    skipped = next(
        (i for i, word in enumerate(lowered) if word not in patterns.LEADING_WORDS), len(words)
    )
    names = [
        word for word in lowered[skipped:] if word not in NAME_PARTICLES and not word.endswith(".")
    ]
    if len(names) < (1 if cued and not skipped else 2):
        return None
    start, end = match.start("name") + words[skipped].start(), match.end("name")
    return PiiSpan(start, end, PiiKind.PERSON, match.string[start:end])


RECOGNIZERS: tuple[Recognizer, ...] = (
    PatternRecognizer(PiiKind.EMAIL, patterns.EMAIL),
    PatternRecognizer(PiiKind.RUT, patterns.RUT, accept=rut.is_valid),
    PatternRecognizer(PiiKind.RUT, patterns.RUT_CUED),
    PatternRecognizer(PiiKind.PHONE, patterns.PHONE_INTERNATIONAL),
    PatternRecognizer(PiiKind.PHONE, patterns.PHONE_MOBILE),
    PatternRecognizer(PiiKind.PHONE, patterns.PHONE_CUED),
    PatternRecognizer(PiiKind.CASE_NUMBER, patterns.CASE_NUMBER),
    PatternRecognizer(PiiKind.CASE_NUMBER, patterns.CASE_NUMBER_CUED),
    PatternRecognizer(PiiKind.BANK_ACCOUNT, patterns.BANK_ACCOUNT),
    PatternRecognizer(PiiKind.ADDRESS, patterns.ADDRESS),
    PatternRecognizer(PiiKind.ADDRESS, patterns.ADDRESS_CUED),
    PatternRecognizer(PiiKind.ADDRESS, patterns.ADDRESS_WITH_UNIT),
    PatternRecognizer(PiiKind.ADDRESS, patterns.ADDRESS_LINE),
    PatternRecognizer(PiiKind.ORGANIZATION, patterns.ORGANIZATION),
    PersonRecognizer(),
)
