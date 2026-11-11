from collections import Counter
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass

from lexguard.domain import PiiKind, PiiSpan
from lexguard.synthetic import LabeledDocument


@dataclass(frozen=True, slots=True)
class Score:
    predicted: int = 0
    correct: int = 0
    labeled: int = 0
    found: int = 0

    @property
    def precision(self) -> float:
        return self.correct / self.predicted if self.predicted else 1.0

    @property
    def recall(self) -> float:
        return self.found / self.labeled if self.labeled else 1.0


def evaluate(
    detect: Callable[[str], Sequence[PiiSpan]],
    documents: Iterable[LabeledDocument],
    *,
    exact: bool = True,
) -> dict[PiiKind, Score]:
    """Precision and recall per kind against the labeled spans.

    Exact matching needs the same kind and the same boundaries. With `exact=False` a span
    counts as soon as it overlaps a span of its kind on the other side, which tells a missed
    entity apart from one found with the wrong boundaries.
    """
    matches = _same if exact else _overlapping
    tally = {kind: Counter[str]() for kind in PiiKind}
    for labeled in documents:
        predicted = detect(labeled.document.text)
        for span in predicted:
            tally[span.kind]["predicted"] += 1
            tally[span.kind]["correct"] += any(matches(span, label) for label in labeled.spans)
        for label in labeled.spans:
            tally[label.kind]["labeled"] += 1
            tally[label.kind]["found"] += any(matches(label, span) for span in predicted)
    return {kind: Score(**counts) for kind, counts in tally.items()}


def _same(a: PiiSpan, b: PiiSpan) -> bool:
    return (a.kind, a.start, a.end) == (b.kind, b.start, b.end)


def _overlapping(a: PiiSpan, b: PiiSpan) -> bool:
    return a.kind is b.kind and a.start < b.end and b.start < a.end
