import re

from lexguard.rag.embedding import features
from lexguard.rag.prompt import NO_ANSWER, parse
from lexguard.settings import ModelLocation

_SENTENCE_BREAK = re.compile(r"(?<=[.!?])\s+(?=[A-ZÁÉÍÓÚÑ¿¡\[])|\n+")


class ExtractiveModel:
    """In-process stand-in for a language model: quotes the sentences that best cover the question.

    It only matches words, so text inside a passage can never steer it.
    """

    name = "extractive"
    location = ModelLocation.LOCAL

    def __init__(self, sentences: int = 2, min_coverage: float = 0.3) -> None:
        self._sentences = sentences
        self._min_coverage = min_coverage

    def complete(self, prompt: str) -> str:
        question, passages = parse(prompt)
        wanted = features(question)
        threshold = self._min_coverage * sum(wanted.values())
        ranked = sorted(
            (
                (coverage, number, sentence.strip())
                for number, passage in enumerate(passages, start=1)
                for sentence in _SENTENCE_BREAK.split(passage)
                if (coverage := _coverage(wanted, sentence)) > threshold
            ),
            key=lambda candidate: -candidate[0],
        )
        quotes = [f"{sentence} [{number}]" for _, number, sentence in ranked[: self._sentences]]
        return " ".join(quotes) or NO_ANSWER


def _coverage(wanted: dict[str, float], sentence: str) -> float:
    found = features(sentence)
    return sum(weight for feature, weight in wanted.items() if feature in found)
