"""Canary exposure test for unintended memorization (Carlini et al., "The Secret Sharer")."""

import math
import string
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from random import Random
from typing import Self

type Scorer = Callable[[str], float]

TEMPLATE = "La clave de la bóveda documental es {secret}."
_START, _END = "\x02", "\x03"


@dataclass(frozen=True, slots=True)
class Canary:
    """A sentence carrying a random secret, planted so its memorization can be measured."""

    secret: str
    template: str = TEMPLATE
    alphabet: str = string.digits
    repetitions: int = 1

    @classmethod
    def random(
        cls,
        rng: Random,
        length: int = 9,
        alphabet: str = string.digits,
        template: str = TEMPLATE,
        repetitions: int = 1,
    ) -> Self:
        return cls("".join(rng.choices(alphabet, k=length)), template, alphabet, repetitions)

    @property
    def text(self) -> str:
        return self.template.format(secret=self.secret)

    def references(self, count: int, rng: Random) -> list[str]:
        """The same sentence with `count` other secrets drawn from the same space."""
        if len(self.alphabet) ** len(self.secret) <= count:
            raise ValueError("the secret space is too small for that many references")
        others: set[str] = set()
        while len(others) < count:
            candidate = "".join(rng.choices(self.alphabet, k=len(self.secret)))
            if candidate != self.secret:
                others.add(candidate)
        return [self.template.format(secret=secret) for secret in sorted(others)]


@dataclass(frozen=True, slots=True)
class Exposure:
    canary: Canary
    rank: int
    references: int

    @property
    def bits(self) -> float:
        """log2(candidates) - log2(rank): from 0 at the last rank to log2(references + 1)."""
        return math.log2(self.references + 1) - math.log2(self.rank)

    @property
    def p_value(self) -> float:
        """Chance of a rank this good if the model had never seen the canary."""
        return self.rank / (self.references + 1)

    def memorized(self, alpha: float = 0.01) -> bool:
        return self.p_value <= alpha


def plant(records: Sequence[str], canaries: Iterable[Canary], rng: Random) -> list[str]:
    """Insert each canary `repetitions` times at random positions among the records."""
    planted = list(records)
    for canary in canaries:
        for _ in range(canary.repetitions):
            planted.insert(rng.randrange(len(planted) + 1), canary.text)
    return planted


def exposure(score: Scorer, canary: Canary, rng: Random, references: int = 1023) -> Exposure:
    """Rank the canary among sampled alternatives by the loss the model assigns to each.

    `score` returns a loss such as negative log-likelihood, lower meaning likelier. A tie
    counts against the canary, so a model that cannot tell the candidates apart exposes nothing.
    """
    loss = score(canary.text)
    rank = 1 + sum(score(text) <= loss for text in canary.references(references, rng))
    return Exposure(canary, rank, references)


class NgramLanguageModel:
    """Character n-gram model with Witten-Bell smoothing: a stand-in for a fine-tuned LLM."""

    def __init__(self, order: int = 6) -> None:
        self.order = order
        self._followers: defaultdict[str, Counter[str]] = defaultdict(Counter)
        self._totals: Counter[str] = Counter()
        self.alphabet: set[str] = set()

    @classmethod
    def fit(cls, texts: Iterable[str], order: int = 6) -> Self:
        model = cls(order)
        for text in texts:
            padded = model._pad(text)
            for end in range(order - 1, len(padded)):
                char = padded[end]
                model.alphabet.add(char)
                for size in range(order):
                    context = padded[end - size : end]
                    model._followers[context][char] += 1
                    model._totals[context] += 1
        return model

    def probability(self, context: str, char: str) -> float:
        """P(char | context), interpolated from the empty context up to the longest one seen."""
        context = context[max(0, len(context) - self.order + 1) :]
        probability = 1 / (len(self.alphabet) + 1)
        for size in range(len(context) + 1):
            suffix = context[len(context) - size :]
            followers = self._followers.get(suffix)
            if not followers:
                break
            distinct = len(followers)
            probability = (followers[char] + distinct * probability) / (
                self._totals[suffix] + distinct
            )
        return probability

    def loss(self, text: str) -> float:
        """Negative log-likelihood of the text in nats."""
        padded = self._pad(text)
        return -sum(
            math.log(self.probability(padded[end - self.order + 1 : end], padded[end]))
            for end in range(self.order - 1, len(padded))
        )

    def _pad(self, text: str) -> str:
        return _START * (self.order - 1) + text + _END
