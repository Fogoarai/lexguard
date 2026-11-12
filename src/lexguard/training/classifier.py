import re
import zlib
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Self

import numpy as np

from lexguard import tokens
from lexguard.domain import DocumentKind

KINDS = tuple(DocumentKind)
_WORD = re.compile(r"\w+")


@dataclass(frozen=True, slots=True)
class DpSgd:
    clip_norm: float = 1.0
    noise_multiplier: float = 1.0

    def __post_init__(self) -> None:
        if not self.clip_norm > 0 or not self.noise_multiplier >= 0:
            raise ValueError("clip_norm must be positive and noise_multiplier non-negative")


@dataclass(frozen=True, slots=True)
class DpReport:
    """The parameters of a DP-SGD run, which is all this module claims about its privacy.

    No epsilon is computed here. Give these values and a delta below 1/examples to an
    established accountant (RDP or PRV, as in Opacus or Google's dp-accounting) to get one.
    """

    noise_multiplier: float
    clip_norm: float
    steps: int
    sampling_rate: float
    examples: int

    def __str__(self) -> str:
        return (
            f"DP-SGD over {self.examples} examples: noise multiplier {self.noise_multiplier}, "
            f"clip norm {self.clip_norm}, {self.steps} steps, Poisson sampling rate "
            f"{self.sampling_rate}. Epsilon is not computed here; an established privacy "
            "accountant must derive it from these values."
        )


def featurize(texts: Sequence[str], n_features: int) -> np.ndarray:
    """L2-normalized hashed word counts.

    Hashing needs no vocabulary fitted on the data, so featurizing spends no privacy budget.
    A pseudonym token counts as its PII kind, so no per-entity digest becomes a feature.
    """
    matrix = np.zeros((len(texts), n_features))
    for row, text in zip(matrix, texts, strict=True):
        kinds_only = tokens.TOKEN.sub(lambda match: f" {match['kind']} ", text)
        for word in _WORD.findall(kinds_only.lower()):
            row[zlib.crc32(word.encode()) % n_features] += 1
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return np.divide(matrix, norms, out=matrix, where=norms > 0)


@dataclass(frozen=True, slots=True)
class KindClassifier:
    """Multinomial logistic regression over hashed bag-of-words features."""

    weights: np.ndarray
    privacy: DpReport | None = None

    @classmethod
    def fit(
        cls,
        pairs: Sequence[tuple[str, DocumentKind]],
        *,
        n_features: int = 1024,
        steps: int = 200,
        sampling_rate: float = 0.25,
        learning_rate: float = 4.0,
        dp: DpSgd | None = None,
        seed: int = 0,
    ) -> Self:
        """Train with SGD over Poisson-sampled lots.

        With `dp`, each example's gradient is clipped to `clip_norm` and Gaussian noise of
        standard deviation `noise_multiplier * clip_norm` is added to every summed lot.
        """
        if not pairs:
            raise ValueError("no training examples")
        if not 0 < sampling_rate <= 1:
            raise ValueError("sampling_rate must be in (0, 1]")
        rng = np.random.default_rng(seed)
        features = featurize([text for text, _ in pairs], n_features)
        targets = np.eye(len(KINDS))[[KINDS.index(kind) for _, kind in pairs]]
        weights = np.zeros((n_features, len(KINDS)))
        expected_lot = sampling_rate * len(pairs)

        for _ in range(steps):
            lot = rng.random(len(pairs)) < sampling_rate
            x = features[lot]
            error = _softmax(x @ weights) - targets[lot]
            if dp:
                norms = np.linalg.norm(x, axis=1) * np.linalg.norm(error, axis=1)
                error *= np.minimum(1.0, dp.clip_norm / np.maximum(norms, 1e-12))[:, None]
            gradient = x.T @ error
            if dp:
                gradient += rng.normal(0.0, dp.noise_multiplier * dp.clip_norm, gradient.shape)
            weights -= learning_rate * gradient / expected_lot

        report = (
            DpReport(dp.noise_multiplier, dp.clip_norm, steps, sampling_rate, len(pairs))
            if dp
            else None
        )
        return cls(weights, report)

    def predict(self, texts: Sequence[str]) -> list[DocumentKind]:
        scores = featurize(texts, self.weights.shape[0]) @ self.weights
        return [KINDS[index] for index in scores.argmax(axis=1)]

    def accuracy(self, pairs: Sequence[tuple[str, DocumentKind]]) -> float:
        predicted = self.predict([text for text, _ in pairs])
        return float(np.mean([p is kind for p, (_, kind) in zip(predicted, pairs, strict=True)]))


def _softmax(logits: np.ndarray) -> np.ndarray:
    exp = np.exp(logits - logits.max(axis=1, keepdims=True))
    return exp / exp.sum(axis=1, keepdims=True)
