import re
import unicodedata
import zlib
from collections import defaultdict
from collections.abc import Sequence

import numpy as np

from lexguard.tokens import TOKEN

_TERM = re.compile(rf"{TOKEN.pattern}|\w+")
_NGRAM = 4
_NGRAM_WEIGHT = 0.25
_STOPWORDS = frozenset({
    "al", "ante", "aun", "como", "con", "cual", "cuales", "cuando", "cuanto", "de", "del",
    "desde", "donde", "el", "ella", "ellos", "en", "entre", "era", "es", "esa", "ese", "eso",
    "esta", "este", "esto", "fue", "ha", "han", "hay", "la", "las", "le", "les", "lo", "los",
    "mas", "mi", "muy", "no", "nos", "para", "pero", "por", "que", "quien", "quienes", "se",
    "sea", "ser", "si", "sin", "sobre", "son", "su", "sus", "tambien", "un", "una", "uno",
    "unas", "unos", "ya",
})  # fmt: skip


def features(text: str) -> dict[str, float]:
    """Weighted lexical features: folded words, their character n-grams, and whole pseudonyms.

    A pseudonym token is one opaque feature, so two passages about the same entity of a
    matter share it while its digest contributes no n-gram noise.
    """
    weights: defaultdict[str, float] = defaultdict(float)
    for match in _TERM.finditer(text):
        if match["digest"]:
            weights[match.group()] += 1
            continue
        word = _fold(match.group())
        if len(word) < 2 or word in _STOPWORDS:
            continue
        weights[word] += 1
        padded = f"<{word}>"
        for start in range(len(padded) - _NGRAM + 1):
            weights[f"#{padded[start : start + _NGRAM]}"] += _NGRAM_WEIGHT
    return dict(weights)


def _fold(word: str) -> str:
    decomposed = unicodedata.normalize("NFKD", word.casefold())
    return "".join(char for char in decomposed if not unicodedata.combining(char))


class HashingEmbedder:
    """Signed feature hashing into L2-normalized rows; deterministic and download-free.

    A vector depends only on its own text, never on corpus statistics, so indexing one
    compartment cannot change how another is scored.
    """

    def __init__(self, dimensions: int = 4096) -> None:
        self.dimensions = dimensions

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        matrix = np.zeros((len(texts), self.dimensions), dtype=np.float32)
        for row, text in zip(matrix, texts, strict=True):
            for feature, weight in features(text).items():
                digest = zlib.crc32(feature.encode())
                row[digest % self.dimensions] += weight if digest >> 31 else -weight
        matrix = np.sign(matrix) * np.log1p(np.abs(matrix))
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        return matrix / np.where(norms == 0, 1, norms)
