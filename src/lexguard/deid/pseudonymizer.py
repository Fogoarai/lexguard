import re
from collections.abc import Sequence
from dataclasses import replace

from lexguard import tokens
from lexguard.deid.detection import detect
from lexguard.deid.recognizers import RECOGNIZERS, Recognizer
from lexguard.deid.vault import SealedTokenVault, TokenNotFound
from lexguard.domain import Document, PiiSpan


class Pseudonymizer:
    """`ports.Deidentifier` over the rule-based recognizers and the sealed vault.

    It takes no access decisions: the caller clears READ_RAW with the Guard before
    `pseudonymize` and REIDENTIFY before `reidentify`.
    """

    def __init__(
        self, vault: SealedTokenVault, recognizers: Sequence[Recognizer] = RECOGNIZERS
    ) -> None:
        self._vault = vault
        self._recognizers = recognizers

    def detect(self, text: str) -> list[PiiSpan]:
        return detect(text, self._recognizers)

    def pseudonymize(self, document: Document) -> Document:
        pieces: list[str] = []
        cursor = 0
        for span in self.detect(document.text):
            token = self._vault.remember(span.kind, span.text, document.matter_id)
            pieces += [document.text[cursor : span.start], token]
            cursor = span.end
        pieces.append(document.text[cursor:])
        return replace(document, text="".join(pieces))

    def reidentify(self, text: str, matter_id: str) -> str:
        """Tokens that are unknown to the matter or were forgotten stay as they are."""

        def restore(match: re.Match[str]) -> str:
            try:
                return self._vault.reveal(match.group(), matter_id)
            except TokenNotFound:
                return match.group()

        return tokens.TOKEN.sub(restore, text)
