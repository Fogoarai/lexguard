from types import SimpleNamespace

import pytest

from lexguard.domain import PiiSpan
from lexguard.guard import Guard
from lexguard.synthetic import generate
from lexguard.testing import LabelDeidentifier, MemoryAuditLog


@pytest.fixture(scope="session")
def corpus():
    return generate()


@pytest.fixture
def deidentifier(corpus):
    return LabelDeidentifier(corpus)


@pytest.fixture
def documents(corpus, deidentifier):
    return [deidentifier.pseudonymize(labeled.document) for labeled in corpus.documents]


@pytest.fixture
def audit():
    return MemoryAuditLog()


@pytest.fixture
def guard_with(audit):
    """Builds a guard from a policy object or from a bare `decide` function."""

    def _guard_with(policy):
        if not hasattr(policy, "decide"):
            policy = SimpleNamespace(decide=policy)
        return Guard(policy, audit)

    return _guard_with


@pytest.fixture
def scan(corpus):
    """Residual-PII scanner that knows every labeled value of the corpus."""
    known = {(span.text, span.kind) for labeled in corpus.documents for span in labeled.spans}

    def _scan(text):
        return [
            PiiSpan(start, start + len(value), kind, value)
            for value, kind in known
            if (start := text.find(value)) >= 0
        ]

    return _scan
