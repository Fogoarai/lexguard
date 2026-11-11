import pytest

from lexguard import tokens
from lexguard.domain import PiiKind

KEY, OTHER_KEY = b"k" * 32, b"o" * 32


def test_pseudonym_is_deterministic_under_one_key():
    first = tokens.pseudonym(PiiKind.PERSON, "María Soto", KEY)
    assert first == tokens.pseudonym(PiiKind.PERSON, "María Soto", KEY)
    assert tokens.TOKEN.fullmatch(first)


def test_pseudonym_is_unlinkable_across_keys_and_kinds():
    variants = {
        tokens.pseudonym(PiiKind.PERSON, "Andes", KEY),
        tokens.pseudonym(PiiKind.PERSON, "Andes", OTHER_KEY),
        tokens.pseudonym(PiiKind.ORGANIZATION, "Andes", KEY),
    }
    assert len(variants) == 3


@pytest.mark.parametrize("kind", PiiKind)
def test_kind_round_trips(kind):
    assert tokens.kind_of(tokens.pseudonym(kind, "x", KEY)) is kind


def test_find_locates_tokens_in_text():
    person = tokens.pseudonym(PiiKind.PERSON, "María Soto", KEY)
    case = tokens.pseudonym(PiiKind.CASE_NUMBER, "C-12-2024", KEY)
    found = [match.group() for match in tokens.find(f"{person} comparece en causa {case}.")]
    assert found == [person, case]


def test_kind_of_rejects_plain_text():
    with pytest.raises(ValueError, match="not a pseudonym"):
        tokens.kind_of("[PERSON]")
