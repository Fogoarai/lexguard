import random
from itertools import pairwise

import pytest

from lexguard import tokens
from lexguard.deid import RECOGNIZERS, detect
from lexguard.deid.detection import resolve
from lexguard.domain import PiiKind, PiiSpan
from lexguard.synthetic import generate

KEY = b"k" * 32


def test_spans_are_ordered_disjoint_and_match_the_text():
    for labeled in generate(seed=3).documents:
        text = labeled.document.text
        spans = detect(text)
        assert spans == sorted(spans, key=lambda span: span.start)
        assert all(a.end <= b.start for a, b in pairwise(spans))
        assert all(text[span.start : span.end] == span.text for span in spans)


def test_longest_span_wins_an_overlap():
    person = PiiSpan(0, 20, PiiKind.PERSON, "Hernán Ossandón Vial")
    organization = PiiSpan(0, 29, PiiKind.ORGANIZATION, "Hernán Ossandón Vial E.I.R.L.")
    assert resolve([person, organization]) == [organization]


def test_equal_spans_go_to_the_more_specific_kind():
    account = PiiSpan(14, 22, PiiKind.BANK_ACCOUNT, "17456321")
    as_rut = PiiSpan(14, 22, PiiKind.RUT, "17456321")
    assert resolve([as_rut, account]) == [account]
    assert [span.kind for span in detect("cuenta RUT N° 17456321")] == [PiiKind.BANK_ACCOUNT]


def test_resolution_does_not_depend_on_recognizer_order():
    text = generate(seed=5).documents[0].document.text
    shuffled = list(RECOGNIZERS)
    random.Random(1).shuffle(shuffled)
    assert detect(text, shuffled) == detect(text)


def test_later_mentions_of_a_known_person():
    text = (
        "Para: Carolina Sanhueza Parra\n\nEstimada Carolina: la señora Sanhueza firmó ayer. "
        "Sanhueza y Parra son apellidos; Carolina también."
    )
    people = [span.text for span in detect(text) if span.kind is PiiKind.PERSON]
    assert people == [
        "Carolina Sanhueza Parra",
        "Carolina",
        "Sanhueza",
        "Sanhueza",
        "Parra",
        "Carolina",
    ]


def test_full_name_in_capitals_is_a_mention_of_a_known_person():
    text = "FIRMA: CAROLINA SANHUEZA PARRA\n\nCarolina Sanhueza Parra declara lo siguiente."
    assert [span.text for span in detect(text)] == [
        "CAROLINA SANHUEZA PARRA",
        "Carolina Sanhueza Parra",
    ]


def test_a_lone_name_with_no_full_name_in_the_document_is_not_flagged():
    assert detect("El finiquito lo firmó Sanhueza ayer.") == []


@pytest.mark.parametrize("kind", PiiKind)
def test_existing_tokens_are_not_flagged(kind):
    token = tokens.pseudonym(kind, "x", KEY)
    text = f"El señor {token}, RUT {token}, cuenta corriente N° {token}, teléfono {token}."
    assert detect(text) == []


def test_pii_next_to_a_token_is_still_found():
    token = tokens.pseudonym(PiiKind.PERSON, "x", KEY)
    spans = detect(f"{token}, RUT 12.345.678-5, domiciliado en Calle Merced 456, Santiago")
    assert [span.kind for span in spans] == [PiiKind.RUT, PiiKind.ADDRESS]


def test_another_recognizer_plugs_in_behind_the_protocol():
    class Nicknames:
        def find(self, text):
            start = text.find("Pancho")
            return [PiiSpan(start, start + 6, PiiKind.PERSON, "Pancho")] if start >= 0 else []

    text = "Pancho: llama al +56 9 7645 3392."
    assert [span.kind for span in detect(text)] == [PiiKind.PHONE]
    assert [span.kind for span in detect(text, [*RECOGNIZERS, Nicknames()])] == [
        PiiKind.PERSON,
        PiiKind.PHONE,
    ]
