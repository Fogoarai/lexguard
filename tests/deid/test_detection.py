import random
import unicodedata
from itertools import pairwise

import pytest

from lexguard import tokens
from lexguard.deid import RECOGNIZERS, detect
from lexguard.deid.canonical import plain
from lexguard.deid.detection import resolve
from lexguard.domain import PiiKind, PiiSpan
from lexguard.synthetic import generate

KEY = b"k" * 32
DRESSINGS = {
    "decomposed accents": lambda text: unicodedata.normalize("NFD", text),
    "non-breaking spaces": lambda text: text.replace(" ", "\u00a0"),
    "narrow spaces between digits": lambda text: text.replace("9 ", "9\u202f"),
    "windows line endings": lambda text: text.replace("\n", "\r\n"),
    "non-breaking hyphens": lambda text: text.replace("-", "\u2011"),
    "soft hyphens": lambda text: text.replace("a", "a\u00ad"),
    "ligatures": lambda text: text.replace("fi", "\ufb01"),
    "curly apostrophes": lambda text: text.replace("'", "\u2019"),
}


def test_spans_are_ordered_disjoint_and_match_the_text():
    for labeled in generate(seed=3).documents:
        text = labeled.document.text
        spans = detect(text)
        assert spans == sorted(spans, key=lambda span: span.start)
        assert all(a.end <= b.start for a, b in pairwise(spans))
        assert all(text[span.start : span.end] == span.text for span in spans)


@pytest.mark.parametrize("dressing", DRESSINGS)
def test_detection_does_not_depend_on_how_the_text_is_encoded(dressing, handwritten):
    documents = [*generate(seed=11).documents, *handwritten.values()]
    for labeled in documents:
        text = labeled.document.text
        dressed = DRESSINGS[dressing](text)
        spans = detect(dressed)
        assert all(dressed[span.start : span.end] == span.text for span in spans)
        assert [(span.kind, plain(span.text)) for span in spans] == [
            (span.kind, plain(span.text)) for span in detect(text)
        ]


def test_offsets_refer_to_the_text_as_given():
    name = "Mari\u0301a Jose\u0301 Nu\u0301n\u0303ez"
    number = "12.345.678\u20115"
    token = tokens.pseudonym(PiiKind.ORGANIZATION, "x", KEY)
    text = f"{name}\u00ad, de {token}, RUT {number}."
    at = text.index(number)
    assert detect(text) == [
        PiiSpan(0, len(name), PiiKind.PERSON, name),
        PiiSpan(at, at + len(number), PiiKind.RUT, number),
    ]


def test_longest_span_wins_an_overlap():
    person = PiiSpan(0, 20, PiiKind.PERSON, "Hernán Ossandón Vial")
    organization = PiiSpan(0, 29, PiiKind.ORGANIZATION, "Hernán Ossandón Vial E.I.R.L.")
    assert resolve([person, organization]) == [organization]


def test_equal_spans_go_to_the_more_specific_kind():
    account = PiiSpan(14, 22, PiiKind.BANK_ACCOUNT, "17456321")
    as_rut = PiiSpan(14, 22, PiiKind.RUT, "17456321")
    assert resolve([as_rut, account]) == [account]
    assert [span.kind for span in detect("cuenta RUT N° 17456321")] == [PiiKind.BANK_ACCOUNT]


def test_resolution_keeps_exactly_what_no_preferred_span_overlaps():
    rng = random.Random(4)
    spans = [
        PiiSpan(start, start + rng.randrange(1, 30), PiiKind.PERSON, "x")
        for start in (rng.randrange(2000) for _ in range(1500))
    ]
    expected: list[PiiSpan] = []
    for span in sorted(set(spans), key=lambda span: (span.start - span.end, span.start)):
        if all(span.end <= kept.start or kept.end <= span.start for kept in expected):
            expected.append(span)

    assert resolve(spans) == sorted(expected, key=lambda span: span.start)
    assert resolve(reversed(spans)) == resolve(spans)


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


def test_capitals_without_accents_are_a_mention_of_a_known_person():
    text = "representada por María José Núñez Peña, quien firma.\n\nMARIA JOSE NUÑEZ PEÑA"
    assert [span.text for span in detect(text)] == [
        "María José Núñez Peña",
        "MARIA JOSE NUÑEZ PEÑA",
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


def test_an_empty_span_from_a_recognizer_is_dropped():
    class Empty:
        def find(self, text):
            return [PiiSpan(0, 0, PiiKind.PERSON, ""), PiiSpan(5, 5, PiiKind.PERSON, "")]

    assert detect("Sin datos personales.", [Empty()]) == []


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
