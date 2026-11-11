import pytest

from lexguard.deid import canonicalize
from lexguard.deid.canonical import name_parts
from lexguard.domain import PiiKind


@pytest.mark.parametrize(
    ("kind", "spellings"),
    [
        (PiiKind.RUT, ["12.345.678-5", "12345678-5", " 123456785 "]),
        (PiiKind.RUT, ["8.765.432-K", "8765432-k"]),
        (PiiKind.PERSON, ["Ana Fuentes Leiva", "ANA FUENTES LEIVA", "Ana  Fuentes\tLeiva"]),
        (PiiKind.ORGANIZATION, ["Tres Pinos S.A.", "TRES PINOS S.A", "Tres Pinos SA"]),
        (PiiKind.ORGANIZATION, ["Río Claro Ltda.", "Río Claro Ltda"]),
        (PiiKind.EMAIL, ["M.Lopez@Estudio.Example", "m.lopez@estudio.example"]),
        (PiiKind.PHONE, ["+56 9 7645 3392", "+56976453392", "9 7645 3392", "(+56 9) 7645-3392"]),
        (PiiKind.CASE_NUMBER, ["C-15.234-2022", "c-15234-2022"]),
        (PiiKind.BANK_ACCOUNT, ["000-12345-67", "0001234567"]),
        (PiiKind.ADDRESS, ["Calle Merced 456, Santiago", "calle  Merced 456, SANTIAGO"]),
    ],
)
def test_spellings_of_one_entity_share_a_canonical_form(kind, spellings):
    assert len({canonicalize(kind, spelling) for spelling in spellings}) == 1


def test_rut_uses_the_core_canonical_form():
    assert canonicalize(PiiKind.RUT, "12.345.678-5") == "12345678-5"
    assert canonicalize(PiiKind.RUT, "no es un rut") == "NO ES UN RUT"


def test_different_entities_stay_apart():
    assert canonicalize(PiiKind.PERSON, "Ana Fuentes") != canonicalize(
        PiiKind.PERSON, "Ana Puentes"
    )
    assert canonicalize(PiiKind.RUT, "12.345.678-5") != canonicalize(PiiKind.RUT, "12.345.679-3")


def test_name_parts_leave_out_particles():
    assert name_parts("Pedro Pablo del Río Aguirre") == ["Pedro", "Pablo", "Río", "Aguirre"]
    assert name_parts("Francisca De la Maza") == ["Francisca", "Maza"]
