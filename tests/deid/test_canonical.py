import pytest

from lexguard.deid import canonicalize
from lexguard.deid.canonical import name_parts, plain, without_accents
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
        (
            PiiKind.PERSON,
            ["María José Núñez", "MARIA JOSE NUÑEZ", "Mari\u0301a Jose\u0301 Nu\u0301n\u0303ez"],
        ),
        (
            PiiKind.PERSON,
            ["Ana Fuentes", "Ana\u00a0Fuentes", "Ana Fuen\u00adtes", "Ana Fuentes\ufeff"],
        ),
        (PiiKind.PERSON, ["Björn O'Ryan", "Björn O’Ryan", "BJORN O'RYAN"]),
        (
            PiiKind.ORGANIZATION,
            ["Agrícola Pacífico S.A.", "AGRICOLA PACIFICO S.A.", "Agrícola Pací\ufb01co SA"],
        ),
        (PiiKind.ADDRESS, ["Av. Perú 123, Ñuñoa", "AV. PERU 123, ÑUÑOA"]),
        (PiiKind.RUT, ["12.345.678-5", "12.345.678\u20115", "12.345.678\u20135"]),
        (PiiKind.PHONE, ["+56 9 7645 3392", "+56\u00a09\u00a07645\u202f3392"]),
        (PiiKind.CASE_NUMBER, ["C-123-2024", "C\u2013123\u20132024"]),
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


def test_the_tilde_tells_names_apart_and_accents_do_not():
    assert canonicalize(PiiKind.PERSON, "Rosa Peña") != canonicalize(PiiKind.PERSON, "Rosa Pena")
    assert without_accents("NÚÑEZ Agüero João") == "NUÑEZ Aguero João"


def test_plain_form_keeps_what_the_text_says():
    dressed = "Sen\u0303or\u00a0Juan\u2011Pablo O\u2019Ryan\r\no\ufb01cina\u200b 12"
    assert plain(dressed) == "Señor Juan-Pablo O'Ryan \noficina 12"
    assert plain("Señor Juan-Pablo, oficina 12") == "Señor Juan-Pablo, oficina 12"


def test_name_parts_leave_out_particles():
    assert name_parts("Pedro Pablo del Río Aguirre") == ["Pedro", "Pablo", "Río", "Aguirre"]
    assert name_parts("Francisca De la Maza") == ["Francisca", "Maza"]
