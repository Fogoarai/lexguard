import pytest

from lexguard.deid import detect
from lexguard.domain import PiiKind


def found(text: str, kind: PiiKind) -> list[str]:
    return [span.text for span in detect(text) if span.kind is kind]


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("cédula de identidad N° 12.345.678-5, chilena", ["12.345.678-5"]),
        ("RUN 15432109-8, soltero", ["15432109-8"]),
        ("contribuyente 8.765.432-k registra deuda", ["8.765.432-k"]),
        ("(RUT: 76.543.210-3)", ["76.543.210-3"]),
        ("RUT 123456785 sin guion", ["123456785"]),
        ("RUT 12.345.678\u20115, con guion de no separación", ["12.345.678\u20115"]),
        ("cédula de identidad N°\u00a012.345.678-5", ["12.345.678-5"]),
    ],
)
def test_rut_in_dotted_and_undotted_spellings(text, expected):
    assert found(text, PiiKind.RUT) == expected


def test_rut_needs_a_valid_check_digit_or_a_cue():
    assert found("folio 12.345.678-9 del registro", PiiKind.RUT) == []
    assert found("RUT 12.345.678-9", PiiKind.RUT) == ["12.345.678-9"]
    assert found("la factura 123456785 fue pagada", PiiKind.RUT) == []


def test_email():
    text = "Escribir a m.lopez@aguas-valle.cl o <Tomas_E+legal@Estudio.Example>."
    assert found(text, PiiKind.EMAIL) == ["m.lopez@aguas-valle.cl", "Tomas_E+legal@Estudio.Example"]


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("llámeme al +56 9 7645 3392.", ["+56 9 7645 3392"]),
        ("WhatsApp +56976453392", ["+56976453392"]),
        ("oficina: +56 2 2345 6789", ["+56 2 2345 6789"]),
        ("su celular es 9 8765 4321", ["9 8765 4321"]),
        ("el fijo de la oficina 22 345 6789", ["22 345 6789"]),
        ("fono 2 2987 6543)", ["2 2987 6543"]),
        ("Teléfono (2) 2345 6789, horario de oficina", ["(2) 2345 6789"]),
        ("fono: (032) 268 9900", ["(032) 268 9900"]),
        ("cel 987654321", ["987654321"]),
        ("Tel 22 345 6789 / Fax 22 345 6780", ["22 345 6789", "22 345 6780"]),
        ("llámeme al +56\u00a09\u00a07645\u00a03392.", ["+56\u00a09\u00a07645\u00a03392"]),
    ],
)
def test_phone(text, expected):
    assert found(text, PiiKind.PHONE) == expected


def test_amounts_and_plain_numbers_are_not_phones():
    assert found("pagó $9.876.543 el 9 de marzo de 2024, folio 912345678", PiiKind.PHONE) == []
    assert found("el cartel (12) 345 6789 y el hotel 22 345 6789", PiiKind.PHONE) == []


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("en causa RIT O-45-2023, a US.", ["O-45-2023"]),
        ("causa Rol C-123-2024", ["C-123-2024"]),
        ("autos Rol C-15.234-2022 del 12° Juzgado Civil", ["C-15.234-2022"]),
        ("recurso Rol N° 4521-2023", ["4521-2023"]),
        ("RUC 2300456789-1, RIT 2145-2023", ["2300456789-1", "2145-2023"]),
        ("juicio arbitral Rol CAM N° 5123-2023", ["5123-2023"]),
    ],
)
def test_case_number(text, expected):
    assert found(text, PiiKind.CASE_NUMBER) == expected


def test_statutes_and_articles_are_not_case_numbers():
    text = "Conforme a la Ley N° 19.496 y al artículo 161-2020 del reglamento"
    assert found(text, PiiKind.CASE_NUMBER) == []


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("en la cuenta corriente N° 629659571 del banco", ["629659571"]),
        ("cuenta corriente del Banco del Sur N° 71-23456-8", ["71-23456-8"]),
        ("Cta. Cte. N° 000-12345-67", ["000-12345-67"]),
        ("depositar en cuenta vista 12345678", ["12345678"]),
        ("su cuenta RUT N° 17456321", ["17456321"]),
        ("cta cte 12345678 del banco", ["12345678"]),
        ("cuenta corriente Nro. 629659571", ["629659571"]),
        ("Cuenta Corriente: 00123456", ["00123456"]),
    ],
)
def test_bank_account_is_anchored_on_context(text, expected):
    assert found(text, PiiKind.BANK_ACCOUNT) == expected


def test_numbers_without_account_context_are_not_accounts():
    text = "Teniendo en cuenta 2024-2025, la boleta 629659571 se da por pagada."
    assert found(text, PiiKind.BANK_ACCOUNT) == []


@pytest.mark.parametrize(
    ("context", "address"),
    [
        ("domiciliado en {}, en adelante", "Av. Vitacura 9128, Providencia"),
        ("oficinas de {}.", "calle Condell 1190, oficina 61, Valparaíso"),
        ("vive en {};", "Pasaje Los Aromos N° 45, depto. 302, comuna de Viña del Mar"),
        ("ubicado en {} y", "Av. 11 de Septiembre 1860, Providencia"),
        ("domiciliados para estos efectos en {}, vienen", "Huérfanos 835, Santiago"),
        ("atención en {}.", "Los Militares 5001, oficina 402, Las Condes"),
        ("Señor\nCristian Oyarzún\n{}\nPresente", "Los Olmos 2278, Macul"),
    ],
)
def test_address(context, address):
    assert found(context.format(address), PiiKind.ADDRESS) == [address]


def test_a_city_or_a_street_without_number_is_not_an_address():
    text = "domiciliado en Santiago, cruzó la calle Merced y el Código Civil, artículo 12, rige."
    assert found(text, PiiKind.ADDRESS) == []


@pytest.mark.parametrize(
    ("context", "organization"),
    [
        ("Nuestro cliente {} (RUT", "Inversiones Los Boldos SpA"),
        ("en contra de {}, RUT", "Constructora Río Claro Ltda."),
        ("La contraparte, {}, no conoce", "Agrícola Santa Elvira S.A."),
        ("a nombre de {}.", "Comercial e Industrial Tres Pinos Limitada"),
        ("DEMANDADO: {}", "COMERCIAL TRES PINOS S.A."),
        ("la firma {} declara", "Lagos y Cía. Ltda."),
    ],
)
def test_organization_by_legal_form(context, organization):
    assert found(context.format(organization), PiiKind.ORGANIZATION) == [organization]


def test_sole_proprietorship_is_an_organization_not_a_person():
    spans = detect("factura emitida por Hernán Ossandón Vial E.I.R.L. en mayo")
    assert [(span.kind, span.text) for span in spans] == [
        (PiiKind.ORGANIZATION, "Hernán Ossandón Vial E.I.R.L.")
    ]


@pytest.mark.parametrize(
    ("context", "person"),
    [
        ("comparecen {}, cédula", "Hernán Ossandón Vial"),
        ("De: {} <p@x.example>", "Paula Riquelme"),
        ("don {}, chileno", "Pedro Pablo del Río Aguirre"),
        ("los abogados {} y otros", "Francisca De la Maza Urrutia"),
        ("y {}, ambos del estudio", "José Tomás Errázuriz Mc-Kay"),
        ("titular don {}, cédula", "Björn Åke Lindqvist O'Ryan"),
        ("Atentamente,\n\n{}\nGerente", "Patricio Elgueta S."),
        ("El señor {} se negó a firmar.", "Inostroza"),
        ("Hola {},\n\nTe cuento", "Cata"),
        ("Un abrazo,\n{}", "Nico"),
    ],
)
def test_person(context, person):
    assert found(context.format(person), PiiKind.PERSON) == [person]


@pytest.mark.parametrize(
    "text",
    [
        "DEMANDANTE: ROSA ELENA CAMPOS VERGARA\nRUT: 12.345.678-5",
        "en representación de doña ROSA ELENA CAMPOS VERGARA, pensionada",
        "ROSA ELENA CAMPOS VERGARA, abogada, por el imputado",
    ],
)
def test_person_in_capitals_needs_a_cue(text):
    assert found(text, PiiKind.PERSON) == ["ROSA ELENA CAMPOS VERGARA"]
    assert found("CONTRATO DE ARRENDAMIENTO\n\nPOR TANTO, ruego", PiiKind.PERSON) == []


@pytest.mark.parametrize(
    "text",
    [
        "En Santiago, a 3 de julio de 2024, comparecen",
        "PRIMERO: El Arrendador da en arrendamiento el Inmueble.",
        "del Banco de Chile y del Banco Santander, según el Código Civil",
        "ante la Corte de Apelaciones de Santiago y el Servicio de Impuestos Internos",
        "Señor Juez de Letras: el Sr. Gerente General no compareció.",
        "Estimados señores:",
        "Cuenta Corriente: 00123456",
    ],
)
def test_capitalized_phrases_that_are_not_people(text):
    assert found(text, PiiKind.PERSON) == []
