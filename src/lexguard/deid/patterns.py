"""Shapes and context cues of PII in Chilean legal Spanish.

The word lists are generic cues (function words, honorifics, institutional nouns, street
types, legal forms). None of them is a list of known names, companies or streets.
"""

import re

from lexguard.deid.canonical import NAME_PARTICLES


def _words(text: str) -> frozenset[str]:
    return frozenset(text.split())


def _either(words: frozenset[str]) -> str:
    return "|".join(sorted(words, key=lambda word: (-len(word), word)))


LEADING_WORDS = _words("""
    a al ante bajo con contra de del desde durante en entre hacia hasta mediante para por
    según sin sobre tras el la los las lo un una unos unas y e o u ni que si no como cuando
    donde además asimismo también este esta estos estas ese esa dicho dicha mi su sus
    nuestro nuestra nuestros nuestras don doña señor señora señorita señores sr sra srta
    estimado estimada estimados estimadas querido querida hola abogado abogada gerente
    general juez jueza ministro ministra notario notaria presidente presidenta director
    directora secretario secretaria doctor doctora
""")
NOT_PERSONS = _words("""
    banco tribunal tribunales juzgado corte ministerio servicio registro notaría conservador
    tesorería dirección inspección superintendencia municipalidad universidad república
    región estado fisco poder código ley decreto reglamento constitución diario comisión
    contraloría fiscalía defensoría cámara senado congreso gobierno instituto fundación
    corporación sociedad compañía empresa hospital clínica colegio policía carabineros
    unidad unidades chile calle avenida pasaje camino
""")

_UP = "A-ZÀ-ÖØ-Þ"
_LOW = "a-zß-öø-ÿ"
_SP = r"[ \t]+"
_NUMBER_SIGN = r"(?:N[°ºo]\.?|[Nn]úmero|#)"
_TITLE_WORD = rf"(?:[{_UP}]['’])?[{_UP}][{_LOW}]+(?:-[{_UP}][{_LOW}]+)?"
_CAPS_WORD = rf"[{_UP}]{{2,}}(?:-[{_UP}]{{2,}})?"
_ANY_CASE_WORD = rf"[{_UP}][{_UP}{_LOW}'’-]+"

EMAIL = re.compile(r"(?<![\w.+-])[\w.+-]+@[\w-]+(?:\.[\w-]+)+")

RUT = re.compile(r"(?<![\w.-])(?:\d{1,2}(?:\.\d{3}){2}|\d{7,8})-[\dkK](?![\w-])")
RUT_CUED = re.compile(
    r"(?i:\bR\.?U\.?[TN]\b\.?|\bc[ée]dula(?:\s+nacional)?\s+de\s+identidad|\bC\.I\.)"
    rf"\s*:?\s*(?:{_NUMBER_SIGN}\s*)?(?P<pii>\d{{1,2}}\.?\d{{3}}\.?\d{{3}}-?[\dkK])(?![\w-])"
)

PHONE_INTERNATIONAL = re.compile(r"(?<![\w+])\+56(?:[ ()-]{0,2}\d){9}(?!\d)")
PHONE_MOBILE = re.compile(r"(?<![\w+-])9 ?\d{4} \d{4}(?![\w-])")
PHONE_CUED = re.compile(
    r"(?i:(?<!\w)(?:tel[ée]fonos?|fonos?|celular(?:es)?|m[óo]vil|fijo|whatsapp|tel\.|cel\.))"
    r"[^\d\n+]{0,25}?(?P<pii>(?:\d[ -]?){7,8}\d)(?![\w-])"
)

_SEQUENCE = r"(?:\d{1,3}\.\d{3}|\d{1,6})-(?:19|20)\d{2}"
CASE_NUMBER = re.compile(rf"(?<![\w-])[A-Z]{{1,2}}-{_SEQUENCE}(?![\w-])")
CASE_NUMBER_CUED = re.compile(
    rf"\b(?:[Rr]ol|ROL|RIT|RUC)\b(?:{_SP}[^\W\d]{{1,12}})?"
    rf"\s*:?\s*(?:{_NUMBER_SIGN}\s*)?"
    rf"(?P<pii>{_SEQUENCE}|\d{{10}}-[\dkK]|\d{{2}}-\d-\d{{7}}-[\dkK])(?![\w-])"
)

_ACCOUNT_TYPE = r"(?i:corriente|cte\.?|vista|rut|bancaria|de\s+ahorros?)"
BANK_ACCOUNT = re.compile(
    rf"(?i:(?<!\w)(?:cuenta|cta\.))(?:\s+{_ACCOUNT_TYPE}[^\d\n.;]{{0,40}}?"
    rf"|[^\d\n.;]{{0,40}}?{_NUMBER_SIGN}\s*|\s*:\s*)(?P<pii>\d[\d-]{{4,18}}\d)(?![\w-])"
)

_STREET_TYPE = (
    r"(?:(?i:avenida|avda\.|av\.|calle|pasaje|psje\.|pje\.)"
    r"|Camino|Paseo|Ruta|Carretera|Alameda|Costanera|Callejón)"
)
_STREET_WORD = rf"[{_UP}0-9][\w'’]*\.?"
_STREET_LINK = r"(?:de|del|la|las|los|el|y|a|al)"
_STREET_NAME = (
    rf"(?:{_STREET_LINK}{_SP})?{_STREET_WORD}"
    rf"(?:{_SP}(?:{_STREET_LINK}{_SP}){{0,2}}{_STREET_WORD}){{0,4}}"
)
_HOUSE_NUMBER = rf"{_SP}(?:{_NUMBER_SIGN}[ \t]*)?\d{{1,5}}(?!\w|\.\d)"
_UNIT = (
    rf",?{_SP}(?i:oficina|of\.|departamento|depto\.?|dpto\.?|piso|local|casa|torre|block)"
    rf"{_SP}(?:{_NUMBER_SIGN}[ \t]*)?[\w-]{{1,6}}"
)
_LOCALITY = (
    rf",{_SP}(?:(?i:comuna|ciudad){_SP}de{_SP})?{_TITLE_WORD}"
    rf"(?:{_SP}(?:(?:de|del|la|las|los){_SP}){{0,2}}{_TITLE_WORD}){{0,3}}"
)
_PREMISES = rf"{_STREET_NAME}{_HOUSE_NUMBER}(?:{_UNIT})*"
_ADDRESS_TAIL = rf"{_PREMISES}(?:{_LOCALITY}){{0,2}}"
ADDRESS = re.compile(rf"(?<!\w){_STREET_TYPE}{_SP}{_ADDRESS_TAIL}")
ADDRESS_CUED = re.compile(
    r"(?i:domiciliad[oa]s?|domicilio|ubicad[oa]s?)\b[^.;\n]{0,30}?\ben\s+"
    rf"(?P<pii>(?:{_STREET_TYPE}{_SP})?{_ADDRESS_TAIL})"
)
ADDRESS_WITH_UNIT = re.compile(
    rf"(?<![\w.])(?=[{_UP}]){_STREET_NAME}{_HOUSE_NUMBER}(?:{_UNIT})+(?:{_LOCALITY}){{0,2}}"
)
ADDRESS_LINE = re.compile(
    rf"^[ \t]*(?P<pii>(?=[{_UP}]){_PREMISES}(?:{_LOCALITY}){{1,2}})[ \t]*$", re.MULTILINE
)

_LEGAL_FORM = r"(?:S\.p\.A\.|SpA|SPA|Ltda\.?|LTDA\.?|Limitada|LIMITADA|S\.A\.|SA|E\.I\.R\.L\.|EIRL)"
_ORGANIZATION_WORD = rf"(?:[{_UP}0-9][\w&'’-]*|C[ií]a\.)"
ORGANIZATION = re.compile(
    rf"(?<![\w.])(?!(?i:{_either(LEADING_WORDS)})[ \t])(?=[{_UP}]){_ORGANIZATION_WORD}"
    rf"(?:{_SP}(?:(?:de|del|la|las|los|y|e|&){_SP}){{0,2}}{_ORGANIZATION_WORD}){{0,6}}"
    rf"{_SP}{_LEGAL_FORM}(?!\w)"
)

_PARTICLES = rf"(?:(?:{_either(NAME_PARTICLES)}){_SP}){{0,2}}"
_HONORIFIC = (
    r"(?:don|doña|señor(?:a|ita|es)?|sr\.?|sra\.?|srta\.?|sres\.?|estimad[oa]s?|querid[oa]s?|hola)"
)
_PARTY_LABEL = (
    r"(?:demandante|demandad[oa]|abogad[oa](?:[ \t]+patrocinante)?|apoderad[oa]"
    r"|representante(?:[ \t]+legal)?|recurrente|recurrid[oa]|querellante|querellad[oa]"
    r"|ejecutante|ejecutad[oa]|solicitante|denunciante|denunciad[oa])[ \t]*:"
)
_SIGN_OFF = r"(?:saludos|atentamente|cordialmente|abrazos?|slds|atte)[,.]?"
_IDENTITY_CUE = (
    r"(?:(?:abogad[oa]|chilen[oa]|casad[oa]|solter[oa]|viud[oa]|divorciad[oa]|domiciliad[oa]"
    r"|c[ée]dula|RU[TN])\b|C\.I\.)"
)
NAME = re.compile(
    rf"(?<![\w.@])(?P<name>{_TITLE_WORD}(?:{_SP}{_PARTICLES}{_TITLE_WORD})*"
    rf"(?:{_SP}[{_UP}]\.)?)(?![\w@])"
)
NAME_AFTER_CUE = re.compile(
    rf"(?<!\w)(?i:(?:{_HONORIFIC}|{_PARTY_LABEL}){_SP}|{_SIGN_OFF}\s+)"
    rf"(?P<name>{_ANY_CASE_WORD}(?:{_SP}(?i:{_PARTICLES}){_ANY_CASE_WORD})*)"
)
NAME_IN_CAPITALS = re.compile(
    rf"(?<![\w.])(?P<name>{_CAPS_WORD}(?:{_SP}{_CAPS_WORD}){{1,6}})(?=,\s+(?i:{_IDENTITY_CUE}))"
)
