"""Synthetic Chilean legal corpus with labeled PII spans. Depicts no real person or case."""

import random
import unicodedata
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from string import Formatter
from typing import NamedTuple

from lexguard.domain import (
    Document,
    DocumentKind,
    Matter,
    PiiKind,
    PiiSpan,
    Principal,
    Purpose,
    Role,
    Sensitivity,
)
from lexguard.rut import format_rut

_FIRST_NAMES = (
    "María", "José", "Camila", "Matías", "Valentina", "Benjamín", "Francisca", "Sebastián",
    "Javiera", "Ignacio", "Catalina", "Felipe", "Constanza", "Rodrigo", "Daniela", "Cristóbal",
)  # fmt: skip
_SURNAMES = (
    "González", "Muñoz", "Rojas", "Díaz", "Pérez", "Soto", "Contreras", "Silva",
    "Martínez", "Sepúlveda", "Morales", "Rodríguez", "Castillo", "Araya", "Espinoza", "Tapia",
)  # fmt: skip
_COMPANY_FORMS = (
    "Inversiones {} SpA", "Constructora {} Ltda.", "Agrícola {} S.A.",
    "Transportes {} Ltda.", "Minera {} SpA", "Inmobiliaria {} S.A.",
)  # fmt: skip
_COMPANY_STEMS = (
    "Andes", "Pacífico", "Austral", "Cordillera", "Maipo", "Atacama", "Patagonia", "Biobío",
)  # fmt: skip
_STREETS = (
    "Av. Providencia", "Av. Apoquindo", "Calle Huérfanos", "Av. Vitacura",
    "Calle Merced", "Calle Agustinas", "Av. Irarrázaval", "Av. Los Leones",
)  # fmt: skip
_COMMUNES = ("Providencia", "Las Condes", "Santiago", "Ñuñoa", "Vitacura", "Maipú", "La Florida")
_BANKS = ("Banco de Chile", "BancoEstado", "Banco Santander", "Banco BCI")
_MAIL_DOMAINS = ("correo.example", "empresa.example", "estudio.example")
_MONTHS = (
    "enero", "febrero", "marzo", "abril", "mayo", "junio",
    "julio", "agosto", "septiembre", "octubre", "noviembre", "diciembre",
)  # fmt: skip
_MATTER_TITLES = (
    "Arbitraje comercial", "Juicio laboral", "Compraventa de activos",
    "Arrendamiento de oficinas", "Reorganización societaria", "Cobranza judicial",
)  # fmt: skip


class Pii(NamedTuple):
    kind: PiiKind
    text: str

    def __str__(self) -> str:
        return self.text


type Fields = dict[str, Pii | str]


@dataclass(frozen=True, slots=True)
class LabeledDocument:
    document: Document
    spans: tuple[PiiSpan, ...]


@dataclass(frozen=True, slots=True)
class Corpus:
    matters: tuple[Matter, ...]
    documents: tuple[LabeledDocument, ...]
    principals: tuple[Principal, ...]

    @property
    def matter_index(self) -> dict[str, Matter]:
        return {matter.id: matter for matter in self.matters}

    def principal(self, principal_id: str) -> Principal:
        return next(p for p in self.principals if p.id == principal_id)


def render(template: str, fields: Mapping[str, Pii | str]) -> tuple[str, tuple[PiiSpan, ...]]:
    parts: list[str] = []
    spans: list[PiiSpan] = []
    cursor = 0
    for literal, name, _, _ in Formatter().parse(template):
        parts.append(literal)
        cursor += len(literal)
        if name is None:
            continue
        value = fields[name]
        text = str(value)
        if isinstance(value, Pii):
            spans.append(PiiSpan(cursor, cursor + len(text), value.kind, text))
        parts.append(text)
        cursor += len(text)
    return "".join(parts), tuple(spans)


def generate(seed: int = 7, matters: int = 4, documents_per_matter: int = 6) -> Corpus:
    rng = random.Random(seed)
    fake = _Fake(rng)
    registry: list[Matter] = []
    documents: list[LabeledDocument] = []
    principals: list[Principal] = []

    for number, client in enumerate(fake.companies(matters), start=1):
        matter = Matter(
            id=f"M-{number:03d}",
            client=client.text,
            title=rng.choice(_MATTER_TITLES),
            purposes=_purposes(number),
            walled_off=frozenset({"lawyer-1"}) if number == 2 else frozenset(),
        )
        registry.append(matter)
        principals.append(Principal(f"lawyer-{number}", Role.LAWYER, frozenset({matter.id})))

        for index in range(documents_per_matter):
            template = _TEMPLATES[index % len(_TEMPLATES)]
            text, spans = render(template.body, template.cast(fake, client))
            document = Document(
                id=f"{matter.id}-D{index + 1:03d}",
                matter_id=matter.id,
                kind=template.kind,
                sensitivity=template.sensitivity,
                text=text,
                created_at=datetime(2024, 1, 1, tzinfo=UTC) + timedelta(days=rng.randrange(700)),
            )
            documents.append(LabeledDocument(document, spans))

    every_matter = frozenset(matter.id for matter in registry)
    if matters >= 2:
        principals[0] = Principal("lawyer-1", Role.LAWYER, frozenset({"M-001", "M-002"}))
    principals += [
        Principal("paralegal-1", Role.PARALEGAL, every_matter),
        Principal("engineer-1", Role.DATA_ENGINEER),
        Principal("scientist-1", Role.DATA_SCIENTIST),
        Principal("dpo-1", Role.PRIVACY_OFFICER),
        Principal("svc-pipeline", Role.SERVICE),
    ]
    return Corpus(tuple(registry), tuple(documents), tuple(principals))


def _purposes(number: int) -> frozenset[Purpose]:
    if number % 2:
        return frozenset({Purpose.LEGAL_ADVICE, Purpose.MODEL_DEVELOPMENT, Purpose.ANALYTICS})
    return frozenset({Purpose.LEGAL_ADVICE})


def _slug(text: str) -> str:
    return unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode().lower()


def _part(person: Pii, index: int) -> Pii:
    return Pii(PiiKind.PERSON, person.text.split()[index])


class _Fake:
    def __init__(self, rng: random.Random) -> None:
        self._rng = rng

    def person(self) -> Pii:
        first = self._rng.choice(_FIRST_NAMES)
        paternal, maternal = self._rng.sample(_SURNAMES, 2)
        return Pii(PiiKind.PERSON, f"{first} {paternal} {maternal}")

    def company(self) -> Pii:
        form, stem = self._rng.choice(_COMPANY_FORMS), self._rng.choice(_COMPANY_STEMS)
        return Pii(PiiKind.ORGANIZATION, form.format(stem))

    def companies(self, count: int) -> list[Pii]:
        names = [form.format(stem) for form in _COMPANY_FORMS for stem in _COMPANY_STEMS]
        return [Pii(PiiKind.ORGANIZATION, name) for name in self._rng.sample(names, count)]

    def rut(self, low: int = 5_000_000, high: int = 25_000_000) -> Pii:
        return Pii(PiiKind.RUT, format_rut(self._rng.randrange(low, high)))

    def company_rut(self) -> Pii:
        return self.rut(76_000_000, 78_000_000)

    def address(self) -> Pii:
        street, commune = self._rng.choice(_STREETS), self._rng.choice(_COMMUNES)
        return Pii(PiiKind.ADDRESS, f"{street} {self._rng.randrange(100, 9900)}, {commune}")

    def email(self, person: Pii) -> Pii:
        first, paternal, *_ = _slug(person.text).split()
        return Pii(PiiKind.EMAIL, f"{first}.{paternal}@{self._rng.choice(_MAIL_DOMAINS)}")

    def phone(self) -> Pii:
        number = self._rng.randrange(10_000_000, 100_000_000)
        return Pii(PiiKind.PHONE, f"+56 9 {number // 10_000:04d} {number % 10_000:04d}")

    def case_number(self, prefix: str) -> Pii:
        sequence, year = self._rng.randrange(1, 9999), self._rng.randrange(2019, 2026)
        return Pii(PiiKind.CASE_NUMBER, f"{prefix}-{sequence}-{year}")

    def account(self) -> Pii:
        return Pii(PiiKind.BANK_ACCOUNT, str(self._rng.randrange(10_000_000, 1_000_000_000)))

    def bank(self) -> str:
        return self._rng.choice(_BANKS)

    def date(self) -> str:
        day, year = self._rng.randrange(1, 29), self._rng.randrange(2021, 2026)
        return f"{day} de {self._rng.choice(_MONTHS)} de {year}"

    def amount(self) -> str:
        return f"{self._rng.randrange(350, 45_000) * 1000:,}".replace(",", ".")


def _lease(fake: _Fake, client: Pii) -> Fields:
    landlord, tenant = fake.person(), fake.person()
    return {
        "date": fake.date(),
        "landlord": landlord,
        "landlord_rut": fake.rut(),
        "landlord_address": fake.address(),
        "landlord_email": fake.email(landlord),
        "company": client,
        "company_rut": fake.company_rut(),
        "tenant": tenant,
        "tenant_rut": fake.rut(),
        "tenant_email": fake.email(tenant),
        "property": fake.address(),
        "rent": fake.amount(),
        "account": fake.account(),
        "bank": fake.bank(),
    }


def _claim(fake: _Fake, client: Pii) -> Fields:
    plaintiff, manager = fake.person(), fake.person()
    return {
        "plaintiff": plaintiff,
        "plaintiff_rut": fake.rut(),
        "plaintiff_address": fake.address(),
        "plaintiff_email": fake.email(plaintiff),
        "case": fake.case_number("O"),
        "company": client,
        "company_rut": fake.company_rut(),
        "company_address": fake.address(),
        "manager": manager,
        "manager_surname": _part(manager, 1),
        "date": fake.date(),
        "salary": fake.amount(),
        "phone": fake.phone(),
    }


def _advice(fake: _Fake, client: Pii) -> Fields:
    lawyer, contact = fake.person(), fake.person()
    return {
        "lawyer": lawyer,
        "lawyer_email": fake.email(lawyer),
        "contact": contact,
        "contact_first_name": _part(contact, 0),
        "contact_email": fake.email(contact),
        "case": fake.case_number("C"),
        "company": client,
        "counterpart": fake.company(),
        "amount": fake.amount(),
        "date": fake.date(),
        "phone": fake.phone(),
    }


def _memo(fake: _Fake, client: Pii) -> Fields:
    manager = fake.person()
    return {
        "author": fake.person(),
        "date": fake.date(),
        "company": client,
        "company_rut": fake.company_rut(),
        "buyer": fake.company(),
        "amount": fake.amount(),
        "manager": manager,
        "manager_rut": fake.rut(),
        "manager_email": fake.email(manager),
        "case": fake.case_number("C"),
        "property": fake.address(),
        "phone": fake.phone(),
    }


@dataclass(frozen=True, slots=True)
class _Template:
    kind: DocumentKind
    sensitivity: Sensitivity
    cast: Callable[[_Fake, Pii], Fields]
    body: str


_TEMPLATES = (
    _Template(
        DocumentKind.CONTRACT,
        Sensitivity.CONFIDENTIAL,
        _lease,
        """\
CONTRATO DE ARRENDAMIENTO

En Santiago, a {date}, comparecen {landlord}, cédula de identidad N° {landlord_rut}, \
domiciliado en {landlord_address}, en adelante el "Arrendador", y {company}, \
RUT {company_rut}, representada por {tenant}, cédula de identidad N° {tenant_rut}, \
en adelante el "Arrendatario", quienes convienen lo siguiente:

PRIMERO: El Arrendador da en arrendamiento el inmueble ubicado en {property}.

SEGUNDO: La renta mensual será de ${rent}, pagadera dentro de los primeros cinco días \
de cada mes en la cuenta corriente N° {account} del {bank}.

TERCERO: Para efectos de notificaciones, las partes fijan los correos electrónicos \
{landlord_email} y {tenant_email}.
""",
    ),
    _Template(
        DocumentKind.FILING,
        Sensitivity.INTERNAL,
        _claim,
        """\
EN LO PRINCIPAL: Demanda por despido injustificado y cobro de prestaciones.

S.J.L. DEL TRABAJO DE SANTIAGO

{plaintiff}, RUT {plaintiff_rut}, domiciliado en {plaintiff_address}, en causa RIT {case}, \
a US. respetuosamente digo:

Que interpongo demanda en contra de {company}, RUT {company_rut}, representada legalmente \
por {manager}, ambos con domicilio en {company_address}.

Con fecha {date} fui despedido sin expresión de causa. A esa fecha mi remuneración mensual \
ascendía a ${salary}. El señor {manager_surname} se negó a firmar el finiquito.

POR TANTO, ruego a US. tener por interpuesta la demanda y acogerla en todas sus partes. \
Para notificaciones señalo el correo {plaintiff_email} y el teléfono {phone}.
""",
    ),
    _Template(
        DocumentKind.EMAIL,
        Sensitivity.PRIVILEGED,
        _advice,
        """\
De: {lawyer} <{lawyer_email}>
Para: {contact} <{contact_email}>
Asunto: Estrategia en causa Rol {case}

Estimado {contact_first_name}:

Revisé los antecedentes de {company} y mi recomendación es no contestar todavía el \
requerimiento, sino preparar una oferta de avenimiento por ${amount} antes de la audiencia \
del {date}. La contraparte, {counterpart}, no conoce aún el informe pericial.

Esta comunicación es confidencial y está amparada por el secreto profesional. \
Cualquier duda, llámeme al {phone}.

Saludos,
{lawyer}
""",
    ),
    _Template(
        DocumentKind.MEMO,
        Sensitivity.PRIVILEGED,
        _memo,
        """\
MEMORÁNDUM INTERNO

Para: Equipo del asunto
De: {author}
Fecha: {date}

Nuestro cliente {company} (RUT {company_rut}) recibió una oferta de compra de {buyer} \
por ${amount}. El gerente general, {manager} (RUT {manager_rut}), pide evaluar \
contingencias antes de firmar.

Pendientes: revisar la causa Rol {case} y confirmar la titularidad del inmueble de \
{property}. Contacto del cliente: {manager_email}, {phone}.
""",
    ),
)
