from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from typing import Self


class Zone(StrEnum):
    RAW = "raw"
    DEIDENTIFIED = "deidentified"
    SERVING = "serving"
    TRAINING = "training"


class Sensitivity(StrEnum):
    INTERNAL = "internal"
    CONFIDENTIAL = "confidential"
    PRIVILEGED = "privileged"


class DocumentKind(StrEnum):
    CONTRACT = "contract"
    FILING = "filing"
    EMAIL = "email"
    MEMO = "memo"


class PiiKind(StrEnum):
    PERSON = "person"
    ORGANIZATION = "organization"
    RUT = "rut"
    EMAIL = "email"
    PHONE = "phone"
    ADDRESS = "address"
    CASE_NUMBER = "case_number"
    BANK_ACCOUNT = "bank_account"


class Role(StrEnum):
    LAWYER = "lawyer"
    PARALEGAL = "paralegal"
    DATA_ENGINEER = "data_engineer"
    DATA_SCIENTIST = "data_scientist"
    PRIVACY_OFFICER = "privacy_officer"
    SERVICE = "service"


class Action(StrEnum):
    INGEST = "ingest"
    READ_RAW = "read_raw"
    READ_DEIDENTIFIED = "read_deidentified"
    RETRIEVE = "retrieve"
    REIDENTIFY = "reidentify"
    EXPORT_TRAINING = "export_training"
    AGGREGATE = "aggregate"
    EGRESS = "egress"
    ERASE = "erase"


class Purpose(StrEnum):
    LEGAL_ADVICE = "legal_advice"
    MODEL_DEVELOPMENT = "model_development"
    ANALYTICS = "analytics"
    DATA_SUBJECT_REQUEST = "data_subject_request"
    SECURITY_AUDIT = "security_audit"
    RETENTION = "retention"


class Obligation(StrEnum):
    KEEP_PSEUDONYMS = "keep_pseudonyms"
    NO_EGRESS = "no_egress"


@dataclass(frozen=True, slots=True)
class Matter:
    """A client engagement; `purposes` is what its engagement letter permits."""

    id: str
    client: str
    title: str
    purposes: frozenset[Purpose] = frozenset({Purpose.LEGAL_ADVICE})
    walled_off: frozenset[str] = frozenset()


@dataclass(frozen=True, slots=True)
class Document:
    id: str
    matter_id: str
    kind: DocumentKind
    sensitivity: Sensitivity
    text: str
    created_at: datetime


@dataclass(frozen=True, slots=True)
class Chunk:
    id: str
    document_id: str
    matter_id: str
    sensitivity: Sensitivity
    ordinal: int
    text: str


@dataclass(frozen=True, slots=True)
class PiiSpan:
    start: int
    end: int
    kind: PiiKind
    text: str


@dataclass(frozen=True, slots=True)
class Principal:
    id: str
    role: Role
    matters: frozenset[str] = frozenset()


@dataclass(frozen=True, slots=True)
class Resource:
    zone: Zone
    matter_id: str | None = None
    sensitivity: Sensitivity = Sensitivity.CONFIDENTIAL


@dataclass(frozen=True, slots=True)
class AccessRequest:
    principal: Principal
    action: Action
    purpose: Purpose
    resource: Resource
    at: datetime = field(default_factory=lambda: datetime.now(UTC))


@dataclass(frozen=True, slots=True)
class Decision:
    allowed: bool
    reason: str
    obligations: frozenset[Obligation] = frozenset()


@dataclass(frozen=True, slots=True)
class AuditEvent:
    at: datetime
    principal_id: str
    role: Role
    action: Action
    purpose: Purpose
    zone: Zone
    matter_id: str | None
    allowed: bool
    reason: str
    detail: Mapping[str, str] = field(default_factory=dict)

    @classmethod
    def of(
        cls,
        request: AccessRequest,
        decision: Decision,
        detail: Mapping[str, str] | None = None,
    ) -> Self:
        return cls(
            at=request.at,
            principal_id=request.principal.id,
            role=request.principal.role,
            action=request.action,
            purpose=request.purpose,
            zone=request.resource.zone,
            matter_id=request.resource.matter_id,
            allowed=decision.allowed,
            reason=decision.reason,
            detail=dict(detail or {}),
        )
