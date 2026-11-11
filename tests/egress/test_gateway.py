import re
from contextlib import suppress
from dataclasses import dataclass, field

import pytest

from lexguard.domain import (
    AccessRequest,
    Action,
    Decision,
    Obligation,
    PiiKind,
    PiiSpan,
    Principal,
    Purpose,
    Role,
    Sensitivity,
    Zone,
)
from lexguard.egress import EgressGateway
from lexguard.errors import EgressBlocked
from lexguard.guard import Guard
from lexguard.settings import Boundary, ModelLocation, Settings
from lexguard.testing import AllowAll, MembershipPolicy, MemoryAuditLog

LAWYER = Principal("lawyer-1", Role.LAWYER, frozenset({"M-001", "M-002"}))
CONFIDENTIAL = ("M-001", Sensitivity.CONFIDENTIAL)
PRIVILEGED = ("M-001", Sensitivity.PRIVILEGED)
OTHER_MATTER = ("M-002", Sensitivity.CONFIDENTIAL)
CLEAN = "Resume la cláusula de renta pactada con [PERSON_3fa9c2d1e0b4]."
LEAKY = "Resume la cláusula de renta pactada con 12.345.678-5 y 9.876.543-3."
REACHABLE = {
    Boundary.LOCAL_ONLY: {ModelLocation.LOCAL},
    Boundary.PRIVATE_CLOUD: {ModelLocation.LOCAL, ModelLocation.PRIVATE_CLOUD},
    Boundary.DEIDENTIFIED_EGRESS: set(ModelLocation),
}
_RUT = re.compile(r"\d{1,2}\.\d{3}\.\d{3}-[\dkK]")
_DIGITS = re.compile(r"\d{8,}")


@dataclass
class FakeModel:
    location: ModelLocation
    name: str = "fake-model"
    prompts: list[str] = field(default_factory=list)

    def complete(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return "respuesta"


class NoEgress:
    def decide(self, request: AccessRequest) -> Decision:
        return Decision(True, "allowed inside the firm", frozenset({Obligation.NO_EGRESS}))


def scan_ruts(text: str) -> list[PiiSpan]:
    return [PiiSpan(m.start(), m.end(), PiiKind.RUT, m.group()) for m in _RUT.finditer(text)]


def scan_accounts(text: str) -> list[PiiSpan]:
    return [
        PiiSpan(m.start(), m.end(), PiiKind.BANK_ACCOUNT, m.group()) for m in _DIGITS.finditer(text)
    ]


def unreachable_scan(text: str) -> list[PiiSpan]:
    raise AssertionError("a prompt for a local model must not be scanned")


@pytest.fixture
def audit():
    return MemoryAuditLog()


@pytest.fixture
def send(audit):
    def _send(
        model,
        prompt=CLEAN,
        *,
        policy=None,
        boundary=Boundary.DEIDENTIFIED_EGRESS,
        scan=scan_ruts,
        compartments=(CONFIDENTIAL,),
        obligations=(),
    ):
        gateway = EgressGateway(Guard(policy or AllowAll(), audit), Settings(boundary), scan)
        return gateway.complete(
            model,
            prompt,
            principal=LAWYER,
            purpose=Purpose.LEGAL_ADVICE,
            compartments=compartments,
            obligations=obligations,
        )

    return _send


def outcomes(audit):
    return [event for event in audit.events if "prompt_chars" in event.detail]


def policy_decisions(audit):
    return [event for event in audit.events if "prompt_chars" not in event.detail]


@pytest.mark.parametrize("boundary", Boundary)
@pytest.mark.parametrize("location", ModelLocation)
def test_boundary_decides_which_locations_are_reachable(send, audit, boundary, location):
    model = FakeModel(location)

    if location in REACHABLE[boundary]:
        assert send(model, boundary=boundary) == "respuesta"
        assert model.prompts == [CLEAN]
    else:
        with pytest.raises(EgressBlocked, match=f"{boundary} does not admit {location}"):
            send(model, boundary=boundary)
        assert model.prompts == []
        assert policy_decisions(audit) == []

    (outcome,) = outcomes(audit)
    assert outcome.allowed is (location in REACHABLE[boundary])
    assert outcome.detail["boundary"] == boundary
    assert outcome.detail["location"] == location


def test_policy_is_asked_once_per_compartment_on_behalf_of_the_principal(send, audit):
    send(
        FakeModel(ModelLocation.LOCAL),
        compartments=[CONFIDENTIAL, PRIVILEGED, OTHER_MATTER, CONFIDENTIAL, PRIVILEGED],
    )

    asked = policy_decisions(audit)
    assert [(event.matter_id, event.detail["sensitivity"]) for event in asked] == [
        ("M-001", "confidential"),
        ("M-001", "privileged"),
        ("M-002", "confidential"),
    ]
    for event in asked:
        assert (event.principal_id, event.action, event.zone, event.purpose) == (
            "lawyer-1",
            Action.EGRESS,
            Zone.SERVING,
            Purpose.LEGAL_ADVICE,
        )
        assert event.detail["model"] == "fake-model"


@pytest.mark.parametrize("location", ModelLocation)
def test_policy_denial_blocks_every_location(send, audit, location):
    model = FakeModel(location)
    walled = ("M-003", Sensitivity.CONFIDENTIAL)

    with pytest.raises(EgressBlocked, match=r"policy denied egress of M-003/confidential"):
        send(model, policy=MembershipPolicy(), compartments=[CONFIDENTIAL, walled])

    assert model.prompts == []
    assert [event.allowed for event in policy_decisions(audit)] == [True, False]
    assert [(event.matter_id, event.allowed) for event in outcomes(audit)] == [
        ("M-001", False),
        ("M-003", False),
    ]


@pytest.mark.parametrize("location", ModelLocation)
def test_no_egress_obligation_means_local_models_only(send, location):
    model = FakeModel(location)

    if location is ModelLocation.LOCAL:
        assert send(model, policy=NoEgress()) == "respuesta"
    else:
        with pytest.raises(EgressBlocked, match=f"no_egress obligation forbids {location}"):
            send(model, policy=NoEgress())
        assert model.prompts == []


def test_obligations_carried_with_the_data_are_honoured(send):
    carried = [Obligation.NO_EGRESS]
    external = FakeModel(ModelLocation.EXTERNAL)

    with pytest.raises(EgressBlocked, match="no_egress"):
        send(external, obligations=carried)

    assert external.prompts == []
    assert send(FakeModel(ModelLocation.LOCAL), obligations=carried) == "respuesta"


@pytest.mark.parametrize("location", [ModelLocation.PRIVATE_CLOUD, ModelLocation.EXTERNAL])
def test_residual_pii_blocks_a_model_that_is_not_local(send, audit, location):
    model = FakeModel(location)

    with pytest.raises(EgressBlocked) as blocked:
        send(model, LEAKY)

    assert str(blocked.value) == "residual PII in outbound prompt: 2 rut"
    assert model.prompts == []
    (outcome,) = outcomes(audit)
    assert not outcome.allowed
    assert outcome.reason == str(blocked.value)


def test_a_local_model_is_not_scanned(send):
    assert send(FakeModel(ModelLocation.LOCAL), LEAKY, scan=unreachable_scan) == "respuesta"


def test_a_scanner_match_inside_a_pseudonym_is_not_residual_pii(send):
    external = FakeModel(ModelLocation.EXTERNAL)
    masked = "Depositar en la cuenta [BANK_ACCOUNT_123456789012]."

    assert send(external, masked, scan=scan_accounts) == "respuesta"
    with pytest.raises(EgressBlocked, match="1 bank_account"):
        send(external, "Depositar en la cuenta 123456789.", scan=scan_accounts)


@pytest.mark.parametrize("prompt", [CLEAN, LEAKY])
def test_every_attempt_is_audited_without_the_prompt(send, audit, prompt):
    model = FakeModel(ModelLocation.EXTERNAL)
    with suppress(EgressBlocked):
        send(model, prompt, compartments=[CONFIDENTIAL, PRIVILEGED])

    (outcome,) = outcomes(audit)
    assert outcome.allowed is (prompt == CLEAN)
    assert (outcome.principal_id, outcome.action, outcome.matter_id) == (
        "lawyer-1",
        Action.EGRESS,
        "M-001",
    )
    assert outcome.detail == {
        "model": "fake-model",
        "location": "external",
        "boundary": "deidentified_egress",
        "matters": "1",
        "sensitivities": "confidential/privileged",
        "prompt_chars": str(len(prompt)),
        "pseudonyms": "1" if prompt == CLEAN else "0",
    }
    recorded = repr(audit.events)
    assert prompt not in recorded
    assert "12.345.678-5" not in recorded
    assert "[PERSON_3fa9c2d1e0b4]" not in recorded


def test_the_outcome_is_audited_per_matter_before_the_model_runs(send, audit):
    class Failing(FakeModel):
        def complete(self, prompt: str) -> str:
            raise ConnectionError("model is down")

    with pytest.raises(ConnectionError):
        send(Failing(ModelLocation.LOCAL), compartments=[OTHER_MATTER, PRIVILEGED, CONFIDENTIAL])

    assert [
        (event.matter_id, event.allowed, event.reason, event.detail["sensitivities"])
        for event in outcomes(audit)
    ] == [
        ("M-001", True, "admitted", "confidential/privileged"),
        ("M-002", True, "admitted", "confidential"),
    ]
    assert {event.detail["matters"] for event in outcomes(audit)} == {"2"}


def test_audit_detail_values_are_short_identifiers(send, audit):
    identifier = re.compile(r"[\w./-]{1,64}")

    send(FakeModel(ModelLocation.LOCAL), compartments=[CONFIDENTIAL, PRIVILEGED, OTHER_MATTER])

    values = [value for event in audit.events for value in event.detail.values()]
    assert all(identifier.fullmatch(value) for value in values)


def test_content_read_in_another_zone_is_judged_in_that_zone(audit):
    gateway = EgressGateway(Guard(AllowAll(), audit), Settings(), scan_ruts)

    gateway.complete(
        FakeModel(ModelLocation.LOCAL),
        CLEAN,
        principal=LAWYER,
        purpose=Purpose.MODEL_DEVELOPMENT,
        compartments=[CONFIDENTIAL],
        zone=Zone.DEIDENTIFIED,
    )

    assert {(event.zone, event.purpose) for event in audit.events} == {
        (Zone.DEIDENTIFIED, Purpose.MODEL_DEVELOPMENT)
    }


def test_a_prompt_that_declares_no_compartment_is_blocked(send, audit):
    model = FakeModel(ModelLocation.LOCAL)

    with pytest.raises(EgressBlocked, match="no compartment"):
        send(model, compartments=[])

    assert model.prompts == []
    assert [(event.matter_id, event.allowed) for event in audit.events] == [(None, False)]
