import pytest

from lexguard.domain import AccessRequest, Action, Principal, Purpose, Resource, Role, Zone
from lexguard.errors import AccessDenied
from lexguard.guard import Guard
from lexguard.settings import Boundary, Settings
from lexguard.testing import AllowAll, DenyAll, MembershipPolicy, MemoryAuditLog

LAWYER = Principal("lawyer-1", Role.LAWYER, frozenset({"M-001"}))


def request(matter_id: str) -> AccessRequest:
    return AccessRequest(
        LAWYER, Action.RETRIEVE, Purpose.LEGAL_ADVICE, Resource(Zone.SERVING, matter_id)
    )


def test_allowed_request_is_audited():
    audit = MemoryAuditLog()
    decision = Guard(AllowAll(), audit).authorize(request("M-001"), query="plazo")
    assert decision.allowed
    (event,) = audit.events
    assert (event.principal_id, event.matter_id, event.allowed) == ("lawyer-1", "M-001", True)
    assert event.detail == {"query": "plazo"}


def test_denied_request_raises_after_auditing():
    audit = MemoryAuditLog()
    with pytest.raises(AccessDenied, match="deny-all"):
        Guard(DenyAll(), audit).authorize(request("M-001"))
    assert [event.allowed for event in audit.events] == [False]


def test_evaluate_does_not_raise():
    guard = Guard(MembershipPolicy(), MemoryAuditLog())
    assert guard.evaluate(request("M-001")).allowed
    assert not guard.evaluate(request("M-002")).allowed


def test_settings_default_to_the_strictest_boundary():
    assert Settings.from_env({}).boundary is Boundary.LOCAL_ONLY


def test_settings_read_the_environment():
    settings = Settings.from_env(
        {"LEXGUARD_BOUNDARY": "deidentified_egress", "LEXGUARD_MIN_GROUP_SIZE": "10"}
    )
    assert settings == Settings(Boundary.DEIDENTIFIED_EGRESS, min_group_size=10)
