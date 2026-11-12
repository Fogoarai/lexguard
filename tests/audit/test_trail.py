from datetime import UTC, datetime, timedelta

import pytest

from lexguard.audit import AuditTrail, ChainedAuditLog
from lexguard.domain import Action, AuditEvent, Principal, Purpose, Role, Zone
from lexguard.errors import AccessDenied
from lexguard.guard import Guard
from lexguard.testing import AllowAll, DenyAll

OFFICER = Principal("dpo-1", Role.PRIVACY_OFFICER)
START = datetime(2026, 12, 1, 9, tzinfo=UTC)


def retrieval(principal_id: str, matter_id: str, allowed: bool, minute: int) -> AuditEvent:
    return AuditEvent(
        at=START + timedelta(minutes=minute),
        principal_id=principal_id,
        role=Role.LAWYER,
        action=Action.RETRIEVE,
        purpose=Purpose.LEGAL_ADVICE,
        zone=Zone.SERVING,
        matter_id=matter_id,
        allowed=allowed,
        reason="scripted",
    )


HISTORY = [
    retrieval("lawyer-1", "M-001", True, 0),
    retrieval("lawyer-1", "M-002", False, 1),
    retrieval("lawyer-2", "M-002", True, 2),
]


@pytest.fixture
def log(tmp_path):
    log = ChainedAuditLog(tmp_path / "audit.jsonl", b"k" * 32)
    for event in HISTORY:
        log.record(event)
    return log


def test_a_cleared_reader_gets_the_events_asked_for(log):
    trail = AuditTrail(log, Guard(AllowAll(), log))

    assert trail.events(OFFICER, principal_id="lawyer-1") == HISTORY[:2]
    assert trail.events(OFFICER, matter_id="M-002", denied_only=True) == [HISTORY[1]]


def test_a_read_of_the_trail_is_asked_as_a_security_audit_and_lands_on_it(log):
    trail = AuditTrail(log, Guard(AllowAll(), log))

    seen = trail.events(OFFICER, principal_id="lawyer-1", matter_id="M-002", denied_only=True)

    read = log.events()[-1]
    assert seen == [HISTORY[1]]
    assert (read.principal_id, read.action, read.zone, read.purpose, read.matter_id) == (
        "dpo-1",
        Action.READ_AUDIT,
        Zone.AUDIT,
        Purpose.SECURITY_AUDIT,
        "M-002",
    )
    assert read.detail == {"about": "lawyer-1", "denied_only": "true"}
    assert trail.events(OFFICER)[-2:] == [read, log.events()[-1]]


def test_a_refused_reader_gets_nothing_and_the_refusal_is_recorded(log):
    trail = AuditTrail(log, Guard(DenyAll(), log))

    with pytest.raises(AccessDenied):
        trail.events(Principal("lawyer-1", Role.LAWYER))

    refusal = log.events()[-1]
    assert (refusal.principal_id, refusal.action, refusal.allowed) == (
        "lawyer-1",
        Action.READ_AUDIT,
        False,
    )
    assert log.verify() == len(HISTORY) + 1
