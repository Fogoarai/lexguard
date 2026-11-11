import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from lexguard.audit import ChainedAuditLog
from lexguard.audit.log import GENESIS
from lexguard.domain import (
    AccessRequest,
    Action,
    AuditEvent,
    Principal,
    Purpose,
    Resource,
    Role,
    Zone,
)
from lexguard.errors import AccessDenied, IntegrityError
from lexguard.guard import Guard
from lexguard.synthetic import generate
from lexguard.testing import MembershipPolicy

KEY, OTHER_KEY = b"k" * 32, b"o" * 32
START = datetime(2026, 12, 1, 9, tzinfo=UTC)


def event(
    principal_id: str = "lawyer-1",
    matter_id: str = "M-001",
    allowed: bool = True,
    minute: int = 0,
    **detail: str,
) -> AuditEvent:
    return AuditEvent(
        at=START + timedelta(minutes=minute),
        principal_id=principal_id,
        role=Role.LAWYER,
        action=Action.RETRIEVE,
        purpose=Purpose.LEGAL_ADVICE,
        zone=Zone.SERVING,
        matter_id=matter_id,
        allowed=allowed,
        reason="lawyer-on-assigned-matter" if allowed else "ethical-wall",
        detail=detail,
    )


HISTORY = [
    event("lawyer-1", "M-001", minute=0, document="M-001-D001"),
    event("lawyer-1", "M-002", allowed=False, minute=1),
    event("lawyer-2", "M-002", minute=2),
    event("lawyer-2", "M-001", allowed=False, minute=3),
]


@pytest.fixture
def path(tmp_path) -> Path:
    return tmp_path / "audit" / "log.jsonl"


@pytest.fixture
def log(path) -> ChainedAuditLog:
    log = ChainedAuditLog(path, KEY)
    for entry in HISTORY:
        log.record(entry)
    return log


def rewrite(path: Path, edit) -> None:
    lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
    path.write_text("".join(edit(lines)), encoding="utf-8")


def test_events_round_trip_through_the_file(log, path):
    assert log.events() == HISTORY
    assert ChainedAuditLog(path, KEY).events() == HISTORY


def test_each_record_is_one_json_line_chained_to_the_previous_one(log, path):
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert [record["seq"] for record in records] == [1, 2, 3, 4]
    assert [record["prev"] for record in records] == [
        GENESIS,
        *(record["mac"] for record in records[:-1]),
    ]
    assert len({record["mac"] for record in records}) == 4


def test_intact_log_verifies(log):
    assert log.verify() == len(HISTORY)


def test_empty_log_verifies(path):
    assert ChainedAuditLog(path, KEY).verify() == 0


def test_altered_record_is_pinpointed(log, path):
    def turn_denial_into_approval(lines):
        assert '"allowed":false' in lines[1]
        return [lines[0], lines[1].replace('"allowed":false', '"allowed":true'), *lines[2:]]

    rewrite(path, turn_denial_into_approval)
    with pytest.raises(IntegrityError, match="record 2 was altered"):
        log.verify()


def test_removed_record_is_pinpointed(log, path):
    rewrite(path, lambda lines: [lines[0], *lines[2:]])
    with pytest.raises(IntegrityError, match="record 2 is missing or out of order: record 3"):
        log.verify()


def test_reordered_records_are_pinpointed(log, path):
    rewrite(path, lambda lines: [lines[0], lines[1], lines[3], lines[2]])
    with pytest.raises(IntegrityError, match="record 3 is missing or out of order: record 4"):
        log.verify()


def test_replayed_record_is_pinpointed(log, path):
    rewrite(path, lambda lines: [*lines, lines[0]])
    with pytest.raises(IntegrityError, match="record 5 is missing or out of order: record 1"):
        log.verify()


def test_record_spliced_from_another_log_under_the_same_key_does_not_fit(log, path, tmp_path):
    other_path = tmp_path / "other.jsonl"
    other = ChainedAuditLog(other_path, KEY)
    other.record(event("lawyer-3", "M-003"))
    other.record(event("lawyer-1", "M-002", allowed=True, minute=1))
    approval = other_path.read_text(encoding="utf-8").splitlines(keepends=True)[1]

    rewrite(path, lambda lines: [lines[0], approval, *lines[2:]])
    with pytest.raises(IntegrityError, match="record 2 is missing or out of order"):
        log.verify()


def test_stripped_final_newline_is_reported_before_it_can_corrupt_an_append(log, path):
    rewrite(path, lambda lines: [*lines[:-1], lines[-1].rstrip("\n")])
    with pytest.raises(IntegrityError, match="record 4 was altered"):
        ChainedAuditLog(path, KEY)


def test_garbage_is_reported_as_an_alteration(log, path):
    rewrite(path, lambda lines: [*lines[:2], "not json\n", *lines[2:]])
    with pytest.raises(IntegrityError, match="record 3 was altered"):
        log.verify()


def test_truncated_tail_is_noticed_by_an_instance_that_saw_it(log, path):
    rewrite(path, lambda lines: lines[:-1])
    with pytest.raises(IntegrityError, match="record 4 is missing"):
        log.verify()

    log.record(event(minute=5))
    with pytest.raises(IntegrityError, match="record 4 is missing or out of order: record 5"):
        ChainedAuditLog(path, KEY)


def test_chain_cannot_be_rebuilt_without_the_key(log, path, tmp_path):
    forged_path = tmp_path / "forged.jsonl"
    forger = ChainedAuditLog(forged_path, OTHER_KEY)
    for entry in HISTORY:
        forger.record(event(entry.principal_id, entry.matter_id, allowed=True))
    assert forger.verify() == len(HISTORY)

    path.write_bytes(forged_path.read_bytes())
    with pytest.raises(IntegrityError, match="record 1 was altered"):
        log.verify()


def test_reloading_continues_the_chain(log, path):
    reloaded = ChainedAuditLog(path, KEY)
    reloaded.record(event("paralegal-1", "M-003", minute=4))

    assert reloaded.verify() == len(HISTORY) + 1
    *_, fourth, fifth = (json.loads(line) for line in path.read_text().splitlines())
    assert (fifth["seq"], fifth["prev"]) == (5, fourth["mac"])


def test_reader_opened_earlier_still_verifies_after_new_records(log, path):
    reader = ChainedAuditLog(path, KEY)
    log.record(event(minute=4))
    assert reader.verify() == len(HISTORY) + 1
    assert reader.events()[-1] == event(minute=4)


def test_tampered_log_cannot_be_reopened_or_queried(log, path):
    rewrite(path, lambda lines: lines[1:])
    with pytest.raises(IntegrityError, match="record 1"):
        ChainedAuditLog(path, KEY)
    with pytest.raises(IntegrityError, match="record 1"):
        log.events(denied_only=True)


def test_privacy_officer_queries(log):
    assert log.events(principal_id="lawyer-1") == HISTORY[:2]
    assert log.events(matter_id="M-002") == HISTORY[1:3]
    assert log.events(denied_only=True) == [HISTORY[1], HISTORY[3]]
    assert log.events(principal_id="lawyer-1", denied_only=True) == [HISTORY[1]]
    assert log.events(principal_id="dpo-1") == []


def test_free_text_is_fingerprinted_never_stored(path):
    labeled = generate(matters=1, documents_per_matter=1).documents[0]
    query = "¿Cuál es la renta mensual pactada?"
    log = ChainedAuditLog(path, KEY)
    log.record(event(document=labeled.document.id, query=query, excerpt=labeled.document.text))

    stored = path.read_text(encoding="utf-8")
    assert "renta mensual" not in stored
    assert not any(span.text in stored for span in labeled.spans)

    (recorded,) = log.events()
    assert recorded.detail == {
        "document": labeled.document.id,
        "query": log.fingerprint(query),
        "excerpt": log.fingerprint(labeled.document.text),
    }


@pytest.mark.parametrize(
    "value", ["M-001-D003", "3", "privileged", "local-llm:v2", "[PERSON_0123456789ab]"]
)
def test_identifiers_are_kept_verbatim(path, value):
    log = ChainedAuditLog(path, KEY)
    log.record(event(value=value))
    assert log.events()[0].detail == {"value": value}


@pytest.mark.parametrize(
    "value", ["María Soto", "maria.soto@correo.example", "+56 9 1234 5678", "x" * 65, ""]
)
def test_anything_else_is_fingerprinted(path, value):
    log = ChainedAuditLog(path, KEY)
    log.record(event(value=value))
    assert log.events()[0].detail == {"value": log.fingerprint(value)}


def test_fingerprint_is_keyed_and_stable(path):
    log, other = ChainedAuditLog(path, KEY), ChainedAuditLog(path, OTHER_KEY)
    assert log.fingerprint("María Soto") == log.fingerprint("María Soto")
    assert log.fingerprint("María Soto") != other.fingerprint("María Soto")
    assert "María" not in log.fingerprint("María Soto")


def test_guard_decisions_land_in_the_log(path):
    log = ChainedAuditLog(path, KEY)
    guard = Guard(MembershipPolicy(), log)
    lawyer = Principal("lawyer-1", Role.LAWYER, frozenset({"M-001"}))

    def retrieve(matter_id: str) -> AccessRequest:
        resource = Resource(Zone.SERVING, matter_id)
        return AccessRequest(lawyer, Action.RETRIEVE, Purpose.LEGAL_ADVICE, resource)

    guard.authorize(retrieve("M-001"), document="M-001-D001")
    with pytest.raises(AccessDenied):
        guard.authorize(retrieve("M-002"))

    assert log.verify() == 2
    (denial,) = log.events(denied_only=True)
    assert (denial.principal_id, denial.matter_id, denial.reason) == (
        "lawyer-1",
        "M-002",
        "not assigned",
    )


def test_empty_key_is_rejected(path):
    with pytest.raises(ValueError, match="key"):
        ChainedAuditLog(path, b"")
