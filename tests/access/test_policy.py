from dataclasses import replace
from datetime import UTC, datetime, timedelta
from itertools import product

import pytest

from lexguard.access import Grant, Policy
from lexguard.access.rules import EGRESS_READS, ENGAGEMENT, PERMISSIONS, PROHIBITIONS
from lexguard.domain import (
    AccessRequest,
    Action,
    Decision,
    Obligation,
    Principal,
    Purpose,
    Resource,
    Role,
    Sensitivity,
    Zone,
)
from lexguard.errors import AccessDenied
from lexguard.guard import Guard
from lexguard.synthetic import generate
from lexguard.testing import MemoryAuditLog

CORPUS = generate()
MATTERS = CORPUS.matter_index
CAST = [principal.id for principal in CORPUS.principals]
NOW = datetime(2026, 12, 1, 9, tzinfo=UTC)
SECOND_OFFICER = Principal("dpo-2", Role.PRIVACY_OFFICER)

type Shape = tuple[Action, Zone, Purpose]

# The rows of the "Guard requests" table in CLAUDE.md.
LAND = (Action.INGEST, Zone.RAW, Purpose.LEGAL_ADVICE)
READ_RAW = (Action.READ_RAW, Zone.RAW, Purpose.LEGAL_ADVICE)
RETRIEVE = (Action.RETRIEVE, Zone.SERVING, Purpose.LEGAL_ADVICE)
REIDENTIFY = (Action.REIDENTIFY, Zone.SERVING, Purpose.LEGAL_ADVICE)
PROMPT = (Action.EGRESS, Zone.SERVING, Purpose.LEGAL_ADVICE)
EXPORT = (Action.EXPORT_TRAINING, Zone.DEIDENTIFIED, Purpose.MODEL_DEVELOPMENT)
AGGREGATE = (Action.AGGREGATE, Zone.DEIDENTIFIED, Purpose.ANALYTICS)
LOCATE = (Action.READ_DEIDENTIFIED, Zone.DEIDENTIFIED, Purpose.DATA_SUBJECT_REQUEST)
ERASE_SUBJECT = (Action.ERASE, Zone.DEIDENTIFIED, Purpose.DATA_SUBJECT_REQUEST)
CLOSE_MATTER = (Action.ERASE, Zone.RAW, Purpose.RETENTION)

READ_FOR_TRAINING = (Action.READ_DEIDENTIFIED, Zone.DEIDENTIFIED, Purpose.MODEL_DEVELOPMENT)
SEND_FOR_TRAINING = (Action.EGRESS, Zone.DEIDENTIFIED, Purpose.MODEL_DEVELOPMENT)
SEND_FOR_SUBJECT = (Action.EGRESS, Zone.DEIDENTIFIED, Purpose.DATA_SUBJECT_REQUEST)
LAND_FOR_ANY_ENGAGEMENT = {(Action.INGEST, Zone.RAW, purpose) for purpose in ENGAGEMENT}
READ_RAW_FOR_ANY_ENGAGEMENT = {(Action.READ_RAW, Zone.RAW, purpose) for purpose in ENGAGEMENT}

LAWYER = {RETRIEVE, REIDENTIFY, PROMPT}
PARALEGAL = {RETRIEVE, PROMPT}
SCIENTIST = {READ_FOR_TRAINING, EXPORT, AGGREGATE, SEND_FOR_TRAINING}
OFFICER = {LOCATE, ERASE_SUBJECT, SEND_FOR_SUBJECT, CLOSE_MATTER}
PIPELINE = LAND_FOR_ANY_ENGAGEMENT | READ_RAW_FOR_ANY_ENGAGEMENT
NOTHING: set[Shape] = set()


def request(
    who: str | Principal,
    shape: Shape,
    matter: str | None = "M-001",
    sensitivity: Sensitivity = Sensitivity.CONFIDENTIAL,
    at: datetime = NOW,
) -> AccessRequest:
    principal = CORPUS.principal(who) if isinstance(who, str) else who
    action, zone, purpose = shape
    return AccessRequest(principal, action, purpose, Resource(zone, matter, sensitivity), at)


def ask(
    who: str | Principal,
    shape: Shape,
    matter: str | None = "M-001",
    sensitivity: Sensitivity = Sensitivity.CONFIDENTIAL,
    *,
    grants: tuple[Grant, ...] = (),
    at: datetime = NOW,
) -> Decision:
    return Policy(MATTERS, grants).decide(request(who, shape, matter, sensitivity, at))


def grant(
    who: str,
    shape: Shape,
    matter: str = "M-001",
    *,
    hours: float = 1,
    approver: Principal = SECOND_OFFICER,
) -> Grant:
    action, _, purpose = shape
    return Grant(who, action, matter, purpose, NOW + timedelta(hours=hours), approver)


def everything_allowed(who: str, matter: str, sensitivity: Sensitivity) -> set[Shape]:
    return {
        shape
        for shape in product(Action, Zone, Purpose)
        if ask(who, shape, matter, sensitivity).allowed
    }


@pytest.mark.parametrize(
    ("who", "matter", "sensitivity", "expected"),
    [
        # M-001 permits legal advice, model development and analytics.
        ("lawyer-1", "M-001", Sensitivity.CONFIDENTIAL, LAWYER),
        ("lawyer-2", "M-001", Sensitivity.CONFIDENTIAL, NOTHING),
        ("paralegal-1", "M-001", Sensitivity.CONFIDENTIAL, PARALEGAL),
        ("scientist-1", "M-001", Sensitivity.CONFIDENTIAL, SCIENTIST),
        ("engineer-1", "M-001", Sensitivity.CONFIDENTIAL, LAND_FOR_ANY_ENGAGEMENT),
        ("svc-pipeline", "M-001", Sensitivity.CONFIDENTIAL, PIPELINE),
        ("dpo-1", "M-001", Sensitivity.CONFIDENTIAL, OFFICER),
        # Privileged content stays with the matter's lawyers and the machinery around them.
        ("lawyer-1", "M-001", Sensitivity.PRIVILEGED, LAWYER),
        ("paralegal-1", "M-001", Sensitivity.PRIVILEGED, NOTHING),
        ("scientist-1", "M-001", Sensitivity.PRIVILEGED, NOTHING),
        ("engineer-1", "M-001", Sensitivity.PRIVILEGED, LAND_FOR_ANY_ENGAGEMENT),
        ("svc-pipeline", "M-001", Sensitivity.PRIVILEGED, PIPELINE),
        ("dpo-1", "M-001", Sensitivity.PRIVILEGED, OFFICER),
        # M-002 permits legal advice only, and lawyer-1 is behind its ethical wall.
        ("lawyer-1", "M-002", Sensitivity.CONFIDENTIAL, NOTHING),
        ("lawyer-2", "M-002", Sensitivity.CONFIDENTIAL, LAWYER),
        ("paralegal-1", "M-002", Sensitivity.CONFIDENTIAL, PARALEGAL),
        ("scientist-1", "M-002", Sensitivity.CONFIDENTIAL, NOTHING),
        ("engineer-1", "M-002", Sensitivity.CONFIDENTIAL, {LAND}),
        ("svc-pipeline", "M-002", Sensitivity.CONFIDENTIAL, {LAND, READ_RAW}),
        ("dpo-1", "M-002", Sensitivity.CONFIDENTIAL, OFFICER),
    ],
)
def test_each_principal_may_do_exactly_this(who, matter, sensitivity, expected):
    assert everything_allowed(who, matter, sensitivity) == expected


def test_lawyer_works_only_on_assigned_matters():
    assert ask("lawyer-1", RETRIEVE).reason == "lawyer-on-assigned-matter"
    assert ask("lawyer-1", REIDENTIFY, sensitivity=Sensitivity.PRIVILEGED).allowed
    assert not ask("lawyer-1", RETRIEVE).obligations

    outsider = ask("lawyer-3", RETRIEVE)
    assert not outsider.allowed
    assert outsider.reason == "matter-assignment: lawyer-3 is not assigned to M-001"


def test_paralegal_is_bound_to_assigned_matters_too():
    paralegal = Principal("paralegal-2", Role.PARALEGAL, frozenset({"M-001"}))
    assert ask(paralegal, RETRIEVE, "M-001").allowed
    assert ask(paralegal, RETRIEVE, "M-003").reason.startswith("matter-assignment")


def test_ethical_wall_overrides_assignment():
    assert "M-002" in CORPUS.principal("lawyer-1").matters
    for shape in (RETRIEVE, REIDENTIFY, PROMPT):
        decision = ask("lawyer-1", shape, "M-002")
        assert not decision.allowed
        assert "ethical-wall: lawyer-1 is walled off from M-002" in decision.reason


def test_paralegal_keeps_pseudonyms_and_never_sees_privileged_content():
    for matter in MATTERS:
        decision = ask("paralegal-1", RETRIEVE, matter)
        assert decision.allowed
        assert decision.obligations == {Obligation.KEEP_PSEUDONYMS}
        assert decision.reason == "paralegal-on-assigned-matter [keep_pseudonyms]"

    assert ask("paralegal-1", REIDENTIFY).reason == "reidentify-lawyers-only"
    privileged = ask("paralegal-1", RETRIEVE, sensitivity=Sensitivity.PRIVILEGED)
    assert privileged.reason == "paralegal-never-privileged"


def test_egress_follows_the_read_that_produced_the_content():
    confidential = ask("lawyer-1", PROMPT)
    assert confidential.allowed
    assert not confidential.obligations
    assert confidential.reason == "egress-follows-retrieve: lawyer-on-assigned-matter"

    privileged = ask("lawyer-1", PROMPT, sensitivity=Sensitivity.PRIVILEGED)
    assert privileged.allowed
    assert privileged.obligations == {Obligation.NO_EGRESS}
    assert privileged.reason.endswith("[no_egress]")

    assert not ask("paralegal-1", PROMPT, sensitivity=Sensitivity.PRIVILEGED).allowed
    assert "matter-assignment" in ask("lawyer-3", PROMPT).reason
    assert "purpose-limitation" in ask("scientist-1", SEND_FOR_TRAINING, "M-002").reason


@pytest.mark.parametrize("zone", [Zone.RAW, Zone.TRAINING])
def test_nothing_leaves_the_raw_or_training_zones(zone):
    for who, purpose in product(CAST, Purpose):
        decision = ask(who, (Action.EGRESS, zone, purpose))
        assert not decision.allowed
        assert decision.reason.startswith("egress-zone")


def test_purpose_limitation_follows_the_engagement_letter():
    for matter in ("M-001", "M-003"):
        assert ask("scientist-1", EXPORT, matter).reason == "scientist-develops-models"
        assert ask("scientist-1", AGGREGATE, matter).reason == "scientist-aggregates"
    for matter in ("M-002", "M-004"):
        export = ask("scientist-1", EXPORT, matter)
        assert export.reason == f"purpose-limitation: {matter} does not permit model_development"
        assert not ask("scientist-1", AGGREGATE, matter).allowed
        assert not ask(
            "svc-pipeline", (Action.READ_RAW, Zone.RAW, Purpose.ANALYTICS), matter
        ).allowed


def test_statutory_purposes_do_not_need_the_engagement_letter():
    assert MATTERS["M-002"].purposes == {Purpose.LEGAL_ADVICE}
    for shape in (LOCATE, ERASE_SUBJECT, CLOSE_MATTER):
        assert ask("dpo-1", shape, "M-002").allowed


def test_scientist_never_touches_raw_identities_or_privileged_content():
    assert ask("scientist-1", READ_RAW).reason == "scientist-never-raw"
    assert ask("scientist-1", REIDENTIFY).reason == "reidentify-lawyers-only"
    privileged = ask("scientist-1", EXPORT, sensitivity=Sensitivity.PRIVILEGED)
    assert privileged.reason == "scientist-never-privileged"

    row_level_for_analytics = (Action.READ_DEIDENTIFIED, Zone.DEIDENTIFIED, Purpose.ANALYTICS)
    assert ask("scientist-1", row_level_for_analytics).reason.startswith("default-deny")


def test_privacy_officer_retrieves_no_content():
    for shape in (RETRIEVE, REIDENTIFY, PROMPT, READ_RAW, EXPORT, AGGREGATE):
        assert not ask("dpo-1", shape).allowed
    erase_raw_for_subject = (Action.ERASE, Zone.RAW, Purpose.DATA_SUBJECT_REQUEST)
    assert ask("dpo-1", erase_raw_for_subject).reason.startswith("default-deny")


def test_human_reads_raw_only_under_a_live_grant():
    assert ask("engineer-1", READ_RAW).reason.startswith("default-deny")

    break_glass = grant("engineer-1", READ_RAW, hours=1)
    allowed = ask("engineer-1", READ_RAW, grants=(break_glass,))
    assert allowed.allowed
    assert allowed.reason.startswith("time-bound-grant: approved by dpo-2 until 2026-12-01T10:00")

    an_hour_later = NOW + timedelta(hours=1)
    lapsed = ask("engineer-1", READ_RAW, grants=(break_glass,), at=an_hour_later)
    assert lapsed.reason.startswith("grant-expired")

    renewed = (break_glass, grant("engineer-1", READ_RAW, hours=2))
    assert ask("engineer-1", READ_RAW, grants=renewed, at=an_hour_later).allowed


@pytest.mark.parametrize(
    ("who", "approver"),
    [("engineer-1", "engineer-1"), ("engineer-1", "lawyer-2"), ("dpo-1", "dpo-1")],
)
def test_grant_needs_a_second_person_who_is_a_privacy_officer(who, approver):
    self_or_unqualified = grant(who, READ_RAW, approver=CORPUS.principal(approver))
    decision = ask(who, READ_RAW, grants=(self_or_unqualified,))
    assert decision.reason == f"grant-dual-control: {approver} cannot approve this grant"

    assert ask(who, READ_RAW, grants=(grant(who, READ_RAW, approver=SECOND_OFFICER),)).allowed


def test_approver_behind_the_wall_cannot_open_the_matter():
    walled = replace(MATTERS["M-001"], walled_off=frozenset({"dpo-2"}))
    policy = Policy(MATTERS | {"M-001": walled}, (grant("engineer-1", READ_RAW),))
    assert policy.decide(request("engineer-1", READ_RAW)).reason.startswith("grant-dual-control")


def test_grant_never_beats_the_ethical_wall_or_purpose_limitation():
    walled = ask("lawyer-1", RETRIEVE, "M-002", grants=(grant("lawyer-1", RETRIEVE, "M-002"),))
    assert walled.reason.startswith("ethical-wall")

    off_purpose = ask(
        "scientist-1", EXPORT, "M-002", grants=(grant("scientist-1", EXPORT, "M-002"),)
    )
    assert off_purpose.reason.startswith("purpose-limitation")


def test_grant_never_unlocks_a_prohibition_or_privileged_content():
    def with_grant(who: str, shape: Shape, sensitivity=Sensitivity.CONFIDENTIAL) -> str:
        return ask(who, shape, sensitivity=sensitivity, grants=(grant(who, shape),)).reason

    assert with_grant("scientist-1", READ_RAW) == "scientist-never-raw"
    assert with_grant("paralegal-1", REIDENTIFY) == "reidentify-lawyers-only"
    assert with_grant("engineer-1", REIDENTIFY) == "reidentify-lawyers-only"
    assert with_grant("engineer-1", READ_RAW, Sensitivity.PRIVILEGED) == "grant-excludes-privileged"
    assert with_grant("lawyer-3", RETRIEVE, Sensitivity.PRIVILEGED) == "grant-excludes-privileged"


def test_grant_covers_only_what_it_names():
    grants = (grant("engineer-1", READ_RAW),)
    assert not ask("engineer-1", READ_RAW, "M-003", grants=grants).allowed
    assert not ask(
        "engineer-1", (Action.READ_RAW, Zone.RAW, Purpose.ANALYTICS), grants=grants
    ).allowed
    assert not ask("engineer-1", RETRIEVE, grants=grants).allowed
    assert not ask("dpo-1", READ_RAW, grants=grants).allowed
    assert not ask(
        "engineer-1", (Action.READ_RAW, Zone.SERVING, Purpose.LEGAL_ADVICE), grants=grants
    ).allowed


def test_granted_retrieval_stays_pseudonymous_until_reidentification_is_granted_too():
    retrieve_only = (grant("lawyer-3", RETRIEVE),)
    covering = ask("lawyer-3", RETRIEVE, grants=retrieve_only)
    assert covering.allowed
    assert covering.obligations == {Obligation.KEEP_PSEUDONYMS}
    assert ask("lawyer-3", PROMPT, grants=retrieve_only).allowed
    assert not ask("lawyer-3", REIDENTIFY, grants=retrieve_only).allowed

    both = (*retrieve_only, grant("lawyer-3", REIDENTIFY))
    assert not ask("lawyer-3", RETRIEVE, grants=both).obligations
    assert ask("lawyer-3", REIDENTIFY, grants=both).allowed


def test_egress_cannot_be_granted_on_its_own():
    assert not ask("lawyer-3", PROMPT, grants=(grant("lawyer-3", PROMPT),)).allowed


def test_grant_lends_only_what_some_role_may_already_do():
    reidentify_for_training = (Action.REIDENTIFY, Zone.SERVING, Purpose.MODEL_DEVELOPMENT)
    erase_raw_for_subject = (Action.ERASE, Zone.RAW, Purpose.DATA_SUBJECT_REQUEST)
    read_raw_for_audit = (Action.READ_RAW, Zone.RAW, Purpose.SECURITY_AUDIT)

    for who, shape, matter in [
        ("lawyer-1", reidentify_for_training, "M-001"),
        ("engineer-1", erase_raw_for_subject, "M-001"),
        ("engineer-1", read_raw_for_audit, "M-002"),
    ]:
        decision = ask(who, shape, matter, grants=(grant(who, shape, matter),))
        assert not decision.allowed
        assert decision.reason.startswith("grant-exceeds-roles")

    assert ask(
        "lawyer-1", reidentify_for_training, grants=(grant("lawyer-1", reidentify_for_training),)
    ).reason == (
        "grant-exceeds-roles: no role may reidentify in the serving zone for model_development"
    )


def test_security_audit_opens_no_content_even_under_a_grant():
    for who, action, zone in product(CAST, Action, Zone):
        shape = (action, zone, Purpose.SECURITY_AUDIT)
        assert not ask(who, shape, grants=(grant(who, shape),)).allowed


def test_grant_is_not_honored_at_a_time_without_a_timezone():
    naive = datetime(2026, 12, 1, 9)
    break_glass = grant("engineer-1", READ_RAW)

    decision = ask("engineer-1", READ_RAW, grants=(break_glass,), at=naive)
    assert decision.reason.startswith("grant-time-unverifiable")
    assert ask("lawyer-1", RETRIEVE, at=naive).allowed


def test_grant_added_to_a_live_collection_applies_at_once():
    grants: set[Grant] = set()
    policy = Policy(MATTERS, grants)

    assert not policy.decide(request("engineer-1", READ_RAW)).allowed
    grants.add(grant("engineer-1", READ_RAW))
    assert policy.decide(request("engineer-1", READ_RAW)).allowed


def test_grant_expiry_must_be_timezone_aware():
    with pytest.raises(ValueError, match="timezone-aware"):
        Grant(
            "engineer-1",
            Action.READ_RAW,
            "M-001",
            Purpose.LEGAL_ADVICE,
            datetime(2026, 12, 2),
            SECOND_OFFICER,
        )


@pytest.mark.parametrize("matter", ["M-999", None])
def test_unknown_matter_is_denied(matter):
    for who in CAST:
        assert ask(who, RETRIEVE, matter).reason == f"unknown-matter: {matter}"


def test_action_in_the_wrong_zone_is_denied():
    misplaced = ask("lawyer-1", (Action.RETRIEVE, Zone.RAW, Purpose.LEGAL_ADVICE))
    assert misplaced.reason == "zone-mismatch: retrieve does not take place in the raw zone"


def test_guard_raises_with_the_deciding_rule():
    guard = Guard(Policy(MATTERS), MemoryAuditLog())
    with pytest.raises(AccessDenied, match="ethical-wall"):
        guard.authorize(request("lawyer-1", RETRIEVE, "M-002"))
    assert guard.audit.events[0].reason.startswith("ethical-wall")


RULES = {rule.name for rule in PERMISSIONS + PROHIBITIONS} | {
    "unknown-matter",
    "ethical-wall",
    "purpose-limitation",
    "zone-mismatch",
    "matter-assignment",
    "default-deny",
    "egress-zone",
    "time-bound-grant",
    "grant-exceeds-roles",
    "grant-dual-control",
    "grant-expired",
    "grant-time-unverifiable",
    "grant-excludes-privileged",
}


def deciding_rule(decision: Decision) -> str:
    reason = decision.reason
    if reason.startswith("egress-follows-"):
        reason = reason.partition(": ")[2]
    return reason.partition(":")[0].partition(" [")[0]


def test_invariants_hold_for_every_request_whatever_grants_the_principal_holds():
    for who, shape, matter, sensitivity in product(
        CAST, product(Action, Zone, Purpose), MATTERS, Sensitivity
    ):
        action, zone, purpose = shape
        role = CORPUS.principal(who).role
        grants = tuple(grant(who, (granted, zone, purpose), matter) for granted in Action)
        decision = ask(who, shape, matter, sensitivity, grants=grants)
        case = (who, shape, matter, sensitivity, decision.reason)

        assert deciding_rule(decision) in RULES, case
        if not decision.allowed:
            assert not decision.obligations, case
            continue
        assert who not in MATTERS[matter].walled_off, case
        assert purpose not in ENGAGEMENT - MATTERS[matter].purposes, case
        assert action is not Action.REIDENTIFY or role is Role.LAWYER, case
        if role is Role.DATA_SCIENTIST:
            assert zone is not Zone.RAW, case
        if "time-bound-grant" in decision.reason:
            performed = EGRESS_READS[zone] if action is Action.EGRESS else action
            assert any(
                performed in rule.actions and zone in rule.zones and purpose in rule.purposes
                for rule in PERMISSIONS
            ), case
        if sensitivity is Sensitivity.PRIVILEGED:
            assert role not in {Role.PARALEGAL, Role.DATA_SCIENTIST}, case
            assert "time-bound-grant" not in decision.reason, case
            assert action is not Action.EGRESS or Obligation.NO_EGRESS in decision.obligations, case
