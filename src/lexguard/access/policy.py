from collections.abc import Collection, Mapping
from dataclasses import dataclass, replace

from lexguard.access.grants import Grant
from lexguard.access.rules import (
    EGRESS_READS,
    MATTER_BOUND,
    PERMISSIONS,
    PROHIBITIONS,
    STATUTORY,
    ZONES,
)
from lexguard.domain import AccessRequest, Action, Decision, Matter, Obligation, Sensitivity


@dataclass(frozen=True, slots=True)
class Policy:
    """Deny-by-default decision point over the tables in `rules`.

    Checks run from the least to the most overridable: matter, ethical wall, purpose
    limitation, zone, prohibitions, role permission with matter assignment, grant.
    `grants` is read at decision time, so a grant added to the collection applies at once.
    """

    matters: Mapping[str, Matter]
    grants: Collection[Grant] = ()

    def decide(self, request: AccessRequest) -> Decision:
        if request.action is Action.EGRESS:
            return self._egress(request)
        decision = self._evaluate(request)
        if request.action is Action.RETRIEVE and decision.allowed:
            reidentify = self._evaluate(replace(request, action=Action.REIDENTIFY))
            if not reidentify.allowed:
                return _allow(decision.reason, Obligation.KEEP_PSEUDONYMS)
        return decision

    def _egress(self, request: AccessRequest) -> Decision:
        resource = request.resource
        read = EGRESS_READS.get(resource.zone)
        if read is None:
            return _deny(f"egress-zone: nothing leaves the {resource.zone} zone")
        decision = self._evaluate(replace(request, action=read))
        reason = f"egress-follows-{read}: {decision.reason}"
        if not decision.allowed:
            return _deny(reason)
        if resource.sensitivity is Sensitivity.PRIVILEGED:
            return _allow(reason, Obligation.NO_EGRESS)
        return _allow(reason)

    def _evaluate(self, request: AccessRequest) -> Decision:
        principal, resource = request.principal, request.resource
        matter = self.matters.get(resource.matter_id or "")
        if matter is None:
            return _deny(f"unknown-matter: {resource.matter_id}")
        if principal.id in matter.walled_off:
            return _deny(f"ethical-wall: {principal.id} is walled off from {matter.id}")
        if request.purpose not in STATUTORY | matter.purposes:
            return _deny(f"purpose-limitation: {matter.id} does not permit {request.purpose}")
        if resource.zone not in ZONES[request.action]:
            return _deny(
                f"zone-mismatch: {request.action} does not take place in the {resource.zone} zone"
            )
        for rule in PROHIBITIONS:
            if rule.covers(request):
                return _deny(rule.name)

        permission = next((rule for rule in PERMISSIONS if rule.covers(request)), None)
        assigned = principal.role not in MATTER_BOUND or matter.id in principal.matters
        if permission and assigned:
            return _allow(permission.name)
        if (granted := self._granted(request, matter)) is not None:
            return granted
        if permission:
            return _deny(f"matter-assignment: {principal.id} is not assigned to {matter.id}")
        return _deny(
            f"default-deny: no rule lets a {principal.role} {request.action} "
            f"in the {resource.zone} zone for {request.purpose}"
        )

    def _granted(self, request: AccessRequest, matter: Matter) -> Decision | None:
        named = [grant for grant in self.grants if grant.names(request)]
        if not named:
            return None
        if not any(rule.describes(request) for rule in PERMISSIONS):
            return _deny(
                f"grant-exceeds-roles: no role may {request.action} "
                f"in the {request.resource.zone} zone for {request.purpose}"
            )
        refusals = []
        for grant in named:
            refusal = grant.refusal(request, matter)
            if refusal is None:
                return _allow(
                    f"time-bound-grant: approved by {grant.approved_by.id} "
                    f"until {grant.expires_at.isoformat()}"
                )
            refusals.append(refusal)
        return _deny(min(refusals))


def _allow(reason: str, *obligations: Obligation) -> Decision:
    """Obligations are spelled out in the reason because audit events record only the reason."""
    if obligations:
        reason = f"{reason} [{', '.join(sorted(obligations))}]"
    return Decision(True, reason, frozenset(obligations))


def _deny(reason: str) -> Decision:
    return Decision(False, reason)
