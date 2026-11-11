from dataclasses import dataclass

from lexguard.domain import AccessRequest, AuditEvent, Decision
from lexguard.errors import AccessDenied
from lexguard.ports import AuditLog, PolicyDecisionPoint


@dataclass(frozen=True, slots=True)
class Guard:
    """Single enforcement point: every decision is audited before it takes effect."""

    policy: PolicyDecisionPoint
    audit: AuditLog

    def evaluate(self, request: AccessRequest, **detail: str) -> Decision:
        decision = self.policy.decide(request)
        self.audit.record(AuditEvent.of(request, decision, detail))
        return decision

    def authorize(self, request: AccessRequest, **detail: str) -> Decision:
        decision = self.evaluate(request, **detail)
        if not decision.allowed:
            raise AccessDenied(decision.reason)
        return decision
