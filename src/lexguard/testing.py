"""In-memory stand-ins for the ports, so each module can be built and tested on its own."""

from dataclasses import dataclass, field

from lexguard.domain import AccessRequest, AuditEvent, Decision


class AllowAll:
    def decide(self, request: AccessRequest) -> Decision:
        return Decision(allowed=True, reason="allow-all")


class DenyAll:
    def decide(self, request: AccessRequest) -> Decision:
        return Decision(allowed=False, reason="deny-all")


class MembershipPolicy:
    """Allows a request only when the principal is assigned to the resource's matter."""

    def decide(self, request: AccessRequest) -> Decision:
        assigned = request.resource.matter_id in request.principal.matters
        return Decision(allowed=assigned, reason="assigned" if assigned else "not assigned")


@dataclass(slots=True)
class MemoryAuditLog:
    events: list[AuditEvent] = field(default_factory=list)

    def record(self, event: AuditEvent) -> None:
        self.events.append(event)
