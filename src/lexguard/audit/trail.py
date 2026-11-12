from dataclasses import dataclass

from lexguard.audit.log import ChainedAuditLog
from lexguard.domain import (
    AccessRequest,
    Action,
    AuditEvent,
    Principal,
    Purpose,
    Resource,
    Zone,
)
from lexguard.guard import Guard


@dataclass(frozen=True, slots=True)
class AuditTrail:
    """Reads of the log cleared by the guard, so every read is itself on the record."""

    log: ChainedAuditLog
    guard: Guard

    def events(
        self,
        reader: Principal,
        *,
        principal_id: str | None = None,
        matter_id: str | None = None,
        denied_only: bool = False,
    ) -> list[AuditEvent]:
        """Raises `AccessDenied`. The matter may be one that was closed since."""
        resource = Resource(Zone.AUDIT, matter_id)
        request = AccessRequest(reader, Action.READ_AUDIT, Purpose.SECURITY_AUDIT, resource)
        scope = {"about": principal_id} if principal_id else {}
        self.guard.authorize(request, **scope, denied_only=str(denied_only).lower())
        return self.log.events(
            principal_id=principal_id, matter_id=matter_id, denied_only=denied_only
        )
