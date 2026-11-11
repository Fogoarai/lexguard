from dataclasses import dataclass
from datetime import datetime

from lexguard.domain import (
    AccessRequest,
    Action,
    Matter,
    Principal,
    Purpose,
    Role,
    Sensitivity,
)


@dataclass(frozen=True, slots=True)
class Grant:
    """Time-bound exception that stands in for a missing role permission or matter assignment.

    Dual control: the approver is a privacy officer other than the grantee and outside the
    matter's ethical wall. A grant never reaches privileged content.
    """

    principal_id: str
    action: Action
    matter_id: str
    purpose: Purpose
    expires_at: datetime
    approved_by: Principal

    def __post_init__(self) -> None:
        if self.expires_at.tzinfo is None:
            raise ValueError("grant expiry must be timezone-aware")

    def names(self, request: AccessRequest) -> bool:
        return (self.principal_id, self.action, self.matter_id, self.purpose) == (
            request.principal.id,
            request.action,
            request.resource.matter_id,
            request.purpose,
        )

    def refusal(self, request: AccessRequest, matter: Matter) -> str | None:
        """Why a request this grant names is still not covered by it, if it is not."""
        approver = self.approved_by
        if (
            approver.role is not Role.PRIVACY_OFFICER
            or approver.id == self.principal_id
            or approver.id in matter.walled_off
        ):
            return f"grant-dual-control: {approver.id} cannot approve this grant"
        if request.at >= self.expires_at:
            return f"grant-expired: lapsed at {self.expires_at.isoformat()}"
        if request.resource.sensitivity is Sensitivity.PRIVILEGED:
            return "grant-excludes-privileged"
        return None
