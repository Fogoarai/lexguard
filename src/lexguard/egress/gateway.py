from collections import Counter
from collections.abc import Callable, Iterable, Sequence
from datetime import UTC, datetime

from lexguard import tokens
from lexguard.domain import (
    AccessRequest,
    Action,
    AuditEvent,
    Obligation,
    PiiKind,
    PiiSpan,
    Principal,
    Purpose,
    Resource,
    Sensitivity,
    Zone,
)
from lexguard.errors import EgressBlocked
from lexguard.guard import Guard
from lexguard.ports import LanguageModel
from lexguard.settings import ModelLocation, Settings

type Scanner = Callable[[str], Sequence[PiiSpan]]


class EgressGateway:
    """The single door to a language model.

    A prompt leaves only if the trust boundary admits the model's location, the policy allows
    egress of every compartment the prompt contains and, for a model that is not local, no
    obligation forbids it and an outbound scan finds no residual PII. The outcome is audited
    before the model is called, once per matter, and never includes the prompt text.
    """

    def __init__(self, guard: Guard, settings: Settings, scan: Scanner) -> None:
        self._guard = guard
        self._settings = settings
        self._scan = scan

    def complete(
        self,
        model: LanguageModel,
        prompt: str,
        *,
        principal: Principal,
        purpose: Purpose,
        compartments: Iterable[tuple[str, Sensitivity]],
        obligations: Iterable[Obligation] = (),
        zone: Zone = Zone.SERVING,
    ) -> str:
        """Send `prompt` to `model` or raise `EgressBlocked`.

        `compartments` are the (matter, sensitivity) pairs whose data the prompt contains,
        `zone` is where that data was read, and `obligations` are those already attached to
        it by earlier decisions.
        """
        scope = sorted(set(compartments))
        at = datetime.now(UTC)
        requests = [
            AccessRequest(principal, Action.EGRESS, purpose, Resource(zone, matter_id, level), at)
            for matter_id, level in scope
        ]
        objection = self._objection(model, prompt, requests, frozenset(obligations))

        matters = sorted({matter_id for matter_id, _ in scope})
        detail = {
            "model": model.name,
            "location": model.location.value,
            "boundary": self._settings.boundary.value,
            "matters": str(len(matters)),
            "prompt_chars": str(len(prompt)),
            "pseudonyms": str(sum(1 for _ in tokens.find(prompt))),
        }
        for matter_id in matters or [None]:
            levels = "/".join(level for owner, level in scope if owner == matter_id)
            self._guard.audit.record(
                AuditEvent(
                    at=at,
                    principal_id=principal.id,
                    role=principal.role,
                    action=Action.EGRESS,
                    purpose=purpose,
                    zone=zone,
                    matter_id=matter_id,
                    allowed=objection is None,
                    reason=objection or "admitted",
                    detail=detail | {"sensitivities": levels} if levels else detail,
                )
            )
        if objection is not None:
            raise EgressBlocked(objection)
        return model.complete(prompt)

    def _objection(
        self,
        model: LanguageModel,
        prompt: str,
        requests: Sequence[AccessRequest],
        carried: frozenset[Obligation],
    ) -> str | None:
        if not requests:
            return "prompt declares no compartment"
        if not self._settings.boundary.admits(model.location):
            return f"boundary {self._settings.boundary} does not admit {model.location} models"

        decisions = [
            self._guard.evaluate(
                request,
                sensitivity=request.resource.sensitivity.value,
                model=model.name,
                location=model.location.value,
            )
            for request in requests
        ]
        denied = [
            f"{request.resource.matter_id}/{request.resource.sensitivity} ({decision.reason})"
            for request, decision in zip(requests, decisions, strict=True)
            if not decision.allowed
        ]
        if denied:
            return "policy denied egress of " + ", ".join(denied)

        if model.location == ModelLocation.LOCAL:
            return None
        if Obligation.NO_EGRESS in carried.union(*(d.obligations for d in decisions)):
            return f"no_egress obligation forbids {model.location} models"
        residual = self._residual(prompt)
        if residual:
            found = ", ".join(f"{count} {kind}" for kind, count in sorted(residual.items()))
            return f"residual PII in outbound prompt: {found}"
        return None

    def _residual(self, prompt: str) -> Counter[PiiKind]:
        """Count scanner findings by kind, ignoring matches that lie inside a pseudonym token."""
        shielded = [match.span() for match in tokens.find(prompt)]
        return Counter(
            span.kind
            for span in self._scan(prompt)
            if not any(start <= span.start and span.end <= end for start, end in shielded)
        )
