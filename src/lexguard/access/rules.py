"""The policy as data: what each role may do, and what no grant can unlock."""

from collections.abc import Mapping, Set
from dataclasses import dataclass

from lexguard.domain import AccessRequest, Action, Purpose, Role, Sensitivity, Zone

ENGAGEMENT = frozenset({Purpose.LEGAL_ADVICE, Purpose.MODEL_DEVELOPMENT, Purpose.ANALYTICS})
STATUTORY = frozenset(Purpose) - ENGAGEMENT

MATTER_BOUND = frozenset({Role.LAWYER, Role.PARALEGAL})


@dataclass(frozen=True, slots=True)
class Rule:
    """Covers a request whose every attribute is in scope; an omitted scope covers all values."""

    name: str
    roles: Set[Role] = frozenset(Role)
    actions: Set[Action] = frozenset(Action)
    zones: Set[Zone] = frozenset(Zone)
    purposes: Set[Purpose] = frozenset(Purpose)
    sensitivities: Set[Sensitivity] = frozenset(Sensitivity)

    def covers(self, request: AccessRequest) -> bool:
        return (
            request.principal.role in self.roles
            and request.action in self.actions
            and request.resource.zone in self.zones
            and request.purpose in self.purposes
            and request.resource.sensitivity in self.sensitivities
        )


PERMISSIONS: tuple[Rule, ...] = (
    Rule(
        "lawyer-on-assigned-matter",
        roles={Role.LAWYER},
        actions={Action.RETRIEVE, Action.REIDENTIFY},
        zones={Zone.SERVING},
        purposes={Purpose.LEGAL_ADVICE},
    ),
    Rule(
        "paralegal-on-assigned-matter",
        roles={Role.PARALEGAL},
        actions={Action.RETRIEVE},
        zones={Zone.SERVING},
        purposes={Purpose.LEGAL_ADVICE},
    ),
    Rule(
        "scientist-develops-models",
        roles={Role.DATA_SCIENTIST},
        actions={Action.READ_DEIDENTIFIED, Action.EXPORT_TRAINING},
        zones={Zone.DEIDENTIFIED},
        purposes={Purpose.MODEL_DEVELOPMENT},
    ),
    Rule(
        "scientist-aggregates",
        roles={Role.DATA_SCIENTIST},
        actions={Action.AGGREGATE},
        zones={Zone.DEIDENTIFIED},
        purposes={Purpose.ANALYTICS},
    ),
    Rule(
        "pipeline-lands-and-deidentifies",
        roles={Role.SERVICE},
        actions={Action.INGEST, Action.READ_RAW},
        zones={Zone.RAW},
        purposes=ENGAGEMENT,
    ),
    Rule(
        "engineer-lands-documents",
        roles={Role.DATA_ENGINEER},
        actions={Action.INGEST},
        zones={Zone.RAW},
        purposes=ENGAGEMENT,
    ),
    Rule(
        "officer-serves-data-subject",
        roles={Role.PRIVACY_OFFICER},
        actions={Action.READ_DEIDENTIFIED, Action.ERASE},
        zones={Zone.DEIDENTIFIED},
        purposes={Purpose.DATA_SUBJECT_REQUEST},
    ),
    Rule(
        "officer-closes-matter",
        roles={Role.PRIVACY_OFFICER},
        actions={Action.ERASE},
        zones={Zone.RAW},
        purposes={Purpose.RETENTION},
    ),
)

# Checked before permissions and grants, so neither can override them.
PROHIBITIONS: tuple[Rule, ...] = (
    Rule(
        "paralegal-never-privileged",
        roles={Role.PARALEGAL},
        sensitivities={Sensitivity.PRIVILEGED},
    ),
    Rule(
        "scientist-never-privileged",
        roles={Role.DATA_SCIENTIST},
        sensitivities={Sensitivity.PRIVILEGED},
    ),
    Rule(
        "scientist-never-raw",
        roles={Role.DATA_SCIENTIST},
        zones={Zone.RAW},
    ),
    Rule(
        "reidentify-lawyers-only",
        roles=frozenset(Role) - {Role.LAWYER},
        actions={Action.REIDENTIFY},
    ),
)

# A grant changes who may act, never where: an action stays in the zones where some
# role already performs it.
ZONES: Mapping[Action, frozenset[Zone]] = {
    action: frozenset(zone for rule in PERMISSIONS if action in rule.actions for zone in rule.zones)
    for action in Action
}

# Egress has no permission of its own: content leaves a zone only for a principal who
# could make this read of it, for the same purpose and compartment.
EGRESS_READS: Mapping[Zone, Action] = {
    Zone.DEIDENTIFIED: Action.READ_DEIDENTIFIED,
    Zone.SERVING: Action.RETRIEVE,
}
