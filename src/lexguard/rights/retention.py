from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime

from lexguard.domain import AccessRequest, Action, Principal, Purpose, Resource, Zone
from lexguard.errors import KeyUnavailable, LexguardError
from lexguard.guard import Guard
from lexguard.ports import DatasetRegistry
from lexguard.storage import ZonedStore

_PROBE = "probe"


class ClosureIncomplete(LexguardError):
    """Part of the matter was still readable after the purge."""


@dataclass(frozen=True, slots=True)
class Closure:
    matter_id: str
    closed_at: datetime
    destroyed: Mapping[Zone, int]
    tainted_datasets: tuple[str, ...]


def close_matter(
    store: ZonedStore,
    guard: Guard,
    datasets: DatasetRegistry,
    principal: Principal,
    matter_id: str,
) -> Closure:
    """Crypto-shred a matter at the end of its retention period.

    Returns only after checking, zone by zone, that no object is left and that no key can
    be derived for the matter, so a restored backup of its ciphertext stays unreadable.
    """
    resource = Resource(Zone.RAW, matter_id)
    guard.authorize(AccessRequest(principal, Action.ERASE, Purpose.RETENTION, resource))

    zones = Zone.of_matters()
    destroyed = {zone: len(store.names(zone, matter_id)) for zone in zones}
    tainted = datasets.retract(lambda owner, _: owner == matter_id, "matter closed")
    store.purge(matter_id)
    if readable := [zone for zone in zones if _readable(store, zone, matter_id)]:
        raise ClosureIncomplete(f"matter {matter_id} still readable in {', '.join(readable)}")
    return Closure(
        matter_id=matter_id,
        closed_at=datetime.now(UTC),
        destroyed=destroyed,
        tainted_datasets=tuple(tainted),
    )


def _readable(store: ZonedStore, zone: Zone, matter_id: str) -> bool:
    if store.names(zone, matter_id):
        return True
    try:
        store.get(zone, matter_id, _PROBE)
    except KeyUnavailable:
        return False
    except FileNotFoundError:
        pass
    return True
