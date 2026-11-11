from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime

from lexguard.domain import AccessRequest, Action, Principal, Purpose, Resource, Zone
from lexguard.errors import KeyUnavailable, LexguardError
from lexguard.guard import Guard
from lexguard.storage import ZonedStore
from lexguard.training.manifest import ManifestRegistry

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
    manifests: ManifestRegistry,
    principal: Principal,
    matter_id: str,
) -> Closure:
    """Crypto-shred a matter at the end of its retention period.

    Returns only after checking, zone by zone, that no object is left and that no key can
    be derived for the matter, so a restored backup of its ciphertext stays unreadable.
    """
    resource = Resource(Zone.RAW, matter_id)
    guard.authorize(AccessRequest(principal, Action.ERASE, Purpose.RETENTION, resource))

    destroyed = {zone: len(store.names(zone, matter_id)) for zone in Zone}
    tainted = manifests.taint(lambda entry: entry.matter_id == matter_id, "matter closed")
    store.purge(matter_id)
    if readable := [zone for zone in Zone if _readable(store, zone, matter_id)]:
        raise ClosureIncomplete(f"matter {matter_id} still readable in {', '.join(readable)}")
    return Closure(
        matter_id=matter_id,
        closed_at=datetime.now(UTC),
        destroyed=destroyed,
        tainted_datasets=tuple(manifest.dataset_id for manifest in tainted),
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
