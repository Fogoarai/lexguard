import json
import re
from collections.abc import Callable, Iterator
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any

from lexguard.domain import Purpose, Sensitivity

_SUFFIX = ".manifest.json"
_SAFE_ID = re.compile(r"\w[\w.-]*")


class Reason(StrEnum):
    PURPOSE_NOT_PERMITTED = "purpose_not_permitted"
    POLICY_DENIED = "policy_denied"
    PRIVILEGED = "privileged"
    RESIDUAL_PII = "residual_pii"


@dataclass(frozen=True, slots=True)
class Entry:
    document_id: str
    matter_id: str
    sensitivity: Sensitivity
    sha256: str


@dataclass(frozen=True, slots=True)
class Exclusion:
    """Recorded at the level it was decided: a matter, a sensitivity compartment or a document."""

    matter_id: str
    reason: Reason
    sensitivity: Sensitivity | None = None
    document_id: str | None = None
    detail: str = ""


@dataclass(frozen=True, slots=True)
class Taint:
    at: datetime
    reason: str
    document_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class Manifest:
    """Provenance of a training set: identifiers, hashes and decisions, never document text."""

    dataset_id: str
    created_at: datetime
    purpose: Purpose
    entries: tuple[Entry, ...]
    exclusions: tuple[Exclusion, ...]
    taints: tuple[Taint, ...] = ()

    @property
    def tainted(self) -> bool:
        """True once data it used was erased: rebuild the dataset and retrain its models."""
        return bool(self.taints)


def encode(manifest: Manifest) -> str:
    return json.dumps(asdict(manifest), default=datetime.isoformat, indent=2)


def decode(data: str) -> Manifest:
    fields = json.loads(data)
    return Manifest(
        dataset_id=fields["dataset_id"],
        created_at=datetime.fromisoformat(fields["created_at"]),
        purpose=Purpose(fields["purpose"]),
        entries=tuple(
            Entry(e["document_id"], e["matter_id"], Sensitivity(e["sensitivity"]), e["sha256"])
            for e in fields["entries"]
        ),
        exclusions=tuple(_exclusion(e) for e in fields["exclusions"]),
        taints=tuple(
            Taint(datetime.fromisoformat(t["at"]), t["reason"], tuple(t["document_ids"]))
            for t in fields["taints"]
        ),
    )


def _exclusion(fields: dict[str, Any]) -> Exclusion:
    sensitivity = fields["sensitivity"]
    return Exclusion(
        matter_id=fields["matter_id"],
        reason=Reason(fields["reason"]),
        sensitivity=Sensitivity(sensitivity) if sensitivity else None,
        document_id=fields["document_id"],
        detail=fields["detail"],
    )


class ManifestRegistry:
    """Manifests as plain files outside the encrypted zones.

    They hold no text, and they must outlive the matters they list: after a matter is
    shredded its manifest is the only record that a model was trained on it.
    """

    def __init__(self, root: Path) -> None:
        self._root = root

    def __contains__(self, dataset_id: str) -> bool:
        return self._path(dataset_id).is_file()

    def __iter__(self) -> Iterator[Manifest]:
        for path in sorted(self._root.glob(f"*{_SUFFIX}")):
            yield decode(path.read_text(encoding="utf-8"))

    def save(self, manifest: Manifest) -> None:
        self._root.mkdir(parents=True, exist_ok=True)
        self._path(manifest.dataset_id).write_text(encode(manifest), encoding="utf-8")

    def load(self, dataset_id: str) -> Manifest:
        return decode(self._path(dataset_id).read_text(encoding="utf-8"))

    def taint(self, affected: Callable[[Entry], bool], reason: str) -> list[Manifest]:
        """Mark every dataset that used an affected document; returns the marked manifests."""
        at = datetime.now(UTC)
        marked = []
        for manifest in self:
            if used := tuple(e.document_id for e in manifest.entries if affected(e)):
                tainted = replace(manifest, taints=(*manifest.taints, Taint(at, reason, used)))
                self.save(tainted)
                marked.append(tainted)
        return marked

    def _path(self, dataset_id: str) -> Path:
        if not _SAFE_ID.fullmatch(dataset_id):
            raise ValueError(f"unsafe dataset id: {dataset_id!r}")
        return self._root / (dataset_id + _SUFFIX)
