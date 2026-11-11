import hashlib
import json
import secrets
from collections.abc import Callable, Iterable, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from itertools import groupby
from operator import attrgetter

from lexguard.documents import DocumentStore
from lexguard.domain import (
    AccessRequest,
    Action,
    Document,
    DocumentKind,
    Matter,
    PiiSpan,
    Principal,
    Purpose,
    Resource,
    Sensitivity,
    Zone,
)
from lexguard.errors import IntegrityError, LexguardError
from lexguard.guard import Guard
from lexguard.storage import ZonedStore
from lexguard.training.manifest import Entry, Exclusion, Manifest, ManifestRegistry, Reason

type Scanner = Callable[[str], Sequence[PiiSpan]]

PURPOSE = Purpose.MODEL_DEVELOPMENT


class DatasetTainted(LexguardError):
    """The dataset used data that has since been erased; rebuild it and retrain."""


@dataclass(frozen=True, slots=True)
class Example:
    document_id: str
    kind: DocumentKind
    text: str


@dataclass(frozen=True, slots=True)
class TrainingSet:
    manifest: Manifest
    examples: tuple[Example, ...]

    def fine_tuning_records(self) -> list[str]:
        return [example.text for example in self.examples]

    def classification_pairs(self) -> list[tuple[str, DocumentKind]]:
        return [(example.text, example.kind) for example in self.examples]


def shard_name(dataset_id: str) -> str:
    return f"{dataset_id}.shard.jsonl"


@dataclass(frozen=True, slots=True)
class TrainingSets:
    """Exports pseudonymized documents into per-matter shards of the training zone.

    A shard is sealed under its matter's key, so shredding the matter removes its
    contribution from every dataset. The manifest is what remains.
    """

    store: ZonedStore
    manifests: ManifestRegistry
    guard: Guard
    residual_pii: Scanner
    include_privileged: bool = False

    def build(
        self, principal: Principal, matters: Iterable[Matter], dataset_id: str | None = None
    ) -> Manifest:
        created_at = datetime.now(UTC)
        dataset_id = dataset_id or f"ds-{created_at:%Y%m%dT%H%M%S}-{secrets.token_hex(4)}"
        if dataset_id in self.manifests:
            raise ValueError(f"dataset {dataset_id} already exists")

        documents = DocumentStore(self.store)
        entries: list[Entry] = []
        exclusions: list[Exclusion] = []
        for matter in {matter.id: matter for matter in matters}.values():
            if PURPOSE not in matter.purposes:
                exclusions.append(Exclusion(matter.id, Reason.PURPOSE_NOT_PERMITTED))
                continue
            allowed = self._compartments(principal, matter.id, dataset_id, exclusions)
            if not allowed:
                continue
            shard: list[Example] = []
            for document in documents.scan(Zone.DEIDENTIFIED, matter.id):
                if document.sensitivity not in allowed:
                    continue
                if found := self.residual_pii(document.text):
                    exclusions.append(_residual(document, found))
                    continue
                shard.append(Example(document.id, document.kind, document.text))
                entries.append(
                    Entry(document.id, matter.id, document.sensitivity, _digest(document.text))
                )
            if shard:
                self.store.put(Zone.TRAINING, matter.id, shard_name(dataset_id), _encode(shard))

        manifest = Manifest(dataset_id, created_at, PURPOSE, tuple(entries), tuple(exclusions))
        self.manifests.save(manifest)
        return manifest

    def load(self, principal: Principal, dataset_id: str) -> TrainingSet:
        """Read a dataset back for a training run.

        Asks the export question again for every compartment the dataset drew from, so a
        revoked grant stops the run just as a taint or a shredded matter does.
        """
        manifest = self.manifests.load(dataset_id)
        if manifest.tainted:
            raise DatasetTainted(f"{dataset_id}: {manifest.taints[-1].reason}")
        compartments = {(entry.matter_id, entry.sensitivity) for entry in manifest.entries}
        for matter_id, sensitivity in sorted(compartments):
            self.guard.authorize(_request(principal, matter_id, sensitivity), dataset=dataset_id)

        examples: list[Example] = []
        for matter_id, entries in groupby(manifest.entries, key=attrgetter("matter_id")):
            shard = _decode(self.store.get(Zone.TRAINING, matter_id, shard_name(dataset_id)))
            if [(e.document_id, _digest(e.text)) for e in shard] != [
                (e.document_id, e.sha256) for e in entries
            ]:
                raise IntegrityError(f"shard of {matter_id} does not match manifest {dataset_id}")
            examples += shard
        return TrainingSet(manifest, tuple(examples))

    def _compartments(
        self, principal: Principal, matter_id: str, dataset_id: str, exclusions: list[Exclusion]
    ) -> set[Sensitivity]:
        allowed = set()
        for sensitivity in Sensitivity:
            if sensitivity is Sensitivity.PRIVILEGED and not self.include_privileged:
                exclusions.append(Exclusion(matter_id, Reason.PRIVILEGED, sensitivity))
                continue
            decision = self.guard.evaluate(
                _request(principal, matter_id, sensitivity), dataset=dataset_id
            )
            if decision.allowed:
                allowed.add(sensitivity)
            else:
                exclusions.append(
                    Exclusion(matter_id, Reason.POLICY_DENIED, sensitivity, detail=decision.reason)
                )
        return allowed


def _request(principal: Principal, matter_id: str, sensitivity: Sensitivity) -> AccessRequest:
    resource = Resource(Zone.DEIDENTIFIED, matter_id, sensitivity)
    return AccessRequest(principal, Action.EXPORT_TRAINING, PURPOSE, resource)


def _residual(document: Document, found: Sequence[PiiSpan]) -> Exclusion:
    kinds = ",".join(sorted({span.kind for span in found}))
    return Exclusion(
        document.matter_id, Reason.RESIDUAL_PII, document.sensitivity, document.id, kinds
    )


def _digest(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def _encode(examples: Iterable[Example]) -> bytes:
    return "\n".join(json.dumps(asdict(e), ensure_ascii=False) for e in examples).encode()


def _decode(data: bytes) -> list[Example]:
    return [
        Example(fields["document_id"], DocumentKind(fields["kind"]), fields["text"])
        for fields in map(json.loads, data.decode().split("\n"))
    ]
