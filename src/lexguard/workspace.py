import os
from collections.abc import Collection, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Self

from lexguard.access import Grant, Policy
from lexguard.analytics import Aggregator
from lexguard.audit import ChainedAuditLog
from lexguard.crypto import KeyRing
from lexguard.deid import Pseudonymizer, SealedTokenVault
from lexguard.documents import DocumentStore
from lexguard.domain import Document, Matter, Principal, Zone
from lexguard.egress import EgressGateway
from lexguard.guard import Guard
from lexguard.pipeline import Landing, Pipeline
from lexguard.rag import Answerer, ExtractiveModel, HashingEmbedder, SecureIndex
from lexguard.rights import Closure, Erasure, Identifier, SubjectRequests, close_matter
from lexguard.settings import Settings
from lexguard.storage import ZonedStore
from lexguard.training import ManifestRegistry, TrainingSets

AUDIT_LOG = Path("audit", "decisions.jsonl")


@dataclass(frozen=True, slots=True)
class Workspace:
    """Composition root: every component shares one guard, one key ring and one store."""

    matters: dict[str, Matter]
    settings: Settings
    guard: Guard
    audit: ChainedAuditLog
    store: ZonedStore
    vault: SealedTokenVault
    deidentifier: Pseudonymizer
    pipeline: Pipeline
    index: SecureIndex
    gateway: EgressGateway
    answerer: Answerer
    training: TrainingSets
    analytics: Aggregator
    subjects: SubjectRequests
    manifests: ManifestRegistry

    @classmethod
    def create(
        cls,
        root: Path,
        matters: Iterable[Matter],
        settings: Settings | None = None,
        grants: Collection[Grant] = (),
    ) -> Self:
        settings = settings or Settings.from_env()
        registry = {matter.id: matter for matter in matters}
        keys = KeyRing.generate()
        store = ZonedStore(root / "zones", keys)
        audit = ChainedAuditLog(root / AUDIT_LOG, os.urandom(32))
        guard = Guard(Policy(registry, grants), audit)
        vault = SealedTokenVault(keys, root / "vault")
        deidentifier = Pseudonymizer(vault)
        index = SecureIndex(HashingEmbedder(), guard)
        gateway = EgressGateway(guard, settings, scan=deidentifier.detect)
        manifests = ManifestRegistry(root / "manifests")
        return cls(
            matters=registry,
            settings=settings,
            guard=guard,
            audit=audit,
            store=store,
            vault=vault,
            deidentifier=deidentifier,
            pipeline=Pipeline(guard, keys, DocumentStore(store), deidentifier),
            index=index,
            gateway=gateway,
            answerer=Answerer(index, gateway, ExtractiveModel(), deidentifier, guard),
            training=TrainingSets(store, manifests, guard, residual_pii=deidentifier.detect),
            analytics=Aggregator(store, guard, settings),
            subjects=SubjectRequests(store, vault, guard, manifests),
            manifests=manifests,
        )

    def ingest(self, principal: Principal, batch: Iterable[Document]) -> Landing:
        """Land what the policy admits, pseudonymize it and publish it to the serving index."""
        batch = list(batch)
        landing = self.pipeline.land(principal, batch)
        accepted = set(landing.accepted)
        for matter_id in sorted({d.matter_id for d in batch if d.id in accepted}):
            self.pipeline.promote(principal, matter_id)
            self.index.add(self.pipeline.documents.scan(Zone.DEIDENTIFIED, matter_id))
        return landing

    def erase(
        self,
        principal: Principal,
        subject: Collection[Identifier],
        *,
        remove_documents: bool = False,
    ) -> Erasure:
        erasure = self.subjects.erase(
            principal, subject, list(self.matters), remove_documents=remove_documents
        )
        if remove_documents:
            for hit in erasure.hits:
                self.index.remove(hit.matter_id, hit.document_id)
        return erasure

    def close(self, principal: Principal, matter_id: str) -> Closure:
        closure = close_matter(self.store, self.guard, self.manifests, principal, matter_id)
        self.vault.discard(matter_id)
        self.index.remove(matter_id)
        del self.matters[matter_id]
        return closure
