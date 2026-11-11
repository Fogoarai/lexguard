"""End-to-end walkthrough on the synthetic corpus. Run with `python -m lexguard`."""

import tempfile
from collections import Counter
from dataclasses import replace
from pathlib import Path
from random import Random
from textwrap import shorten

from lexguard.analytics import BudgetExhausted, Dimension
from lexguard.deid import evaluate
from lexguard.domain import PiiKind, Purpose, Zone
from lexguard.egress import EgressGateway
from lexguard.errors import EgressBlocked, IntegrityError
from lexguard.rag import NO_ANSWER, Answerer
from lexguard.rights import Identifier
from lexguard.settings import Boundary, ModelLocation, Settings
from lexguard.synthetic import Corpus, generate
from lexguard.training import (
    Canary,
    DatasetTainted,
    DpSgd,
    KindClassifier,
    NgramLanguageModel,
    exposure,
    plant,
)
from lexguard.workspace import AUDIT_LOG, Workspace

PARTIES = "¿Quiénes comparecen y con qué cédula de identidad y domicilio?"
STRATEGY = "¿Cuál es la recomendación sobre la oferta de avenimiento antes de la audiencia?"


class HostedModel:
    """Stands in for an LLM API outside the trust boundary."""

    name = "hosted-llm"
    location = ModelLocation.EXTERNAL

    def __init__(self) -> None:
        self.received = 0

    def complete(self, prompt: str) -> str:
        self.received += len(prompt)
        return NO_ANSWER


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="lexguard-") as root:
        run(Path(root))


def run(root: Path) -> None:
    corpus = generate(matters=6, documents_per_matter=8)
    workspace = Workspace.create(root, corpus.matters)
    _ingest(workspace, corpus)
    _retrieve(workspace, corpus)
    _egress(workspace, corpus)
    _train(workspace, corpus)
    _aggregate(workspace, corpus)
    _rights(workspace, corpus)
    _audit(workspace, root / AUDIT_LOG)


def _ingest(workspace: Workspace, corpus: Corpus) -> None:
    _section("Ingest: only what the policy admits leaves the lake")
    lake = [labeled.document for labeled in corpus.documents]
    stray = replace(lake[0], id="M-999-D001", matter_id="M-999")

    refused = workspace.ingest(corpus.principal("scientist-1"), lake[:1]).refused
    print(f"scientist-1 tries to land a document -> {next(iter(refused.values()))}")
    landing = workspace.ingest(corpus.principal("svc-pipeline"), [*lake, stray])
    print(f"svc-pipeline lands {len(landing.accepted)} documents; refused {landing.refused}")

    email = lake[2]
    stored = workspace.pipeline.documents.load(Zone.DEIDENTIFIED, email.matter_id, email.id)
    print(f"in the lake : {email.text.splitlines()[0]}")
    print(f"as stored   : {stored.text.splitlines()[0]}")
    left = sum(len(workspace.store.names(Zone.RAW, matter_id)) for matter_id in workspace.matters)
    print(f"documents left in the raw zone: {left}")

    scores = evaluate(workspace.deidentifier.detect, corpus.documents)
    kind, worst = min(scores.items(), key=lambda item: item[1].recall)
    print(f"detection on this labeled corpus: lowest recall {worst.recall:.0%} ({kind})")


def _retrieve(workspace: Workspace, corpus: Corpus) -> None:
    _section("Retrieve: one index, a different view per person")
    cases = (
        ("lawyer-1", "M-001", PARTIES, "assigned lawyer reads names"),
        ("paralegal-1", "M-001", PARTIES, "paralegal reads pseudonyms"),
        ("lawyer-1", "M-001", STRATEGY, "privileged advice, lawyer"),
        ("paralegal-1", "M-001", STRATEGY, "privileged advice, paralegal"),
        ("lawyer-1", "M-002", PARTIES, "assigned but behind an ethical wall"),
        ("scientist-1", "M-001", PARTIES, "no retrieval role"),
    )
    for who, matter_id, question, note in cases:
        answer = workspace.answerer.answer(
            question, corpus.principal(who), Purpose.LEGAL_ADVICE, matter_id=matter_id
        )
        print(f"{who} on {matter_id} ({note}):\n    {shorten(answer.text, 150)}")


def _egress(workspace: Workspace, corpus: Corpus) -> None:
    _section("Egress: what may reach a hosted model")
    opened = EgressGateway(
        workspace.guard,
        Settings(boundary=Boundary.DEIDENTIFIED_EGRESS),
        scan=workspace.deidentifier.detect,
    )
    cases = (
        (workspace.settings.boundary, workspace.gateway, "paralegal-1", PARTIES),
        (Boundary.DEIDENTIFIED_EGRESS, opened, "paralegal-1", PARTIES),
        (Boundary.DEIDENTIFIED_EGRESS, opened, "lawyer-1", STRATEGY),
    )
    for boundary, gateway, who, question in cases:
        model = HostedModel()
        answerer = Answerer(
            workspace.index, gateway, model, workspace.deidentifier, workspace.guard
        )
        try:
            answerer.answer(
                question, corpus.principal(who), Purpose.LEGAL_ADVICE, matter_id="M-001"
            )
            outcome = f"sent {model.received} pseudonymized characters"
        except EgressBlocked as blocked:
            outcome = f"blocked, {blocked}"
        print(f"{boundary}, {who}: {outcome}")


def _train(workspace: Workspace, corpus: Corpus) -> None:
    _section("Train: only matters that agreed, never privileged, never raw")
    scientist = corpus.principal("scientist-1")
    manifest = workspace.training.build(scientist, corpus.matters, dataset_id="demo")
    matters = sorted({entry.matter_id for entry in manifest.entries})
    reasons = Counter(exclusion.reason.value for exclusion in manifest.exclusions)
    print(f"dataset 'demo': {len(manifest.entries)} documents from {', '.join(matters)}")
    print(f"exclusions: {dict(reasons)}")

    dataset = workspace.training.load(scientist, "demo")
    held_out = matters[-1]
    train = [(e.text, e.kind) for e in dataset.examples if not e.document_id.startswith(held_out)]
    test = [(e.text, e.kind) for e in dataset.examples if e.document_id.startswith(held_out)]
    plain = KindClassifier.fit(train)
    private = KindClassifier.fit(train, dp=DpSgd(noise_multiplier=1.0))
    print(
        f"document-kind classifier on held-out {held_out}: "
        f"{plain.accuracy(test):.0%} plain, {private.accuracy(test):.0%} with DP-SGD"
    )
    print(private.privacy)

    rng = Random(0)
    planted = Canary.random(rng)
    model = NgramLanguageModel.fit(plant(dataset.fine_tuning_records(), [planted], rng))
    for label, canary in (("planted", planted), ("unseen", Canary.random(rng))):
        result = exposure(model.loss, canary, rng)
        print(f"{label} canary: {result.bits:.1f} bits exposed, memorized={result.memorized()}")


def _aggregate(workspace: Workspace, corpus: Corpus) -> None:
    _section("Aggregate: small groups suppressed, noise added, budget enforced")
    scientist = corpus.principal("scientist-1")
    for dimension in (Dimension.KIND, Dimension.MATTER):
        release = workspace.analytics.count_by(scientist, corpus.matters, dimension, epsilon=0.4)
        print(
            f"count by {dimension}: {dict(release.counts)} with Laplace scale "
            f"{release.noise_scale:g}; {release.suppressed} groups under "
            f"{workspace.settings.min_group_size} suppressed"
        )
    try:
        workspace.analytics.count_by(scientist, corpus.matters, Dimension.MONTH, epsilon=0.4)
    except BudgetExhausted as exhausted:
        print(f"third query refused: {exhausted}")


def _rights(workspace: Workspace, corpus: Corpus) -> None:
    _section("Rights: erase a person, close a matter")
    officer = corpus.principal("dpo-1")
    lease = corpus.documents[0]
    person = next(span.text for span in lease.spans if span.kind is PiiKind.PERSON)
    rut = next(span.text for span in lease.spans if span.kind is PiiKind.RUT)
    subject = [Identifier(PiiKind.PERSON, person), Identifier(PiiKind.RUT, rut)]

    location = workspace.subjects.locate(officer, subject, list(workspace.matters))
    print(f"{person} appears in {[hit.document_id for hit in location.hits]}")
    erasure = workspace.erase(officer, subject)
    print(f"tokens forgotten; datasets tainted: {list(erasure.tainted_datasets)}")
    try:
        workspace.training.load(corpus.principal("scientist-1"), "demo")
    except DatasetTainted as tainted:
        print(f"next training run refused: {tainted}")
    answer = workspace.answerer.answer(
        PARTIES, corpus.principal("lawyer-1"), Purpose.LEGAL_ADVICE, matter_id="M-001"
    )
    print(f"lawyer-1 now reads:\n    {shorten(answer.text, 150)}")

    closure = workspace.close(officer, "M-003")
    print(f"M-003 closed: {sum(closure.destroyed.values())} objects shredded")
    answer = workspace.answerer.answer(
        PARTIES, corpus.principal("lawyer-3"), Purpose.LEGAL_ADVICE, matter_id="M-003"
    )
    print(f"lawyer-3 on M-003:\n    {answer.text}")


def _audit(workspace: Workspace, log: Path) -> None:
    _section("Audit: every decision recorded, tampering detected")
    records = workspace.audit.verify()
    denials = Counter(event.principal_id for event in workspace.audit.events(denied_only=True))
    print(f"{records} records, chain verified; denials by principal: {dict(denials)}")
    log.write_text(log.read_text().replace('"allowed":false', '"allowed":true', 1))
    try:
        workspace.audit.verify()
    except IntegrityError as broken:
        print(f"after flipping one denial to allowed: {broken}")


def _section(title: str) -> None:
    print(f"\n== {title} ==")
