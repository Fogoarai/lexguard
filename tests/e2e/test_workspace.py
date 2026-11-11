import pytest

from lexguard import tokens
from lexguard.analytics import BudgetExhausted, Dimension
from lexguard.demo import PARTIES, STRATEGY, HostedModel, run
from lexguard.domain import PiiKind, Purpose, Sensitivity, Zone
from lexguard.errors import EgressBlocked, IntegrityError, KeyUnavailable
from lexguard.rag import NO_ANSWER, Answerer
from lexguard.rights import Identifier
from lexguard.synthetic import generate
from lexguard.training import DatasetTainted, Reason
from lexguard.workspace import AUDIT_LOG, Workspace


@pytest.fixture(scope="module")
def corpus():
    return generate(matters=6, documents_per_matter=8)


@pytest.fixture
def workspace(tmp_path, corpus):
    workspace = Workspace.create(tmp_path, corpus.matters)
    lake = [labeled.document for labeled in corpus.documents]
    landing = workspace.ingest(corpus.principal("svc-pipeline"), lake)
    assert not landing.refused
    return workspace


def ask(workspace, corpus, who, question, matter_id="M-001"):
    principal = corpus.principal(who)
    return workspace.answerer.answer(question, principal, Purpose.LEGAL_ADVICE, matter_id=matter_id)


def test_nothing_labeled_survives_ingestion(workspace, corpus):
    documents = workspace.pipeline.documents
    for labeled in corpus.documents:
        original = labeled.document
        stored = documents.load(Zone.DEIDENTIFIED, original.matter_id, original.id)
        assert not any(span.text in stored.text for span in labeled.spans)
        assert workspace.store.names(Zone.RAW, original.matter_id) == []


def test_the_same_person_gets_a_different_token_in_each_matter(workspace, corpus):
    client = corpus.matters[0].client
    vault = workspace.vault
    issued = {
        vault.token_for(PiiKind.ORGANIZATION, client, matter_id) for matter_id in ("M-001", "M-002")
    }
    assert len(issued) == 2


def test_lawyer_reads_names_and_paralegal_reads_pseudonyms(workspace, corpus):
    landlord = next(s.text for s in corpus.documents[0].spans if s.kind is PiiKind.PERSON)

    lawyer = ask(workspace, corpus, "lawyer-1", PARTIES)
    paralegal = ask(workspace, corpus, "paralegal-1", PARTIES)

    assert lawyer.reidentified and landlord in lawyer.text
    assert not list(tokens.find(lawyer.text))
    assert not paralegal.reidentified and landlord not in paralegal.text
    assert list(tokens.find(paralegal.text))


def test_privileged_advice_reaches_only_the_matter_lawyer(workspace, corpus):
    assert "avenimiento" in ask(workspace, corpus, "lawyer-1", STRATEGY).text
    assert ask(workspace, corpus, "paralegal-1", STRATEGY).text == NO_ANSWER


@pytest.mark.parametrize(
    ("who", "matter_id"),
    [("lawyer-1", "M-002"), ("lawyer-2", "M-001"), ("scientist-1", "M-001"), ("dpo-1", "M-001")],
)
def test_walls_assignment_and_roles_hold_at_retrieval(workspace, corpus, who, matter_id):
    answer = ask(workspace, corpus, who, PARTIES, matter_id)
    assert (answer.text, answer.citations) == (NO_ANSWER, ())


def test_a_hosted_model_is_blocked_by_default(workspace, corpus):
    model = HostedModel()
    hosted = Answerer(
        workspace.index, workspace.gateway, model, workspace.deidentifier, workspace.guard
    )
    with pytest.raises(EgressBlocked, match="local_only"):
        hosted.answer(PARTIES, corpus.principal("lawyer-1"), Purpose.LEGAL_ADVICE)
    assert model.received == 0


def test_training_uses_only_consenting_matters_and_no_privileged_text(workspace, corpus):
    scientist = corpus.principal("scientist-1")
    manifest = workspace.training.build(scientist, corpus.matters, dataset_id="e2e")

    assert {entry.matter_id for entry in manifest.entries} == {"M-001", "M-003", "M-005"}
    assert Sensitivity.PRIVILEGED not in {entry.sensitivity for entry in manifest.entries}
    refused = {e.matter_id for e in manifest.exclusions if e.reason is Reason.PURPOSE_NOT_PERMITTED}
    assert refused == {"M-002", "M-004", "M-006"}

    labels = [span.text for labeled in corpus.documents for span in labeled.spans]
    records = workspace.training.load(scientist, "e2e").fine_tuning_records()
    assert records and not any(label in record for label in labels for record in records)


def test_analytics_spends_a_finite_budget(workspace, corpus):
    scientist = corpus.principal("scientist-1")
    release = workspace.analytics.count_by(scientist, corpus.matters, Dimension.MATTER, 0.6)
    assert release.counts == {} and release.suppressed == 3
    with pytest.raises(BudgetExhausted):
        workspace.analytics.count_by(scientist, corpus.matters, Dimension.KIND, 0.6)


def test_erasing_a_person_is_irreversible_and_taints_training(workspace, corpus):
    scientist, officer = corpus.principal("scientist-1"), corpus.principal("dpo-1")
    lease = corpus.documents[0]
    landlord = next(s.text for s in lease.spans if s.kind is PiiKind.PERSON)
    rut = next(s.text for s in lease.spans if s.kind is PiiKind.RUT)
    workspace.training.build(scientist, corpus.matters, dataset_id="e2e")

    erasure = workspace.erase(
        officer, [Identifier(PiiKind.PERSON, landlord), Identifier(PiiKind.RUT, rut)]
    )

    assert [hit.document_id for hit in erasure.hits] == [lease.document.id]
    assert erasure.tainted_datasets == ("e2e",)
    with pytest.raises(DatasetTainted):
        workspace.training.load(scientist, "e2e")
    answer = ask(workspace, corpus, "lawyer-1", PARTIES)
    assert landlord not in answer.text and rut not in answer.text


def test_closing_a_matter_leaves_nothing_readable(workspace, corpus, tmp_path):
    workspace.close(corpus.principal("dpo-1"), "M-003")

    assert ask(workspace, corpus, "lawyer-3", PARTIES, "M-003").text == NO_ANSWER
    assert "M-003" not in workspace.matters
    assert not (tmp_path / "vault" / "M-003").exists()
    with pytest.raises(KeyUnavailable):
        workspace.store.get(Zone.DEIDENTIFIED, "M-003", "M-003-D001.doc.json")
    assert ask(workspace, corpus, "lawyer-1", PARTIES).reidentified


def test_every_decision_is_audited_and_tampering_is_detected(workspace, corpus, tmp_path):
    before = workspace.audit.verify()
    ask(workspace, corpus, "lawyer-1", PARTIES, "M-002")
    denials = workspace.audit.events(principal_id="lawyer-1", denied_only=True)

    assert workspace.audit.verify() > before
    assert denials and all("ethical-wall" in event.reason for event in denials)

    log = tmp_path / AUDIT_LOG
    log.write_text(log.read_text().replace('"allowed":false', '"allowed":true', 1))
    with pytest.raises(IntegrityError):
        workspace.audit.verify()


def test_demo_runs(tmp_path, capsys):
    run(tmp_path)
    output = capsys.readouterr().out
    assert "chain verified" in output
    assert "blocked, boundary local_only" in output
