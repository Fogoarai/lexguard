import numpy as np
import pytest

from lexguard import tokens
from lexguard.domain import DocumentKind, PiiKind
from lexguard.synthetic import generate
from lexguard.testing import LabelDeidentifier
from lexguard.training.classifier import DpSgd, KindClassifier, featurize

KEY, OTHER_KEY = b"k" * 32, b"o" * 32


@pytest.fixture(scope="module")
def splits():
    """Pseudonymized (text, kind) pairs, with whole matters held out for evaluation."""
    corpus = generate(matters=12, documents_per_matter=8)
    deidentifier = LabelDeidentifier(corpus)
    documents = [deidentifier.pseudonymize(labeled.document) for labeled in corpus.documents]
    held_out = {matter.id for matter in corpus.matters[8:]}
    train = [(d.text, d.kind) for d in documents if d.matter_id not in held_out]
    test = [(d.text, d.kind) for d in documents if d.matter_id in held_out]
    return train, test


def test_features_are_unit_vectors_blind_to_pseudonym_digests():
    texts = [
        f"Demanda de {tokens.pseudonym(PiiKind.PERSON, 'María Soto', key)} por despido."
        for key in (KEY, OTHER_KEY)
    ]
    first, second = featurize(texts, 64)

    assert np.array_equal(first, second)
    assert np.linalg.norm(first) == pytest.approx(1.0)
    assert not featurize([""], 64).any()


def test_learns_document_kinds_on_held_out_matters(splits):
    train, test = splits
    model = KindClassifier.fit(train)

    assert model.privacy is None
    assert model.accuracy(test) >= 0.95
    assert set(model.predict([text for text, _ in test])) == set(DocumentKind)


def test_dp_run_reports_its_parameters_and_no_epsilon(splits):
    train, _ = splits
    dp = DpSgd(clip_norm=0.5, noise_multiplier=1.1)
    report = KindClassifier.fit(train, steps=50, sampling_rate=0.2, dp=dp).privacy

    assert (report.noise_multiplier, report.clip_norm) == (1.1, 0.5)
    assert (report.steps, report.sampling_rate, report.examples) == (50, 0.2, len(train))
    assert not hasattr(report, "epsilon")
    assert "accountant" in str(report)


@pytest.mark.parametrize("sampling_rate", [0.0, -1.0, 2.0])
def test_reported_sampling_rate_must_be_the_one_applied(splits, sampling_rate):
    train, _ = splits

    with pytest.raises(ValueError, match="sampling_rate"):
        KindClassifier.fit(train, sampling_rate=sampling_rate, dp=DpSgd())


@pytest.mark.parametrize(("clip_norm", "noise_multiplier"), [(0.0, 1.0), (-1.0, 0.0), (1.0, -0.5)])
def test_dp_parameters_must_describe_a_real_mechanism(clip_norm, noise_multiplier):
    with pytest.raises(ValueError, match="clip_norm"):
        DpSgd(clip_norm, noise_multiplier)


def test_fit_needs_examples():
    with pytest.raises(ValueError, match="no training examples"):
        KindClassifier.fit([])


def test_clipping_bounds_what_any_example_contributes(splits):
    train, _ = splits
    one_full_step = {"steps": 1, "sampling_rate": 1.0, "learning_rate": 1.0}
    clip_only = DpSgd(clip_norm=0.1, noise_multiplier=0.0)

    clipped = KindClassifier.fit(train, dp=clip_only, **one_full_step)
    unclipped = KindClassifier.fit(train, **one_full_step)

    assert np.linalg.norm(clipped.weights) <= 0.1 + 1e-9
    assert np.linalg.norm(unclipped.weights) > 0.1


def test_noise_follows_the_multiplier_and_the_clip_norm(splits):
    train, _ = splits
    dp = DpSgd(clip_norm=2.0, noise_multiplier=3.0)
    one_full_step = {"steps": 1, "sampling_rate": 1.0, "learning_rate": 1.0, "dp": dp}

    first, second, repeat = (
        KindClassifier.fit(train, seed=seed, **one_full_step).weights for seed in (1, 2, 1)
    )

    assert np.array_equal(first, repeat)
    expected = np.sqrt(2) * dp.noise_multiplier * dp.clip_norm / len(train)
    assert np.std(first - second) == pytest.approx(expected, rel=0.1)


def test_dp_training_still_separates_the_kinds(splits):
    train, test = splits
    model = KindClassifier.fit(train, dp=DpSgd(clip_norm=1.0, noise_multiplier=1.0))

    assert model.accuracy(test) >= 0.9
