import math
from random import Random

import pytest

from lexguard.training.memorization import Canary, NgramLanguageModel, exposure, plant


@pytest.fixture(scope="module")
def records(pseudonymized):
    return [document.text for document in pseudonymized]


def test_plant_inserts_each_canary_as_often_as_asked(records):
    once, thrice = Canary("111222333"), Canary("444555666", repetitions=3)

    planted = plant(records, [once, thrice], Random(0))

    assert planted.count(once.text) == 1
    assert planted.count(thrice.text) == 3
    assert [record for record in planted if record in records] == records


def test_references_share_the_shape_of_the_canary_but_not_its_secret():
    canary = Canary.random(Random(0), length=6)
    references = canary.references(200, Random(1))

    assert len(set(references)) == 200
    assert canary.text not in references
    assert {len(reference) for reference in references} == {len(canary.text)}


def test_references_need_a_large_enough_secret_space():
    with pytest.raises(ValueError, match="too small"):
        Canary("01", alphabet="01").references(10, Random(0))


def test_exposure_is_maximal_for_the_best_rank_and_zero_for_a_tie():
    canary = Canary.random(Random(0))

    singled_out = exposure(lambda text: float(text != canary.text), canary, Random(1), 255)
    indifferent = exposure(lambda text: 1.0, canary, Random(1), 255)

    assert (singled_out.rank, singled_out.bits) == (1, 8.0)
    assert singled_out.memorized()
    assert (indifferent.rank, indifferent.bits) == (256, 0.0)
    assert not indifferent.memorized()


def test_language_model_is_a_distribution_that_prefers_what_it_saw(records):
    model = NgramLanguageModel.fit(records[:4], order=4)
    context = "RUT"

    total = sum(model.probability(context, char) for char in model.alphabet)
    unseen = model.probability(context, "¤")

    assert "¤" not in model.alphabet
    assert total + unseen == pytest.approx(1.0)
    seen = records[0][:200]
    assert model.loss(seen) < model.loss("".join(Random(0).sample(seen, len(seen))))


def test_harness_flags_a_memorized_canary_and_not_an_unseen_one(records):
    rng = Random(7)
    planted, unseen = Canary.random(rng), Canary.random(rng)
    model = NgramLanguageModel.fit(plant(records, [planted], rng))

    memorized = exposure(model.loss, planted, rng)
    fresh = exposure(model.loss, unseen, rng)

    assert memorized.rank == 1
    assert memorized.bits == pytest.approx(math.log2(1024))
    assert memorized.memorized()
    assert fresh.bits < 5
    assert not fresh.memorized()


def test_a_model_that_never_saw_the_canary_does_not_expose_it(records):
    rng = Random(7)
    canary = Canary.random(rng)

    assert not exposure(NgramLanguageModel.fit(records).loss, canary, rng).memorized()
