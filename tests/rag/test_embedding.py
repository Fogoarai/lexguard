import subprocess
import sys

import numpy as np

from lexguard.rag.embedding import HashingEmbedder, features

PERSON = "[PERSON_3fa9c2d1e0b4]"
OTHER_PERSON = "[PERSON_0b4c2d1e3fa9]"


def test_rows_are_unit_length_and_deterministic():
    texts = ["La renta mensual será de $4.404.000.", "Demanda por despido injustificado."]
    vectors = HashingEmbedder().embed(texts)

    assert vectors.shape == (2, 4096)
    assert np.allclose(np.linalg.norm(vectors, axis=1), 1)
    assert np.array_equal(vectors, HashingEmbedder().embed(texts))


def test_vectors_do_not_depend_on_the_process(monkeypatch):
    script = (
        "import sys; from lexguard.rag import HashingEmbedder; "
        "sys.stdout.buffer.write(HashingEmbedder(dimensions=64).embed([sys.argv[1]]).tobytes())"
    )
    text = f"La renta mensual que paga {PERSON} será de $4.404.000."
    monkeypatch.setenv("PYTHONHASHSEED", "12345")

    elsewhere = subprocess.run(
        [sys.executable, "-c", script, text], capture_output=True, check=True
    )

    assert elsewhere.stdout == HashingEmbedder(dimensions=64).embed([text]).tobytes()


def test_dimensions_are_configurable():
    assert HashingEmbedder(dimensions=256).embed(["renta"]).shape == (1, 256)


def test_related_text_scores_higher_than_unrelated_text():
    query, rent, dismissal = HashingEmbedder().embed(
        [
            "¿Cuánto es la renta del arrendamiento?",
            "El arrendador fija la renta mensual en $4.404.000.",
            "Fui despedido sin expresión de causa.",
        ]
    )
    assert query @ rent > 0.2 > query @ dismissal


def test_case_accents_and_stopwords_do_not_matter():
    assert features("la NOTIFICACIÓN de la demanda") == features("notificacion demanda")
    assert features("de la que el") == {}


def test_inflections_share_character_ngrams():
    dismissal, dismissed, rent = HashingEmbedder().embed(["despido", "despedido", "renta"])
    assert dismissal @ dismissed > 0.15
    assert dismissal @ rent == 0


def test_a_pseudonym_is_a_single_opaque_feature():
    assert features(f"{PERSON} y {PERSON}") == {PERSON: 2}

    same, other = HashingEmbedder().embed([PERSON, OTHER_PERSON])
    assert np.count_nonzero(same) == 1
    assert same @ other == 0


def test_text_without_features_embeds_to_zero():
    assert not HashingEmbedder().embed(["", "de la"]).any()
