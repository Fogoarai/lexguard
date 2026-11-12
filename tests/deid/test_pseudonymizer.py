import unicodedata
from dataclasses import replace

import pytest

from lexguard import tokens
from lexguard.deid import canonicalize, detect
from lexguard.domain import PiiKind
from lexguard.errors import KeyUnavailable
from lexguard.synthetic import generate

SEEDS = range(5)


@pytest.fixture(scope="module")
def corpus():
    return generate(matters=2)


def test_each_labeled_span_becomes_its_keyed_token(pseudonymizer, keys, corpus):
    for labeled in corpus.documents:
        document = labeled.document
        key = keys.key_for(document.matter_id, "pseudonym")
        result = pseudonymizer.pseudonymize(document)
        for span in labeled.spans:
            token = tokens.pseudonym(span.kind, canonicalize(span.kind, span.text), key)
            assert token in result.text
            assert span.text not in result.text


def test_only_the_text_changes(pseudonymizer, corpus):
    document = corpus.documents[0].document
    result = pseudonymizer.pseudonymize(document)
    assert result.text != document.text
    assert replace(result, text=document.text) == document


def test_token_for_locates_what_pseudonymize_produced(pseudonymizer, vault, corpus):
    for labeled in corpus.documents:
        result = pseudonymizer.pseudonymize(labeled.document)
        for span in labeled.spans:
            token = vault.token_for(span.kind, span.text, result.matter_id)
            assert token in result.text
            assert vault.reveal(token, result.matter_id) == span.text


def test_an_entity_keeps_one_token_within_a_matter_and_none_across(pseudonymizer, vault, corpus):
    first, second = corpus.matters
    texts = {
        matter.id: [
            pseudonymizer.pseudonymize(labeled.document).text
            for labeled in corpus.documents
            if labeled.document.matter_id == matter.id
        ]
        for matter in corpus.matters
    }
    token = vault.token_for(PiiKind.ORGANIZATION, first.client, first.id)
    assert all(token in text for text in texts[first.id])
    assert not any(token in text for text in texts[second.id])
    assert vault.token_for(PiiKind.ORGANIZATION, first.client, second.id) != token


def test_spellings_of_one_entity_share_a_token(pseudonymizer, corpus):
    text = (
        "Comparece doña ANA MARÍA FUENTES LEIVA, RUT 12.345.678-5. "
        "La cédula 12345678-5 pertenece a Ana María Fuentes Leiva."
    )
    result = pseudonymizer.pseudonymize(replace(corpus.documents[0].document, text=text))
    found = [match.group() for match in tokens.find(result.text)]
    assert [tokens.kind_of(token) for token in found] == [
        PiiKind.PERSON,
        PiiKind.RUT,
        PiiKind.RUT,
        PiiKind.PERSON,
    ]
    assert found[0] == found[3]
    assert found[1] == found[2]


def test_capitals_without_accents_share_the_token_of_the_accented_name(pseudonymizer, corpus):
    text = (
        "DEMANDANTE: MARIA JOSE NUÑEZ PEÑA\nDEMANDADO: AGRICOLA PACIFICO S.A.\n\n"
        "María José Núñez Peña demanda a Agrícola Pacífico S.A. por despido."
    )
    result = pseudonymizer.pseudonymize(replace(corpus.documents[0].document, text=text))
    person, organization, same_person, same_organization = (
        match.group() for match in tokens.find(result.text)
    )
    assert (tokens.kind_of(person), tokens.kind_of(organization)) == (
        PiiKind.PERSON,
        PiiKind.ORGANIZATION,
    )
    assert (person, organization) == (same_person, same_organization)


def test_text_in_another_encoding_gets_the_same_tokens_and_is_restored_as_it_was(
    pseudonymizer, vault, corpus
):
    entities = {
        PiiKind.PERSON: "María José Núñez Peña",
        PiiKind.RUT: "12.345.678-5",
        PiiKind.PHONE: "+56 9 7645 3392",
        PiiKind.ADDRESS: "Avenida Perú 123, Ñuñoa",
    }
    person, number, phone, address = entities.values()
    text = f"Comparece doña {person}, RUT {number}, teléfono {phone}, domiciliada en {address}."
    dressed = unicodedata.normalize("NFD", text).replace(" ", "\u00a0").replace("-", "\u2011")
    document = replace(corpus.documents[0].document, text=dressed)

    result = pseudonymizer.pseudonymize(document)

    assert [match.group() for match in tokens.find(result.text)] == [
        vault.token_for(kind, value, document.matter_id) for kind, value in entities.items()
    ]
    assert detect(result.text) == []
    assert pseudonymizer.reidentify(result.text, document.matter_id) == dressed


@pytest.mark.parametrize("seed", SEEDS)
def test_reidentify_restores_the_original(pseudonymizer, seed):
    for labeled in generate(seed=seed, matters=2).documents:
        document = labeled.document
        result = pseudonymizer.pseudonymize(document)
        assert pseudonymizer.reidentify(result.text, document.matter_id) == document.text


@pytest.mark.parametrize("seed", SEEDS)
def test_no_residual_pii_in_pseudonymized_text(pseudonymizer, seed):
    for labeled in generate(seed=seed, matters=2).documents:
        result = pseudonymizer.pseudonymize(labeled.document)
        assert detect(result.text) == []
        assert pseudonymizer.pseudonymize(result) == result


def test_forgotten_token_stays_a_token(pseudonymizer, vault, corpus):
    labeled = corpus.documents[0]
    person = next(span for span in labeled.spans if span.kind is PiiKind.PERSON)
    result = pseudonymizer.pseudonymize(labeled.document)
    token = vault.token_for(PiiKind.PERSON, person.text, result.matter_id)

    vault.forget(token, result.matter_id)
    restored = pseudonymizer.reidentify(result.text, result.matter_id)

    assert token in restored
    assert person.text not in restored
    assert restored.replace(token, person.text) == labeled.document.text
    assert token in pseudonymizer.pseudonymize(labeled.document).text
    assert token in pseudonymizer.reidentify(result.text, result.matter_id)


def test_erasing_a_person_also_silences_partial_mentions(pseudonymizer, vault, corpus):
    document = replace(
        corpus.documents[0].document,
        text="Para: Carolina Sanhueza Parra\n\nEstimada Carolina: la señora Sanhueza firmó.",
    )
    result = pseudonymizer.pseudonymize(document)
    vault.forget(vault.token_for(PiiKind.PERSON, "Carolina Sanhueza Parra", "M-001"), "M-001")
    restored = pseudonymizer.reidentify(result.text, "M-001")
    assert restored == result.text
    assert "Carolina" not in restored
    assert "Sanhueza" not in restored


def test_tokens_of_another_matter_are_left_alone(pseudonymizer, corpus):
    result = pseudonymizer.pseudonymize(corpus.documents[0].document)
    assert pseudonymizer.reidentify(result.text, "M-002") == result.text


def test_shredded_matter_cannot_be_reidentified(pseudonymizer, keys, corpus):
    document = corpus.documents[0].document
    result = pseudonymizer.pseudonymize(document)
    keys.shred(document.matter_id)
    with pytest.raises(KeyUnavailable):
        pseudonymizer.reidentify(result.text, document.matter_id)
    with pytest.raises(KeyUnavailable):
        pseudonymizer.pseudonymize(document)
