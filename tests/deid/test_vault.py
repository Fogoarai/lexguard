import pytest

from lexguard import tokens
from lexguard.deid import SealedTokenVault, TokenNotFound, canonicalize
from lexguard.domain import PiiKind
from lexguard.errors import IntegrityError, KeyUnavailable

PERSON = PiiKind.PERSON


def entries(tmp_path):
    return sorted(path.name for path in (tmp_path / "vault").rglob("*.token"))


def test_token_is_the_keyed_pseudonym_of_the_canonical_value(vault, keys):
    key = keys.key_for("M-001", "pseudonym")
    expected = tokens.pseudonym(PiiKind.RUT, canonicalize(PiiKind.RUT, "12.345.678-5"), key)
    assert vault.token_for(PiiKind.RUT, "12.345.678-5", "M-001") == expected
    assert vault.token_for(PiiKind.RUT, "12345678-5", "M-001") == expected


def test_token_for_only_computes(vault, tmp_path):
    token = vault.token_for(PERSON, "Ana Fuentes Leiva", "M-001")
    assert entries(tmp_path) == []
    with pytest.raises(TokenNotFound):
        vault.reveal(token, "M-001")


def test_remember_then_reveal_keeps_the_first_spelling(vault):
    token = vault.remember(PERSON, "Ana Fuentes Leiva", "M-001")
    assert vault.remember(PERSON, "ANA  FUENTES LEIVA", "M-001") == token
    assert token == vault.token_for(PERSON, "ana fuentes leiva", "M-001")
    assert vault.reveal(token, "M-001") == "Ana Fuentes Leiva"


def test_nothing_is_stored_in_the_clear(vault, tmp_path):
    vault.remember(PERSON, "Ana Fuentes Leiva", "M-001")
    (entry,) = (tmp_path / "vault" / "M-001").iterdir()
    assert b"Fuentes" not in entry.read_bytes()
    assert "Fuentes" not in entry.name


def test_matters_do_not_share_tokens_or_entries(vault):
    token = vault.remember(PERSON, "Ana Fuentes Leiva", "M-001")
    assert vault.token_for(PERSON, "Ana Fuentes Leiva", "M-002") != token
    with pytest.raises(TokenNotFound):
        vault.reveal(token, "M-002")


def test_forgotten_token_cannot_be_revealed_or_recorded_again(vault):
    token = vault.remember(PiiKind.EMAIL, "ana@estudio.example", "M-001")
    vault.forget(token, "M-001")
    with pytest.raises(TokenNotFound):
        vault.reveal(token, "M-001")

    assert vault.remember(PiiKind.EMAIL, "ana@estudio.example", "M-001") == token
    with pytest.raises(TokenNotFound):
        vault.reveal(token, "M-001")


def test_forgetting_before_the_value_was_ever_seen_still_blocks_it(vault):
    token = vault.token_for(PERSON, "Ana Fuentes Leiva", "M-001")
    vault.forget(token, "M-001")
    vault.remember(PERSON, "Ana Fuentes Leiva", "M-001")
    with pytest.raises(TokenNotFound):
        vault.reveal(token, "M-001")


def test_a_person_may_be_mentioned_by_any_shorter_form_of_the_name(vault, tmp_path):
    full = vault.token_for(PERSON, "Pedro Pablo del Río Aguirre", "M-001")
    partial = {
        vault.token_for(PERSON, form, "M-001")
        for form in ("Pedro", "Aguirre", "Pedro Pablo", "Pablo del Río", "Río Aguirre")
    }

    mentions = vault.mention_tokens(PERSON, "Pedro Pablo del Río Aguirre", "M-001")

    assert {full} | partial <= mentions
    assert vault.token_for(PERSON, "del", "M-001") not in mentions
    assert vault.token_for(PERSON, "Pablo del", "M-001") not in mentions
    assert vault.token_for(PERSON, "Ana Fuentes Leiva", "M-001") not in mentions
    assert mentions.isdisjoint(vault.mention_tokens(PERSON, "Pedro Pablo del Río Aguirre", "M-002"))
    assert entries(tmp_path) == []


def test_only_a_person_has_shorter_forms(vault):
    assert vault.mention_tokens(PiiKind.RUT, "12.345.678-5", "M-001") == {
        vault.token_for(PiiKind.RUT, "12.345.678-5", "M-001")
    }
    assert len(vault.mention_tokens(PiiKind.ORGANIZATION, "Agrícola Maipo S.A.", "M-001")) == 1


def test_forgetting_one_token_leaves_the_others(vault):
    full = vault.remember(PERSON, "Ana Fuentes Leiva", "M-001")
    surname = vault.remember(PERSON, "Fuentes", "M-001")

    vault.forget(full, "M-001")

    assert vault.reveal(surname, "M-001") == "Fuentes"


def test_shredding_the_matter_kills_its_vault(vault, keys, tmp_path):
    token = vault.remember(PERSON, "Ana Fuentes Leiva", "M-001")
    kept = vault.remember(PERSON, "Ana Fuentes Leiva", "M-002")

    keys.shred("M-001")

    assert entries(tmp_path)
    with pytest.raises(KeyUnavailable):
        vault.reveal(token, "M-001")
    with pytest.raises(KeyUnavailable):
        vault.token_for(PERSON, "Ana Fuentes Leiva", "M-001")
    assert vault.reveal(kept, "M-002") == "Ana Fuentes Leiva"


def test_discard_removes_the_dead_entries_of_one_matter(vault, keys, tmp_path):
    vault.remember(PERSON, "Ana Fuentes Leiva", "M-001")
    kept = vault.remember(PERSON, "Ana Fuentes Leiva", "M-002")
    keys.shred("M-001")

    vault.discard("M-001")
    vault.discard("M-001")

    assert not (tmp_path / "vault" / "M-001").exists()
    assert vault.reveal(kept, "M-002") == "Ana Fuentes Leiva"
    with pytest.raises(ValueError, match="unsafe"):
        vault.discard("../M-002")


def test_an_entry_cannot_be_moved_to_another_token(vault, tmp_path):
    ana = vault.remember(PERSON, "Ana Fuentes Leiva", "M-001")
    pedro = vault.remember(PERSON, "Pedro del Río", "M-001")
    directory = tmp_path / "vault" / "M-001"
    (directory / f"{pedro[1:-1]}.token").write_bytes(
        (directory / f"{ana[1:-1]}.token").read_bytes()
    )
    with pytest.raises(IntegrityError):
        vault.reveal(pedro, "M-001")


def test_a_second_vault_on_the_same_directory_sees_the_same_entries(vault, keys, tmp_path):
    token = vault.remember(PERSON, "Ana Fuentes Leiva", "M-001")
    assert SealedTokenVault(keys, tmp_path / "vault").reveal(token, "M-001") == "Ana Fuentes Leiva"


@pytest.mark.parametrize("token", ["PERSON_0123456789ab", "[PERSON]", "[../../x_0123456789ab]", ""])
def test_rejects_what_is_not_a_token(vault, token):
    with pytest.raises(ValueError, match="not a pseudonym"):
        vault.reveal(token, "M-001")
    with pytest.raises(ValueError, match="not a pseudonym"):
        vault.forget(token, "M-001")
