import pytest

from lexguard import rut


@pytest.mark.parametrize(
    ("number", "digit"),
    [(12_345_678, "5"), (11_111_111, "1"), (20_000_011, "0"), (6_000_000, "K")],
)
def test_check_digit(number, digit):
    assert rut.check_digit(number) == digit


@pytest.mark.parametrize("text", ["12.345.678-5", "12345678-5", "123456785", "6.000.000-k"])
def test_accepts_common_spellings(text):
    assert rut.is_valid(text)


@pytest.mark.parametrize("text", ["12.345.678-9", "1234", "12.345.678", "abc"])
def test_rejects_invalid(text):
    assert not rut.is_valid(text)


def test_format_and_canonical_round_trip():
    assert rut.format_rut(12_345_678) == "12.345.678-5"
    assert rut.canonical("12.345.678-5") == "12345678-5"


def test_canonical_rejects_garbage():
    with pytest.raises(ValueError, match="not a RUT"):
        rut.canonical("hola")
