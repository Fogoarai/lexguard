import re
from itertools import cycle

_RUT = re.compile(r"(\d{1,2})\.?(\d{3})\.?(\d{3})-?([\dkK])")


def check_digit(number: int) -> str:
    total = sum(int(d) * w for d, w in zip(reversed(str(number)), cycle(range(2, 8)), strict=False))
    return "0123456789K"[-total % 11]


def format_rut(number: int) -> str:
    return f"{number:,}".replace(",", ".") + f"-{check_digit(number)}"


def parse(text: str) -> tuple[int, str] | None:
    match = _RUT.fullmatch(text.strip())
    if match is None:
        return None
    *digits, verifier = match.groups()
    return int("".join(digits)), verifier.upper()


def is_valid(text: str) -> bool:
    parsed = parse(text)
    return parsed is not None and check_digit(parsed[0]) == parsed[1]


def canonical(text: str) -> str:
    parsed = parse(text)
    if parsed is None:
        raise ValueError(f"not a RUT: {text!r}")
    number, verifier = parsed
    return f"{number}-{verifier}"
