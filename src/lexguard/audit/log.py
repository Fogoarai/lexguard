import hashlib
import hmac
import json
import re
from collections.abc import Iterator
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, NamedTuple

from lexguard.domain import Action, AuditEvent, Purpose, Role, Zone
from lexguard.errors import IntegrityError

GENESIS = "0" * 64

_IDENTIFIER = re.compile(r"[\w.:/\[\]-]{1,64}")


class _Record(NamedTuple):
    seq: int
    mac: str
    event: dict[str, Any]


class ChainedAuditLog:
    """Append-only JSON-lines audit log in which every record is HMAC-chained to the last.

    A record holds its sequence number, the MAC of the previous record and its own MAC
    over both and the event, so it cannot be altered, dropped or moved without the key.
    Detail values that are not identifier-like are stored as keyed fingerprints, never as
    text. One writer per file.
    """

    def __init__(self, path: Path, key: bytes) -> None:
        if not key:
            raise ValueError("audit key must not be empty")
        self._path = path
        self._key = key
        path.parent.mkdir(parents=True, exist_ok=True)
        self._count, self._head = self._tail()

    def record(self, event: AuditEvent) -> None:
        body = {"seq": self._count + 1, "prev": self._head, "event": self._encode(event)}
        mac = self._mac(_canonical(body))
        with self._path.open("a", encoding="utf-8") as file:
            file.write(_canonical(body | {"mac": mac}) + "\n")
        self._count, self._head = self._count + 1, mac

    def verify(self) -> int:
        """Return the record count, or raise `IntegrityError` naming the first broken record.

        A tail cut off cleanly is only noticed by an instance that has seen the longer log.
        """
        count, _ = self._tail()
        if count < self._count:
            raise IntegrityError(f"audit record {count + 1} is missing")
        return count

    def events(
        self,
        *,
        principal_id: str | None = None,
        matter_id: str | None = None,
        denied_only: bool = False,
    ) -> list[AuditEvent]:
        """Events in order, narrowed by the given filters; never returns unverified records."""
        events = (_decode(record.event) for record in self._replay())
        return [
            event
            for event in events
            if (principal_id is None or event.principal_id == principal_id)
            and (matter_id is None or event.matter_id == matter_id)
            and not (denied_only and event.allowed)
        ]

    def fingerprint(self, text: str) -> str:
        """The value stored in place of free text, to match a known text against the log."""
        return "hmac:" + self._mac(f"detail\x1f{text}")[:32]

    def _encode(self, event: AuditEvent) -> dict[str, Any]:
        detail = {name: self._scrub(str(value)) for name, value in event.detail.items()}
        return asdict(event) | {"at": event.at.isoformat(), "detail": detail}

    def _scrub(self, value: str) -> str:
        return value if _IDENTIFIER.fullmatch(value) else self.fingerprint(value)

    def _mac(self, message: str) -> str:
        return hmac.new(self._key, message.encode(), hashlib.sha256).hexdigest()

    def _tail(self) -> tuple[int, str]:
        count, head = 0, GENESIS
        for record in self._replay():
            count, head = record.seq, record.mac
        return count, head

    def _replay(self) -> Iterator[_Record]:
        if not self._path.exists():
            return
        head = GENESIS
        with self._path.open("rb") as lines:
            for position, line in enumerate(lines, start=1):
                try:
                    fields = json.loads(line)
                    mac = fields.pop("mac")
                    genuine = line.endswith(b"\n") and hmac.compare_digest(
                        mac, self._mac(_canonical(fields))
                    )
                except (ValueError, TypeError, AttributeError, KeyError):
                    genuine = False
                if not genuine:
                    raise IntegrityError(f"audit record {position} was altered")
                if fields["prev"] != head:
                    raise IntegrityError(
                        f"audit record {position} is missing or out of order: "
                        f"record {fields['seq']} is in its place"
                    )
                head = mac
                yield _Record(position, mac, fields["event"])


def _canonical(fields: dict[str, Any]) -> str:
    return json.dumps(fields, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _decode(fields: dict[str, Any]) -> AuditEvent:
    return AuditEvent(
        at=datetime.fromisoformat(fields["at"]),
        principal_id=fields["principal_id"],
        role=Role(fields["role"]),
        action=Action(fields["action"]),
        purpose=Purpose(fields["purpose"]),
        zone=Zone(fields["zone"]),
        matter_id=fields["matter_id"],
        allowed=fields["allowed"],
        reason=fields["reason"],
        detail=fields["detail"],
    )
