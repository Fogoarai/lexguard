import json
import math
from collections.abc import Collection
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from lexguard.errors import LexguardError


class BudgetExhausted(LexguardError):
    """The remaining privacy budget cannot cover the query."""


@dataclass(frozen=True, slots=True)
class Charge:
    at: datetime
    principal_id: str
    query: str
    epsilon: float
    matters: tuple[str, ...]


@dataclass(slots=True)
class PrivacyLedger:
    """What each matter has spent of its budget, whoever asked.

    A query spends its epsilon in every matter it reads. Within a matter the epsilons add
    up (sequential composition); matters hold disjoint documents, so what one spends leaves
    the others' budgets alone (parallel composition). With a `path`, charges are appended
    to that file and read back from it, so a restart does not refill a budget.
    """

    budget: float
    path: Path | None = None
    charges: list[Charge] = field(default_factory=list, init=False)

    def __post_init__(self) -> None:
        if self.path is not None and self.path.exists():
            lines = self.path.read_text(encoding="utf-8").splitlines()
            self.charges = [_decode(line) for line in lines]

    def spent(self, matter_id: str) -> float:
        return math.fsum(c.epsilon for c in self.charges if matter_id in c.matters)

    def remaining(self, matter_id: str) -> float:
        return max(0.0, self.budget - self.spent(matter_id))

    def charge(
        self, principal_id: str, query: str, epsilon: float, matters: Collection[str]
    ) -> None:
        """All or nothing: refused if any of the matters cannot cover the query."""
        if not epsilon > 0:
            raise ValueError("epsilon must be positive")
        for matter_id in sorted(matters):
            total = math.fsum([self.spent(matter_id), epsilon])
            if total > self.budget and not math.isclose(total, self.budget):
                raise BudgetExhausted(
                    f"query needs epsilon {epsilon:g}, "
                    f"{matter_id} has {self.remaining(matter_id):g} of {self.budget:g} left"
                )
        charge = Charge(datetime.now(UTC), principal_id, query, epsilon, tuple(sorted(matters)))
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as file:
                file.write(json.dumps(asdict(charge) | {"at": charge.at.isoformat()}) + "\n")
        self.charges.append(charge)


def _decode(line: str) -> Charge:
    fields = json.loads(line)
    return Charge(
        at=datetime.fromisoformat(fields["at"]),
        principal_id=fields["principal_id"],
        query=fields["query"],
        epsilon=fields["epsilon"],
        matters=tuple(fields["matters"]),
    )
