import math
from dataclasses import dataclass, field
from datetime import UTC, datetime

from lexguard.errors import LexguardError


class BudgetExhausted(LexguardError):
    """The remaining privacy budget cannot cover the query."""


@dataclass(frozen=True, slots=True)
class Charge:
    at: datetime
    principal_id: str
    query: str
    epsilon: float


@dataclass(slots=True)
class PrivacyLedger:
    """Sequential composition: the epsilons of every answered query add up against one budget."""

    budget: float
    charges: list[Charge] = field(default_factory=list)

    @property
    def spent(self) -> float:
        return math.fsum(charge.epsilon for charge in self.charges)

    @property
    def remaining(self) -> float:
        return max(0.0, self.budget - self.spent)

    def charge(self, principal_id: str, query: str, epsilon: float) -> None:
        if not epsilon > 0:
            raise ValueError("epsilon must be positive")
        total = math.fsum([self.spent, epsilon])
        if total > self.budget and not math.isclose(total, self.budget):
            raise BudgetExhausted(
                f"query needs epsilon {epsilon:g}, {self.remaining:g} of {self.budget:g} left"
            )
        self.charges.append(Charge(datetime.now(UTC), principal_id, query, epsilon))
