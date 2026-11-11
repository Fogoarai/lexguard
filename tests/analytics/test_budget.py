import pytest

from lexguard.analytics.budget import BudgetExhausted, PrivacyLedger


def test_charges_add_up_against_the_budget():
    ledger = PrivacyLedger(1.0)
    ledger.charge("scientist-1", "count by kind", 0.25)
    ledger.charge("scientist-1", "count by month", 0.5)

    assert ledger.spent == 0.75
    assert ledger.remaining == 0.25
    assert [(c.principal_id, c.query, c.epsilon) for c in ledger.charges] == [
        ("scientist-1", "count by kind", 0.25),
        ("scientist-1", "count by month", 0.5),
    ]


def test_budget_can_be_spent_exactly():
    ledger = PrivacyLedger(1.0)
    for _ in range(10):
        ledger.charge("scientist-1", "count by kind", 0.1)

    assert ledger.remaining == pytest.approx(0.0, abs=1e-12)
    with pytest.raises(BudgetExhausted):
        ledger.charge("scientist-1", "count by kind", 0.01)


def test_refused_charge_leaves_the_ledger_untouched():
    ledger = PrivacyLedger(1.0)
    ledger.charge("scientist-1", "count by kind", 0.8)

    with pytest.raises(BudgetExhausted, match=r"0\.2 of 1 left"):
        ledger.charge("scientist-2", "count by matter", 0.5)
    assert ledger.spent == 0.8
    assert len(ledger.charges) == 1


@pytest.mark.parametrize("epsilon", [0.0, -1.0, float("nan")])
def test_epsilon_must_be_positive(epsilon):
    with pytest.raises(ValueError, match="positive"):
        PrivacyLedger(1.0).charge("scientist-1", "count by kind", epsilon)
