import pytest

from lexguard.analytics.budget import BudgetExhausted, PrivacyLedger

ONE = ["M-001"]


def test_charges_add_up_within_a_matter():
    ledger = PrivacyLedger(1.0)
    ledger.charge("scientist-1", "count by kind", 0.25, ONE)
    ledger.charge("scientist-1", "count by month", 0.5, ONE)

    assert ledger.spent("M-001") == 0.75
    assert ledger.remaining("M-001") == 0.25
    assert [(c.principal_id, c.query, c.epsilon, c.matters) for c in ledger.charges] == [
        ("scientist-1", "count by kind", 0.25, ("M-001",)),
        ("scientist-1", "count by month", 0.5, ("M-001",)),
    ]


def test_a_query_spends_in_every_matter_it_reads_and_in_no_other():
    ledger = PrivacyLedger(1.0)
    ledger.charge("scientist-1", "count by kind", 0.75, ["M-003", "M-001"])

    assert ledger.spent("M-001") == ledger.spent("M-003") == 0.75
    assert ledger.remaining("M-005") == 1.0
    ledger.charge("scientist-1", "count by kind", 1.0, ["M-005"])


def test_one_exhausted_matter_refuses_the_whole_query():
    ledger = PrivacyLedger(1.0)
    ledger.charge("scientist-1", "count by kind", 0.8, ONE)

    with pytest.raises(BudgetExhausted, match=r"M-001 has 0\.2 of 1 left"):
        ledger.charge("scientist-1", "count by matter", 0.5, ["M-003", "M-001"])
    assert ledger.spent("M-003") == 0
    assert len(ledger.charges) == 1


def test_the_budget_belongs_to_the_data_not_to_who_asks():
    ledger = PrivacyLedger(1.0)
    ledger.charge("scientist-1", "count by kind", 0.8, ONE)

    with pytest.raises(BudgetExhausted):
        ledger.charge("scientist-2", "count by kind", 0.5, ONE)


def test_budget_can_be_spent_exactly():
    ledger = PrivacyLedger(1.0)
    for _ in range(10):
        ledger.charge("scientist-1", "count by kind", 0.1, ONE)

    assert ledger.remaining("M-001") == pytest.approx(0.0, abs=1e-12)
    with pytest.raises(BudgetExhausted):
        ledger.charge("scientist-1", "count by kind", 0.01, ONE)


def test_a_restart_does_not_refill_the_budget(tmp_path):
    path = tmp_path / "analytics" / "budget.jsonl"
    first = PrivacyLedger(1.0, path)
    first.charge("scientist-1", "count by kind", 0.8, ["M-001", "M-003"])

    reopened = PrivacyLedger(1.0, path)

    assert reopened.charges == first.charges
    assert reopened.remaining("M-003") == pytest.approx(0.2)
    with pytest.raises(BudgetExhausted):
        reopened.charge("scientist-1", "count by kind", 0.5, ONE)
    assert len(path.read_text().splitlines()) == 1


@pytest.mark.parametrize("epsilon", [0.0, -1.0, float("nan")])
def test_epsilon_must_be_positive(epsilon):
    with pytest.raises(ValueError, match="positive"):
        PrivacyLedger(1.0).charge("scientist-1", "count by kind", epsilon, ONE)
