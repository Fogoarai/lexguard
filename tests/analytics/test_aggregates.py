from collections import Counter
from types import SimpleNamespace

import numpy as np
import pytest

from lexguard.analytics.aggregates import Aggregator, Dimension
from lexguard.analytics.budget import BudgetExhausted
from lexguard.domain import Action, Decision, Principal, Purpose, Role, Sensitivity, Zone
from lexguard.errors import AccessDenied
from lexguard.guard import Guard
from lexguard.settings import Settings
from lexguard.testing import AllowAll, DenyAll, MembershipPolicy

SCIENTIST = Principal("scientist-1", Role.DATA_SCIENTIST)


def test_counts_by_kind_over_matters_that_permit_analytics(aggregator, corpus):
    release = aggregator(min_group_size=1).count_by(SCIENTIST, corpus.matters, Dimension.KIND, 0.5)

    assert release.counts == {"contract": 4, "email": 2, "filing": 4, "memo": 2}
    assert release.suppressed == 0


def test_matters_without_the_analytics_purpose_are_never_read(aggregator, corpus, audit):
    release = aggregator().count_by(SCIENTIST, corpus.matters, Dimension.MATTER, 0.5)

    assert release.counts == {"M-001": 6, "M-003": 6}
    assert {event.matter_id for event in audit.events} == {"M-001", "M-003"}


def test_counts_by_month(aggregator, corpus):
    release = aggregator(min_group_size=1).count_by(SCIENTIST, corpus.matters, Dimension.MONTH, 0.5)

    assert release.counts == Counter(
        f"{labeled.document.created_at:%Y-%m}"
        for labeled in corpus.documents
        if labeled.document.matter_id in {"M-001", "M-003"}
    )


@pytest.mark.parametrize(
    ("min_group_size", "released", "suppressed"),
    [(3, {"contract": 4, "filing": 4}, 2), (Settings().min_group_size, {}, 4)],
)
def test_groups_under_the_minimum_size_are_suppressed(
    aggregator, corpus, min_group_size, released, suppressed
):
    release = aggregator(min_group_size=min_group_size).count_by(
        SCIENTIST, corpus.matters, Dimension.KIND, 0.5
    )

    assert release.counts == released
    assert release.suppressed == suppressed


def test_noise_is_scaled_to_sensitivity_over_epsilon(aggregator, corpus, noise):
    noise.value = 2.4
    release = aggregator(min_group_size=3).count_by(SCIENTIST, corpus.matters, Dimension.KIND, 0.25)

    assert release.noise_scale == 4.0
    assert noise.scales == [4.0, 4.0]
    assert release.counts == {"contract": 6, "filing": 6}


def test_noisy_counts_are_never_negative(aggregator, corpus, noise):
    noise.value = -50.0
    release = aggregator().count_by(SCIENTIST, corpus.matters, Dimension.MATTER, 0.5)

    assert release.counts == {"M-001": 0, "M-003": 0}


def test_guard_is_asked_per_compartment_with_the_aggregate_shape(aggregator, corpus, audit):
    aggregator().count_by(SCIENTIST, corpus.matters, Dimension.KIND, 0.5)

    assert {(event.action, event.zone, event.purpose) for event in audit.events} == {
        (Action.AGGREGATE, Zone.DEIDENTIFIED, Purpose.ANALYTICS)
    }
    assert Counter(event.matter_id for event in audit.events) == {"M-001": 3, "M-003": 3}
    assert audit.events[0].detail == {"dimension": "kind"}


def test_denied_compartments_are_not_counted(aggregator, corpus):
    no_privileged = SimpleNamespace(
        decide=lambda request: Decision(
            request.resource.sensitivity is not Sensitivity.PRIVILEGED, "no privileged"
        )
    )
    by_kind = aggregator(no_privileged, min_group_size=1).count_by(
        SCIENTIST, corpus.matters, Dimension.KIND, 0.5
    )
    assert by_kind.counts == {"contract": 4, "filing": 4}

    member = Principal("scientist-2", Role.DATA_SCIENTIST, frozenset({"M-001"}))
    by_matter = aggregator(MembershipPolicy()).count_by(
        member, corpus.matters, Dimension.MATTER, 0.5
    )
    assert by_matter.counts == {"M-001": 6}


def test_query_denied_everywhere_is_refused_without_spending_budget(aggregator, corpus):
    denied = aggregator(DenyAll())

    with pytest.raises(AccessDenied):
        denied.count_by(SCIENTIST, corpus.matters, Dimension.KIND, 0.5)
    assert denied.ledger.spent == 0


def test_queries_are_refused_once_the_budget_is_spent(aggregator, corpus, noise):
    analytics = aggregator(epsilon=1.0)
    analytics.count_by(SCIENTIST, corpus.matters, Dimension.KIND, 0.5)
    analytics.count_by(SCIENTIST, corpus.matters, Dimension.MATTER, 0.5)
    drawn = len(noise.scales)

    with pytest.raises(BudgetExhausted):
        analytics.count_by(SCIENTIST, corpus.matters, Dimension.MONTH, 0.1)

    assert len(noise.scales) == drawn
    assert analytics.ledger.spent == 1.0
    assert [charge.query for charge in analytics.ledger.charges] == [
        "count by kind",
        "count by matter",
    ]


def test_defaults_come_from_settings_and_draw_real_laplace_noise(store, corpus, audit):
    analytics = Aggregator(store, Guard(AllowAll(), audit), rng=np.random.default_rng(0))

    release = analytics.count_by(SCIENTIST, corpus.matters, Dimension.MATTER, 0.5)

    assert analytics.ledger.budget == Settings().epsilon
    assert set(release.counts) == {"M-001", "M-003"}
    assert all(type(count) is int for count in release.counts.values())
    assert release.counts != {"M-001": 6, "M-003": 6}
