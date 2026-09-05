"""A solution phase records, per period, the signature its kernels were grouped by.

Two periods carry the same signature exactly when every per-period grouping that
built the regime's solve kernels put them in one group, so the signature is a
sound key for a compiled program: equal signature implies identical grouped
inputs.
"""

from tests.solution.test_dcegm_age_specialized_function import _twin
from tests.test_models.deterministic.regression import get_model

_N_PERIODS = 5


def test_period_signatures_cover_exactly_the_regimes_active_periods() -> None:
    """Every active period of a regime carries a signature, and no other period does."""
    regime = get_model(n_periods=_N_PERIODS)._regimes["working_life"]
    assert set(regime.solution.period_signatures) == set(regime.active_periods)


def test_an_age_invariant_regime_shares_one_signature_across_equal_continuations() -> (
    None
):
    """Periods reaching the same continuation targets share one signature.

    `working_life` reaches both itself and `dead` from periods 0-2 and only
    `dead` from its last active period, and nothing in the model is
    age-specialized, so the first three periods are one group.
    """
    regime = get_model(n_periods=_N_PERIODS)._regimes["working_life"]
    signatures = regime.solution.period_signatures
    assert signatures[0] == signatures[1] == signatures[2]


def test_a_period_reaching_different_targets_gets_its_own_signature() -> None:
    """The last active period, reaching only `dead`, is not grouped with the rest."""
    regime = get_model(n_periods=_N_PERIODS)._regimes["working_life"]
    signatures = regime.solution.period_signatures
    assert signatures[3] != signatures[0]


def test_a_terminal_regime_gives_every_period_one_signature() -> None:
    """A terminal regime's kernels are period-invariant, so all its periods group."""
    regime = get_model(n_periods=_N_PERIODS)._regimes["dead"]
    assert len(set(regime.solution.period_signatures.values())) == 1


def test_age_specialized_periods_get_distinct_signatures() -> None:
    """A utility that is a different function at every age splits every period."""
    regime = _twin(solver_kind="brute_force", age_specialized=True)._regimes[
        "working_life"
    ]
    signatures = regime.solution.period_signatures
    assert len(set(signatures.values())) == len(signatures)


def test_signatures_are_stable_across_model_construction() -> None:
    """Building the same model twice yields the same signatures, value for value."""
    first = get_model(n_periods=_N_PERIODS)._regimes["working_life"]
    second = get_model(n_periods=_N_PERIODS)._regimes["working_life"]
    assert dict(first.solution.period_signatures) == dict(
        second.solution.period_signatures
    )


def test_signatures_are_hashable() -> None:
    """Every signature can be used as a dictionary key."""
    regime = get_model(n_periods=_N_PERIODS)._regimes["working_life"]
    assert isinstance(hash(tuple(regime.solution.period_signatures.values())), int)
