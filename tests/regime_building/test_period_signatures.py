"""A solution phase records, per period, the signature the engine grouped it by.

Two periods carry the same signature exactly when both engine-side per-period
groupings — the decision grouping and the gated-edge fold grouping — put them in
one group. A solver's own per-period grouping inside `build_period_kernels` is a
separate component of a compiled program's identity and is not represented here,
so a consumer keying a compiled executable on this signature carries that
component alongside it.
"""

from collections.abc import Hashable
from typing import cast

from tests.solution.test_dcegm_age_specialized_function import _twin
from tests.solution.test_gated_edge_reference_age_specialized_axes import (
    _build_model as _build_gated_model,
)
from tests.test_models.deterministic.regression import get_model

_N_PERIODS = 5


def _component(*, signature: Hashable, name: str) -> Hashable:
    """Read one named component out of a period signature."""
    components = cast("tuple[tuple[str, Hashable], ...]", signature)[2:]
    return dict(components)[name]


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


def test_periods_reading_different_reference_grids_get_distinct_gated_components() -> (
    None
):
    """A gated edge's fingerprint moves with the grids it lands on.

    `saver` folds its edge into `account` at the period it lands in, and the
    gate reference `index` carries an `AgeSpecializedGrid` whose nodes drop
    between those two landing periods. The two source periods therefore never
    share the compiled fold, and their gated components say so.
    """
    signatures = _build_gated_model()._regimes["saver"].solution.period_signatures
    assert _component(signature=signatures[0], name="gated-edges") != _component(
        signature=signatures[1], name="gated-edges"
    )


def test_a_regime_declaring_no_gated_edge_has_no_gated_component() -> None:
    """A regime with no gated edge contributes `None` in the gated slot."""
    regime = get_model(n_periods=_N_PERIODS)._regimes["working_life"]
    signature = regime.solution.period_signatures[0]
    assert _component(signature=signature, name="gated-edges") is None


def test_a_source_period_losing_a_target_gets_its_own_decision_component() -> None:
    """Losing a declared target splits a gated source through its decision component.

    `saver` can continue as itself from period 0 but not from period 1. The
    gated fingerprint knows nothing about which regimes are active, so where two
    landing periods resolve the same grids it is the decision component's target
    tuple that keeps the periods apart.
    """
    signatures = _build_gated_model()._regimes["saver"].solution.period_signatures
    assert _component(signature=signatures[0], name="decision") != _component(
        signature=signatures[1], name="decision"
    )
