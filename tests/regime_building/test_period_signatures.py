"""A solution phase records, per period, the signature the engine grouped it by.

Two periods carry the same signature exactly when both engine-side per-period
groupings — the decision grouping and the gated-edge fold grouping — put them in
one group. A solver's own per-period grouping inside `build_period_kernels` is a
separate component of a compiled program's identity and is not represented here,
so a consumer keying a compiled executable on this signature carries that
component alongside it.

Typed incoming queries wrap that grouping signature with their exact date.
Their stored-reference checks inspect the bound read and support directly.
"""

from collections.abc import Hashable
from typing import cast

import numpy as np

from _lcm.policy_queries import QueryKind, _PolicyQueryKernel
from tests.solution.test_dcegm_age_specialized_function import _twin
from tests.solution.test_gated_edge_reference_age_specialized_axes import (
    _build_model as _build_participation_model,
)
from tests.test_models.deterministic.regression import get_model

_N_PERIODS = 5


def _component(*, signature: Hashable, name: str) -> Hashable:
    """Read the underlying grouping component through a typed-query wrapper."""
    parts = cast("tuple[Hashable, ...]", signature)
    if len(parts) == 3 and parts[1] == "typed-exit-query":
        parts = cast("tuple[Hashable, ...]", parts[0])
    components = cast("tuple[tuple[str, Hashable], ...]", parts[2:])
    return dict(components)[name]


def test_period_signatures_cover_exactly_the_regimes_covered_periods() -> None:
    """Every covered period of a regime carries a signature, and no other does."""
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


def test_participation_references_bind_their_dated_stored_support() -> None:
    """The represented index read uses its actual landing date and grid."""
    regime = _build_participation_model()._regimes["saver"]
    for source_period, expected in ((0, [0.0, 4.0, 8.0]), (1, [0.0, 2.0, 4.0])):
        kernel = regime.solution.period_kernels[source_period]
        assert isinstance(kernel, _PolicyQueryKernel)
        plan = kernel.root_context.participation_queries[source_period]["account"]
        assert plan.period == source_period + 1
        represented = [
            node
            for node in plan.queries.definitions.values()
            if node.read.kind is QueryKind.INTEGRATED_FULL
        ]
        assert len(represented) == 1
        node = represented[0]
        assert node.read.target == "index"
        assert node.read.period == plan.period
        assert len(node.operand_value_reads) == 1
        stored = node.operand_value_reads[0].target
        assert (stored.regime, stored.period) == ("index", plan.period)
        dated = node.context.period_to_regime_v_interp
        assert dated is not None
        np.testing.assert_array_equal(
            dated[plan.period]["index"].continuous_states["level"].to_jax(), expected
        )


def test_a_regime_declaring_no_gated_edge_has_no_gated_component() -> None:
    """A regime with no gated edge contributes `None` in the gated slot."""
    regime = get_model(n_periods=_N_PERIODS)._regimes["working_life"]
    signature = regime.solution.period_signatures[0]
    assert _component(signature=signature, name="gated-edges") is None


def test_a_source_period_losing_a_target_gets_its_own_decision_component() -> None:
    """Losing the self target splits the underlying decision grouping."""
    regime = _build_participation_model()._regimes["saver"]
    for period in (0, 1):
        kernel = regime.solution.period_kernels[period]
        assert isinstance(kernel, _PolicyQueryKernel)
        graph = kernel.root_context.solution_reachability
        assert graph.has_edge(period=period, source="saver", target="account")
        assert graph.has_edge(period=period, source="saver", target="saver") is (
            period == 0
        )
    signatures = regime.solution.period_signatures
    assert _component(signature=signatures[0], name="decision") != _component(
        signature=signatures[1], name="decision"
    )
