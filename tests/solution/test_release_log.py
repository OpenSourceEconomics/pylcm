"""What a solve releases, and what it leaves alone.

`log_level="debug"` logs every release with the artifact key and the closing
dispatch; a solve at any other level publishes the same values. A release only
ever frees a buffer the engine produced, so an array the model declares stays
readable even when a solver hands it through as a continuation leaf.
"""

import logging

import jax.numpy as jnp
import numpy as np

from _lcm.execution.value_transfer import ValueArtifactKind
from lcm import AgeGrid, LinSpacedGrid, Model, categorical
from lcm.consumption_savings_regime import (
    ConsumptionSavingsRegime,
    LiquidMargin,
    post_decision_lower_bound,
)
from lcm.regime import Regime
from lcm.solver_api import SolutionResult
from lcm.solvers import EGM
from lcm.typing import ContinuousAction, ContinuousState, FloatND, ScalarInt
from tests.regime_building.test_gated_edges_collective_solve import (
    EKLRegimeId,
    _make_full_topology_regimes,
)


def _params() -> dict[str, float]:
    """Discount factor and consent-threshold parameters of the mini-EKL model."""
    return {"discount_factor": 0.95, "delta_f": 0.5, "delta_m": 0.2}


class _Records(logging.Handler):
    """Collect the release records the solve emits, in emission order."""

    def __init__(self) -> None:
        super().__init__(level=logging.DEBUG)
        self.records: list[logging.LogRecord] = []

    def emit(self, record: logging.LogRecord) -> None:
        if hasattr(record, "artifact_key"):
            self.records.append(record)


def _model() -> Model:
    return Model(
        regimes=_make_full_topology_regimes(),
        ages=AgeGrid(start=0, stop=3, step="Y"),
        regime_id_class=EKLRegimeId,
    )


def _solve_with_release_log() -> tuple[Model, list[logging.LogRecord], SolutionResult]:
    model = _model()
    handler = _Records()
    logger = logging.getLogger("lcm")
    logger.addHandler(handler)
    try:
        solution = model.solve(params=_params(), log_level="debug")
    finally:
        logger.removeHandler(handler)
    return model, handler.records, solution


def test_a_gated_continuation_is_released_by_the_source_that_reads_it() -> None:
    """Every released Wbar names the source's own dispatch as the closer."""
    _, records, _ = _solve_with_release_log()
    released = [
        (r.artifact_key, r.closing_dispatch)  # ty: ignore[unresolved-attribute]
        for r in records
        if r.artifact_key.kind  # ty: ignore[unresolved-attribute]
        is ValueArtifactKind.GATED_CONTINUATION
    ]

    assert released
    assert all(
        dispatch == (artifact.period - 1, artifact.regime)
        for artifact, dispatch in released
    )


def test_no_regime_value_is_ever_released() -> None:
    """Retained values stay on device for the whole solve."""
    _, records, solution = _solve_with_release_log()

    assert not any(
        r.artifact_key.kind  # ty: ignore[unresolved-attribute]
        is ValueArtifactKind.REGIME_VALUE
        for r in records
    )
    assert not any(
        value.is_deleted()
        for period in solution.values.values()
        for value in period.values()
    )


def test_releasing_changes_no_published_value() -> None:
    """A solve with releases publishes the values a silent solve publishes."""
    model, _, debug = _solve_with_release_log()
    silent = model.solve(params=_params(), log_level="off")

    np.testing.assert_array_equal(
        np.asarray(debug.values[1]["married"]), np.asarray(silent.values[1]["married"])
    )


_PASS_THROUGH_WEALTH_GRID = LinSpacedGrid(start=1.0, stop=5.0, n_points=6)
_PASS_THROUGH_CONSUMPTION_GRID = LinSpacedGrid(start=0.1, stop=5.0, n_points=7)
_PASS_THROUGH_SAVINGS_GRID = LinSpacedGrid(start=0.0, stop=5.0, n_points=8)


@categorical(ordered=False)
class _PassThroughRegimeId:
    working: ScalarInt
    dead: ScalarInt


def _pass_through_next_regime() -> ScalarInt:
    """One decision period is followed by the terminal regime."""
    return _PassThroughRegimeId.dead


def _pass_through_utility(consumption: ContinuousAction) -> FloatND:
    """Log flow utility of consumption."""
    return jnp.log(consumption)


def _pass_through_terminal_utility(wealth: ContinuousState) -> FloatND:
    """The terminal period consumes the wealth on hand."""
    return jnp.log1p(wealth)


def _pass_through_savings(
    *, wealth: ContinuousState, consumption: ContinuousAction
) -> ContinuousState:
    """Post-decision liquid balance."""
    return wealth - consumption


def _pass_through_next_wealth(savings: ContinuousState) -> ContinuousState:
    """Carry the liquid balance into the terminal period."""
    return savings


def _pass_through_margin() -> LiquidMargin:
    """The single liquid margin of the pass-through model."""
    return LiquidMargin(
        state="wealth",
        action="consumption",
        resources="wealth",
        post_decision_state="savings",
    )


def _pass_through_model() -> Model:
    """A two-period EGM model whose terminal carry reuses the declared grid.

    The terminal regime carries no other state, so its endogenous grid is the
    wealth grid the model declares, unbroadcast. Solving without compilation
    hands that very array to the working regime as a continuation leaf.
    """
    margin = _pass_through_margin()
    working = ConsumptionSavingsRegime(
        transition=_pass_through_next_regime,
        states={"wealth": _PASS_THROUGH_WEALTH_GRID},
        actions={"consumption": _PASS_THROUGH_CONSUMPTION_GRID},
        state_transitions={"wealth": _pass_through_next_wealth},
        functions={
            "utility": _pass_through_utility,
            "savings": _pass_through_savings,
        },
        constraints={
            "borrowing_limit": post_decision_lower_bound(margin=margin, lower=0.0)
        },
        liquid=margin,
        solver=EGM(savings_grid=_PASS_THROUGH_SAVINGS_GRID),
        active=lambda age: age == 0,
    )
    dead = Regime(
        transition=None,
        states={"wealth": _PASS_THROUGH_WEALTH_GRID},
        functions={"utility": _pass_through_terminal_utility},
        active=lambda age: age == 1,
    )
    return Model(
        regimes={"working": working, "dead": dead},
        regime_id_class=_PassThroughRegimeId,
        ages=AgeGrid(start=0, stop=1, step="Y"),
        enable_jit=False,
    )


def _pass_through_params() -> dict[str, dict[str, dict[str, float]]]:
    """Discount factor of the pass-through model."""
    return {"working": {"koopmans_aggregator": {"discount_factor": 0.95}}, "dead": {}}


def test_a_model_whose_grid_is_a_continuation_leaf_solves_twice_alike() -> None:
    """A grid a solve hands through as a continuation leaf stays readable."""
    model = _pass_through_model()
    params = _pass_through_params()

    first = model.solve(params=params, log_level="debug")
    second = model.solve(params=params, log_level="debug")

    np.testing.assert_array_equal(
        np.asarray(second.values[0]["working"]),
        np.asarray(first.values[0]["working"]),
    )


def test_a_model_whose_grid_is_a_continuation_leaf_still_releases_a_leaf() -> None:
    """Handing a declared grid through leaves the engine's own leaves releasable."""
    model = _pass_through_model()
    handler = _Records()
    logger = logging.getLogger("lcm")
    logger.addHandler(handler)
    try:
        model.solve(params=_pass_through_params(), log_level="debug")
    finally:
        logger.removeHandler(handler)

    assert any(
        record.artifact_key.kind  # ty: ignore[unresolved-attribute]
        is ValueArtifactKind.CONTINUATION_LEAF
        for record in handler.records
    )
