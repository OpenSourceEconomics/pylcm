"""`ExecutionConfig.axis_widths` may fix one axis width per regime.

A bare integer under an axis name broadcasts to every regime declaring that
axis; a mapping from regime name to width pins only the regimes it names and
leaves the planner free everywhere else.
"""

import functools
from types import MappingProxyType
from typing import Any

import numpy as np
import pytest

from _lcm.solution import backward_induction
from lcm import ExecutionConfig, Model
from lcm.exceptions import ExecutionPlanningError
from tests.simulation._profile_comparison import (
    assert_values_agree as assert_agrees_to_ulp,
)
from tests.test_models.initial_nodes import initial_nodes_of
from tests.test_models.processes import (
    MultiRegimeId,
    get_multi_regime_model,
    get_multi_regime_params,
)

_N_PERIODS = 6
_CELL_AXIS = "cell"
# The leaves whose low-income, bad-health entries are born by cancellation: every
# working-life period, each a small sum of a negative flow utility and a larger
# positive continuation.
_CANCELLATION_LEAVES = frozenset({(0, "work"), (1, "work"), (2, "work")})


def _base_model() -> Model:
    return get_multi_regime_model(n_periods=_N_PERIODS, distribution_type="normal")


def _model_with(config: ExecutionConfig) -> Model:
    base = _base_model()
    return Model(
        edges=base.edges,
        regimes=base.user_regimes,
        ages=base.ages,
        regime_id_class=MultiRegimeId,
        fixed_params=dict(base.fixed_params),
        execution_config=config,
        initial_nodes=initial_nodes_of(model=base),
    )


# keyword-only-exempt: library-callback=_group_cores_by_regime_period
def _capture_grouping(
    cores_by_triple: Any,
    *,
    original: Any,
    sink: dict[tuple[str, int, str], dict[str, int]],
) -> Any:
    """Record each compiled core's lowering widths under its (regime, period, core)."""
    for triple, core in cores_by_triple.items():
        sink[triple] = dict(core.tile_widths)
    return original(cores_by_triple)


def _solve_and_collect_widths(
    *, config: ExecutionConfig
) -> tuple[dict[tuple[str, int, str], dict[str, int]], Any]:
    """Solve the two-regime model and return the widths every core was lowered at."""
    model = _model_with(config)
    observed: dict[tuple[str, int, str], dict[str, int]] = {}
    with pytest.MonkeyPatch.context() as patcher:
        patcher.setattr(
            backward_induction,
            "_group_cores_by_regime_period",
            functools.partial(
                _capture_grouping,
                original=backward_induction._group_cores_by_regime_period,
                sink=observed,
            ),
        )
        solution = model.solve(
            params=get_multi_regime_params("normal"),
            log_level="off",
        )
    return observed, solution


def _cell_widths_by_regime(
    widths: dict[tuple[str, int, str], dict[str, int]],
) -> dict[str, set[int]]:
    """Collapse the per-core record to the cell widths each regime was lowered at."""
    by_regime: dict[str, set[int]] = {}
    for (regime_name, _period, _core), core_widths in widths.items():
        if _CELL_AXIS in core_widths:
            by_regime.setdefault(regime_name, set()).add(core_widths[_CELL_AXIS])
    return by_regime


def test_a_bare_integer_broadcasts_to_every_regime() -> None:
    """One integer under an axis name fixes that axis in every regime."""
    observed, _ = _solve_and_collect_widths(
        config=ExecutionConfig(axis_widths={_CELL_AXIS: 2})
    )

    assert _cell_widths_by_regime(observed) == {"work": {2}, "retire": {2}}


def test_a_per_regime_width_leaves_every_other_regime_planned() -> None:
    """Pinning one regime's cell width does not narrow the other regime's."""
    planned, _ = _solve_and_collect_widths(config=ExecutionConfig())
    pinned, _ = _solve_and_collect_widths(
        config=ExecutionConfig(axis_widths={_CELL_AXIS: {"retire": 2}}),
    )

    planned_by_regime = _cell_widths_by_regime(planned)
    pinned_by_regime = _cell_widths_by_regime(pinned)

    assert pinned_by_regime["retire"] == {2}
    assert pinned_by_regime["work"] == planned_by_regime["work"]
    assert planned_by_regime["retire"] == planned_by_regime["work"] != {2}


def test_a_per_regime_width_preserves_the_solved_values() -> None:
    """Chunking one regime finer partitions the work without changing the answer."""
    _, planned_solution = _solve_and_collect_widths(config=ExecutionConfig())
    _, pinned_solution = _solve_and_collect_widths(
        config=ExecutionConfig(axis_widths={_CELL_AXIS: {"retire": 2}}),
    )

    planned_values = planned_solution._engine_view.values
    pinned_values = pinned_solution._engine_view.values
    assert set(pinned_values) == set(planned_values)
    for period, by_regime in planned_values.items():
        for regime_name, expected in by_regime.items():
            got = pinned_values[period][regime_name]
            err_msg = f"{regime_name} period {period}"
            if (period, regime_name) not in _CANCELLATION_LEAVES:
                assert_agrees_to_ulp(
                    got=got, expected=expected, n_ulp=8, err_msg=err_msg
                )
                continue
            # A cancellation leaf's low-income, bad-health entries are a small
            # sum of a flow utility and a continuation of opposite sign. A
            # reordered reduction, here or in a later period whose value the
            # continuation averages, moves such an entry by roundings of those
            # operands, which are many of the entry's own steps, so these leaves
            # are bounded by each entry's own two operands.
            flow, continuation = _bellman_operands(
                values=planned_values, period=period, regime_name=regime_name
            )
            _assert_within_operand_rounding_bound(
                got=got,
                expected=expected,
                flow=flow,
                continuation=continuation,
                n_ulp=8,
                err_msg=err_msg,
            )


def _continuation_regime(*, period: int) -> str:
    """Return the regime a period-`period` value continues into.

    Mirrors the model's edges, under which age equals the period: `work` moves
    to `retire` at age `n_periods // 2 - 1` and `retire` to `dead` at age
    `n_periods - 2`.
    """
    if period >= _N_PERIODS - 2:
        return "dead"
    if period >= _N_PERIODS // 2 - 1:
        return "retire"
    return "work"


def _bellman_operands(
    *, values: Any, period: int, regime_name: str
) -> tuple[np.ndarray, np.ndarray]:
    """Return the two terms whose sum is each value of one leaf.

    The terminal `dead` regime's value is its flow utility alone. Every other
    leaf is `V = u(c*) + E[V_next(next period)]` with discount factor 1:

    - flow utility `u = log(c*) * (1 - 0.3 * (1 - health))`;
    - continuation: the next-period value of the regime the law moves to, at
      wealth `wealth - c*`, linear in wealth (extrapolated below the grid),
      averaged over the Gauss-Hermite income nodes and the two equally likely
      health states; zero when that regime is `dead`.

    The optimal consumption `c*` is found by enumerating the feasible
    consumption grid. Arrays are indexed `(income, health, wealth)` like the
    value leaf. The reconstruction must reproduce the solved leaf, or the
    operands are not the ones the solver summed. Where several consumption
    choices attain the solved value within the reconstruction tolerance, each
    operand takes its largest magnitude among them, since the solver summed
    one of them.
    """
    leaf = np.asarray(values[period][regime_name])
    if regime_name == "dead":
        return leaf, np.zeros(leaf.shape)
    target = _continuation_regime(period=period)
    following = (
        np.zeros((5, 2, 5))
        if target == "dead"
        else np.asarray(values[period + 1][target], dtype=np.float64)
    )
    wealth = np.linspace(1.0, 5.0, 5)
    consumption = np.linspace(0.1, 2.0, 4)
    nodes, weights = np.polynomial.hermite_e.hermegauss(5)
    weights = weights / weights.sum()
    tolerance = 1e3 * np.finfo(leaf.dtype).eps
    flow = np.empty(leaf.shape)
    continuation = np.empty(leaf.shape)
    for (income, health, wealth_index), solved in np.ndenumerate(leaf):
        candidates = []
        for choice in consumption:
            if wealth[wealth_index] - choice + np.exp(nodes[income]) < 0:
                continue
            utility = np.log(choice) * (1.0 - 0.3 * (1.0 - health))
            expected_next = sum(
                0.5
                * weights[node]
                * _interpolate(
                    x=wealth[wealth_index] - choice,
                    grid=wealth,
                    values=following[node, next_health],
                )
                for node in range(len(nodes))
                for next_health in (0, 1)
            )
            candidates.append((utility + expected_next, utility, expected_next))
        best = max(total for total, _, _ in candidates)
        assert abs(best - solved) <= tolerance
        attaining = [c for c in candidates if best - c[0] <= tolerance]
        flow[income, health, wealth_index] = max(abs(c[1]) for c in attaining)
        continuation[income, health, wealth_index] = max(abs(c[2]) for c in attaining)
    return flow, continuation


def _interpolate(*, x: float, grid: np.ndarray, values: np.ndarray) -> float:
    """Interpolate linearly on `grid`, extending the end segments beyond it."""
    segment = int(np.clip(np.searchsorted(grid, x) - 1, 0, len(grid) - 2))
    weight = (x - grid[segment]) / (grid[segment + 1] - grid[segment])
    return float(values[segment] * (1 - weight) + values[segment + 1] * weight)


def _assert_within_operand_rounding_bound(
    *,
    got: Any,
    expected: Any,
    flow: np.ndarray,
    continuation: np.ndarray,
    n_ulp: int,
    err_msg: str,
) -> None:
    """Hold each element to its own steps or to the rounding of its own operands.

    An element passes if it moved at most `n_ulp` of its own representable steps,
    or if it is born by cancellation and `|got - expected|` is at most the operand
    rounding bound `n_ulp * (spacing(|flow|) + spacing(|continuation|))`, with
    spacings in the leaf's format. Each element's bound uses only that element's
    two operands; a reordered sum moves a value born by their cancellation by
    roundings of the operands, not of the value. An element is born by
    cancellation when it is smaller in magnitude than its larger operand, which
    happens exactly when the two operands have opposite signs; any other element
    is held to its own steps alone.
    """
    actual = np.asarray(got)
    reference = np.asarray(expected)
    dtype = reference.dtype
    bound = n_ulp * (
        np.spacing(np.abs(flow).astype(dtype)).astype(np.float64)
        + np.spacing(np.abs(continuation).astype(dtype)).astype(np.float64)
    )
    distance = np.abs(actual.astype(np.float64) - reference.astype(np.float64))
    cancels = np.abs(reference.astype(np.float64)) < np.maximum(
        np.abs(flow), np.abs(continuation)
    )
    by_operands = cancels & (distance <= bound)
    assert_agrees_to_ulp(
        got=actual[~by_operands],
        expected=reference[~by_operands],
        n_ulp=n_ulp,
        err_msg=err_msg,
    )


def test_a_mapping_and_an_integer_may_share_one_declaration() -> None:
    """A regime the mapping does not name keeps the broadcast width."""
    observed, _ = _solve_and_collect_widths(
        config=ExecutionConfig(
            axis_widths={_CELL_AXIS: 4, "action_product": {"retire": 1}}
        ),
    )

    assert _cell_widths_by_regime(observed) == {"work": {4}, "retire": {4}}


def test_per_regime_widths_are_read_only_after_construction() -> None:
    """The caller's nested dict cannot be mutated into the stored configuration."""
    widths: dict[str, Any] = {_CELL_AXIS: {"retire": 2}}
    config = ExecutionConfig(axis_widths=widths)

    widths[_CELL_AXIS]["retire"] = 8

    assert config.axis_widths[_CELL_AXIS] == MappingProxyType({"retire": 2})
    with pytest.raises(TypeError):
        config.axis_widths[_CELL_AXIS]["retire"] = 8  # ty: ignore[invalid-assignment]


def test_two_configurations_with_equal_per_regime_widths_are_equal() -> None:
    """Configuration identity survives the nested form."""
    first = ExecutionConfig(axis_widths={_CELL_AXIS: {"retire": 2, "work": 4}})
    second = ExecutionConfig(axis_widths={_CELL_AXIS: {"work": 4, "retire": 2}})

    assert first == second


def test_per_regime_widths_reject_a_non_positive_width() -> None:
    """A width of zero is refused at construction, naming axis and regime."""
    with pytest.raises(
        ValueError, match=r"axis_widths\['cell'\]\['retire'\] must be positive"
    ):
        ExecutionConfig(axis_widths={_CELL_AXIS: {"retire": 0}})


def test_per_regime_widths_reject_a_bool_width() -> None:
    """Widths are exact ints, so a bool is refused."""
    with pytest.raises(TypeError, match=r"axis_widths\['cell'\]\['retire'\]"):
        ExecutionConfig(axis_widths={_CELL_AXIS: {"retire": True}})


def test_per_regime_widths_reject_an_empty_mapping() -> None:
    """An empty mapping names no regime, so it cannot be what the caller meant."""
    with pytest.raises(ValueError, match=r"axis_widths\['cell'\] names no regime"):
        ExecutionConfig(axis_widths={_CELL_AXIS: {}})


def test_per_regime_widths_reject_an_empty_regime_name() -> None:
    """Regime keys are non-empty strings."""
    with pytest.raises(TypeError, match=r"axis_widths\['cell'\] keys"):
        ExecutionConfig(axis_widths={_CELL_AXIS: {"": 2}})


def test_model_rejects_a_per_regime_width_for_an_unknown_regime() -> None:
    """A regime no model declares is refused at model build, listing the known ones."""
    with pytest.raises(
        ExecutionPlanningError, match=r"axis_widths\['cell'\] names regime 'nope'"
    ):
        _model_with(ExecutionConfig(axis_widths={_CELL_AXIS: {"nope": 2}}))


def test_model_rejects_a_per_regime_width_for_an_unknown_axis() -> None:
    """A per-regime width for an axis no core program declares is refused."""
    with pytest.raises(ExecutionPlanningError, match="axis_widths names 'nope'"):
        _model_with(ExecutionConfig(axis_widths={"nope": {"retire": 2}}))


def test_model_rejects_a_per_regime_width_for_a_simulation_only_axis() -> None:
    """Simulation plans without a regime in hand, so it takes the broadcast form."""
    with pytest.raises(ExecutionPlanningError, match="subject"):
        _model_with(ExecutionConfig(axis_widths={"subject": {"retire": 2}}))
