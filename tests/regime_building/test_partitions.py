"""Partition-dimension behavior on non-trivial models.

Fixed states (`state_transitions[name] = None` on a `DiscreteGrid`) are lifted
out of each regime's state-action space and iterated externally. The tests
below check that the user-visible behavior is exactly what a monolithic
model with those states vmap'd in-place would produce, and that the
partition axes surface at the expected position in returned V-arrays.
"""

import ast
import inspect
import textwrap
from types import MappingProxyType

import jax.numpy as jnp
import numpy as np
import pytest

from lcm import (
    AgeGrid,
    DiscreteGrid,
    DispatchStrategy,
    LinSpacedGrid,
    Model,
    Regime,
    categorical,
)
from lcm.exceptions import ModelInitializationError
from lcm.regime_building.partitions import (
    detect_model_partitions,
    iterate_partition_points,
    lift_partitions_from_regime,
    stack_partition_scalars,
)
from lcm.solution import solve_brute
from lcm.typing import (
    BoolND,
    ContinuousAction,
    ContinuousState,
    DiscreteState,
    FloatND,
    InternalParams,
    ScalarInt,
)


@categorical(ordered=False)
class TypeGrid:
    type_a: int
    type_b: int
    type_c: int


@categorical(ordered=False)
class _TwoCat:
    low: int
    high: int


@categorical(ordered=False)
class _RegimeId:
    alive: int
    dead: int


def _next_regime(age: float, final_age_alive: float) -> ScalarInt:
    return jnp.where(age >= final_age_alive, _RegimeId.dead, _RegimeId.alive)


def _utility(consumption: ContinuousAction, pref_type: DiscreteState) -> FloatND:
    """Scale log-consumption by `pref_type`'s integer code.

    `pref_type` is used directly by utility (no H-DAG machinery needed on
    this branch). `type_c` gets 3x, `type_a` gets 1x.
    """
    scale = 1.0 + pref_type.astype(jnp.float32)
    return jnp.log(consumption) * scale


def _next_wealth(
    wealth: ContinuousState, consumption: ContinuousAction
) -> ContinuousState:
    return wealth - consumption + 1.0


def _borrowing(consumption: ContinuousAction, wealth: ContinuousState) -> BoolND:
    return consumption <= wealth


_FINAL_AGE = 2


def _make_model() -> Model:
    alive = Regime(
        actions={
            "consumption": LinSpacedGrid(start=0.1, stop=5.0, n_points=20),
        },
        states={
            "wealth": LinSpacedGrid(start=0.1, stop=5.0, n_points=10),
            "pref_type": DiscreteGrid(
                TypeGrid, dispatch=DispatchStrategy.PARTITION_SCAN
            ),
        },
        state_transitions={
            "wealth": _next_wealth,
            "pref_type": None,  # partition
        },
        constraints={"borrowing": _borrowing},
        functions={"utility": _utility},
        transition=_next_regime,
        active=lambda age: age <= _FINAL_AGE,
    )

    def dead_utility() -> float:
        return 0.0

    dead = Regime(
        transition=None,
        functions={"utility": dead_utility},
        active=lambda age: age > _FINAL_AGE,
    )

    return Model(
        regimes={"alive": alive, "dead": dead},
        ages=AgeGrid(start=0, stop=_FINAL_AGE + 1, step="Y"),
        regime_id_class=_RegimeId,
    )


def test_detect_partitions_identifies_opt_in_dispatch():
    """Picks up a DiscreteGrid that opted into partition-lifted dispatch."""
    model = _make_model()
    detected = detect_model_partitions(regimes=model.regimes)
    assert set(detected) == {"pref_type"}
    assert detected["pref_type"].categories == ("type_a", "type_b", "type_c")


def test_detect_partitions_skips_default_dispatch():
    """A DiscreteGrid with default dispatch (FUSED_VMAP) is not a partition.

    Even with `state_transitions[name] = None`, a discrete state without
    an explicit partition-lifted dispatch stays in the state-action space
    with an identity transition.
    """
    regime = Regime(
        actions={},
        states={"x": DiscreteGrid(TypeGrid)},
        state_transitions={"x": None},
        functions={"utility": lambda x: x},
        transition=lambda: 0,
        active=lambda age: age < 1,
    )
    assert detect_model_partitions(regimes={"r": regime}) == {}


def test_detect_partitions_rejects_non_none_transition():
    """A partition-lifted state with a non-identity transition is an error."""
    regime = Regime(
        actions={},
        states={"x": DiscreteGrid(TypeGrid, dispatch=DispatchStrategy.PARTITION_SCAN)},
        state_transitions={"x": lambda x: x},
        functions={"utility": lambda x: x},
        transition=lambda: 0,
        active=lambda age: age < 1,
    )
    with pytest.raises(ModelInitializationError, match="partition-lifted"):
        detect_model_partitions(regimes={"r": regime})


def test_detect_partitions_skips_continuous():
    """Continuous grids have no `dispatch` kwarg, so they are never partition-lifted."""
    regime = Regime(
        actions={"consumption": LinSpacedGrid(start=0.1, stop=1.0, n_points=3)},
        states={"wealth": LinSpacedGrid(start=0.1, stop=1.0, n_points=3)},
        state_transitions={"wealth": None},
        functions={"utility": lambda consumption, wealth: wealth + consumption},
        transition=lambda: 0,
        active=lambda age: age < 1,
    )
    assert detect_model_partitions(regimes={"r": regime}) == {}


def test_lift_removes_partition_from_states_and_transitions():
    """The reduced regime no longer contains the partition state."""
    model = _make_model()
    alive = model.regimes["alive"]
    reduced, partitions = lift_partitions_from_regime(
        regime=alive, partition_names=frozenset({"pref_type"})
    )
    assert "pref_type" not in reduced.states
    assert "pref_type" not in reduced.state_transitions
    assert set(partitions) == {"pref_type"}


def test_internal_regime_exposes_partitions():
    """`InternalRegime.partitions` carries the lifted grid; `grids` does not."""
    model = _make_model()
    alive = model.internal_regimes["alive"]
    assert "pref_type" in alive.partitions
    assert "pref_type" not in alive.grids


def test_model_partition_grid_union():
    """`Model._partition_grid` aggregates partitions across all regimes."""
    model = _make_model()
    assert set(model._partition_grid) == {"pref_type"}


def test_iterate_partition_points_cardinality():
    """The product iterator yields one dict per category code."""
    model = _make_model()
    points = list(iterate_partition_points(partition_grid=model._partition_grid))
    assert len(points) == 3  # type_a, type_b, type_c
    assert {int(p["pref_type"]) for p in points} == {0, 1, 2}


def test_stack_partition_scalars_builds_leading_axis():
    """Declared partition names become 1-D arrays of length `prod(partition_shape)`.

    Undeclared names and non-partition scalars stay untouched. Regimes that
    do not declare the partition (here: `dead`) receive no stacked entry.
    """
    model = _make_model()
    internal_params = model._process_params(
        {
            "discount_factor": 0.9,
            "alive": {"next_regime": {"final_age_alive": _FINAL_AGE}},
        }
    )
    stacked, partition_shape = stack_partition_scalars(
        internal_params=internal_params,
        partition_grid=model._partition_grid,
        regime_partitions=model._regime_partitions,
    )
    assert partition_shape == (3,)
    # Alive declares pref_type → stacked to a 1-D array with every code.
    alive_codes = stacked["alive"]["pref_type"]
    assert jnp.asarray(alive_codes).shape == (3,)
    assert {int(c) for c in jnp.asarray(alive_codes)} == {0, 1, 2}
    # Non-partition entries are carried through unchanged.
    assert stacked["alive"].keys() == internal_params["alive"].keys() | {"pref_type"}
    # Dead does not declare pref_type → no stacked entry appears there.
    assert "pref_type" not in stacked["dead"]


def test_stack_partition_scalars_empty_is_identity():
    """Empty `partition_grid` returns the input mapping and shape `()`."""
    internal_params: InternalParams = MappingProxyType(
        {"regime_a": MappingProxyType({"beta": 0.95})}
    )
    stacked, partition_shape = stack_partition_scalars(
        internal_params=internal_params,
        partition_grid=MappingProxyType({}),
        regime_partitions=MappingProxyType({"regime_a": MappingProxyType({})}),
    )
    assert partition_shape == ()
    assert stacked is internal_params


def test_solve_handles_multi_dimensional_partition_shape():
    """Multi-axis partition_shape — e.g. `(3, 2)` — solves without errors.

    Regression guard for a bug in `run_compiled_solve`'s deferred-
    diagnostic reductions: it used to treat `len(partition_shape)` as
    the number of leading axes on `V_arr`, but `V_arr` always has a
    single flattened leading axis of size `prod(partition_shape)`.
    Multi-dim partitions broke `jnp.stack(reductions)` with
    "All input arrays must have the same shape" as soon as one regime
    had a different inner state shape than another.
    """

    def _extra_next_regime(age: float, final_age_alive: float) -> ScalarInt:
        return jnp.where(age >= final_age_alive, _RegimeId.dead, _RegimeId.alive)

    def _extra_utility(
        consumption: ContinuousAction,
        pref_type: DiscreteState,
        extra: DiscreteState,
    ) -> FloatND:
        scale = 1.0 + pref_type.astype(jnp.float32) + extra.astype(jnp.float32)
        return jnp.log(consumption) * scale

    alive = Regime(
        actions={"consumption": LinSpacedGrid(start=0.1, stop=5.0, n_points=20)},
        states={
            "wealth": LinSpacedGrid(start=0.1, stop=5.0, n_points=10),
            "pref_type": DiscreteGrid(
                TypeGrid, dispatch=DispatchStrategy.PARTITION_SCAN
            ),
            "extra": DiscreteGrid(_TwoCat, dispatch=DispatchStrategy.PARTITION_SCAN),
        },
        state_transitions={
            "wealth": _next_wealth,
            "pref_type": None,
            "extra": None,
        },
        constraints={"borrowing": _borrowing},
        functions={"utility": _extra_utility},
        transition=_extra_next_regime,
        active=lambda age: age <= _FINAL_AGE,
    )

    def dead_utility() -> float:
        return 0.0

    dead = Regime(
        transition=None,
        functions={"utility": dead_utility},
        active=lambda age: age > _FINAL_AGE,
    )
    model = Model(
        regimes={"alive": alive, "dead": dead},
        ages=AgeGrid(start=0, stop=_FINAL_AGE + 1, step="Y"),
        regime_id_class=_RegimeId,
    )
    assert len(model._partition_grid) == 2  # pref_type (3) x extra (2)

    V = model.solve(
        params={
            "discount_factor": 0.9,
            "alive": {"next_regime": {"final_age_alive": _FINAL_AGE}},
        },
        log_level="off",
    )
    # Alive at period 0: (pref_type, extra, wealth) = (3, 2, 10).
    assert V[0]["alive"].shape == (3, 2, 10)


def test_reshape_leading_partition_axis_rejects_shape_mismatch():
    """`_reshape_leading_partition_axis` asserts the leading-axis invariant.

    Guards the row-major contract between `stack_partition_scalars` and the
    reshape. A V-array whose leading axis does not match
    `prod(partition_shape)` must raise — silently reshaping would
    produce misattributed values.
    """
    from lcm.model import _reshape_leading_partition_axis  # noqa: PLC0415

    bad_V = jnp.zeros((5, 4))  # leading axis 5 ≠ 2 * 3
    raw = MappingProxyType({0: MappingProxyType({"alive": bad_V})})
    with pytest.raises(AssertionError, match="row-major contract"):
        _reshape_leading_partition_axis(raw=raw, partition_shape=(2, 3))


def test_solve_V_has_leading_partition_axis():
    """Solved V for alive has shape (n_pref_type, ...); partition axes are leading."""
    model = _make_model()
    V = model.solve(
        params={
            "discount_factor": 0.9,
            "alive": {"next_regime": {"final_age_alive": _FINAL_AGE}},
        },
        log_level="off",
    )
    # Alive at period 0: partition axis 3, wealth grid size 10 → shape (3, 10).
    assert V[0]["alive"].shape[0] == 3
    assert V[0]["alive"].ndim == 2


def test_solve_V_monotone_in_pref_type():
    """Higher pref_type code ⇒ higher utility ⇒ higher V at the same state.

    The partition axis is leading, so `V[0]`, `V[1]`, `V[2]` index into the
    three pref-type sub-solves. `_utility` scales with the code, so
    `V[2] > V[1] > V[0]` at non-terminal periods.
    """
    model = _make_model()
    V = model.solve(
        params={
            "discount_factor": 0.9,
            "alive": {"next_regime": {"final_age_alive": _FINAL_AGE}},
        },
        log_level="off",
    )
    non_terminal_periods = [p for p in V if p < max(V.keys())]
    for period in non_terminal_periods:
        v = V[period]["alive"]
        v_a = float(jnp.mean(v[0]))
        v_b = float(jnp.mean(v[1]))
        v_c = float(jnp.mean(v[2]))
        assert v_a < v_b < v_c, (
            f"V non-monotone in pref_type at period {period}: "
            f"{v_a:.4f}, {v_b:.4f}, {v_c:.4f}"
        )


def test_simulate_routes_subjects_to_correct_partition():
    """Subjects with different pref_types get dispatched per sub-solution.

    All subjects start at the same wealth; the partition dispatch groups
    them by `pref_type`. The realised DataFrame must preserve each
    subject's pref_type label across all periods (it is a fixed state).
    """
    model = _make_model()
    n_per_type = 4
    n_subjects = 3 * n_per_type
    initial_conditions = {
        "wealth": jnp.full(n_subjects, 2.0),
        "pref_type": jnp.array(
            [TypeGrid.type_a] * n_per_type
            + [TypeGrid.type_b] * n_per_type
            + [TypeGrid.type_c] * n_per_type
        ),
        "age": jnp.zeros(n_subjects),
        "regime": jnp.full(n_subjects, _RegimeId.alive),
    }
    result = model.simulate(
        params={
            "discount_factor": 0.9,
            "alive": {"next_regime": {"final_age_alive": _FINAL_AGE}},
        },
        initial_conditions=initial_conditions,
        period_to_regime_to_V_arr=None,
        log_level="off",
        seed=0,
    )
    df = result.to_dataframe(use_labels=False)
    # pref_type is fixed: every subject's pref_type should be constant across periods.
    per_subject = df.groupby("subject_id")["pref_type"].nunique()
    assert (per_subject == 1).all()
    # All three types must appear in the output.
    assert set(df["pref_type"].dropna().unique()) == {0.0, 1.0, 2.0}


def test_simulate_subject_ids_globally_unique():
    """After per-partition dispatch, concatenated subject_ids cover [0, n)."""
    model = _make_model()
    n_subjects = 7
    initial_conditions = {
        "wealth": jnp.full(n_subjects, 2.0),
        "pref_type": jnp.array(
            [TypeGrid.type_a, TypeGrid.type_b, TypeGrid.type_c] * 2 + [TypeGrid.type_a]
        ),
        "age": jnp.zeros(n_subjects),
        "regime": jnp.full(n_subjects, _RegimeId.alive),
    }
    result = model.simulate(
        params={
            "discount_factor": 0.9,
            "alive": {"next_regime": {"final_age_alive": _FINAL_AGE}},
        },
        initial_conditions=initial_conditions,
        period_to_regime_to_V_arr=None,
        log_level="off",
    )
    df = result.to_dataframe(use_labels=False)
    # Each period's subject_ids must be exactly [0, 1, ..., n_subjects-1] with no gaps.
    for period, sub in df.groupby("period"):
        assert set(sub["subject_id"].astype(int)) == set(range(n_subjects)), (
            f"Period {period} missing subject_ids"
        )


@pytest.mark.parametrize("reorder_subjects", [False, True])
def test_simulate_invariant_to_subject_ordering(*, reorder_subjects: bool):
    """Simulation results depend only on the (pref_type, wealth) pair per subject.

    Reordering the subjects in `initial_conditions` must yield the same
    per-(subject_id) rows in the output after sorting — the partition
    dispatch should not introduce order-dependent drift.
    """
    model = _make_model()
    n_subjects = 6
    wealth = jnp.array([1.0, 2.0, 3.0, 1.5, 2.5, 3.5])
    types = jnp.array([TypeGrid.type_a, TypeGrid.type_b, TypeGrid.type_c] * 2)

    if reorder_subjects:
        perm = jnp.array([3, 0, 4, 1, 5, 2])
        wealth = wealth[perm]
        types = types[perm]

    initial_conditions = {
        "wealth": wealth,
        "pref_type": types,
        "age": jnp.zeros(n_subjects),
        "regime": jnp.full(n_subjects, _RegimeId.alive),
    }
    result = model.simulate(
        params={
            "discount_factor": 0.9,
            "alive": {"next_regime": {"final_age_alive": _FINAL_AGE}},
        },
        initial_conditions=initial_conditions,
        period_to_regime_to_V_arr=None,
        log_level="off",
        seed=42,
    )
    df = result.to_dataframe(use_labels=False)
    # After sorting by subject_id, each (period, subject_id) row corresponds to
    # the subject at that index in initial_conditions.
    alive_period_0 = df[(df["regime"] == "alive") & (df["period"] == 0)].sort_values(
        "subject_id"
    )
    # The 0-th subject's wealth must match initial_conditions["wealth"][0].
    np.testing.assert_allclose(
        alive_period_0["wealth"].to_numpy(), np.asarray(wealth), atol=1e-10
    )


def _borrowing_uses_pref_type(
    consumption: ContinuousAction,
    wealth: ContinuousState,
    pref_type: DiscreteState,
) -> BoolND:
    """Feasibility function that references a partition state."""
    return (consumption <= wealth) & (pref_type >= 0)


def test_simulate_feasibility_validation_sees_partition_states():
    """Feasibility check during `validate_initial_conditions` must see partition values.

    Regression guard: the Mahler-Yum GPU regression test failed with
    `InvalidFunctionArgumentsError: missing required arguments:
    education, productivity` because partition states were lifted out
    of `variable_info` but the feasibility validator still drew its
    per-subject state set from there. Now `internal_regime.partitions`
    is walked alongside `variable_info`.
    """
    alive = Regime(
        actions={
            "consumption": LinSpacedGrid(start=0.1, stop=5.0, n_points=20),
        },
        states={
            "wealth": LinSpacedGrid(start=0.1, stop=5.0, n_points=10),
            "pref_type": DiscreteGrid(TypeGrid),
        },
        state_transitions={
            "wealth": _next_wealth,
            "pref_type": None,
        },
        constraints={"borrowing": _borrowing_uses_pref_type},
        functions={"utility": _utility},
        transition=_next_regime,
        active=lambda age: age <= _FINAL_AGE,
    )

    def dead_utility() -> float:
        return 0.0

    dead = Regime(
        transition=None,
        functions={"utility": dead_utility},
        active=lambda age: age > _FINAL_AGE,
    )

    model = Model(
        regimes={"alive": alive, "dead": dead},
        ages=AgeGrid(start=0, stop=_FINAL_AGE + 1, step="Y"),
        regime_id_class=_RegimeId,
    )

    n_subjects = 3
    result = model.simulate(
        params={
            "discount_factor": 0.9,
            "alive": {"next_regime": {"final_age_alive": _FINAL_AGE}},
        },
        initial_conditions={
            "wealth": jnp.full(n_subjects, 2.0),
            "pref_type": jnp.array([TypeGrid.type_a, TypeGrid.type_b, TypeGrid.type_c]),
            "age": jnp.zeros(n_subjects),
            "regime": jnp.full(n_subjects, _RegimeId.alive),
        },
        period_to_regime_to_V_arr=None,
        log_level="off",
    )
    df = result.to_dataframe(use_labels=False)
    # Each subject kept their pref_type across all periods, confirming the
    # feasibility check actually saw the partition value (and so didn't just
    # skip validation silently).
    per_subject_pref_types = df.groupby("subject_id")["pref_type"].nunique()
    assert (per_subject_pref_types == 1).all()
    assert set(df["pref_type"].dropna().unique()) == {0.0, 1.0, 2.0}


def test_invalid_partition_code_raises():
    """A user-supplied partition code outside the grid must fail validation.

    Regression guard: `_validate_discrete_state_values` used to skip
    partition states entirely because they were absent from
    `variable_info`. A bad code (e.g. 99 for a 3-category partition)
    would silently propagate to sub-solution dispatch.
    """
    model = _make_model()
    n_subjects = 3
    with pytest.raises(Exception, match=r"(?i)invalid.*pref_type"):
        model.simulate(
            params={
                "discount_factor": 0.9,
                "alive": {"next_regime": {"final_age_alive": _FINAL_AGE}},
            },
            initial_conditions={
                "wealth": jnp.full(n_subjects, 2.0),
                "pref_type": jnp.array([0, 1, 99]),
                "age": jnp.zeros(n_subjects),
                "regime": jnp.full(n_subjects, _RegimeId.alive),
            },
            period_to_regime_to_V_arr=None,
            log_level="off",
        )


def test_solve_compiles_once_for_multi_point_partition(monkeypatch):
    """`Model.solve` compiles the solve kernel once regardless of partition cardinality.

    Regression guard for the #326 performance issue: calling
    `_compile_all_functions` inside the partition loop made every point
    pay a full AOT pass, turning Mahler-Yum's 3.8s main baseline into
    42s on #326. `Model.solve` now calls `compile_solve` outside the
    loop and `run_compiled_solve` inside it, so only one compile
    happens per solve call even with a 3-point (or larger) partition.
    """
    original = solve_brute._compile_all_functions
    call_count = 0

    def counting_compile(**kwargs):
        nonlocal call_count
        call_count += 1
        return original(**kwargs)

    monkeypatch.setattr(solve_brute, "_compile_all_functions", counting_compile)

    model = _make_model()
    assert len(model._partition_grid["pref_type"].codes) >= 2
    model.solve(
        params={
            "discount_factor": 0.9,
            "alive": {"next_regime": {"final_age_alive": _FINAL_AGE}},
        },
        log_level="off",
    )
    assert call_count == 1, (
        f"_compile_all_functions called {call_count} times for a "
        f"{len(model._partition_grid['pref_type'].codes)}-point partition; "
        "expected 1."
    )


def test_compile_simulate_hoisted_above_partition_loop():
    """`Model.simulate` calls `compile_simulate` exactly once, outside any loop.

    Structural guard against re-introducing per-partition-point compile
    overhead. Asserted on the AST of `Model.simulate`'s source so the check
    does not depend on any mocking and is immune to import-order quirks.
    """
    source = textwrap.dedent(inspect.getsource(Model.simulate))
    tree = ast.parse(source)

    class _Visitor(ast.NodeVisitor):
        def __init__(self) -> None:
            self.outside_loop = 0
            self.inside_loop = 0
            self._loop_depth = 0

        def visit_For(self, node: ast.For) -> None:
            self._loop_depth += 1
            self.generic_visit(node)
            self._loop_depth -= 1

        def visit_While(self, node: ast.While) -> None:
            self._loop_depth += 1
            self.generic_visit(node)
            self._loop_depth -= 1

        def visit_Call(self, node: ast.Call) -> None:
            if isinstance(node.func, ast.Name) and node.func.id == "compile_simulate":
                if self._loop_depth:
                    self.inside_loop += 1
                else:
                    self.outside_loop += 1
            self.generic_visit(node)

    visitor = _Visitor()
    visitor.visit(tree)
    assert visitor.outside_loop == 1, (
        f"expected exactly 1 `compile_simulate(...)` call outside the "
        f"partition loop, saw {visitor.outside_loop}"
    )
    assert visitor.inside_loop == 0, (
        f"`compile_simulate(...)` must not be called inside the partition "
        f"loop, saw {visitor.inside_loop} nested call(s)"
    )
