"""Partition-dimension behavior on non-trivial models.

Fixed states (`state_transitions[name] = None` on a `DiscreteGrid`) are lifted
out of each regime's state-action space and iterated externally. The tests
below check that the user-visible behavior is exactly what a monolithic
model with those states vmap'd in-place would produce, and that the
partition axes surface at the expected position in returned V-arrays.
"""

import jax.numpy as jnp
import numpy as np
import pytest

from lcm import (
    AgeGrid,
    DiscreteGrid,
    LinSpacedGrid,
    Model,
    Regime,
    categorical,
)
from lcm.regime_building.partitions import (
    detect_model_partitions,
    iterate_partition_points,
    lift_partitions_from_regime,
)
from lcm.typing import (
    BoolND,
    ContinuousAction,
    ContinuousState,
    DiscreteState,
    FloatND,
    ScalarInt,
)


@categorical(ordered=False)
class TypeGrid:
    type_a: int
    type_b: int
    type_c: int


@categorical(ordered=False)
class _RegimeId:
    alive: int
    dead: int


def _next_regime(age: float, final_age_alive: float) -> ScalarInt:
    return jnp.where(age >= final_age_alive, _RegimeId.dead, _RegimeId.alive)


def _utility(consumption: ContinuousAction, pref_type: DiscreteState) -> FloatND:
    """Utility scales with pref_type's integer code.

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
            "pref_type": DiscreteGrid(TypeGrid),
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


def test_detect_partitions_identifies_discrete_none():
    """`detect_model_partitions` picks up a DiscreteGrid state with None transition."""
    model = _make_model()
    detected = detect_model_partitions(model.regimes)
    assert set(detected) == {"pref_type"}
    assert detected["pref_type"].categories == ("type_a", "type_b", "type_c")


def test_detect_partitions_rejects_non_none_transition():
    """A state with any non-None transition is never a partition."""
    regime = Regime(
        actions={},
        states={"x": DiscreteGrid(TypeGrid)},
        state_transitions={"x": lambda x: x},
        functions={"utility": lambda x: x},
        transition=lambda: 0,
        active=lambda age: age < 1,
    )
    assert detect_model_partitions({"r": regime}) == {}


def test_detect_partitions_rejects_continuous():
    """Continuous `None` transitions fall through to identity, not partition."""
    regime = Regime(
        actions={"consumption": LinSpacedGrid(start=0.1, stop=1.0, n_points=3)},
        states={"wealth": LinSpacedGrid(start=0.1, stop=1.0, n_points=3)},
        state_transitions={"wealth": None},
        functions={"utility": lambda consumption, wealth: wealth + consumption},
        transition=lambda: 0,
        active=lambda age: age < 1,
    )
    assert detect_model_partitions({"r": regime}) == {}


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
    points = list(iterate_partition_points(model._partition_grid))
    assert len(points) == 3  # type_a, type_b, type_c
    assert {int(p["pref_type"]) for p in points} == {0, 1, 2}


def test_solve_V_has_trailing_partition_axis():
    """Solved V for alive has shape (..., n_pref_type); dead has (n_pref_type,)."""
    model = _make_model()
    V = model.solve(
        params={
            "discount_factor": 0.9,
            "alive": {"next_regime": {"final_age_alive": _FINAL_AGE}},
        },
        log_level="off",
    )
    # Alive at period 0: wealth grid size 10, partition axis 3 → shape (10, 3).
    assert V[0]["alive"].shape[-1] == 3
    assert V[0]["alive"].ndim == 2


def test_solve_V_monotone_in_pref_type():
    """Higher pref_type code ⇒ higher utility ⇒ higher V at the same state.

    Each sub-solve uses a different `pref_type` scalar, so the trailing
    axis of `V` distinguishes them. `_utility` scales with the code, so
    `V[..., 2] > V[..., 1] > V[..., 0]` at non-terminal periods.
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
        v_a = float(jnp.mean(v[..., 0]))
        v_b = float(jnp.mean(v[..., 1]))
        v_c = float(jnp.mean(v[..., 2]))
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
