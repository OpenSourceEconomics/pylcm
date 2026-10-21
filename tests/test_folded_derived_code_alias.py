"""A derived code read keeps the folded process that produces it.

Reading `kind_code` is a read of `kind`, and `kind(z)` reads the current value of
the process `z`. Every dependency walk that runs before user functions are
adapted to their code arguments has to follow that chain, including the walk
that decides which folded processes are draw-only. A folded process read within
the period keeps its state: folding averages the stored value over the shock's
nodes, it does not take the shock out of the period.

Literal reference: the two-node `UniformIIDProcess(start=-1, stop=1)` has nodes
`(-1, 1)` with weight `1/2` each, so `kind` has codes `(0, 1)`, the utility
`1 + 2 * kind_code` is `(1, 3)`, and the folded value is `2`. The terminal
regime contributes zero. An unrelated model-level state nobody reads is pruned.
"""

from typing import Literal

import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest

from lcm import (
    AgeGrid,
    DiscreteGrid,
    Model,
    Phased,
    Regime,
    UniformIIDProcess,
    categorical,
    fixed_transition,
)
from lcm.typing import ContinuousState, DiscreteState, FloatND, ScalarInt

type _Placement = Literal["model", "regime"]
type _CodePhase = Literal["both", "solve", "simulate"]


@categorical(ordered=False)
class _RegimeId:
    live: ScalarInt
    dead: ScalarInt


@categorical(ordered=True)
class _Kind:
    low: ScalarInt
    high: ScalarInt


def _kind_from_z(*, z: ContinuousState) -> DiscreteState:
    return jnp.where(z > 0, _Kind.high, _Kind.low)


def _valued_kind_from_z(*, z: ContinuousState) -> DiscreteState:
    return jnp.where(z > 0, 20, 10)


def _utility_of_code(*, kind_code: DiscreteState) -> FloatND:
    return 1.0 + 2.0 * kind_code


def _constant_utility() -> FloatND:
    return jnp.asarray(1.0)


def _zero() -> FloatND:
    return jnp.asarray(0.0)


_CASES = pytest.mark.parametrize(
    ("placement", "process_placement", "values", "fold", "code_phase", "enable_jit"),
    [
        pytest.param(
            placement,
            process_placement,
            values,
            fold,
            code_phase,
            enable_jit,
            id=(
                f"{'jit' if enable_jit else 'eager'}-{code_phase}-"
                f"{'folded' if fold else 'unfolded'}-"
                f"{'value-grid' if values else 'category'}-"
                f"{process_placement}-{placement}"
            ),
        )
        for placement in ("model", "regime")
        for process_placement in ("model", "regime")
        for values in (False, True)
        for fold in (True, False)
        for code_phase in ("both", "solve", "simulate")
        for enable_jit in (False, True)
    ],
)


def _make_model(
    *,
    placement: _Placement,
    process_placement: _Placement,
    values: bool,
    fold: bool,
    code_phase: _CodePhase,
    enable_jit: bool,
) -> Model:
    declaration = {
        "kind": (
            DiscreteGrid(values=np.array([10, 20], dtype=np.int32))
            if values
            else DiscreteGrid(category_class=_Kind)
        )
    }
    utility = (
        _utility_of_code
        if code_phase == "both"
        else Phased(
            solve=_utility_of_code if code_phase == "solve" else _constant_utility,
            simulate=_utility_of_code
            if code_phase == "simulate"
            else _constant_utility,
        )
    )
    process = UniformIIDProcess(start=-1.0, stop=1.0, n_points=2, fold=fold)
    return Model(
        regimes={
            "live": Regime(
                states={"z": process} if process_placement == "regime" else {},
                functions={
                    "utility": utility,
                    "kind": _valued_kind_from_z if values else _kind_from_z,
                },
                derived_categoricals=declaration if placement == "regime" else {},
                state_transitions={"unread": fixed_transition("unread")},
            ),
            "dead": Regime(functions={"utility": _zero}),
        },
        states={
            **({"z": process} if process_placement == "model" else {}),
            "unread": DiscreteGrid(category_class=_Kind),
        },
        derived_categoricals=declaration if placement == "model" else {},
        ages=AgeGrid(start=0, inclusive_stop=1, step="Y"),
        initial_nodes={0: "live"},
        edges={"live": {"dead": 0}},
        regime_id_class=_RegimeId,
        enable_jit=enable_jit,
    )


def _simulated_flow(model: Model) -> np.ndarray:
    params = {"discount_factor": 1.0}
    panel: pd.DataFrame = model.simulate(
        params=params,
        solution=model.solve(params=params, log_level="off"),
        initial_conditions={
            "age": jnp.zeros(2),
            "regime_id": jnp.zeros(2, dtype=jnp.int32),
            "z": jnp.asarray([-0.5, 0.5]),
        },
        seed=0,
        log_level="off",
    ).to_dataframe(additional_targets=["utility"], use_categorical_codes=True)
    return panel.loc[panel["period"] == 0, "utility"].to_numpy()


@_CASES
def test_derived_code_keeps_its_process_state(
    *,
    placement: _Placement,
    process_placement: _Placement,
    values: bool,
    fold: bool,
    code_phase: _CodePhase,
    enable_jit: bool,
) -> None:
    """The process feeding a derived code read stays a state of the regime."""
    model = _make_model(
        placement=placement,
        process_placement=process_placement,
        values=values,
        fold=fold,
        code_phase=code_phase,
        enable_jit=enable_jit,
    )
    assert "z" in model.user_regimes["live"].states


@_CASES
def test_derived_code_keeps_its_process_initial_state(
    *,
    placement: _Placement,
    process_placement: _Placement,
    values: bool,
    fold: bool,
    code_phase: _CodePhase,
    enable_jit: bool,
) -> None:
    """The process feeding a derived code read is an initial condition."""
    model = _make_model(
        placement=placement,
        process_placement=process_placement,
        values=values,
        fold=fold,
        code_phase=code_phase,
        enable_jit=enable_jit,
    )
    assert "z" in model.initial_state_names(regime_name="live")


@_CASES
def test_unread_model_state_is_still_pruned(
    *,
    placement: _Placement,
    process_placement: _Placement,
    values: bool,
    fold: bool,
    code_phase: _CodePhase,
    enable_jit: bool,
) -> None:
    """A model-level state nobody reads leaves the regime."""
    model = _make_model(
        placement=placement,
        process_placement=process_placement,
        values=values,
        fold=fold,
        code_phase=code_phase,
        enable_jit=enable_jit,
    )
    assert "unread" not in model.user_regimes["live"].states


@_CASES
def test_derived_code_value_is_the_literal_fold(
    *,
    placement: _Placement,
    process_placement: _Placement,
    values: bool,
    fold: bool,
    code_phase: _CodePhase,
    enable_jit: bool,
) -> None:
    """The solved value is `2` folded, `[1, 3]` unfolded, `1` if solve ignores it."""
    model = _make_model(
        placement=placement,
        process_placement=process_placement,
        values=values,
        fold=fold,
        code_phase=code_phase,
        enable_jit=enable_jit,
    )
    solution = model.solve(params={"discount_factor": 1.0}, log_level="off")
    if code_phase == "simulate":
        expected = 1.0 if fold else [1.0, 1.0]
    else:
        expected = 2.0 if fold else [1.0, 3.0]
    np.testing.assert_array_equal(np.asarray(solution.values[0]["live"]), expected)


@_CASES
def test_derived_code_flow_is_the_literal_utility(
    *,
    placement: _Placement,
    process_placement: _Placement,
    values: bool,
    fold: bool,
    code_phase: _CodePhase,
    enable_jit: bool,
) -> None:
    """Simulated flow at `z = (-0.5, 0.5)` is `[1, 3]`, or `[1, 1]` unread."""
    model = _make_model(
        placement=placement,
        process_placement=process_placement,
        values=values,
        fold=fold,
        code_phase=code_phase,
        enable_jit=enable_jit,
    )
    expected = [1.0, 1.0] if code_phase == "solve" else [1.0, 3.0]
    np.testing.assert_array_equal(_simulated_flow(model), expected)
