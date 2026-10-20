"""A read of `<name>_code` keeps a model-level discrete variable, as a read of `<name>`.

Model-level states and actions are kept in a regime only where something reads
them, which is decided before user functions are adapted to their code
arguments. Reading the code `x_code` is a read of `x`, so `x` has to survive
there exactly as it does when the regime declares it itself. The utility
`1 + 2 * x_code` gives the literal values `[1, 3]` over a state's two nodes and
`3` at code `1` for an action.

The code is also read through a derived categorical: `kind_code` is the code of
`kind = z`, so the state `z` feeding it is read. An unrelated model-level state
nobody reads must still be pruned.
"""

import jax.numpy as jnp
import numpy as np
import pytest
from dags.tree import flatten_to_qnames

from lcm import (
    AgeGrid,
    DeterministicTransition,
    DiscreteGrid,
    Model,
    Regime,
    Transition,
    categorical,
    fixed_transition,
)
from lcm.typing import DiscreteState, FloatND, ScalarInt


@categorical(ordered=False)
class _RegimeId:
    live: ScalarInt
    dead: ScalarInt


@categorical(ordered=True)
class _Kind:
    low: ScalarInt
    high: ScalarInt


def _utility_of_x_code(*, x_code: DiscreteState) -> FloatND:
    return 1.0 + 2.0 * x_code


def _utility_of_kind_code(*, kind_code: DiscreteState) -> FloatND:
    return 1.0 + 2.0 * kind_code


def _constant_utility() -> FloatND:
    return jnp.asarray(1.0)


def _x_code_is_one(*, x_code: DiscreteState) -> FloatND:
    return x_code == 1


def _x_code_is_a_code(*, x_code: DiscreteState) -> FloatND:
    return x_code >= 0


def _next_regime_by_x_code(*, x_code: DiscreteState) -> ScalarInt:
    return jnp.where(x_code >= 0, _RegimeId.dead, _RegimeId.dead)


def _next_regime() -> ScalarInt:
    return jnp.asarray(_RegimeId.dead)


def _kind(*, z: DiscreteState) -> DiscreteState:
    return z


def _zero() -> FloatND:
    return jnp.asarray(0.0)


def _grid(*, values: bool) -> DiscreteGrid:
    return (
        DiscreteGrid(values=np.array([10, 20]))
        if values
        else DiscreteGrid(category_class=_Kind)
    )


def _model(
    *,
    role: str,
    values: bool,
    reader: str,
    enable_jit: bool = False,
) -> Model:
    """`x` is a model-level `role`; `reader` names the only place its code is read."""
    utility = _utility_of_x_code if reader == "utility" else _constant_utility
    constraints = (
        {"x_rule": _x_code_is_one if role == "action" else _x_code_is_a_code}
        if reader == "constraint"
        else {}
    )
    law = _next_regime_by_x_code if reader == "regime_law" else _next_regime
    model_states = {"unread": DiscreteGrid(category_class=_Kind)}
    model_actions: dict[str, DiscreteGrid] = {}
    (model_states if role == "state" else model_actions)["x"] = _grid(values=values)
    transitions = {"unread": fixed_transition("unread")}
    if role == "state":
        transitions["x"] = fixed_transition("x")
    return Model(
        regimes={
            "live": Regime(
                functions={"utility": utility},
                constraints=constraints,
                state_transitions=transitions,
            ),
            "dead": Regime(functions={"utility": _zero}),
        },
        ages=AgeGrid(start=0, inclusive_stop=1, step="Y"),
        initial_nodes={0: "live"},
        edges={
            "live": Transition(
                targets={"dead": (0,)}, law=DeterministicTransition(func=law)
            )
        },
        regime_id_class=_RegimeId,
        enable_jit=enable_jit,
        states=model_states,
        actions=model_actions,
    )


def _derived_model(*, placement: str, enable_jit: bool = False) -> Model:
    """`kind = z` is a derived categorical declared in the regime or the model."""
    declaration = {"kind": DiscreteGrid(category_class=_Kind)}
    return Model(
        regimes={
            "live": Regime(
                functions={"utility": _utility_of_kind_code, "kind": _kind},
                state_transitions={"z": fixed_transition("z")},
                derived_categoricals=declaration if placement == "regime" else {},
            ),
            "dead": Regime(functions={"utility": _zero}),
        },
        ages=AgeGrid(start=0, inclusive_stop=1, step="Y"),
        initial_nodes={0: "live"},
        edges={"live": {"dead": 0}},
        regime_id_class=_RegimeId,
        enable_jit=enable_jit,
        states={"z": DiscreteGrid(category_class=_Kind)},
        derived_categoricals=declaration if placement == "model" else {},
    )


def _kept(model: Model) -> frozenset[str]:
    regime = model.user_regimes["live"]
    return frozenset(regime.states) | frozenset(regime.actions)


_ROLES_AND_GRIDS = [
    pytest.param(role, values, id=f"{role}-{'value-grid' if values else 'category'}")
    for role in ("state", "action")
    for values in (False, True)
]
_READERS = ["utility", "constraint", "regime_law"]


@pytest.mark.parametrize("reader", _READERS)
@pytest.mark.parametrize(("role", "values"), _ROLES_AND_GRIDS)
def test_a_code_read_keeps_the_model_level_variable(
    *, role: str, values: bool, reader: str
) -> None:
    model = _model(role=role, values=values, reader=reader)
    assert "x" in _kept(model)


@pytest.mark.parametrize("reader", _READERS)
@pytest.mark.parametrize(("role", "values"), _ROLES_AND_GRIDS)
def test_a_code_read_prunes_an_unread_model_level_variable(
    *, role: str, values: bool, reader: str
) -> None:
    model = _model(role=role, values=values, reader=reader)
    assert "unread" not in _kept(model)


@pytest.mark.parametrize(("role", "values"), _ROLES_AND_GRIDS)
def test_a_code_is_no_runtime_parameter(*, role: str, values: bool) -> None:
    model = _model(role=role, values=values, reader="utility")
    qnames = flatten_to_qnames(model.get_params_template())
    assert not [qname for qname in qnames if "x_code" in qname.split("__")]


@pytest.mark.parametrize("enable_jit", [False, True], ids=["eager", "jit"])
@pytest.mark.parametrize(("role", "values"), _ROLES_AND_GRIDS)
def test_a_utility_of_the_code_takes_its_literal_values(
    *, role: str, values: bool, enable_jit: bool
) -> None:
    model = _model(role=role, values=values, reader="utility", enable_jit=enable_jit)
    got = model.solve(params={"discount_factor": 1.0}, log_level="off").values
    # Over the state's two codes; for the action, the maximum at code 1, per
    # node of the unread state, which has been pruned.
    expected = [1.0, 3.0] if role == "state" else 3.0
    np.testing.assert_array_equal(np.asarray(got[0]["live"]), expected)


@pytest.mark.parametrize("values", [False, True], ids=["category", "value-grid"])
def test_a_constraint_on_an_action_code_selects_its_code(*, values: bool) -> None:
    model = _model(role="action", values=values, reader="constraint")
    solution = model.solve(params={"discount_factor": 1.0}, log_level="off")
    panel = model.simulate(
        params={"discount_factor": 1.0},
        initial_conditions={
            "age": jnp.zeros(2),
            "regime_id": jnp.zeros(2, dtype=jnp.int32),
        },
        solution=solution,
        seed=0,
        log_level="off",
    ).to_dataframe(use_categorical_codes=True)
    assert panel.loc[panel["period"] == 0, "x"].tolist() == [1, 1]


@pytest.mark.parametrize("placement", ["regime", "model"])
def test_a_derived_code_read_keeps_the_state_its_producer_reads(
    placement: str,
) -> None:
    assert "z" in _kept(_derived_model(placement=placement))


@pytest.mark.parametrize("enable_jit", [False, True], ids=["eager", "jit"])
@pytest.mark.parametrize("placement", ["regime", "model"])
def test_a_derived_code_read_takes_its_literal_values(
    *, placement: str, enable_jit: bool
) -> None:
    model = _derived_model(placement=placement, enable_jit=enable_jit)
    got = model.solve(params={"discount_factor": 1.0}, log_level="off").values
    np.testing.assert_array_equal(np.asarray(got[0]["live"]), [1.0, 3.0])
