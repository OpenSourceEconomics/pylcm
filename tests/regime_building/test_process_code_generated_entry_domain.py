"""Physical entry must not manufacture a process node ordinal.

Both public producer routes create z=0.25 after valid initial inputs:

* an authored entry law from a source that does not carry z;
* a forced closed participation check's fallback projection, after node-valued
  candidate draws.

There is no node at 0.25 on (-2, -1, 0, 1, 2). A code-reading destination must
refuse before invoking its code consumer. Code-free destinations retain the
physical value. Exact-node controls ensure the routes themselves remain valid.

For the terminal two-action utility, work pays z - 1/8 and staying home pays
zero. Thus value-only evaluation at 1/4 selects work and returns 1/8. Rounding
the coordinate 2.25 to code 2 would reconstruct node 0 and select staying home
instead. All oracle quantities are exact dyadics.
"""

import jax.numpy as jnp
import numpy as np
import pytest

from lcm import (
    AgeGrid,
    ByAge,
    DiscreteGrid,
    Model,
    NormalIIDProcess,
    ParticipationCheck,
    ParticipationRead,
    ProjectedRegimeValue,
    Regime,
    StakeholderRoute,
    Transition,
    categorical,
)
from lcm.exceptions import (
    InvalidSimulationInputError,
    InvalidStateTransitionError,
    ModelInitializationError,
)
from lcm.transition import StochasticTransition
from lcm.typing import (
    BoolND,
    ContinuousState,
    DiscreteAction,
    DiscreteState,
    FloatND,
    ScalarInt,
)


@categorical(ordered=False)
class Work:
    off: ScalarInt
    on: ScalarInt


@categorical(ordered=False)
class EntryRegime:
    source: ScalarInt
    target: ScalarInt


@categorical(ordered=False)
class GatedRegime:
    source: ScalarInt
    landing: ScalarInt
    fallback: ScalarInt


_NODES = (-2.0, -1.0, 0.0, 1.0, 2.0)


def _process():
    return NormalIIDProcess(
        n_points=5,
        mu=0.0,
        sigma=1.0,
        n_std=2.0,
        gauss_hermite=False,
        simulate_draws="nodes",
    )


def _zero() -> FloatND:
    return jnp.asarray(0.0)


def _source_with_z(z: ContinuousState) -> FloatND:
    return 0.0 * z


def _landing(z: ContinuousState) -> FloatND:
    return 5.0 + 0.0 * z


def _value_utility(*, z: ContinuousState, work: DiscreteAction) -> FloatND:
    return work * (z - 0.125)


def _code_utility(*, z_code: DiscreteState, work: DiscreteAction) -> FloatND:
    # Use a normal branch, not an assert statement inside a sealed callable.
    if z_code.dtype != jnp.dtype("int32"):
        raise TypeError("z_code must be int32")
    return work * (jnp.asarray(_NODES)[z_code] - 0.125)


def _entry_quarter() -> ContinuousState:
    return jnp.asarray(0.25)


def _entry_zero() -> ContinuousState:
    return jnp.asarray(0.0)


def _project_quarter(z: ContinuousState) -> ContinuousState:
    # Reads the target's held value and always stays inside fallback support.
    return 0.25 + 0.0 * z


def _project_zero(z: ContinuousState) -> ContinuousState:
    return 0.0 * z


def _closed(V_target: FloatND) -> BoolND:
    return V_target < 0.0


def _landed_z(z: ContinuousState) -> ContinuousState:
    return z


def _prob_one(age: FloatND) -> FloatND:
    return jnp.ones_like(age, dtype=float)


def _destination(*, code: bool) -> Regime:
    return Regime(
        states={"z": _process()},
        actions={"work": DiscreteGrid(category_class=Work)},
        functions={"utility": _code_utility if code else _value_utility},
    )


def _model(*, route: str, code: bool, on_node: bool, enable_jit: bool) -> Model:
    if route == "entry_law":
        return Model(
            regimes={
                "source": Regime(
                    functions={"utility": _zero},
                    state_transitions={
                        "z": {"target": _entry_zero if on_node else _entry_quarter}
                    },
                ),
                "target": _destination(code=code),
            },
            edges={"source": {"target": 0}},
            ages=AgeGrid(start=0, inclusive_stop=1, step="Y"),
            initial_nodes={0: "source"},
            regime_id_class=EntryRegime,
            enable_jit=enable_jit,
        )
    if route != "closed_gate":
        raise ValueError(route)
    return Model(
        regimes={
            "source": Regime(
                states={"z": _process()}, functions={"utility": _source_with_z}
            ),
            "landing": Regime(
                states={"z": _process()}, functions={"utility": _landing}
            ),
            "fallback": _destination(code=code),
        },
        edges={
            "source": Transition(
                targets={"landing": 0, "fallback": 0},
                law=ByAge(cases={0: {"landing": StochasticTransition(func=_prob_one)}}),
                participation={
                    "landing": ParticipationCheck(
                        predicate=_closed,
                        reads={
                            "V_target": ParticipationRead(
                                kind="Raw",
                                regime="landing",
                                projection={"z": _landed_z},
                            )
                        },
                        routes={
                            "only": StakeholderRoute(
                                fallback=ProjectedRegimeValue(
                                    regime="fallback",
                                    projection={
                                        "z": _project_zero
                                        if on_node
                                        else _project_quarter
                                    },
                                )
                            )
                        },
                    )
                },
            )
        },
        ages=AgeGrid(start=0, inclusive_stop=1, step="Y"),
        initial_nodes={0: "source"},
        regime_id_class=GatedRegime,
        enable_jit=enable_jit,
    )


def _simulate(*, route: str, code: bool, on_node: bool, enable_jit: bool):
    model = _model(route=route, code=code, on_node=on_node, enable_jit=enable_jit)
    params = {
        name: (
            {"koopmans_aggregator": {"discount_factor": 1.0}}
            if name == "source"
            else {}
        )
        for name in model.regime_names_to_ids
    }
    initial = {
        "age": jnp.array([0.0]),
        "regime_id": jnp.array([model.regime_names_to_ids["source"]], dtype=jnp.int32),
    }
    if route == "closed_gate":
        initial["z"] = jnp.array([0.0])
    solution = model.solve(params=params, log_level="off")
    result = model.simulate(
        params=params,
        solution=solution,
        initial_conditions=initial,
        seed=0,
        log_level="off",
    )
    return result.to_dataframe().query("period == 1")


@pytest.mark.parametrize("enable_jit", [False, True], ids=["eager", "jit"])
@pytest.mark.parametrize("route", ["entry_law", "closed_gate"])
def test_generated_off_node_value_is_refused_before_code_read(
    *, route: str, enable_jit: bool
):
    # A producer known at build may be refused there; a produced value is refused
    # with a project-typed node-domain error, never rounded to a code or left to
    # fail through an unrelated indexing exception.
    with pytest.raises(
        (
            ModelInitializationError,
            InvalidSimulationInputError,
            InvalidStateTransitionError,
        ),
        match=r"(?i)node",
    ):
        _simulate(route=route, code=True, on_node=False, enable_jit=enable_jit)


@pytest.mark.parametrize("enable_jit", [False, True], ids=["eager", "jit"])
@pytest.mark.parametrize("route", ["entry_law", "closed_gate"])
def test_generated_exact_node_keeps_code_reads_supported(
    *, route: str, enable_jit: bool
):
    panel = _simulate(route=route, code=True, on_node=True, enable_jit=enable_jit)
    destination = "target" if route == "entry_law" else "fallback"
    assert panel["regime_name"].tolist() == [destination]
    np.testing.assert_array_equal(panel["z"].to_numpy(), np.array([0.0]))
    assert panel["work"].tolist() == ["off"]


@pytest.mark.parametrize("enable_jit", [False, True], ids=["eager", "jit"])
@pytest.mark.parametrize("route", ["entry_law", "closed_gate"])
def test_generated_off_node_value_remains_supported_without_code_reads(
    *, route: str, enable_jit: bool
):
    panel = _simulate(route=route, code=False, on_node=False, enable_jit=enable_jit)
    destination = "target" if route == "entry_law" else "fallback"
    assert panel["regime_name"].tolist() == [destination]
    np.testing.assert_array_equal(panel["z"].to_numpy(), np.array([0.25]))
    assert panel["work"].tolist() == ["on"]
