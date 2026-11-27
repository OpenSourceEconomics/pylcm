"""Gates own only agents drawn to their scheduled target.

A real ordinary lottery selects the incoming edge before a gate runs. Separate
incoming gates cannot overwrite one another when their declarations are reversed.
"""

import hashlib
import json
import struct

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
from numpy.testing import assert_array_almost_equal as aaae

from lcm import (
    AgeGrid,
    AgeRange,
    ByAge,
    DiscreteGrid,
    Gate,
    LinSpacedGrid,
    MemberRoute,
    Model,
    ProjectedRegimeValue,
    Regime,
    StochasticTransition,
    Transition,
    categorical,
    fixed_transition,
)
from lcm.solver_api import ResultRetention
from lcm.typing import (
    BoolND,
    ContinuousState,
    DiscreteAction,
    FloatND,
    RegimeName,
    ScalarInt,
)
from tests.conftest import DECIMAL_PRECISION

_BETA = 0.95
_SEED = 41
_N_AGENTS = 64


@categorical(ordered=False)
class _DualRegimeId:
    src: ScalarInt
    target_a: ScalarInt
    target_b: ScalarInt
    fallback_a: ScalarInt
    fallback_b: ScalarInt


@categorical(ordered=True)
class Work:
    leisure: ScalarInt
    work: ScalarInt


def _identity_wage(wage: ContinuousState) -> ContinuousState:
    return wage


def _prob_half(age: FloatND) -> FloatND:
    return 0.5 * jnp.ones_like(age, dtype=float)


def _gate_always_open(wage: ContinuousState) -> BoolND:
    return jnp.ones_like(wage, dtype=bool)


def _u_src(*, wage: ContinuousState, work: DiscreteAction) -> FloatND:
    return wage * work


def _u_target_a(wage: ContinuousState) -> FloatND:
    return wage + 10.0


def _u_target_b(wage: ContinuousState) -> FloatND:
    return wage + 20.0


def _u_fallback_a(wage: ContinuousState) -> FloatND:
    return wage + 1.0


def _u_fallback_b(wage: ContinuousState) -> FloatND:
    return wage + 2.0


def _ordinary_uniforms(*, n_agents: int) -> FloatND:
    """Independently address the reserved period-one regime lottery by agent."""
    encoded = json.dumps(
        ["pylcm.random.v1", "regime", "regime", 0],
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    words = struct.unpack(">8I", hashlib.sha256(encoded).digest())
    root = jax.random.key(_SEED, impl="threefry2x32")
    for word in words:
        root = jax.random.fold_in(root, np.uint32(word))
    draws = []
    for agent in range(n_agents):
        key = jax.random.fold_in(root, np.uint32(agent >> 32))
        key = jax.random.fold_in(key, np.uint32(agent % 2**32))
        key = jax.random.fold_in(key, np.uint32(1))
        draws.append(jax.random.uniform(key, dtype=jnp.asarray(0.0).dtype))
    return jnp.stack(draws)


def _replay(*, model: Model, source: RegimeName, wages: FloatND) -> pd.DataFrame:
    params = {"discount_factor": _BETA}
    solution = model.solve(
        params=params, retention=ResultRetention.VALUES, log_level="off"
    )
    result = model.simulate(
        params=params,
        initial_conditions={
            "wage": wages,
            "age": jnp.zeros(wages.size),
            "regime_id": jnp.full(
                wages.size, model.regime_names_to_ids[source], dtype=jnp.int32
            ),
        },
        solution=solution,
        seed=_SEED,
        log_level="off",
    )
    assert result.randomness["address_version"] == "pylcm.random.v1"
    assert result.randomness["prng_impl"] == "threefry2x32"
    assert result.randomness["precision"] == str(wages.dtype)
    return result.to_dataframe()


_WAGE_2 = LinSpacedGrid(start=1.0, stop=2.0, n_points=2)


def _dual_model(*, edge_order: tuple[RegimeName, RegimeName]) -> Model:
    gates = {
        name: Gate(
            predicate=_gate_always_open,
            routes={
                "only": MemberRoute(
                    fallback=ProjectedRegimeValue(
                        regime=fallback, projection={"wage": _identity_wage}
                    )
                )
            },
        )
        for name, fallback in (("target_a", "fallback_a"), ("target_b", "fallback_b"))
    }
    return Model(
        regimes={
            "src": Regime(
                states={"wage": _WAGE_2},
                state_transitions={"wage": fixed_transition("wage")},
                actions={"work": DiscreteGrid(category_class=Work)},
                functions={"utility": _u_src},
            ),
            "target_a": Regime(
                states={"wage": _WAGE_2}, functions={"utility": _u_target_a}
            ),
            "target_b": Regime(
                states={"wage": _WAGE_2}, functions={"utility": _u_target_b}
            ),
            "fallback_a": Regime(
                states={"wage": _WAGE_2}, functions={"utility": _u_fallback_a}
            ),
            "fallback_b": Regime(
                states={"wage": _WAGE_2}, functions={"utility": _u_fallback_b}
            ),
        },
        ages=AgeGrid(start=0, inclusive_stop=1, step="Y"),
        regime_id_class=_DualRegimeId,
        initial_nodes={0: "src"},
        edges={
            "src": Transition(
                law=ByAge(
                    cases={
                        AgeRange(exclusive_stop=1): {
                            name: StochasticTransition(func=_prob_half)
                            for name in edge_order
                        }
                    }
                ),
                gates={name: gates[name] for name in edge_order},
            )
        },
    )


def test_multiple_gates_preserve_the_original_draw_under_reversed_order() -> None:
    """Each offered target owns its agents under either declaration order."""
    wages = jnp.tile(jnp.asarray([1.0, 2.0]), _N_AGENTS // 2)
    uniform = np.asarray(_ordinary_uniforms(n_agents=_N_AGENTS))
    target_a = uniform < 0.5
    assert target_a.any()
    assert (~target_a).any()
    expected = np.where(target_a, "target_a", "target_b")
    forward = _replay(
        model=_dual_model(edge_order=("target_a", "target_b")),
        source="src",
        wages=wages,
    )
    reverse = _replay(
        model=_dual_model(edge_order=("target_b", "target_a")),
        source="src",
        wages=wages,
    )
    for frame in (forward, reverse):
        landed = frame.loc[frame["period"] == 1].sort_values("agent_id")
        np.testing.assert_array_equal(landed["agent_id"], np.arange(_N_AGENTS))
        np.testing.assert_array_equal(landed["regime_name"].to_numpy(), expected)
        aaae(landed["wage"], wages, decimal=DECIMAL_PRECISION)
        aaae(
            landed["value"],
            np.asarray(wages) + np.where(target_a, 10.0, 20.0),
            decimal=DECIMAL_PRECISION,
        )
    pd.testing.assert_frame_equal(forward, reverse)
