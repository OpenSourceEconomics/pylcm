"""Arrival decisions own only subjects drawn to their scheduled target.

A real ordinary lottery selects the incoming edge before arrival consent runs.
An unrelated draw keeps its regime, state and payoff even when another edge's
consent would open or close there. Separate incoming decisions cannot overwrite
one another when their declarations are reversed.
"""

import hashlib
import json
import struct
from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
from numpy.testing import assert_array_almost_equal as aaae

from lcm import (
    AgeGrid,
    AgeRange,
    ArrivalDecision,
    ByAge,
    DiscreteGrid,
    LinSpacedGrid,
    Model,
    Phased,
    ProjectedRegimeValue,
    Regime,
    StakeholderRoute,
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
from tests.regime_building.test_collective_regime_simulate import (
    _BETA,
    _consent_laws,
    _make_consent_regimes,
    _prob_one,
)

_SEED = 41
_N_SUBJECTS = 64


@categorical(ordered=False)
class _ConsentRegimeId:
    single_f: ScalarInt
    single_f_terminal: ScalarInt
    single_m_terminal: ScalarInt
    married_terminal: ScalarInt


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


def _ordinary_uniforms(*, n_subjects: int) -> FloatND:
    """Independently address the reserved period-one regime lottery by subject."""
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
    for subject in range(n_subjects):
        key = jax.random.fold_in(root, np.uint32(subject >> 32))
        key = jax.random.fold_in(key, np.uint32(subject % 2**32))
        key = jax.random.fold_in(key, np.uint32(1))
        draws.append(jax.random.uniform(key, dtype=jnp.asarray(0.0).dtype))
    return jnp.stack(draws)


def _consent_model(*, lottery: bool) -> Model:
    laws = _consent_laws()
    original = laws["single_f"]
    assert isinstance(original, Transition)
    if lottery:
        original = replace(
            original,
            law=ByAge(
                cases={
                    AgeRange(exclusive_stop=1): {
                        "married_terminal": StochasticTransition(func=_prob_half),
                        "single_m_terminal": StochasticTransition(func=_prob_half),
                    }
                }
            ),
        )
    return Model(
        regimes=_make_consent_regimes(),
        ages=AgeGrid(start=0, inclusive_stop=1, step="Y"),
        regime_id_class=_ConsentRegimeId,
        initial_nodes={0: "single_f"},
        edges={"single_f": original},
    )


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


def test_unrelated_ordinary_draw_keeps_its_own_route_and_state() -> None:
    """Unrelated draws keep the male single payoff under either consent outcome."""
    wages = jnp.tile(jnp.asarray([1.0, 2.0]), _N_SUBJECTS // 2)
    frame = _replay(model=_consent_model(lottery=True), source="single_f", wages=wages)
    landed = frame.loc[frame["period"] == 1].sort_values("subject_id")
    np.testing.assert_array_equal(landed["subject_id"], np.arange(_N_SUBJECTS))
    # The lottery is sorted by the fixed regime codes: single_m precedes married.
    offered = np.asarray(_ordinary_uniforms(n_subjects=_N_SUBJECTS)) >= 0.5
    consent = np.asarray(wages) == 1.0
    for selected in (
        offered & consent,
        offered & ~consent,
        ~offered & consent,
        ~offered & ~consent,
    ):
        assert selected.any()
    expected = np.where(
        offered,
        np.where(consent, "married_terminal", "single_f_terminal"),
        "single_m_terminal",
    )
    np.testing.assert_array_equal(landed["regime_name"].to_numpy(), expected)
    aaae(landed["wage"], np.asarray(wages), decimal=DECIMAL_PRECISION)
    np.testing.assert_array_equal(
        landed["own_stakeholder"].astype("string").fillna("none").to_numpy(),
        np.where(offered & consent, "f", "none"),
    )
    for regime, expected_value in (
        ("single_f_terminal", 1.5 * np.asarray(wages)),
        ("single_m_terminal", np.where(consent, 0.5, 3.0)),
    ):
        selected = expected == regime
        aaae(
            landed.loc[selected, "value"],
            expected_value[selected],
            decimal=DECIMAL_PRECISION,
        )
    married = offered & consent
    aaae(landed.loc[married, "value_f"], 2.0, decimal=DECIMAL_PRECISION)
    aaae(landed.loc[married, "value_m"], 1.0, decimal=DECIMAL_PRECISION)
    source = frame.loc[frame["period"] == 0].sort_values("subject_id")
    aaae(
        source["value"],
        np.asarray(wages) + _BETA * np.where(consent, 1.25, 3.0),
        decimal=DECIMAL_PRECISION,
    )


def test_scheduled_target_routes_open_and_closed_consent() -> None:
    """An offered subject marries at wage one and takes her fallback at wage two."""
    frame = _replay(
        model=_consent_model(lottery=False),
        source="single_f",
        wages=jnp.asarray([1.0, 2.0]),
    )
    landed = frame.loc[frame["period"] == 1].sort_values("subject_id")
    np.testing.assert_array_equal(landed["subject_id"], [0, 1])
    np.testing.assert_array_equal(
        landed["regime_name"].to_numpy(), ["married_terminal", "single_f_terminal"]
    )
    aaae(landed["wage"], [1.0, 2.0], decimal=DECIMAL_PRECISION)
    aaae(landed["value_f"].iloc[0], 2.0, decimal=DECIMAL_PRECISION)
    aaae(landed["value_m"].iloc[0], 1.0, decimal=DECIMAL_PRECISION)
    aaae(landed["value"].iloc[1], 3.0, decimal=DECIMAL_PRECISION)
    aaae(
        frame.loc[frame["period"] == 0].sort_values("subject_id")["value"],
        [2.9, 4.85],
        decimal=DECIMAL_PRECISION,
    )


_WAGE_2 = LinSpacedGrid(start=1.0, stop=2.0, n_points=2)


def _dual_model(*, edge_order: tuple[RegimeName, RegimeName]) -> Model:
    arrivals = {
        name: ArrivalDecision(
            predicate=_gate_always_open,
            routes={
                "only": StakeholderRoute(
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
                arrivals={name: arrivals[name] for name in edge_order},
            )
        },
    )


def test_multiple_arrivals_preserve_the_original_draw_under_reversed_order() -> None:
    """Each offered target owns its subjects under either declaration order."""
    wages = jnp.tile(jnp.asarray([1.0, 2.0]), _N_SUBJECTS // 2)
    uniform = np.asarray(_ordinary_uniforms(n_subjects=_N_SUBJECTS))
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
        landed = frame.loc[frame["period"] == 1].sort_values("subject_id")
        np.testing.assert_array_equal(landed["subject_id"], np.arange(_N_SUBJECTS))
        np.testing.assert_array_equal(landed["regime_name"].to_numpy(), expected)
        aaae(landed["wage"], wages, decimal=DECIMAL_PRECISION)
        aaae(
            landed["value"],
            np.asarray(wages) + np.where(target_a, 10.0, 20.0),
            decimal=DECIMAL_PRECISION,
        )
    pd.testing.assert_frame_equal(forward, reverse)


def test_phased_law_prices_perceived_arrivals_and_settles_realized_draws() -> None:
    """Perceived marriage prices coexist with a realized normalized meeting lottery."""
    original = _consent_laws()["single_f"]
    assert isinstance(original, Transition)
    declaration = replace(
        original,
        law=ByAge(
            cases={
                AgeRange(exclusive_stop=1): Phased(
                    solve={"married_terminal": StochasticTransition(func=_prob_one)},
                    simulate={
                        "married_terminal": StochasticTransition(func=_prob_half),
                        "single_m_terminal": StochasticTransition(func=_prob_half),
                    },
                )
            }
        ),
    )
    model = Model(
        regimes=_make_consent_regimes(),
        ages=AgeGrid(start=0, inclusive_stop=1, step="Y"),
        regime_id_class=_ConsentRegimeId,
        initial_nodes={0: "single_f"},
        edges={"single_f": declaration},
    )
    params = {"discount_factor": _BETA}
    solution = model.solve(
        params=params, retention=ResultRetention.VALUES, log_level="off"
    )
    aaae(solution.values[0]["single_f"], [2.9, 4.85], decimal=DECIMAL_PRECISION)
    wages = jnp.tile(jnp.asarray([1.0, 2.0]), _N_SUBJECTS // 2)
    result = model.simulate(
        params=params,
        initial_conditions={
            "wage": wages,
            "age": jnp.zeros(_N_SUBJECTS),
            "regime_id": jnp.full(
                _N_SUBJECTS, model.regime_names_to_ids["single_f"], dtype=jnp.int32
            ),
        },
        solution=solution,
        seed=_SEED,
        log_level="off",
    )
    assert result.randomness["address_version"] == "pylcm.random.v1"
    assert result.randomness["prng_impl"] == "threefry2x32"
    assert result.randomness["precision"] == str(wages.dtype)
    frame = result.to_dataframe()
    landed = frame.loc[frame["period"] == 1].sort_values("subject_id")
    np.testing.assert_array_equal(landed["subject_id"], np.arange(_N_SUBJECTS))
    offered = np.asarray(_ordinary_uniforms(n_subjects=_N_SUBJECTS)) >= 0.5
    consent = np.asarray(wages) == 1.0
    for selected in (
        offered & consent,
        offered & ~consent,
        ~offered & consent,
        ~offered & ~consent,
    ):
        assert selected.any()
    expected = np.where(
        offered,
        np.where(consent, "married_terminal", "single_f_terminal"),
        "single_m_terminal",
    )
    np.testing.assert_array_equal(landed["regime_name"].to_numpy(), expected)
    aaae(landed["wage"], np.asarray(wages), decimal=DECIMAL_PRECISION)
    np.testing.assert_array_equal(
        landed["own_stakeholder"].astype("string").fillna("none").to_numpy(),
        np.where(offered & consent, "f", "none"),
    )
    for regime, expected_value in (
        ("single_f_terminal", 1.5 * np.asarray(wages)),
        ("single_m_terminal", np.where(consent, 0.5, 3.0)),
    ):
        selected = expected == regime
        aaae(
            landed.loc[selected, "value"],
            expected_value[selected],
            decimal=DECIMAL_PRECISION,
        )
    kept = offered & consent
    aaae(landed.loc[kept, "value_f"], 2.0, decimal=DECIMAL_PRECISION)
    aaae(landed.loc[kept, "value_m"], 1.0, decimal=DECIMAL_PRECISION)
