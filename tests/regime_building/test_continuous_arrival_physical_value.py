"""A typed arrival reads a continuous physical offer at its actual landing.

Raw evaluates the unchanged cubic at the realized offer. It does not interpolate
stored values of that cubic, and a positive-offer rule may retain a payoff below
its declared fallback. Collective participation and process node codes are separate.
"""

import hashlib
import json
import struct

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from lcm import (
    AgeGrid,
    ArrivalDecision,
    ArrivalRead,
    ByAge,
    Model,
    ProjectedRegimeValue,
    Regime,
    StakeholderRoute,
    Transition,
    categorical,
)
from lcm.exceptions import RegimeInitializationError, UnsupportedOperationError
from lcm.shocks import Discrete, Normal, ShockGroup, UserNodes
from lcm.solver_api import ResultRetention
from lcm.typing import BoolND, FloatND, IntND, ReferenceName, ScalarInt, UserFunction
from tests.conftest import DECIMAL_PRECISION

_N_SUBJECTS = 64


@categorical(ordered=False)
class _RegimeId:
    source: ScalarInt
    target: ScalarInt
    fallback: ScalarInt


def _source_flow() -> float:
    return 0.0


def _cubic_offer(wage_shock_f: FloatND) -> FloatND:
    return wage_shock_f**3 + 2.0 * wage_shock_f + 5.0


def _fallback_offer(wage_shock_f: FloatND) -> FloatND:
    return 10.0 + wage_shock_f


def _positive_native_value(*, V_target: FloatND, wage_shock_f: FloatND) -> BoolND:
    reference = (wage_shock_f * wage_shock_f + 2.0) * wage_shock_f + 5.0
    return (wage_shock_f > 0.0) & (jnp.abs(V_target - reference) <= 0.125)


def _addressed_physical_offers() -> FloatND:
    """Independently encode the scalar normal innovation for original rows at t=1."""
    encoded = json.dumps(
        ["pylcm.random.v1", "shock", "wage_shock_f", 0],
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    root = jax.random.key(0, impl="threefry2x32")
    for word in struct.unpack(">8I", hashlib.sha256(encoded).digest()):
        root = jax.random.fold_in(root, np.uint32(word))
    physical = []
    for subject in range(_N_SUBJECTS):
        key = jax.random.fold_in(root, np.uint32(subject >> 32))
        key = jax.random.fold_in(key, np.uint32(subject % 2**32))
        key = jax.random.fold_in(key, np.uint32(1))
        base = jax.random.normal(key, dtype=jnp.asarray(0.0).dtype)
        physical.append(0.0 + 1.0 * base)
    return jnp.stack(physical)


def test_observed_continuous_arrival_reads_the_actual_off_node_value() -> None:
    model = Model(
        regimes={
            "source": Regime(functions={"utility": _source_flow}),
            "target": Regime(
                functions={"utility": _cubic_offer}, shocks=("innovation",)
            ),
            "fallback": Regime(
                functions={"utility": _fallback_offer}, shocks=("innovation",)
            ),
        },
        shocks={
            "innovation": ShockGroup(
                shocks={"wage_shock_f": Normal(mu=0.0, sigma=1.0)},
                nodes=UserNodes(
                    nodes=np.asarray([[-2.0], [-1.0], [0.0], [1.0], [2.0]]),
                    weights=np.asarray([1.0, 4.0, 6.0, 4.0, 1.0]) / 16.0,
                ),
            )
        },
        ages=AgeGrid(start=0, inclusive_stop=1, step="Y"),
        initial_nodes={0: "source"},
        regime_id_class=_RegimeId,
        edges={
            "source": Transition(
                law=ByAge(cases={0: "target"}),
                arrivals={
                    "target": ArrivalDecision(
                        predicate=_positive_native_value,
                        observes=("wage_shock_f",),
                        reads={
                            "V_target": ArrivalRead(
                                kind="Raw",
                                regime="target",
                                projection={},
                                observes=("wage_shock_f",),
                            )
                        },
                        routes={
                            "self": StakeholderRoute(
                                fallback=ProjectedRegimeValue(
                                    regime="fallback", projection={}
                                )
                            )
                        },
                    )
                },
            )
        },
    )
    params = {"discount_factor": 0.5}
    solution = model.solve(
        params=params, retention=ResultRetention.VALUES, log_level="off"
    )
    # Node payoffs: rejected [8,9,10], kept [8,17], weights [1,4,6,4,1]/16.
    # Half their weighted sum is 153/32; maximizing branches would give 165/32.
    np.testing.assert_allclose(solution.values[0]["source"], 153.0 / 32.0)
    np.testing.assert_allclose(solution.values[1]["target"], 5.0)
    np.testing.assert_allclose(solution.values[1]["fallback"], 10.0)
    result = model.simulate(
        params=params,
        solution=solution,
        initial_conditions={
            "age": jnp.zeros(_N_SUBJECTS),
            "regime_id": jnp.full(_N_SUBJECTS, _RegimeId.source, dtype=jnp.int32),
        },
        seed=0,
        log_level="off",
    )
    assert result.randomness["address_version"] == "pylcm.random.v1"
    assert result.randomness["prng_impl"] == "threefry2x32"
    frame = result.to_dataframe()
    source = frame.loc[frame["period"] == 0].sort_values("subject_id")
    landed = frame.loc[frame["period"] == 1].sort_values("subject_id")
    expected_offer = np.asarray(_addressed_physical_offers())
    expected_cubic = (expected_offer * expected_offer + 2.0) * expected_offer + 5.0
    kept = expected_offer > 0.0
    expected_value = np.where(kept, expected_cubic, 10.0 + expected_offer)
    assert (expected_offer < -0.5).any()
    assert (expected_offer > 0.5).any()
    assert not np.isin(expected_offer, [-2.0, -1.0, 0.0, 1.0, 2.0]).any()
    # Positive off-node draws distinguish native cubic from old node interpolation.
    old_interpolation = np.interp(
        expected_offer, [-2.0, -1.0, 0.0, 1.0, 2.0], [-7.0, 2.0, 5.0, 8.0, 17.0]
    )
    assert (kept & (np.abs(old_interpolation - expected_cubic) > 0.125)).any()
    assert (kept & (expected_cubic < 10.0 + expected_offer)).any()
    np.testing.assert_array_equal(source["subject_id"], np.arange(_N_SUBJECTS))
    np.testing.assert_allclose(source["value"], 153.0 / 32.0)
    np.testing.assert_array_equal(landed["subject_id"], np.arange(_N_SUBJECTS))
    np.testing.assert_array_equal(
        landed["regime_name"], np.where(kept, "target", "fallback")
    )
    tolerance = 10.0**-DECIMAL_PRECISION
    np.testing.assert_allclose(
        landed["wage_shock_f"], expected_offer, rtol=tolerance, atol=tolerance
    )
    np.testing.assert_allclose(
        landed["value"], expected_value, rtol=tolerance, atol=tolerance
    )


def _arrival_signal(wage_shock_f: FloatND) -> FloatND:
    return wage_shock_f


def _direct_observation(*, wage_shock_f: FloatND, V_target: FloatND) -> BoolND:
    return (wage_shock_f > 0.0) & (V_target > -1e6)


def _helper_observation(*, arrival_signal: FloatND, V_target: FloatND) -> BoolND:
    return (arrival_signal > 0.0) & (V_target > -1e6)


def _direct_code_observation(*, wage_shock_f_code: ScalarInt) -> BoolND:
    return wage_shock_f_code > 0


def _information_model(*, predicate: UserFunction, observed: bool) -> Model:
    regimes = {
        "source": Regime(functions={"utility": _source_flow}),
        "target": Regime(
            functions={
                "utility": _cubic_offer,
                "arrival_signal": _arrival_signal,
            },
            shocks=("innovation",),
        ),
        "fallback": Regime(
            functions={"utility": _fallback_offer}, shocks=("innovation",)
        ),
    }
    return Model(
        regimes=regimes,
        shocks={
            "innovation": ShockGroup(
                shocks={"wage_shock_f": Normal(mu=0.0, sigma=1.0)},
                nodes=UserNodes(
                    nodes=np.asarray([[-2.0], [-1.0], [0.0], [1.0], [2.0]]),
                    weights=np.asarray([1.0, 4.0, 6.0, 4.0, 1.0]) / 16.0,
                ),
            )
        },
        ages=AgeGrid(start=0, inclusive_stop=1, step="Y"),
        initial_nodes={0: "source"},
        regime_id_class=_RegimeId,
        edges={
            "source": Transition(
                law=ByAge(cases={0: "target"}),
                arrivals={
                    "target": ArrivalDecision(
                        predicate=predicate,
                        observes=("wage_shock_f",) if observed else (),
                        reads={
                            "V_target": ArrivalRead(
                                kind="Raw",
                                regime="target",
                                projection={},
                                observes=("wage_shock_f",),
                            )
                        },
                        routes={
                            "self": StakeholderRoute(
                                fallback=ProjectedRegimeValue(
                                    regime="fallback", projection={}
                                )
                            )
                        },
                    )
                },
            )
        },
    )


@pytest.mark.parametrize(
    "predicate", [_direct_observation, _helper_observation], ids=["direct", "helper"]
)
def test_an_arrival_callback_cannot_read_a_hidden_physical_input(
    *, predicate: UserFunction
) -> None:
    """A read's observed pricing does not reveal its shock to the Boolean callback."""
    with pytest.raises(
        RegimeInitializationError, match=r"Arrival callback.*unobserved.*wage_shock_f"
    ):
        _information_model(predicate=predicate, observed=False)


def test_an_arrival_callback_cannot_read_a_continuous_shock_node_code() -> None:
    """A recognized target Normal draw has no quadrature-node position."""
    with pytest.raises(
        UnsupportedOperationError,
        match=r"wage_shock_f_code.*continuous physical shock.*quadrature-node position",
    ):
        _information_model(predicate=_direct_code_observation, observed=True)


def test_an_observed_target_physical_helper_is_not_a_free_edge_parameter() -> None:
    """The helper reads the actual target physical law, with no extra edge input."""
    model = _information_model(predicate=_helper_observation, observed=True)
    assert "edges" not in model.get_params_template()
    solution = model.solve(params={"discount_factor": 0.5}, log_level="off")
    np.testing.assert_allclose(solution.values[0]["source"], 153.0 / 32.0)


@categorical(ordered=False)
class _DiscreteOffer:
    low: ScalarInt
    high: ScalarInt


def _discrete_payoff(offer_code: IntND) -> FloatND:
    return 2.0 + 4.0 * offer_code


def _discrete_signal(offer_code: IntND) -> IntND:
    return offer_code


def _ten() -> float:
    return 10.0


def _discrete_direct(offer_code: IntND) -> BoolND:
    return offer_code > 0


def _discrete_helper(offer_signal: IntND) -> BoolND:
    return offer_signal > 0


def _discrete_duplicate(*, offer: IntND, offer_code: IntND) -> BoolND:
    return (offer == offer_code) & (offer_code > 0)


def _discrete_read_named_offer(*, offer: FloatND, offer_code: IntND) -> BoolND:
    return (offer > 5.0) & (offer_code > 0)


def _discrete_read_named_code(*, offer_code: FloatND, offer: IntND) -> BoolND:
    return (offer_code > 5.0) & (offer > 0)


def _addressed_discrete_offers() -> IntND:
    """Independently address target categories in original subject order at t=1."""
    encoded = json.dumps(
        ["pylcm.random.v1", "shock", "offer", 0],
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    root = jax.random.key(0, impl="threefry2x32")
    for word in struct.unpack(">8I", hashlib.sha256(encoded).digest()):
        root = jax.random.fold_in(root, np.uint32(word))
    codes = []
    for subject in range(_N_SUBJECTS):
        key = jax.random.fold_in(root, np.uint32(subject >> 32))
        key = jax.random.fold_in(key, np.uint32(subject % 2**32))
        key = jax.random.fold_in(key, np.uint32(1))
        uniform = jax.random.uniform(key, dtype=jnp.asarray(0.0).dtype)
        codes.append((uniform >= 0.5).astype(jnp.int32))
    return jnp.stack(codes)


def _discrete_information_model(
    *, predicate: UserFunction, observed: bool, read_name: ReferenceName | None
) -> Model:
    return Model(
        regimes={
            "source": Regime(functions={"utility": _source_flow}),
            "target": Regime(
                functions={
                    "utility": _discrete_payoff,
                    **(
                        {"offer_signal": _discrete_signal}
                        if predicate is _discrete_helper
                        else {}
                    ),
                },
                shocks=("target_offer",),
            ),
            "fallback": Regime(functions={"utility": _ten}),
        },
        shocks={
            "target_offer": ShockGroup(
                shocks={"offer": Discrete(categories=_DiscreteOffer, probs=(0.5, 0.5))}
            )
        },
        ages=AgeGrid(start=0, inclusive_stop=1, step="Y"),
        initial_nodes={0: "source"},
        regime_id_class=_RegimeId,
        edges={
            "source": Transition(
                law=ByAge(cases={0: "target"}),
                arrivals={
                    "target": ArrivalDecision(
                        predicate=predicate,
                        observes=("offer",) if observed else (),
                        reads={
                            read_name: ArrivalRead(
                                kind="Raw", regime="fallback", projection={}
                            )
                        }
                        if read_name is not None
                        else {},
                        routes={
                            "self": StakeholderRoute(
                                fallback=ProjectedRegimeValue(
                                    regime="fallback", projection={}
                                )
                            )
                        },
                    )
                },
            )
        },
    )


@pytest.mark.parametrize(
    ("predicate", "read_name"),
    [
        (_discrete_direct, None),
        (_discrete_helper, None),
        (_discrete_duplicate, None),
        (_discrete_read_named_offer, "offer"),
        (_discrete_read_named_code, "offer_code"),
    ],
    ids=["direct", "helper", "duplicate", "read-offer", "read-offer-code"],
)
def test_observed_discrete_arrival_code_keeps_its_physical_and_read_channels(
    *, predicate: UserFunction, read_name: ReferenceName | None
) -> None:
    """Named reference10 and actual category0/1 remain distinct Boolean operands."""
    model = _discrete_information_model(
        predicate=predicate, observed=True, read_name=read_name
    )
    assert "edges" not in model.get_params_template()
    params = {"discount_factor": 0.5}
    solution = model.solve(
        params=params, retention=ResultRetention.VALUES, log_level="off"
    )
    # Low falls back to10; high keeps6 even though its fallback would pay10.
    np.testing.assert_array_equal(solution.values[0]["source"], 4.0)
    frame = model.simulate(
        params=params,
        solution=solution,
        initial_conditions={
            "age": jnp.zeros(_N_SUBJECTS),
            "regime_id": jnp.full(_N_SUBJECTS, _RegimeId.source, dtype=jnp.int32),
        },
        seed=0,
        log_level="off",
    ).to_dataframe(use_categorical_codes=True)
    landed = frame.loc[frame["period"] == 1].sort_values("subject_id")
    codes = np.asarray(_addressed_discrete_offers())
    assert (codes == 0).any()
    assert (codes == 1).any()
    kept = codes == 1
    np.testing.assert_array_equal(landed["subject_id"], np.arange(_N_SUBJECTS))
    np.testing.assert_array_equal(
        landed["regime_name"], np.where(kept, "target", "fallback")
    )
    np.testing.assert_array_equal(landed["value"], np.where(kept, 6.0, 10.0))
    np.testing.assert_array_equal(landed.loc[kept, "offer"], codes[kept])


@pytest.mark.parametrize(
    ("predicate", "read_name"),
    [
        (_discrete_direct, None),
        (_discrete_helper, None),
        (_discrete_read_named_offer, "offer"),
    ],
    ids=["direct", "helper", "read-offer"],
)
def test_hidden_discrete_arrival_code_is_not_an_edge_parameter_or_observation(
    *, predicate: UserFunction, read_name: ReferenceName | None
) -> None:
    """A physical category, including its code alias, requires declared visibility."""
    with pytest.raises(
        RegimeInitializationError, match=r"Arrival callback.*unobserved.*offer"
    ):
        _discrete_information_model(
            predicate=predicate, observed=False, read_name=read_name
        )
