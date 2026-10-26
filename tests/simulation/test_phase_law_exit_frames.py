"""Exit frames retain the parameter slots of their actual producing phase."""

import jax.numpy as jnp
import numpy as np
import pytest

from lcm import (
    CollectiveUtility,
    Consent,
    Counterparty,
    Exit,
    ExitTarget,
    InitialNodes,
    Model,
    Phased,
    Regime,
    categorical,
)
from lcm.shocks import GaussHermite, Normal, PhysicalValues, Product, ShockGroup
from lcm.typing import FloatND, ScalarInt
from tests.conftest import DECIMAL_PRECISION


@categorical(ordered=False)
class _PhaseFrameRegimes:
    source: ScalarInt
    couple: ScalarInt
    outside_f: ScalarInt
    outside_m: ScalarInt


def _single_flow() -> float:
    return 1.0


def _female_flow(*, offer: FloatND) -> FloatND:
    return 4.0 + offer


def _male_flow(*, offer: FloatND) -> FloatND:
    return 6.0 + offer


def _solve_mean(*, baseline: FloatND, experience_cap: FloatND) -> FloatND:
    return baseline + 0.0 * experience_cap


def _simulate_mean(*, baseline: FloatND) -> FloatND:
    return baseline


def _offer_location(*, offer_mean: FloatND) -> FloatND:
    return offer_mean


@pytest.mark.coverage(backends=("cpu",), precisions="both")
@pytest.mark.parametrize("seed", [None, 17])
def test_phase_specific_shared_offer_frame_reaches_participating_final_actor(
    *, seed: int | None
) -> None:
    """Consent observes one shared offer and retains the couple's chosen frame."""
    model = Model(
        regimes={
            "source": Regime(
                functions={"utility": _single_flow},
                exits={
                    "marry": Exit(
                        targets={
                            "self": ExitTarget(regime="couple", role="f", projection={})
                        },
                        rule=Consent(
                            counterparty=Counterparty(
                                regime="outside_m", role="m", projection={}
                            )
                        ),
                        observes=("offer",),
                    )
                },
            ),
            "couple": Regime(
                functions={
                    "utility": CollectiveUtility(
                        utilities={"f": _female_flow, "m": _male_flow}
                    ),
                    "offer_mean": Phased(solve=_solve_mean, simulate=_simulate_mean),
                },
                shocks=("offers",),
                exits={
                    "divorce": Exit(
                        targets={
                            "f": ExitTarget(regime="outside_f", projection={}),
                            "m": ExitTarget(regime="outside_m", projection={}),
                        },
                        rule="unilateral",
                        observes=("offer",),
                    )
                },
            ),
            "outside_f": Regime(functions={"utility": _single_flow}),
            "outside_m": Regime(functions={"utility": _single_flow}),
        },
        shocks={
            "offers": ShockGroup(
                shocks={"offer": Normal(mu=_offer_location, sigma=1.0)},
                nodes=Product(offer=GaussHermite(2)),
            )
        },
        n_periods=1,
        regime_id_class=_PhaseFrameRegimes,
        initial_nodes=InitialNodes(by_period={0: "source"}),
        edges={},
        enable_jit=True,
        fixed_params={"offers": {"baseline": 0.0, "experience_cap": 10.0}},
    )
    solution = model.solve(params={}, log_level="off")
    # Both spouses strictly prefer the couple at each original solve node.
    np.testing.assert_array_almost_equal(
        solution.values[0]["source"], 4.0, decimal=DECIMAL_PRECISION
    )
    result = model.simulate(
        params={},
        solution=solution,
        initial_conditions={
            "period": jnp.zeros((2,), dtype=jnp.int32),
            "regime_id": jnp.full(
                (2,), model.regime_names_to_ids["source"], dtype=jnp.int32
            ),
            "offer": PhysicalValues(values=np.asarray([0.0, 1.0])),
        },
        seed=seed,
        log_level="off",
    )
    frame = result.to_dataframe(use_categorical_codes=True)
    np.testing.assert_array_equal(frame["regime_name"], "couple")
    np.testing.assert_array_equal(frame["offer"], [0.0, 1.0])
    np.testing.assert_array_almost_equal(
        frame["value_f"], [4.0, 5.0], decimal=DECIMAL_PRECISION
    )
    np.testing.assert_array_almost_equal(
        frame["value_m"], [6.0, 7.0], decimal=DECIMAL_PRECISION
    )
