"""A joint output may land on a stochastic process's own discretized nodes.

`JointTransition` writes correlated innovations: one shared draw, and one law
per target state mapping that draw to a value. When the target state is a
stochastic process, pylcm has already discretized its grid and stores the value
function on those nodes, so the physical value the output names reaches the
continuation as its coefficients in that node basis — the hat weights of linear
interpolation. On a node those weights are one-hot, so naming a node reads that
node alone; between nodes the continuation is the linear interpolation of the
target's value function, which is the only reading its nodes support.
"""

import jax.numpy as jnp
import numpy as np
import pytest

from lcm import (
    AgeGrid,
    JointTransition,
    LinSpacedGrid,
    Model,
    NormalIIDProcess,
    categorical,
    fixed_transition,
)
from lcm.exceptions import InvalidValueFunctionError
from lcm.regime import Regime
from lcm.typing import FloatND, ScalarInt
from tests.conftest import DECIMAL_PRECISION

_DISCOUNT_FACTOR = 0.95
_WEALTH = (1.0, 5.5, 10.0)
_NODES = (-3.0, 0.0, 3.0)


@categorical(ordered=False)
class RegimeId:
    working: ScalarInt
    dead: ScalarInt


def _next_regime(age: float) -> ScalarInt:
    return jnp.where(age < 61, RegimeId.working, RegimeId.dead)


def _equal_probabilities() -> FloatND:
    return jnp.asarray([0.5, 0.5])


def _next_wage_shock(match: dict[str, FloatND]) -> FloatND:
    """The shared draw, named as a physical value on the process's own axis."""
    return match["innovation"]


def _utility(*, wealth: float, wage_shock: float) -> FloatND:
    """Convex in the shock, so interpolating the nodes is visible in the value."""
    return jnp.asarray(wealth) + wage_shock**2


def _zero_utility() -> FloatND:
    return jnp.asarray(0.0)


def _params() -> dict[str, dict[str, dict[str, float]]]:
    return {
        "working": {"koopmans_aggregator": {"discount_factor": _DISCOUNT_FACTOR}},
        "dead": {},
    }


def _model(*, support: tuple[float, float]) -> Model:
    working = Regime(
        transition=_next_regime,
        active=lambda age: age < 62,
        states={
            "wealth": LinSpacedGrid(
                start=_WEALTH[0], stop=_WEALTH[-1], n_points=len(_WEALTH)
            ),
            "wage_shock": NormalIIDProcess(
                n_points=len(_NODES),
                gauss_hermite=False,
                mu=0.0,
                n_std=3.0,
                sigma=1.0,
            ),
        },
        state_transitions={"wealth": fixed_transition("wealth")},
        functions={"utility": _utility},
        joint_transitions={
            "working": {
                "match": JointTransition(
                    support_size=2,
                    support={"innovation": jnp.asarray(support)},
                    probabilities=_equal_probabilities,
                    outputs={"wage_shock": _next_wage_shock},
                )
            }
        },
    )
    dead = Regime(transition=None, functions={"utility": _zero_utility})
    return Model(
        regimes={"working": working, "dead": dead},
        ages=AgeGrid(start=60, stop=63, step="Y"),
        regime_id_class=RegimeId,
    )


def _expected_V0(*, expected_continuation_shock_square: float) -> np.ndarray:
    """`V0 = u(wealth, shock) + β · E[wealth + interpolated shock²]`.

    The last working period has no continuation, so `V1 = wealth + shock²`, and
    `wealth` is fixed. Each joint node contributes the linear interpolation of
    that surface at the value the output names.

    The axes are `(wage_shock, wealth)`, which is the regime's published solve
    order — process states precede ordinary ones there, so the array is not
    built in the order the regime declares its states.
    """
    shock = np.asarray(_NODES)[:, None]
    wealth = np.asarray(_WEALTH)[None, :]
    continuation = wealth + expected_continuation_shock_square
    return wealth + shock**2 + _DISCOUNT_FACTOR * continuation


def test_a_joint_output_naming_process_nodes_reads_those_nodes_alone() -> None:
    """On-node support: the hat weights are one-hot, so no interpolation enters.

    Both nodes named are `±3`, where `V1` is `wealth + 9`, so the continuation
    is `wealth + 9` regardless of which one the draw lands on.
    """
    solution = _model(support=(-3.0, 3.0)).solve(params=_params(), log_level="debug")

    np.testing.assert_array_almost_equal(
        np.asarray(solution[0]["working"]),
        _expected_V0(expected_continuation_shock_square=9.0),
        decimal=DECIMAL_PRECISION,
    )


def test_an_off_node_joint_output_interpolates_the_targets_node_values() -> None:
    """Off-node support: the continuation is the linear reading of the nodes.

    `-0.1` sits between the nodes `-3` and `0`, so its weights are `0.1/3` and
    `1 - 0.1/3`; against `V1 = wealth + shock²` that reads `wealth + 0.3`, not
    the `wealth + 0.01` the physical value would give off-grid. `+0.1` is the
    mirror image, so both draws contribute the same amount.
    """
    solution = _model(support=(-0.1, 0.1)).solve(params=_params(), log_level="debug")

    np.testing.assert_array_almost_equal(
        np.asarray(solution[0]["working"]),
        _expected_V0(expected_continuation_shock_square=0.1 / 3.0 * 9.0),
        decimal=DECIMAL_PRECISION,
    )


def test_a_joint_output_off_the_targets_support_is_not_priced() -> None:
    """A value the target's nodes cannot express yields no finite continuation.

    The support is the contract: `[-3, 3]` here. A law naming `4` has no
    representation in that basis, so the weights are `NaN` and the caller's
    value function reports it rather than publishing an extrapolation.
    """
    with pytest.raises(InvalidValueFunctionError, match="all values are NaN"):
        _model(support=(-3.0, 4.0)).solve(params=_params(), log_level="debug")


def test_a_simulated_joint_output_realizes_one_of_its_support_points() -> None:
    """Simulation takes the physical value the output names, not a node.

    The node basis exists so the solve can read a stored value function; the
    state itself is the physical value, so a subject's realized `wage_shock` is
    one of the two points the joint support declares.
    """
    model = _model(support=(-0.1, 0.1))
    result = model.simulate(
        params=_params(),
        initial_conditions={
            "wealth": jnp.asarray([1.0, 10.0]),
            "wage_shock": jnp.asarray([0.0, 0.0]),
            "age": jnp.asarray([60.0, 60.0]),
            "regime_id": jnp.asarray([RegimeId.working, RegimeId.working]),
        },
        period_to_regime_to_V_arr=None,
        log_level="debug",
    )

    realized = result.to_dataframe().query("period == 1")["wage_shock"].to_numpy()

    np.testing.assert_array_almost_equal(
        np.abs(realized), 0.1, decimal=DECIMAL_PRECISION
    )
