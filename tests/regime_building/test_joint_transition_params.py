"""Parameter ownership for edge-local joint kernels."""

from typing import Any, cast

import jax.numpy as jnp

from _lcm.params.regime_template import create_regime_params_template
from lcm import JointTransition, MarkovTransition, Phased
from lcm.regime import Regime
from lcm.typing import FloatND


def _target_probability() -> FloatND:
    return jnp.asarray(1.0)


def _support(match_location: float) -> dict[str, FloatND]:
    return {"wealth": jnp.asarray([match_location, match_location + 1])}


def _solve_probabilities(*, match_probability: float, age: float) -> FloatND:  # noqa: ARG001
    return jnp.asarray([match_probability, 1 - match_probability])


def _simulate_probabilities(realized_match_probability: float) -> FloatND:
    return jnp.asarray([realized_match_probability, 1 - realized_match_probability])


def _next_wealth(*, match: dict[str, FloatND], wealth_shift: float) -> FloatND:
    return match["wealth"] + wealth_shift


def _kernel(probabilities: object) -> JointTransition:
    return JointTransition(
        support_size=2,
        support=_support,
        probabilities=probabilities,  # ty: ignore[invalid-argument-type]
        outputs={"wealth": _next_wealth},
    )


def test_joint_kernel_params_follow_role_and_output_ownership() -> None:
    """Support/probability params live under the kernel; outputs keep `next_` paths."""
    regime = Regime(
        transition={"target": MarkovTransition(_target_probability)},
        functions={"utility": lambda: jnp.asarray(0.0)},
        joint_transitions={
            "target": {
                "match": Phased(
                    solve=_kernel(_solve_probabilities),
                    simulate=_kernel(_simulate_probabilities),
                )
            }
        },
    )

    template = cast("Any", create_regime_params_template(user_regime=regime))

    assert template["target"]["match"]["support"] == {"match_location": "float"}
    assert template["target"]["match"]["probabilities"] == {
        "match_probability": "float",
        "realized_match_probability": "float",
    }
    assert template["target"]["next_wealth"] == {"wealth_shift": "float"}
