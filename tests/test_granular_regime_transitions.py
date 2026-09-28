"""Per-target regime transitions: `transition={target: MarkovTransition(prob)}`.

The granular form declares each target regime's transition probability as its
own function; the key set IS the regime's reachability declaration — omitted
targets are structurally unreachable. The coarse forms (one callable over all
regimes) remain valid and reach every regime.
"""

from typing import Any

import jax.numpy as jnp
import pytest

from lcm import (
    AgeGrid,
    LinSpacedGrid,
    MarkovTransition,
    Model,
    Phased,
    categorical,
)
from lcm.exceptions import ModelInitializationError, RegimeInitializationError
from lcm.regime import Regime as UserRegime
from lcm.typing import FloatND, ScalarFloat, ScalarInt
from tests.test_models.schedules import until_exit


@categorical(ordered=False)
class _RegimeId:
    work: ScalarInt
    retired: ScalarInt
    dead: ScalarInt


def _utility(consumption: float) -> FloatND:
    return jnp.log(consumption)


def _next_wealth(*, wealth: float, consumption: float) -> float:
    return wealth - consumption


def _prob_work(age: int) -> ScalarFloat:
    return jnp.where(age < 1, 0.5, 0.0)


def _prob_retired(age: int) -> ScalarFloat:
    return jnp.where(age < 1, 0.5, 1.0) * 0.5


def _prob_dead(age: int) -> ScalarFloat:
    return jnp.asarray(1.0 - _prob_work(age) - _prob_retired(age))


def _granular_transition() -> dict[str, MarkovTransition]:
    return {
        "work": MarkovTransition(_prob_work),
        "retired": MarkovTransition(_prob_retired),
        "dead": MarkovTransition(_prob_dead),
    }


def _build_regime(**overrides: Any) -> UserRegime:
    spec: dict[str, Any] = {
        "regime_transitions": until_exit(
            2, law=_granular_transition(), exits=("retired", "dead")
        ),
        "states": {"wealth": LinSpacedGrid(start=1.0, stop=100.0, n_points=10)},
        "state_transitions": {"wealth": _next_wealth},
        "actions": {"consumption": LinSpacedGrid(start=1.0, stop=10.0, n_points=5)},
        "functions": {"utility": _utility},
    }
    spec.update(overrides)
    return UserRegime(**spec)


def _build_model(*, work: UserRegime, retired: UserRegime | None = None) -> Model:
    # `retired` outlives `work` by one age so that the mass `work` sends it in
    # its final transition lands on an active regime, and hands everything to
    # `dead` in its own final transition.
    if retired is None:
        retired = _build_regime(
            regime_transitions=until_exit(
                3,
                law={
                    "retired": MarkovTransition(
                        lambda age: jnp.where(age < 2, 0.5, 0.0),
                    ),
                    "dead": MarkovTransition(
                        lambda age: jnp.where(age < 2, 0.5, 1.0),
                    ),
                },
                exits=("dead",),
            ),
        )
    dead = UserRegime(
        regime_transitions=None,
        functions={"utility": lambda: 0.0},
    )
    return Model(
        regimes={"work": work, "retired": retired, "dead": dead},
        ages=AgeGrid(start=0, stop=3, step="Y"),
        regime_id_class=_RegimeId,
    )


def test_granular_transition_solves_and_simulates() -> None:
    """A model with per-target regime transitions solves and simulates; the
    realized memberships are drawn from the declared distribution."""
    model = _build_model(work=_build_regime())
    result = model.simulate(
        params={
            "work": {"discount_factor": 0.95},
            "retired": {"discount_factor": 0.95},
        },
        initial_conditions={
            "age": jnp.zeros(40),
            "wealth": jnp.full(40, 50.0),
            "regime_id": jnp.full(40, _RegimeId.work),
        },
        log_level="debug",
        seed=7,
    )
    df = result.to_dataframe()
    last = df.loc[df["period"] == 2, "regime_name"]
    # From period 1 on, work assigns zero probability to itself.
    assert set(last) <= {"retired", "dead"}


def test_template_has_per_target_regime_transition_keys() -> None:
    """Each granular cell's parameters surface under the target's branch."""

    def _prob_dead_with_param(*, age: int, hazard: float) -> ScalarFloat:
        return jnp.clip(hazard * (1.0 + age), 0.0, 1.0)

    work = _build_regime(
        regime_transitions=until_exit(
            3,
            law={
                "work": MarkovTransition(
                    lambda age, hazard: (
                        1.0 - _prob_dead_with_param(age=age, hazard=hazard)
                    )
                ),
                "dead": MarkovTransition(_prob_dead_with_param),
            },
            exits=("dead",),
        ),
    )
    model = _build_model(work=work)
    template = model.get_params_template()
    assert "hazard" in template["work"]["dead"]["next_regime"]
    assert "hazard" in template["work"]["work"]["next_regime"]


def test_plain_callable_cell_is_rejected() -> None:
    """Granular cells must be `MarkovTransition`-wrapped."""
    with pytest.raises(RegimeInitializationError, match=r"MarkovTransition"):
        _build_regime(
            regime_transitions={
                "work": lambda age: 1.0,  # noqa: ARG005
            },
        )


def test_empty_granular_dict_is_rejected() -> None:
    """`regime_transitions={}` is not the terminal spelling; terminality is `None`."""
    with pytest.raises(RegimeInitializationError, match=r"regime_transitions=None"):
        _build_regime(regime_transitions={})


def test_unknown_target_in_granular_dict_raises() -> None:
    """Every granular key must name a regime of the model."""
    work = _build_regime(
        regime_transitions={
            "work": MarkovTransition(lambda age: jnp.asarray(0.5)),  # noqa: ARG005
            "valhalla": MarkovTransition(lambda age: jnp.asarray(0.5)),  # noqa: ARG005
        },
    )
    with pytest.raises(ModelInitializationError, match=r"valhalla"):
        _build_model(work=work)


def test_phased_granular_with_differing_key_sets_raises() -> None:
    """Phase-variant reachability is rejected: simulate could realize a jump
    into a regime whose continuation solve never planned over."""
    with pytest.raises(RegimeInitializationError, match=r"key set|targets"):
        _build_regime(
            regime_transitions=Phased(
                solve={
                    "work": MarkovTransition(lambda age: jnp.asarray(1.0)),  # noqa: ARG005
                },
                simulate={
                    "work": MarkovTransition(lambda age: jnp.asarray(0.5)),  # noqa: ARG005
                    "dead": MarkovTransition(lambda age: jnp.asarray(0.5)),  # noqa: ARG005
                },
            ),
        )


def test_uncovered_reachable_target_raises_with_remedy() -> None:
    """A per-target state law must cover every reachable target carrying the
    state; the error points to the granular transition spelling."""
    work = _build_regime(
        state_transitions={"wealth": {"work": _next_wealth}},
        # Declares retired as reachable.
        regime_transitions=until_exit(
            2, law=_granular_transition(), exits=("retired", "dead")
        ),
    )
    with pytest.raises(
        ModelInitializationError, match=r"retired.*wealth|wealth.*retired"
    ):
        _build_model(work=work)


def test_granular_keys_narrow_reachability() -> None:
    """A per-target state law covering exactly the declared targets is valid —
    the granular key set, not coverage inference, decides reachability."""
    work = _build_regime(
        regime_transitions=until_exit(
            3,
            law={
                "retired": MarkovTransition(lambda age: jnp.asarray(0.7)),  # noqa: ARG005
                "dead": MarkovTransition(lambda age: jnp.asarray(0.3)),  # noqa: ARG005
            },
            exits=("dead",),
        ),
        state_transitions={"wealth": {"retired": _next_wealth}},
    )
    model = _build_model(work=work)
    assert "work" in model.user_regimes


def test_per_target_regime_transition_template_nests_under_target() -> None:
    """Granular cells' params nest under the target in the template."""
    work = _build_regime()
    model = _build_model(work=work)
    template = model.get_params_template()
    assert "next_regime" in template["work"]["retired"]
