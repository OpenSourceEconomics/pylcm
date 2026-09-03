"""Composition boundaries for transition-local joint lotteries."""

from collections.abc import Mapping

import jax.numpy as jnp
import numpy as np
import pytest

from _lcm.regime_building.processing import regime_declares_phased
from lcm import (
    AgeGrid,
    JointTransition,
    LinSpacedGrid,
    MarkovTransition,
    Model,
    Phased,
    Regime,
    categorical,
    fixed_transition,
)
from lcm.exceptions import (
    InvalidNameError,
    ModelInitializationError,
    RegimeInitializationError,
)
from lcm.typing import FloatND, ScalarInt, UserParams


@categorical(ordered=False)
class _OneTargetRegimeId:
    source: ScalarInt
    target: ScalarInt


@categorical(ordered=False)
class _TwoTargetRegimeId:
    source: ScalarInt
    target_a: ScalarInt
    target_b: ScalarInt


def _certain_target() -> FloatND:
    return jnp.asarray(1.0)


def _half_target() -> FloatND:
    return jnp.asarray(0.5)


def _joint_probabilities() -> FloatND:
    return jnp.asarray([0.5, 0.5])


def _one_node_probabilities() -> FloatND:
    return jnp.asarray([1.0])


def _read_match(match: Mapping[str, FloatND]) -> FloatND:
    return match["wealth"]


def _next_wealth(matched_wealth: FloatND) -> FloatND:
    return matched_wealth


def _read_income(*, next_wealth: FloatND, match: Mapping[str, FloatND]) -> FloatND:
    return next_wealth + match["income"]


def _next_income(matched_income: FloatND) -> FloatND:
    return matched_income


def _helper_model() -> Model:
    source = Regime(
        transition={"target": MarkovTransition(_certain_target)},
        active=lambda age: age < 1,
        functions={
            "utility": lambda: jnp.asarray(0.0),
            "matched_wealth": _read_match,
            "matched_income": _read_income,
        },
        joint_transitions={
            "target": {
                "match": JointTransition(
                    support_size=2,
                    support={
                        "wealth": jnp.asarray([0.0, 1.0]),
                        "income": jnp.asarray([0.0, 1.0]),
                    },
                    probabilities=_joint_probabilities,
                    outputs={
                        "wealth": _next_wealth,
                        "income": _next_income,
                    },
                )
            }
        },
    )
    target = Regime(
        transition=None,
        active=lambda age: age >= 1,
        states={
            "wealth": LinSpacedGrid(start=0.0, stop=1.0, n_points=2),
            "income": LinSpacedGrid(start=0.0, stop=2.0, n_points=3),
        },
        functions={"utility": lambda wealth, income: wealth + income},
    )
    return Model(
        regimes={"source": source, "target": target},
        ages=AgeGrid(start=0, stop=1, step="Y"),
        regime_id_class=_OneTargetRegimeId,
        enable_jit=False,
    )


def _helper_params() -> UserParams:
    return {
        "source": {
            "target": {
                "next_regime": {},
                "match": {"support": {}, "probabilities": {}},
                "next_wealth": {},
                "next_income": {},
            },
            "matched_wealth": {},
            "matched_income": {},
            "koopmans_aggregator": {"discount_factor": 1.0},
        },
        "target": {"utility": {}},
    }


def test_joint_output_helpers_inherit_node_and_sibling_output_namespace() -> None:
    """Transition helpers inherit the target's node and next-output namespace."""
    model = _helper_model()

    assert model._params_template["source"]["matched_wealth"] == {}
    assert model._params_template["source"]["matched_income"] == {}

    solution = model.solve(params=_helper_params(), log_level="debug").values

    # Nodes are (wealth, income) = (0, 0) and (1, 2), so E[V] = 1.5.
    np.testing.assert_allclose(np.asarray(solution[0]["source"]), 1.5)


def _next_value(match: FloatND) -> FloatND:
    return match


def _bad_utility(match: FloatND) -> FloatND:
    return match


def _bad_helper(match: FloatND) -> FloatND:
    return match


def _utility_from_bad_helper(bad_helper: FloatND) -> FloatND:
    return bad_helper


@pytest.mark.parametrize(
    "functions",
    [
        {"utility": _bad_utility},
        {"utility": _utility_from_bad_helper, "bad_helper": _bad_helper},
    ],
    ids=["direct", "through-helper"],
)
def test_nontransition_consumers_cannot_read_a_joint_node(
    functions: dict[str, object],
) -> None:
    """A transition-local node cannot be rebound to a user parameter in utility."""
    source = Regime(
        transition={"target": MarkovTransition(_certain_target)},
        active=lambda age: age < 1,
        functions=functions,  # ty: ignore[invalid-argument-type]
        joint_transitions={
            "target": {
                "match": JointTransition(
                    support_size=1,
                    support=jnp.asarray([1.0]),
                    probabilities=_one_node_probabilities,
                    outputs={"value": _next_value},
                )
            }
        },
    )
    target = Regime(
        transition=None,
        active=lambda age: age >= 1,
        states={"value": LinSpacedGrid(start=0.0, stop=1.0, n_points=2)},
        functions={"utility": lambda value: value},
    )

    with pytest.raises(
        InvalidNameError, match=r"utility.*joint.*match|utility.*match.*transition"
    ):
        Model(
            regimes={"source": source, "target": target},
            ages=AgeGrid(start=0, stop=1, step="Y"),
            regime_id_class=_OneTargetRegimeId,
            enable_jit=False,
        )


def _probabilities_reading_match(match: FloatND) -> FloatND:
    return jnp.asarray([1.0 + 0.0 * match])


def test_joint_probabilities_cannot_read_a_joint_node() -> None:
    """Sibling-conditional lotteries remain unsupported and fail at construction."""
    source = Regime(
        transition={"target": MarkovTransition(_certain_target)},
        active=lambda age: age < 1,
        functions={"utility": lambda: jnp.asarray(0.0)},
        joint_transitions={
            "target": {
                "match": JointTransition(
                    support_size=1,
                    support=jnp.asarray([1.0]),
                    probabilities=_probabilities_reading_match,
                    outputs={"value": _next_value},
                )
            }
        },
    )
    target = Regime(
        transition=None,
        active=lambda age: age >= 1,
        states={"value": LinSpacedGrid(start=0.0, stop=1.0, n_points=2)},
        functions={"utility": lambda value: value},
    )

    with pytest.raises(
        InvalidNameError,
        match=r"probabilit.*match.*lotter|probabilit.*joint.*match",  # codespell:ignore
    ):
        Model(
            regimes={"source": source, "target": target},
            ages=AgeGrid(start=0, stop=1, step="Y"),
            regime_id_class=_OneTargetRegimeId,
            enable_jit=False,
        )


def _next_a(match_a: FloatND) -> FloatND:
    return match_a


def _next_b(*, match_a: FloatND, match_b: FloatND) -> FloatND:
    return match_a + match_b


def test_joint_node_is_scoped_to_its_declared_target() -> None:
    """A node on source→A is unavailable to an output on source→B."""
    source = Regime(
        transition={
            "target_a": MarkovTransition(_half_target),
            "target_b": MarkovTransition(_half_target),
        },
        active=lambda age: age < 1,
        functions={"utility": lambda: jnp.asarray(0.0)},
        joint_transitions={
            "target_a": {
                "match_a": JointTransition(
                    support_size=1,
                    support=jnp.asarray([1.0]),
                    probabilities=_one_node_probabilities,
                    outputs={"value_a": _next_a},
                )
            },
            "target_b": {
                "match_b": JointTransition(
                    support_size=1,
                    support=jnp.asarray([2.0]),
                    probabilities=_one_node_probabilities,
                    outputs={"value_b": _next_b},
                )
            },
        },
    )
    target_a = Regime(
        transition=None,
        active=lambda age: age >= 1,
        states={"value_a": LinSpacedGrid(start=0.0, stop=2.0, n_points=3)},
        functions={"utility": lambda value_a: value_a},
    )
    target_b = Regime(
        transition=None,
        active=lambda age: age >= 1,
        states={"value_b": LinSpacedGrid(start=0.0, stop=4.0, n_points=5)},
        functions={"utility": lambda value_b: value_b},
    )

    with pytest.raises(
        ModelInitializationError, match=r"match_a.*target_b|target_b.*match_a"
    ):
        Model(
            regimes={
                "source": source,
                "target_a": target_a,
                "target_b": target_b,
            },
            ages=AgeGrid(start=0, stop=1, step="Y"),
            regime_id_class=_TwoTargetRegimeId,
            enable_jit=False,
        )


def _phase_kernel(probabilities: object) -> JointTransition:
    return JointTransition(
        support_size=1,
        support=jnp.asarray([1.0]),
        probabilities=probabilities,  # ty: ignore[invalid-argument-type]
        outputs={"value": _next_value},
    )


def test_regime_declares_phased_sees_nested_joint_transition_variants() -> None:
    """Phase-sensitive policy reuse sees `Phased` nested below target and kernel."""
    regime = Regime(
        transition={"target": MarkovTransition(_certain_target)},
        functions={"utility": lambda: jnp.asarray(0.0)},
        joint_transitions={
            "target": {
                "match": Phased(
                    solve=_phase_kernel(_one_node_probabilities),
                    simulate=_phase_kernel(_one_node_probabilities),
                )
            }
        },
    )

    assert regime_declares_phased(regime)


def test_identity_invariant_nested_joint_transition_is_not_phased() -> None:
    """One shared kernel object is replay-invariant across both phases."""
    kernel = _phase_kernel(_one_node_probabilities)
    regime = Regime(
        transition={"target": MarkovTransition(_certain_target)},
        functions={"utility": lambda: jnp.asarray(0.0)},
        joint_transitions={"target": {"match": Phased(solve=kernel, simulate=kernel)}},
    )

    assert not regime_declares_phased(regime)


def _support_reading_wealth(wealth: FloatND) -> FloatND:
    return jnp.asarray([wealth])


def _support_reading_next_value(next_value: FloatND) -> FloatND:
    return jnp.asarray([next_value])


@pytest.mark.parametrize(
    ("support", "message"),
    [
        (_support_reading_wealth, r"support.*wealth.*state|support.*runtime.*wealth"),
        (
            _support_reading_next_value,
            r"support.*next_value|next_value.*support",
        ),
    ],
    ids=["source-state", "next-output"],
)
def test_joint_support_cannot_read_runtime_transition_values(
    *, support: object, message: str
) -> None:
    """Declared support is hoistable: only period, age, and params may enter it."""
    source = Regime(
        transition={"target": MarkovTransition(_certain_target)},
        active=lambda age: age < 1,
        states={"wealth": LinSpacedGrid(start=0.0, stop=1.0, n_points=2)},
        state_transitions={"wealth": fixed_transition("wealth")},
        functions={"utility": lambda wealth: wealth},
        joint_transitions={
            "target": {
                "match": JointTransition(
                    support_size=1,
                    support=support,
                    probabilities=_one_node_probabilities,
                    outputs={"value": _next_value},
                )
            }
        },
    )
    target = Regime(
        transition=None,
        active=lambda age: age >= 1,
        states={"value": LinSpacedGrid(start=0.0, stop=1.0, n_points=2)},
        functions={"utility": lambda value: value},
    )

    with pytest.raises(InvalidNameError, match=message):
        Model(
            regimes={"source": source, "target": target},
            ages=AgeGrid(start=0, stop=1, step="Y"),
            regime_id_class=_OneTargetRegimeId,
            enable_jit=False,
        )


def _probabilities_reading_next_value(next_value: FloatND) -> FloatND:
    return jnp.asarray([1.0 + 0.0 * next_value])


def test_joint_probabilities_cannot_read_a_next_output() -> None:
    """Weights are formed before output realization and cannot condition on it."""
    source = Regime(
        transition={"target": MarkovTransition(_certain_target)},
        active=lambda age: age < 1,
        functions={"utility": lambda: jnp.asarray(0.0)},
        joint_transitions={
            "target": {
                "match": JointTransition(
                    support_size=1,
                    support=jnp.asarray([1.0]),
                    probabilities=_probabilities_reading_next_value,
                    outputs={"value": _next_value},
                )
            }
        },
    )
    target = Regime(
        transition=None,
        active=lambda age: age >= 1,
        states={"value": LinSpacedGrid(start=0.0, stop=1.0, n_points=2)},
        functions={"utility": lambda value: value},
    )

    with pytest.raises(
        (InvalidNameError, ModelInitializationError),
        match=r"probabilit.*(next_value|match).*(draw|transition)"  # codespell:ignore
        r"|next_value.*probabilit",  # codespell:ignore
    ):
        Model(
            regimes={"source": source, "target": target},
            ages=AgeGrid(start=0, stop=1, step="Y"),
            regime_id_class=_OneTargetRegimeId,
            enable_jit=False,
        )


def _phase_schema_solve_support() -> dict[str, FloatND]:
    return {"value": jnp.asarray([1.0, 2.0])}


def _phase_schema_simulate_support() -> dict[str, FloatND]:
    return {"value": jnp.asarray([1, 2], dtype=jnp.int32)}


def _phase_schema_output(match: dict[str, FloatND]) -> FloatND:
    return jnp.asarray(match["value"], dtype=float)


def test_callable_phased_support_keeps_one_static_schema() -> None:
    """Params-bound preflight compares callable support schemas across phases."""
    source = Regime(
        transition={"target": MarkovTransition(_certain_target)},
        active=lambda age: age < 1,
        functions={"utility": lambda: jnp.asarray(0.0)},
        joint_transitions={
            "target": {
                "match": Phased(
                    solve=JointTransition(
                        support_size=2,
                        support=_phase_schema_solve_support,
                        probabilities=_joint_probabilities,
                        outputs={"wealth": _phase_schema_output},
                    ),
                    simulate=JointTransition(
                        support_size=2,
                        support=_phase_schema_simulate_support,
                        probabilities=_joint_probabilities,
                        outputs={"wealth": _phase_schema_output},
                    ),
                )
            }
        },
    )
    target = Regime(
        transition=None,
        active=lambda age: age >= 1,
        states={"wealth": LinSpacedGrid(start=0.0, stop=3.0, n_points=4)},
        functions={"utility": lambda wealth: wealth},
    )
    model = Model(
        regimes={"source": source, "target": target},
        ages=AgeGrid(start=0, stop=1, step="Y"),
        regime_id_class=_OneTargetRegimeId,
        enable_jit=False,
    )

    with pytest.raises(
        RegimeInitializationError,
        match=r"support changed.*static pytree signature.*dtypes",
    ):
        model.solve(
            params={
                "source": {
                    "target": {
                        "next_regime": {},
                        "match": {"support": {}, "probabilities": {}},
                        "next_wealth": {},
                    },
                    "koopmans_aggregator": {"discount_factor": 1.0},
                },
                "target": {"utility": {}},
            },
            log_level="debug",
        )
