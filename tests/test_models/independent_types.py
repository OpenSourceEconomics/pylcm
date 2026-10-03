"""A cake-eating model whose preference type never changes, with a brute-force oracle.

An agent holds integer wealth `0..10` and a preference type that
`fixed_transition` keeps for life. Each working period it consumes an integer
amount no larger than its wealth; the rest carries over unchanged, so every
next-period wealth lands exactly on a grid node and linear interpolation reads
the stored value exactly. At the last age a terminal regime values the wealth
left over. Utility weights and curvatures differ by type:

- working: `weight[type] * (1 + consumption) ** exponent[type]`;
- terminal: `bequest_weight[type] * (1 + wealth) ** exponent[type]`.

Each type is therefore an independent dynamic program over the same wealth
grid. `solve_by_enumeration` solves each one by plain Python enumeration over
integer wealth and consumption, sharing no code with pylcm: it is the oracle
the invariant-state work checks the engine against.
"""

import math
from collections.abc import Mapping, Sequence
from types import MappingProxyType

import jax.numpy as jnp

from lcm import (
    AgeGrid,
    ByAge,
    Choose,
    DiscreteGrid,
    LinSpacedGrid,
    Model,
    categorical,
    fixed_transition,
)
from lcm.regime import Regime as UserRegime
from lcm.typing import (
    BoolND,
    ContinuousAction,
    ContinuousState,
    DiscreteState,
    FloatND,
    ScalarInt,
)

# Integer wealth nodes `0..10`; consumption uses the same nodes.
N_WEALTH_POINTS = 11

# Three working ages and one terminal age.
AGES = AgeGrid(start=0, stop=3, step="Y")

DISCOUNT_FACTOR = 0.9

# Per-type utility parameters, indexed by preference-type code.
TYPE_PARAMS = MappingProxyType(
    {
        "weight": (1.0, 1.7, 0.6),
        "exponent": (0.5, 0.3, 0.8),
        "bequest_weight": (0.4, 1.1, 2.3),
    }
)


@categorical(ordered=False)
class PrefType:
    type_0: ScalarInt
    type_1: ScalarInt
    type_2: ScalarInt


@categorical(ordered=False)
class RegimeId:
    working: ScalarInt
    terminal: ScalarInt


def utility(
    *,
    consumption: ContinuousAction,
    pref_type: DiscreteState,
    weight: FloatND,
    exponent: FloatND,
) -> FloatND:
    return weight[pref_type] * (1.0 + consumption) ** exponent[pref_type]


def terminal_utility(
    *,
    wealth: ContinuousState,
    pref_type: DiscreteState,
    bequest_weight: FloatND,
    exponent: FloatND,
) -> FloatND:
    return bequest_weight[pref_type] * (1.0 + wealth) ** exponent[pref_type]


def next_wealth(
    *, wealth: ContinuousState, consumption: ContinuousAction
) -> ContinuousState:
    return wealth - consumption


def affordable(*, wealth: ContinuousState, consumption: ContinuousAction) -> BoolND:
    return consumption <= wealth


def next_regime() -> ScalarInt:
    return RegimeId.working


_WEALTH = LinSpacedGrid(start=0, stop=N_WEALTH_POINTS - 1, n_points=N_WEALTH_POINTS)


def get_model() -> Model:
    """Create the three-type cake-eating model.

    Returns:
        The model, with `pref_type` held fixed by `fixed_transition`.

    """
    last_age = AGES.exact_values[-1]
    working = UserRegime(
        regime_transitions=ByAge.until(
            stop_age_exclusive=last_age,
            law=Choose(func=next_regime, targets=("working",)),
            then=Choose(func=lambda: RegimeId.terminal, targets=("terminal",)),
        ),
        states={"wealth": _WEALTH, "pref_type": DiscreteGrid(category_class=PrefType)},
        state_transitions={
            "wealth": next_wealth,
            "pref_type": fixed_transition("pref_type"),
        },
        actions={"consumption": _WEALTH},
        functions={"utility": utility},
        constraints={"affordable": affordable},
    )
    terminal = UserRegime(
        regime_transitions=None,
        states={"wealth": _WEALTH, "pref_type": DiscreteGrid(category_class=PrefType)},
        functions={"utility": terminal_utility},
    )
    return Model(
        regimes={"working": working, "terminal": terminal},
        ages=AGES,
        regime_id_class=RegimeId,
        initial_regimes={AGES.exact_values[0]: "working"},
        description="Cake eating with a fixed preference type.",
    )


def get_params(
    *,
    type_params: Mapping[str, Sequence[float]] = TYPE_PARAMS,
    discount_factor: float = DISCOUNT_FACTOR,
) -> dict:
    """Return the parameters `get_model()` solves with.

    Args:
        type_params: Per-type `weight`, `exponent` and `bequest_weight`.
        discount_factor: Common discount factor.

    Returns:
        Parameter mapping ready for `Model.solve`.

    """
    exponent = jnp.asarray(type_params["exponent"])
    return {
        "discount_factor": discount_factor,
        "working": {
            "utility": {
                "weight": jnp.asarray(type_params["weight"]),
                "exponent": exponent,
            }
        },
        "terminal": {
            "utility": {
                "bequest_weight": jnp.asarray(type_params["bequest_weight"]),
                "exponent": exponent,
            }
        },
    }


def solve_by_enumeration(
    *,
    type_params: Mapping[str, Sequence[float]] = TYPE_PARAMS,
    discount_factor: float = DISCOUNT_FACTOR,
) -> tuple[dict[int, list[list[float]]], dict[int, list[list[int]]]]:
    """Solve every type's cake-eating problem by enumerating integer choices.

    Each type is solved on its own, with no quantity shared across types. Ties go
    to the smallest consumption, the engine's lowest-action convention.

    Args:
        type_params: Per-type `weight`, `exponent` and `bequest_weight`.
        discount_factor: Common discount factor.

    Returns:
        Values and optimal consumption by period, each indexed `[type][wealth]`
        like the engine's value arrays; the terminal period has values only.

    """
    n_periods = len(AGES.exact_values)
    values: dict[int, list[list[float]]] = {period: [] for period in range(n_periods)}
    policies: dict[int, list[list[int]]] = {
        period: [] for period in range(n_periods - 1)
    }
    for k in range(len(type_params["weight"])):
        weight = type_params["weight"][k]
        exponent = type_params["exponent"][k]
        following = [
            type_params["bequest_weight"][k] * math.pow(1.0 + w, exponent)
            for w in range(N_WEALTH_POINTS)
        ]
        values[n_periods - 1].append(following)
        for period in range(n_periods - 2, -1, -1):
            value_row: list[float] = []
            policy_row: list[int] = []
            for w in range(N_WEALTH_POINTS):
                best_value = -math.inf
                best_choice = -1
                for c in range(w + 1):
                    candidate = (
                        weight * math.pow(1.0 + c, exponent)
                        + discount_factor * following[w - c]
                    )
                    if candidate > best_value:
                        best_value, best_choice = candidate, c
                value_row.append(best_value)
                policy_row.append(best_choice)
            values[period].append(value_row)
            policies[period].append(policy_row)
            following = value_row
    return values, policies
