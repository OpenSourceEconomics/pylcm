"""Target-name introspection and resolution for `to_dataframe`.

The set of "additional targets" a regime exposes — every user function and
constraint that is not the aggregator `H` or an internal stochastic-weight
helper — is computed once here and consumed in two places:

- `simulate` evaluates exactly this set at the realised optimal actions and
  stores the per-subject outputs in `PeriodRegimeSimulationData.intermediates`.
- `SimulationResult.available_targets` reports the union across regimes.

Keeping both on the same helper guarantees the names a result advertises are
exactly the names its `intermediates` carry.
"""

from collections.abc import Mapping
from typing import Literal

from _lcm.engine import Regime
from _lcm.typing import RegimeName
from lcm.exceptions import InvalidAdditionalTargetsError


def _resolve_targets(
    *,
    additional_targets: list[str] | Literal["all"] | None,
    available_targets: list[str],
) -> list[str] | None:
    """Resolve and validate additional targets.

    Args:
        additional_targets: User-provided targets specification.
        available_targets: List of all available target names.

    Returns:
        Resolved list of target names, or None if no targets requested.

    Raises:
        InvalidAdditionalTargetsError: If any target is not available.

    """
    if additional_targets is None:
        return None
    if additional_targets == "all":
        return available_targets

    invalid = set(additional_targets) - set(available_targets)
    if invalid:
        raise InvalidAdditionalTargetsError(
            f"Targets {invalid} not found in any regime. "
            f"Available targets: {available_targets}"
        )

    return additional_targets


def _collect_all_available_targets(
    regimes: Mapping[RegimeName, Regime],
) -> set[str]:
    """Collect all available target names across all regimes."""
    all_targets: set[str] = set()
    for regime in regimes.values():
        all_targets.update(_target_names_for_regime(regime))
    return all_targets


def _target_names_for_regime(regime: Regime) -> set[str]:
    """Return the DAG outputs a regime exposes as additional targets.

    Every user function and constraint, minus the aggregator `H` and the
    internal stochastic-weight helpers. This is exactly the set evaluated
    per-period in `simulate` and stored in `intermediates`.
    """
    excluded = {"H"} | _stochastic_weight_function_names(regime)
    sim = regime.simulate_functions
    return {
        name for name in sim.functions if name not in excluded
    } | sim.constraints.keys()


def _stochastic_weight_function_names(regime: Regime) -> set[str]:
    """Return the names of internal stochastic-weight functions.

    These are functions named `weight_{target_regime}__{transition}` that
    return probability arrays for stochastic state transitions. They are an
    implementation detail of the transition machinery, not user-facing
    targets.
    """
    sim = regime.simulate_functions
    stochastic_transition_names = sim.stochastic_transition_names
    return {
        f"weight_{target_regime}__{transition_name}"
        for target_regime, target_transitions in sim.transitions.items()
        for transition_name in target_transitions
        if transition_name in stochastic_transition_names
    }
