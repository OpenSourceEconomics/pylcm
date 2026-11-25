"""Policy lookup's decision in a regime with EV1 taste shocks.

The body applies the solve's reduction, in the solve's order, to the simulation
decision: it masks the kernel's `Q` with `F`, takes the max over the
continuous-action axes for each discrete-action combination (`Qc`), and passes
`Qc` to `logsum_and_softmax` over the discrete axes. Its `Q` uses the simulate
phase's current utility and aggregator with the solved continuation, so the
smoothed maximum equals the stored solve value at grid rows when both phases
share those primitives and the action grid is the declared one.

A household's body reduces on its decider's own action values with the solve's
household readout, so it publishes one value per member, and the same parts.
"""

import dataclasses
from collections.abc import Callable
from typing import ClassVar, cast

import jax.numpy as jnp

from _lcm.logsum import logsum_and_softmax
from _lcm.regime_building.collective import (
    ParetoWeights,
    decider_choice,
    decider_smoothed_readout,
)
from _lcm.regime_building.max_Q_over_a import (
    TASTE_SHOCK_SCALE_PARAM,
    _stacked_pareto_weights,
)
from _lcm.regime_building.Q_and_F import QAndFParts
from _lcm.simulation.program_types import (
    ACTION_PARTS_ROLES,
    CHOICE_PROBABILITIES,
    CONDITIONAL_ACTION_INDEX,
    CONDITIONAL_VALUE,
    DECISION_VALUE,
)
from _lcm.typing import PytreeValue, QAndFKwargs
from lcm.typing import BoolND, FloatND, IntND, ScalarFloat

# What the taste-shock lookup publishes, in the order its body returns them.
TASTE_SHOCK_LOOKUP_OUTPUT_ROLES = (
    DECISION_VALUE,
    CONDITIONAL_VALUE,
    CHOICE_PROBABILITIES,
    CONDITIONAL_ACTION_INDEX,
    *ACTION_PARTS_ROLES,
)


@dataclasses.dataclass(frozen=True, kw_only=True, eq=False)
class TasteShockLookup:
    """The EV1 decision at one state cell, with the parts it reduces.

    Returns the smoothed maximum, the conditional value `Qc` and choice
    probability of each discrete-action combination, the flat index of the
    continuous action attaining `Qc` (zero where no continuous action is
    feasible), and the kernel's `Q`, `F`, `U` and `CE` over the action product.
    """

    __name__: ClassVar[str] = "logsum_and_softmax_Q_over_a_with_action_parts"
    """Name `dags` reads off the callable when it reports an invalid argument."""

    parts: Callable[..., QAndFParts]
    """The kernel's parts, mapped over the action product."""

    n_discrete_action_axes: int
    """Number of leading discrete-action axes of the action product."""

    def __call__(
        self, **kwargs: PytreeValue
    ) -> tuple[FloatND, FloatND, FloatND, IntND, FloatND, BoolND, FloatND, FloatND]:
        """Return the value, `Qc`, the probabilities, the index and the parts."""
        parts = self.parts(**kwargs)
        Q_masked = jnp.where(parts.F, parts.Q, -jnp.inf)
        Q_by_combination = Q_masked.reshape(
            *Q_masked.shape[: self.n_discrete_action_axes], -1
        )
        Qc = Q_by_combination.max(axis=-1)
        conditional_index = jnp.argmax(Q_by_combination, axis=-1).astype(jnp.int32)
        value, probabilities = logsum_and_softmax(
            values=Qc,
            scale=cast("ScalarFloat", kwargs[TASTE_SHOCK_SCALE_PARAM]),
            axes=tuple(range(Qc.ndim)),
        )
        return (
            value,
            Qc,
            probabilities,
            conditional_index,
            parts.Q,
            parts.F,
            parts.U,
            parts.CE,
        )


@dataclasses.dataclass(frozen=True, kw_only=True, eq=False)
class HouseholdTasteShockLookup:
    """The EV1 decision of a household's one decider at one state cell.

    The decider is the member with the positive Pareto weight. Returns:

    - one value per member, in member order: the decider's smoothed maximum and
      every other member's mean action value under the decider's choice
      probabilities, as the solve reads them off (`decider_smoothed_readout`);
    - the decider's conditional value `Qc` of each discrete-action combination,
      its maximum over the feasible continuous actions;
    - the choice probability `softmax(Qc / scale)` of each combination, zero
      for a combination whose `Qc` is `-inf`;
    - the flat index of the continuous action attaining `Qc`, which every
      member shares (zero where no continuous action is feasible);
    - the kernel's `Q`, `F`, `U` and `CE` over the action product, each value
      with the trailing member axis.
    """

    __name__: ClassVar[str] = "household_logsum_and_softmax_Q_over_a"
    """Name `dags` reads off the callable when it reports an invalid argument."""

    parts: Callable[..., QAndFParts]
    """The household's parts, with a trailing member axis, mapped over the action
    product."""

    n_discrete_action_axes: int
    """Number of leading discrete-action axes of the action product."""

    member_axis: tuple[str, ...]
    """Ordered member names."""

    pareto_weights: ParetoWeights
    """The household's Pareto weight evaluator."""

    def __call__(
        self, **kwargs: PytreeValue
    ) -> tuple[FloatND, FloatND, FloatND, IntND, FloatND, BoolND, FloatND, FloatND]:
        """Return the member values, `Qc`, the probabilities, the index and parts."""
        parts = self.parts(**kwargs)
        Q_arr, F_arr = parts.Q, parts.F
        weights = _stacked_pareto_weights(
            pareto_weights=self.pareto_weights,
            member_axis=self.member_axis,
            states_actions_params=cast("QAndFKwargs", kwargs),
        )
        scale = cast("ScalarFloat", kwargs[TASTE_SHOCK_SCALE_PARAM])
        values, _dissolution = decider_smoothed_readout(
            Q=Q_arr,
            feasibility=F_arr,
            weights=weights,
            n_discrete_action_axes=self.n_discrete_action_axes,
            scale=scale,
        )
        _decider, Qc, conditional_index, _at_choice = decider_choice(
            Q=Q_arr,
            feasibility=F_arr,
            weights=weights,
            n_discrete_action_axes=self.n_discrete_action_axes,
        )
        _smoothed, probabilities = logsum_and_softmax(values=Qc, scale=scale, axes=(0,))
        discrete_shape = F_arr.shape[: self.n_discrete_action_axes]
        return (
            values,
            Qc.reshape(discrete_shape),
            probabilities.reshape(discrete_shape),
            conditional_index.reshape(discrete_shape),
            parts.Q,
            parts.F,
            parts.U,
            parts.CE,
        )
