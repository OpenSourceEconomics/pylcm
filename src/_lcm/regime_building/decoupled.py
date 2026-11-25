"""Reduce over actions for a regime that chooses with one aggregator and stores another.

A `Decoupled` Koopmans aggregator gives each cell two profiles from one
kernel evaluation: `Q_objective`, which the action maximizes, and `Q`, which is
stored and published at that action. The reductions here mirror the dense ones
in `max_Q_over_a` for that pair:

- `argmax_and_readout` is the hard-max reduction at one cell, and
  `decoupled_logsum` the extreme-value one;
- `get_decoupled_Q_over_a` is the reduction over the action product, as
  simulation and policy lookup run it;
- `get_decoupled_max_Q_over_a` maps it over the state product, with the
  same tiling `get_max_Q_over_a` takes, as the solve runs it.

A household's profiles carry a trailing member axis. Under the hard maximum it
chooses on the Pareto-weighted sum of its members' objectives and publishes each
member's stored value at that choice (`household_argmax_and_readout`). Under EV1
taste shocks its one decider chooses on its own objective, and every other
member's value is the mean of its stored profile under the decider's choice
probabilities, as in `decider_smoothed_readout`.
"""

import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import ClassVar, NamedTuple, cast

import jax.numpy as jnp
from dags import with_signature
from jax import Array

from _lcm.logsum import logsum_and_softmax
from _lcm.regime_building.argmax import (
    argmax_and_max,
    max_lexicographic_and_smallest_id,
)
from _lcm.regime_building.collective import (
    ParetoWeights,
    _weighted_sum,
    decider_choice,
)
from _lcm.regime_building.max_Q_over_a import (
    TASTE_SHOCK_SCALE_PARAM,
    BoundedArgmaxOutput,
    BoundedMaxOutput,
    _get_extra_param_names,
    _stacked_pareto_weights,
    _with_bound_summary,
    draw_taste_shock_noise,
)
from _lcm.regime_building.Q_and_F import (
    BoundedQAndF,
    QAndFParts,
    get_Q_and_F_parts,
    runtime_bound_specs,
)
from _lcm.simulation.program_types import (
    ACTION_INDEX,
    ACTION_PARTS_ROLES,
    CHOICE_PROBABILITIES,
    CONDITIONAL_ACTION_INDEX,
    CONDITIONAL_VALUE,
    DECISION_VALUE,
    OBJECTIVE_PROFILE,
)
from _lcm.solution.feasibility import BoundedResult
from _lcm.typing import (
    ActionName,
    MaxQOverAFunction,
    PRNGKeyND,
    QAndFArg,
    QAndFKwargs,
    RegimeName,
    StateName,
)
from _lcm.utils.dispatchers import productmap, tiled_productmap
from _lcm.zero_safe import zero_safe_weighted_term
from lcm.typing import BoolND, FloatND, IntND


def argmax_and_readout(
    *, selection: FloatND, readout: FloatND, where: BoolND
) -> tuple[IntND, FloatND]:
    """Choose the first feasible maximizer of `selection` and read `readout` there.

    Both arrays share one shape and are reduced over all axes:

    - the index is the flat index of the first feasible maximizer of `selection`;
    - the value is `readout` at that index;
    - no feasible element gives index `-1` and value `-inf`;
    - a NaN at any feasible element, in either array, makes the value NaN.

    Args:
        selection: The values the element is chosen on.
        readout: The values published at the chosen element.
        where: Feasibility of each element.

    Returns:
        The flat index of the chosen element and the readout there.

    """
    index, _ = argmax_and_max(a=selection, where=where, initial=-jnp.inf)
    any_feasible = jnp.any(where)
    feasible_nan = jnp.any(where & (jnp.isnan(selection) | jnp.isnan(readout)))
    value = jnp.where(any_feasible, jnp.ravel(readout)[index], -jnp.inf)
    return (
        jnp.where(any_feasible, index, -1).astype(jnp.int32),
        jnp.where(feasible_nan, jnp.nan, value),
    )


def household_argmax_and_readout(
    *, selection: FloatND, readout: FloatND, where: BoolND, weights: FloatND
) -> tuple[IntND, FloatND]:
    """Choose on the Pareto-weighted objectives and read each member's value there.

    `selection` and `readout` carry a trailing member axis that `where` does
    not; the action axes are reduced:

    - the household chooses on the weighted sum of `selection`; ties go to each
      member's objective, then to each member's stored value, both in member
      order, and then to the smallest flat index, so the published values do not
      depend on the order in which the actions are declared;
    - each member's value is its `readout` at that index;
    - no feasible element gives index `-1` and value `-inf`;
    - a NaN household objective at a feasible element makes every value NaN and
      the index zero, and a NaN in a member's own profiles at a feasible element
      makes its value NaN.

    Args:
        selection: Each member's objective profile.
        readout: Each member's stored profile.
        where: Feasibility of each element.
        weights: The Pareto weights in member order.

    Returns:
        The flat index of the chosen element and each member's readout there.

    """
    n_members = weights.shape[0]
    objectives = jnp.reshape(selection, (-1, n_members)).T
    stored = jnp.reshape(readout, (-1, n_members)).T
    feasible = jnp.ravel(jnp.broadcast_to(where, selection.shape[:-1]))
    objective = _weighted_sum(
        member_Q={str(m): objectives[m] for m in range(n_members)},
        weights={str(m): weights[m] for m in range(n_members)},
    )
    values, index = max_lexicographic_and_smallest_id(
        values=(objective, *objectives, *stored),
        ids=jnp.arange(feasible.size, dtype=jnp.int32),
        where=feasible,
    )
    empty = ~jnp.any(feasible)
    household_nan = jnp.any(feasible & jnp.isnan(objective))
    member_nan = jnp.any(feasible & (jnp.isnan(objectives) | jnp.isnan(stored)), axis=1)
    value = jnp.where(empty, -jnp.inf, jnp.stack(values[-n_members:]))
    return (
        jnp.where(empty, -1, jnp.where(household_nan, 0, index)).astype(jnp.int32),
        jnp.where(household_nan | member_nan, jnp.nan, value),
    )


def get_decoupled_Q_over_a(
    *,
    Q_and_F: Callable[..., tuple[FloatND, BoolND]],
    action_names: tuple[ActionName, ...],
    state_names: tuple[StateName, ...],
    publishes_index: bool,
    n_discrete_action_axes: int = 0,
    has_taste_shocks: bool = False,
    member_axis: tuple[str, ...] | None = None,
    pareto_weights: ParetoWeights | None = None,
) -> Callable[..., BoundedArgmaxOutput | BoundedMaxOutput]:
    """Get the reduction over actions of a `Decoupled` regime, at one cell.

    Args:
        Q_and_F: A kernel built by `get_Q_and_F` with an objective aggregator.
        action_names: Tuple of action names, discrete first.
        state_names: Tuple of state names.
        publishes_index: Whether to return `(index, value)`, as simulation
            needs, instead of `(value, empty)`, as the counted solve needs.
            With taste shocks the index is a draw, keyed by `taste_shock_key`.
        n_discrete_action_axes: Number of leading discrete-action axes, which
            taste shocks smooth over.
        has_taste_shocks: Whether the regime declares EV1 taste shocks.
        member_axis: Ordered member names of a household, or `None`.
        pareto_weights: The household's Pareto weight evaluator, or `None`.

    Returns:
        The reduction, taking the kernel's arguments.

    """
    extra_param_names = _get_extra_param_names(
        Q_and_F=Q_and_F, action_names=action_names, state_names=state_names
    )
    bounded = bool(runtime_bound_specs(function=Q_and_F))
    parts = productmap(
        func=BoundedQAndF(function=Q_and_F, parts=True)
        if bounded
        else get_Q_and_F_parts(Q_and_F=Q_and_F),
        variables=action_names,
        batch_sizes=dict.fromkeys(action_names, 0),
        bound_summary=bounded,
    )
    return with_signature(
        _DecoupledQOverA(
            parts=cast("Callable[..., QAndFParts | BoundedResult[QAndFParts]]", parts),
            publishes_index=publishes_index,
            n_discrete_action_axes=n_discrete_action_axes if has_taste_shocks else None,
            member_axis=member_axis,
            pareto_weights=pareto_weights,
        ),
        args=[
            "next_regime_to_V_arr",
            *(("taste_shock_key",) if has_taste_shocks and publishes_index else ()),
            *action_names,
            *state_names,
            *extra_param_names,
        ],
        return_annotation=(
            "tuple[IntND, FloatND]" if publishes_index else "tuple[FloatND, BoolND]"
        ),
        enforce=False,
    )


def get_decoupled_max_Q_over_a(
    *,
    Q_and_F: Callable[..., tuple[FloatND, BoolND]],
    batch_sizes: dict[StateName, int],
    action_names: tuple[ActionName, ...],
    state_names: tuple[StateName, ...],
    cell_width_keyword: str | None = None,
    untiled_state_names: tuple[StateName, ...] = (),
    broadcast_state_names: tuple[StateName, ...] = (),
    n_discrete_action_axes: int = 0,
    has_taste_shocks: bool = False,
    member_axis: tuple[str, ...] | None = None,
    pareto_weights: ParetoWeights | None = None,
) -> MaxQOverAFunction[BoundedMaxOutput]:
    """Get the solve's value function of a `Decoupled` regime.

    Maps `get_decoupled_Q_over_a` over the state product exactly as
    `get_max_Q_over_a` does, and returns `(V, empty)` with `empty`
    marking the cells without a feasible action.
    """
    cell = get_decoupled_Q_over_a(
        Q_and_F=Q_and_F,
        action_names=action_names,
        state_names=state_names,
        publishes_index=False,
        n_discrete_action_axes=n_discrete_action_axes,
        has_taste_shocks=has_taste_shocks,
        member_axis=member_axis,
        pareto_weights=pareto_weights,
    )
    mapped = (
        tiled_productmap(
            func=cell,
            variables=state_names,
            width_keyword=cell_width_keyword,
            untiled_variables=untiled_state_names,
            broadcast_variables=broadcast_state_names,
            bound_summary=bool(runtime_bound_specs(function=Q_and_F)),
        )
        if cell_width_keyword is not None
        else productmap(
            func=cell,
            variables=state_names,
            batch_sizes={name: batch_sizes[name] for name in state_names},
            bound_summary=bool(runtime_bound_specs(function=Q_and_F)),
        )
    )
    return cast("MaxQOverAFunction[BoundedMaxOutput]", mapped)


@dataclass(frozen=True, kw_only=True, eq=False)
class _DecoupledQOverA:
    """The stored value at the objective's choice, at one state cell.

    The caller wraps it in `dags.with_signature` to advertise the arguments the
    state productmap binds by name.
    """

    __name__: ClassVar[str] = "decoupled_Q_over_a"
    """Name `dags` reads off the callable when it reports an invalid argument."""

    parts: Callable[..., QAndFParts | BoundedResult[QAndFParts]]
    """The regime's kernel parts, mapped over the action product."""

    publishes_index: bool
    """Whether to return the chosen flat index in place of the empty flag."""

    n_discrete_action_axes: int | None
    """Leading discrete-action axes the taste shocks smooth over, or `None`
    for the hard maximum."""

    member_axis: tuple[str, ...] | None = None
    """Ordered member names of a household, or `None`."""

    pareto_weights: ParetoWeights | None = None
    """The household's Pareto weight evaluator, or `None`."""

    def __call__(
        self,
        next_regime_to_V_arr: Mapping[RegimeName, FloatND],
        **states_actions_params: QAndFArg | PRNGKeyND,
    ) -> BoundedArgmaxOutput | BoundedMaxOutput:
        """Return `(index, value)` or `(value, empty)` at this cell."""
        taste_shock_key = states_actions_params.pop("taste_shock_key", None)
        evaluated = self.parts(
            next_regime_to_V_arr=next_regime_to_V_arr,
            **states_actions_params,
        )
        parts = cast(
            "QAndFParts",
            evaluated.output if type(evaluated) is BoundedResult else evaluated,
        )
        empty = ~jnp.any(parts.F)
        weights = _household_weights(
            member_axis=self.member_axis,
            pareto_weights=self.pareto_weights,
            states_actions_params=cast("QAndFKwargs", states_actions_params),
        )
        if self.n_discrete_action_axes is None:
            index, value = (
                argmax_and_readout(
                    selection=cast("FloatND", parts.Q_objective),
                    readout=parts.Q,
                    where=parts.F,
                )
                if weights is None
                else household_argmax_and_readout(
                    selection=cast("FloatND", parts.Q_objective),
                    readout=parts.Q,
                    where=parts.F,
                    weights=weights,
                )
            )
            return _with_bound_summary(
                evaluated=evaluated,
                output=(index, value) if self.publishes_index else (value, empty),
            )
        scale = cast("FloatND", states_actions_params[TASTE_SHOCK_SCALE_PARAM])
        smoothed = smoothed_profiles(
            parts=parts,
            weights=weights,
            n_discrete_action_axes=self.n_discrete_action_axes,
        )
        conditional = smoothed.conditional
        if not self.publishes_index:
            value, _ = smoothed_logsum(smoothed=smoothed, scale=scale)
            return _with_bound_summary(evaluated=evaluated, output=(value, empty))
        # One draw per flat discrete combination, as the one-profile route
        # draws, so the same key gives the same shocks.
        noise = draw_taste_shock_noise(
            key=cast("Array", taste_shock_key),
            shape=(conditional.objective.size,),
            scale=scale,
        )
        discrete = jnp.argmax(jnp.ravel(conditional.objective) + noise)
        n_continuous = math.prod(parts.F.shape[self.n_discrete_action_axes :])
        flat_index = (
            discrete * n_continuous + jnp.ravel(conditional.continuous_index)[discrete]
        )
        return _with_bound_summary(
            evaluated=evaluated,
            output=(
                jnp.where(empty, -1, flat_index).astype(jnp.int32),
                jnp.where(
                    empty, -jnp.inf, _stored_at(smoothed=smoothed, flat=discrete)
                ),
            ),
        )


def _household_weights(
    *,
    member_axis: tuple[str, ...] | None,
    pareto_weights: ParetoWeights | None,
    states_actions_params: QAndFKwargs,
) -> FloatND | None:
    """Evaluate a household's Pareto weights in member order; `None` for a single."""
    if member_axis is None:
        return None
    return _stacked_pareto_weights(
        pareto_weights=pareto_weights,
        member_axis=member_axis,
        states_actions_params=states_actions_params,
    )


class ConditionalProfiles(NamedTuple):
    """Both profiles per discrete-action combination, at the objective's choice."""

    objective: FloatND
    """Max of the masked objective profile over the continuous actions; `-inf`
    for a combination without a feasible continuous action."""

    value: FloatND
    """The value profile at that continuous choice."""

    continuous_index: IntND
    """Flat index of that continuous choice; zero where none is feasible."""


def get_conditional_profiles(
    *, parts: QAndFParts, n_discrete_action_axes: int
) -> ConditionalProfiles:
    """Choose the continuous actions on the objective profile, per combination."""
    shape = parts.F.shape[:n_discrete_action_axes]
    objective = jnp.where(
        parts.F, cast("FloatND", parts.Q_objective), -jnp.inf
    ).reshape(*shape, -1)
    continuous_index = jnp.argmax(objective, axis=-1)
    return ConditionalProfiles(
        objective=jnp.take_along_axis(objective, continuous_index[..., None], axis=-1)[
            ..., 0
        ],
        value=jnp.where(
            jnp.any(parts.F.reshape(*shape, -1), axis=-1),
            jnp.take_along_axis(
                parts.Q.reshape(*shape, -1), continuous_index[..., None], axis=-1
            )[..., 0],
            -jnp.inf,
        ),
        continuous_index=continuous_index.astype(jnp.int32),
    )


class SmoothedProfiles(NamedTuple):
    """The profiles EV1 shocks smooth, per discrete-action combination."""

    conditional: ConditionalProfiles
    """The chooser's conditional profiles: a single's, or a household decider's."""

    stored: FloatND
    """The stored value at each combination's choice; a household's carries the
    trailing member axis, `-inf` where a combination has no feasible action."""

    parts: QAndFParts
    """The chooser's own kernel parts."""

    decider: IntND | None
    """Member index of a household's decider, the member with the positive
    weight; `None` for a single."""


def smoothed_profiles(
    *, parts: QAndFParts, weights: FloatND | None, n_discrete_action_axes: int
) -> SmoothedProfiles:
    """Choose the continuous actions on the chooser's objective, per combination.

    A single chooses on its own profiles. A household's decider chooses on its
    own objective, and every member's stored value is read at that choice.
    """
    if weights is None:
        conditional = get_conditional_profiles(
            parts=parts, n_discrete_action_axes=n_discrete_action_axes
        )
        return SmoothedProfiles(
            conditional=conditional, stored=conditional.value, parts=parts, decider=None
        )
    shape = parts.F.shape[:n_discrete_action_axes]
    n_members = parts.Q.shape[-1]
    decider, objective, continuous_index, _ = decider_choice(
        Q=cast("FloatND", parts.Q_objective),
        feasibility=parts.F,
        weights=weights,
        n_discrete_action_axes=n_discrete_action_axes,
    )
    feasible = jnp.any(parts.F.reshape(objective.shape[0], -1), axis=1)
    stored = jnp.take_along_axis(
        parts.Q.reshape(objective.shape[0], -1, n_members),
        continuous_index[:, None, None],
        axis=1,
    )[:, 0]
    stored = jnp.where(
        jnp.isnan(objective)[:, None],
        jnp.nan,
        jnp.where(feasible[:, None], stored, -jnp.inf),
    )
    return SmoothedProfiles(
        conditional=ConditionalProfiles(
            objective=objective.reshape(shape),
            value=jnp.take(stored, decider, axis=-1).reshape(shape),
            continuous_index=continuous_index.reshape(shape),
        ),
        stored=stored.reshape(*shape, n_members),
        parts=parts._replace(
            Q=jnp.take(parts.Q, decider, axis=-1),
            Q_objective=jnp.take(cast("FloatND", parts.Q_objective), decider, axis=-1),
        ),
        decider=decider,
    )


def smoothed_logsum(
    *, smoothed: SmoothedProfiles, scale: FloatND
) -> tuple[FloatND, FloatND]:
    """Return the expected stored value under EV1 shocks and the probabilities.

    The chooser's value is `decoupled_logsum` of its own profiles. Every other
    member of a household gets the mean of its stored value under the decider's
    choice probabilities, which a combination of probability zero leaves out.
    """
    value, probabilities = decoupled_logsum(
        conditional=smoothed.conditional, scale=scale, parts=smoothed.parts
    )
    if smoothed.decider is None:
        return value, probabilities
    means = jnp.sum(
        zero_safe_weighted_term(
            weight=probabilities[..., None],
            value=smoothed.stored,
            subnormal_is_accounted_for=False,
        ),
        axis=tuple(range(probabilities.ndim)),
    )
    n_members = smoothed.stored.shape[-1]
    values = jnp.where(jnp.arange(n_members) == smoothed.decider, value, means)
    return jnp.where(jnp.any(smoothed.parts.F), values, -jnp.inf), probabilities


def _stored_at(*, smoothed: SmoothedProfiles, flat: Array) -> FloatND:
    """Read the stored value at one flat discrete combination."""
    if smoothed.decider is None:
        return jnp.ravel(smoothed.stored)[flat]
    n_members = smoothed.stored.shape[-1]
    return jnp.reshape(smoothed.stored, (-1, n_members))[flat]


def decoupled_logsum(
    *, conditional: ConditionalProfiles, scale: FloatND, parts: QAndFParts
) -> tuple[FloatND, FloatND]:
    """Return the expected stored value under EV1 shocks and the probabilities.

    With `P = softmax(objective / scale)` over the feasible combinations, the
    value is `scale * logsumexp(objective / scale) + sum_k P_k (value_k -
    objective_k)`: the expected value of the stored profile at the noisy choice,
    shock included. It is evaluated as `sum_k P_k value_k + scale * H(P)`, the
    same quantity with `H` the entropy of `P`, so the objective's level never
    enters and cannot cancel the stored value in rounding. A feasible
    combination has positive mass even where its probability underflows, so
    an infinite value there makes the result infinite (NaN for both signs). No
    feasible combination gives `-inf`; a NaN at any feasible action, in either
    profile, gives NaN. Values are masked before they are weighted, so an
    infeasible `-inf` cannot turn a gradient into NaN.
    """
    _, probabilities = logsum_and_softmax(
        values=conditional.objective,
        scale=scale,
        axes=tuple(range(conditional.objective.ndim)),
    )
    chosen = probabilities > 0
    # Mask before multiplying so an inactive -inf cannot poison the adjoint.
    safe_value = jnp.where(chosen, conditional.value, 0.0)
    expected_value = jnp.sum(jnp.where(chosen, probabilities * safe_value, 0.0))
    entropy = -jnp.sum(
        jnp.where(
            chosen, probabilities * jnp.log(jnp.where(chosen, probabilities, 1.0)), 0.0
        )
    )
    value = jnp.where(jnp.any(chosen), expected_value + scale * entropy, -jnp.inf)
    # A finite objective has strictly positive EV1 mass even if its floating
    # probability underflows. Such mass cannot discard an infinite readout.
    support = jnp.isfinite(conditional.objective)
    has_negative_infinity = jnp.any(support & jnp.isneginf(conditional.value))
    has_positive_infinity = jnp.any(support & jnp.isposinf(conditional.value))
    value = jnp.where(
        has_negative_infinity,
        jnp.where(has_positive_infinity, jnp.nan, -jnp.inf),
        jnp.where(has_positive_infinity, jnp.inf, value),
    )
    feasible_nan = jnp.any(
        parts.F & (jnp.isnan(cast("FloatND", parts.Q_objective)) | jnp.isnan(parts.Q))
    )
    return jnp.where(feasible_nan, jnp.nan, value), probabilities


# What the dense lookup of a `Decoupled` regime publishes, in the order its
# body returns them: the objective's pair, the kernel's parts over the action
# product, then the objective profile.
DECOUPLED_LOOKUP_OUTPUT_ROLES = (
    ACTION_INDEX,
    DECISION_VALUE,
    *ACTION_PARTS_ROLES,
    OBJECTIVE_PROFILE,
)


@dataclass(frozen=True, kw_only=True, eq=False)
class DecoupledWithActionParts:
    """The dense lookup body of a `Decoupled` regime at one state cell.

    Returns the chosen flat index and the stored value there, followed by `Q`,
    `F`, `U`, `CE` and `Q_objective` over the action product, all from one
    evaluation of the regime's kernel.
    """

    __name__: ClassVar[str] = "decoupled_Q_over_a_with_action_parts"
    """Name `dags` reads off the callable when it reports an invalid argument."""

    parts: Callable[..., QAndFParts]
    """The kernel's parts, mapped over the action product."""

    member_axis: tuple[str, ...] | None = None
    """Ordered member names of a household, or `None`."""

    pareto_weights: ParetoWeights | None = None
    """The household's Pareto weight evaluator, or `None`."""

    def __call__(
        self, **kwargs: QAndFArg
    ) -> tuple[IntND, FloatND, FloatND, BoolND, FloatND, FloatND, FloatND]:
        """Return the index, the value, and `Q`, `F`, `U`, `CE`, `Q_objective`."""
        parts = self.parts(**kwargs)
        Q_objective = cast("FloatND", parts.Q_objective)
        weights = _household_weights(
            member_axis=self.member_axis,
            pareto_weights=self.pareto_weights,
            states_actions_params=cast("QAndFKwargs", kwargs),
        )
        index, value = (
            argmax_and_readout(selection=Q_objective, readout=parts.Q, where=parts.F)
            if weights is None
            else household_argmax_and_readout(
                selection=Q_objective, readout=parts.Q, where=parts.F, weights=weights
            )
        )
        return index, value, parts.Q, parts.F, parts.U, parts.CE, Q_objective


# What the EV1 lookup of a `Decoupled` regime publishes, in the order its
# body returns them.
DECOUPLED_TASTE_SHOCK_LOOKUP_OUTPUT_ROLES = (
    DECISION_VALUE,
    CONDITIONAL_VALUE,
    CHOICE_PROBABILITIES,
    CONDITIONAL_ACTION_INDEX,
    *ACTION_PARTS_ROLES,
    OBJECTIVE_PROFILE,
)


@dataclass(frozen=True, kw_only=True, eq=False)
class DecoupledTasteShockLookup:
    """The EV1 lookup body of a `Decoupled` regime at one state cell.

    Returns the expected stored value, the stored profile `Qc` and the choice
    probability of each discrete-action combination, the flat index of the
    continuous action chosen there, and `Q`, `F`, `U`, `CE` and `Q_objective`
    over the action product.
    """

    __name__: ClassVar[str] = "decoupled_logsum_Q_over_a_with_action_parts"
    """Name `dags` reads off the callable when it reports an invalid argument."""

    parts: Callable[..., QAndFParts]
    """The kernel's parts, mapped over the action product."""

    n_discrete_action_axes: int
    """Number of leading discrete-action axes of the action product."""

    member_axis: tuple[str, ...] | None = None
    """Ordered member names of a household, or `None`."""

    pareto_weights: ParetoWeights | None = None
    """The household's Pareto weight evaluator, or `None`."""

    def __call__(self, **kwargs: QAndFArg) -> tuple[FloatND | IntND | BoolND, ...]:
        """Return the value, `Qc`, the probabilities, the index and the parts.

        A household's value and `Qc` carry the trailing member axis.
        """
        parts = self.parts(**kwargs)
        smoothed = smoothed_profiles(
            parts=parts,
            weights=_household_weights(
                member_axis=self.member_axis,
                pareto_weights=self.pareto_weights,
                states_actions_params=cast("QAndFKwargs", kwargs),
            ),
            n_discrete_action_axes=self.n_discrete_action_axes,
        )
        value, probabilities = smoothed_logsum(
            smoothed=smoothed, scale=cast("FloatND", kwargs[TASTE_SHOCK_SCALE_PARAM])
        )
        return (
            value,
            smoothed.stored,
            probabilities,
            smoothed.conditional.continuous_index,
            parts.Q,
            parts.F,
            parts.U,
            parts.CE,
            cast("FloatND", parts.Q_objective),
        )
