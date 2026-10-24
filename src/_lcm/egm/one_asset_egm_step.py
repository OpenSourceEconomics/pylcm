"""One backward EGM step for a 1-D consumption--saving problem.

The retired phase of the DS pension model is a plain consumption--saving problem: a
single continuous state `liquid`, a single continuous action `consumption`, and no
discrete choice. With one continuous state the endogenous grid method needs no upper
envelope — invert the consumption Euler equation

$$u'(c) = \\beta\\, g'(s)\\, V'_{\\text{next}}\\big(g(s)\\big)$$

on a post-decision savings grid $s = \\text{liquid} - c$, read the next-period value and
its marginal off the prior arrays, and map the resulting endogenous liquid back onto the
regular liquid grid. Here $g$ is the regime's own law of motion, taken as the two
readings `next_liquid` and `marginal_return` rather than assumed; for the conventional
law $g(s) = (1+r)s + y$ it is the gross return that discounts the continuation marginal.
At the savings grid's lower bound the borrowing constraint binds and the agent consumes
everything above it.

The step carries the **marginal value of liquid** $V'(\\text{liquid})$ between periods
as the economic envelope approximation $u'(c)$ rather than finite-differencing
the value array. This channel is distinct from the derivative of the represented
value interpolant: interpolated policies need not satisfy the Euler equation exactly
between candidates. Each step consumes `next_marginal` and publishes its associated
consumption marginal.
"""

from collections.abc import Iterable
from typing import TYPE_CHECKING, NamedTuple, cast

import jax
import jax.numpy as jnp

from _lcm.egm.euler import invert_euler
from _lcm.egm.interp import interp_on_regular_grid
from _lcm.egm.preferences import Preferences
from lcm.typing import Bool1D, Float1D, Int1D, ScalarFloat

if TYPE_CHECKING:
    from _lcm.typing import PytreeChild
else:
    # EGM records are imported while the engine type vocabulary is initialized.
    type PytreeChild = object


class RetiredEGMResult(NamedTuple):
    """A retired EGM step's value, marginal value of liquid, and consumption policy."""

    value: Float1D
    """This period's value on `liquid_grid`."""
    marginal: Float1D
    """This period's marginal value of liquid `V' = u'(c*)` on `liquid_grid`."""
    consumption: Float1D
    """This period's optimal consumption on `liquid_grid`."""
    savings: Float1D | None = None
    """Associated original saving candidates on the regular row."""
    corner_continuation: ScalarFloat | None = None
    """Already evaluated continuation at the shifted floor."""
    coverage_missing: Bool1D | None = None
    """Legal cash queries without represented viable support."""
    rescore_interior: Bool1D | None = None
    """Strict original adjacent interiors whose reconstructed actions need valuation."""


class _EGMSavingsSupport(NamedTuple):
    """Identify the declared floor and represented augmented savings candidates."""

    floor: ScalarFloat
    """Evaluated lower bound, independent of augmented sorting positions."""
    active: Bool1D
    """Participation of original nodes and admitted boundary candidates."""
    candidate_id: Int1D
    """Original index or negative flattened boundary-side construction index."""


def _flatten_egm_savings_support(
    support: _EGMSavingsSupport,
) -> tuple[tuple[PytreeChild, ...], None]:
    return tuple(support), None


def _flatten_egm_savings_support_with_keys(
    support: _EGMSavingsSupport,
) -> tuple[tuple[tuple[jax.tree_util.GetAttrKey, PytreeChild], ...], None]:
    return tuple(
        (jax.tree_util.GetAttrKey(name), leaf)
        for name, leaf in zip(_EGMSavingsSupport._fields, support, strict=True)
    ), None


# keyword-only-exempt: library-callback=jax.tree_util.register_pytree_with_keys
def _unflatten_egm_savings_support(
    _aux: None, children: Iterable[PytreeChild]
) -> _EGMSavingsSupport:
    """Restore compiler descriptors and temporary batch axes without scalar checks."""
    leaves = tuple(children)
    if len(leaves) != len(_EGMSavingsSupport._fields):
        raise TypeError("Savings support requires all its associated fields.")
    return cast("_EGMSavingsSupport", tuple.__new__(_EGMSavingsSupport, leaves))


jax.tree_util.register_pytree_with_keys(
    _EGMSavingsSupport,
    _flatten_egm_savings_support_with_keys,
    _unflatten_egm_savings_support,
    _flatten_egm_savings_support,
)


def egm_one_asset_step(
    *,
    next_value: Float1D,
    next_marginal: Float1D,
    liquid_grid: Float1D,
    next_liquid_grid: Float1D,
    savings_grid: Float1D,
    discount_factor: ScalarFloat | float,
    preferences: Preferences,
    next_liquid: Float1D,
    marginal_return: Float1D,
    continuation_value: Float1D | None = None,
    continuation_marginal: Float1D | None = None,
    feature_floor: bool = False,
    savings_support: _EGMSavingsSupport | None = None,
) -> RetiredEGMResult:
    """Solve one period of the 1-D consumption--saving problem by EGM.

    The two grids are distinct roles, not a convenience: this period's value is
    published on `liquid_grid`, while every next-period array is tabulated on
    `next_liquid_grid` and must be read there. They coincide unless the liquid
    state is an `AgeSpecializedGrid`, in which case reading `next_value` on this
    period's nodes evaluates the continuation at wealth levels the next period's
    grid never covers.

    Args:
        next_value: Next period's value on `next_liquid_grid`, shape `(n_liquid,)`.
        next_marginal: Next period's marginal value of liquid `V'` on
            `next_liquid_grid`, shape `(n_liquid,)`. For a terminal bequest
            `u(liquid)` this is `u'(liquid)`.
        liquid_grid: Regular liquid-state grid for this period (ascending).
        next_liquid_grid: The same state's grid in the *next* period (ascending);
            the abscissae of `next_value` and `next_marginal`.
        savings_grid: Post-decision savings grid `s = liquid - consumption` (ascending),
            shape `(n_savings,)`. Its lower bound is the borrowing limit: zero for a
            household that cannot borrow, minus the limit for one that can.
        discount_factor: Discount factor `beta`.
        preferences: The regime's felicity `u`, its marginal `u'`, and its
            inverse marginal `(u')^-1`, bound to this solve's parameters.
        next_liquid: The regime's own law of motion evaluated at each savings
            node, shape `(n_savings,)`. Where a given level of savings lands
            next period.
        marginal_return: That law's derivative with respect to savings, same
            shape. How the landing point moves when savings move, which is the
            factor the Euler equation discounts the continuation marginal by.
            For the conventional law this is the gross return at every node.

    Returns:
        This period's value, marginal value of liquid, and consumption policy on
        `liquid_grid`.

    """
    value_next = (
        continuation_value
        if continuation_value is not None
        else interp_on_regular_grid(
            x_query=next_liquid, xp=next_liquid_grid, fp=next_value
        )
    )
    marginal_next = (
        continuation_marginal
        if continuation_marginal is not None
        else interp_on_regular_grid(
            x_query=next_liquid, xp=next_liquid_grid, fp=next_marginal
        )
    )
    if feature_floor or savings_support is not None:
        result = _solve_viable_feature_row(
            value_next=value_next,
            marginal_next=marginal_next,
            liquid_grid=liquid_grid,
            savings_grid=savings_grid,
            discount_factor=discount_factor,
            preferences=preferences,
            marginal_return=marginal_return,
            next_liquid=next_liquid,
            savings_support=savings_support,
        )
        return result if feature_floor else result._replace(coverage_missing=None)

    consumption = invert_euler(
        expected_marginal_continuation=marginal_return * marginal_next,
        discount_factor=discount_factor,
        inverse_marginal_utility=preferences.inverse_marginal_utility,
    )
    liquid_endog = consumption + savings_grid
    value_endog = preferences.utility(consumption) + discount_factor * value_next

    # Interior: interpolate the endogenous value and consumption onto the regular grid.
    interior_value = jnp.interp(liquid_grid, liquid_endog, value_endog)
    interior_consumption = jnp.interp(liquid_grid, liquid_endog, consumption)
    interior_savings = jnp.interp(liquid_grid, liquid_endog, savings_grid)
    lower = jnp.clip(
        jnp.searchsorted(liquid_endog, liquid_grid, side="right") - 1,
        0,
        savings_grid.size - 2,
    )
    coordinates_supported = jnp.all(jnp.isfinite(liquid_endog)) & jnp.all(
        liquid_endog[1:] > liquid_endog[:-1]
    )
    local_tuple_finite = (
        jnp.isfinite(value_endog[lower])
        & jnp.isfinite(value_endog[lower + 1])
        & jnp.isfinite(consumption[lower])
        & jnp.isfinite(consumption[lower + 1])
        & jnp.isfinite(savings_grid[lower])
        & jnp.isfinite(savings_grid[lower + 1])
    )
    rescore_interior = (
        coordinates_supported
        & local_tuple_finite
        & (liquid_grid > liquid_endog[lower])
        & (liquid_grid < liquid_endog[lower + 1])
        & jnp.isfinite(interior_value)
        & jnp.isfinite(interior_consumption)
        & (interior_consumption > 0.0)
        & jnp.isfinite(interior_savings)
    )

    # Constrained: below the smallest endogenous liquid the household would want savings
    # the grid's lower bound forbids, so it takes that bound and consumes everything
    # above it. The bound is where the borrowing limit lives — zero for a household that
    # cannot borrow, negative for one that can — so both the corner consumption and the
    # landing point it reaches are read off the grid and the law rather than assumed.
    min_savings = savings_grid[0]
    constrained = liquid_grid < liquid_endog[0]
    constrained_consumption = liquid_grid - min_savings
    value_at_min_savings = (
        value_next[0]
        if continuation_value is not None
        else interp_on_regular_grid(
            x_query=next_liquid[0],
            xp=next_liquid_grid,
            fp=next_value,
        )
    )
    constrained_value = (
        preferences.utility(constrained_consumption)
        + discount_factor * value_at_min_savings
    )
    consumption_on_grid = jnp.where(
        constrained, constrained_consumption, interior_consumption
    )
    value = jnp.where(constrained, constrained_value, interior_value)
    # Keep the economic envelope approximation separate from value interpolation.
    marginal = preferences.marginal_utility(consumption_on_grid)
    return RetiredEGMResult(
        value=value,
        marginal=marginal,
        consumption=consumption_on_grid,
        savings=jnp.where(constrained, min_savings, interior_savings),
        rescore_interior=rescore_interior,
    )


def _solve_viable_feature_row(
    *,
    value_next: Float1D,
    marginal_next: Float1D,
    liquid_grid: Float1D,
    savings_grid: Float1D,
    discount_factor: ScalarFloat | float,
    preferences: Preferences,
    marginal_return: Float1D,
    next_liquid: Float1D,
    savings_support: _EGMSavingsSupport | None = None,
) -> RetiredEGMResult:
    """Interpolate viable candidate tuples without crossing missing original links.

    Economic minus-infinity continuations have neutral marginal placeholders, not
    Euler targets. Compact only finite candidates into fixed-capacity support, retaining
    original indices to distinguish adjacent support from a gap. The declared floor
    owns its separate corner; the first viable saving never replaces it.
    """
    logical_rank = None
    floor_index = 0
    if savings_support is None:
        invalid = jnp.any(
            jnp.isnan(value_next)
            | jnp.isposinf(value_next)
            | ~jnp.isfinite(marginal_next)
            | (marginal_next < 0.0)
            | ~jnp.isfinite(marginal_return)
        )
        viable = jnp.isfinite(value_next)
    else:
        invalid = jnp.any(
            savings_support.active
            & (
                jnp.isnan(value_next)
                | jnp.isposinf(value_next)
                | ~jnp.isfinite(marginal_next)
                | (marginal_next < 0.0)
                | ~jnp.isfinite(marginal_return)
            )
        )
        active, logical_rank = _prepare_augmented_savings_support(
            savings_grid=savings_grid,
            next_liquid=next_liquid,
            marginal_return=marginal_return,
            value_next=value_next,
            marginal_next=marginal_next,
            savings_support=savings_support,
        )
        floor_index = jnp.argmax(savings_support.candidate_id == 0)
        viable = active & jnp.isfinite(value_next)
    consumption = invert_euler(
        expected_marginal_continuation=jnp.where(
            viable, marginal_return * marginal_next, 1.0
        ),
        discount_factor=discount_factor,
        inverse_marginal_utility=preferences.inverse_marginal_utility,
    )
    consumption = jnp.where(viable, consumption, 1.0)
    liquid_endog = consumption + savings_grid
    value_endog = preferences.utility(consumption) + discount_factor * jnp.where(
        viable, value_next, 0.0
    )
    invalid |= jnp.any(
        viable
        & (
            ~jnp.isfinite(consumption)
            | (consumption <= 0.0)
            | ~jnp.isfinite(liquid_endog)
            | ~jnp.isfinite(value_endog)
        )
    )

    capacity = savings_grid.size
    count = jnp.sum(viable, dtype=jnp.int32)
    original_indices = jnp.nonzero(viable, size=capacity, fill_value=0)[0]
    positions = jnp.arange(capacity)
    last = jnp.maximum(count - 1, 0)
    indices = jnp.where(positions < count, original_indices, original_indices[last])
    support = jnp.where(positions < count, liquid_endog[indices], jnp.inf)
    # Canonical duplicate inputs have already folded. Distinct remaining tuples
    # may share endogenous cash; there the rightmost tuple (the highest saving)
    # wins, both for an exact read and as the left end of the next interval.
    ordered = jnp.all(
        jnp.where(
            positions[1:] < count, support[1:] >= support[:-1], jnp.asarray(a=True)
        )
    )
    lower = jnp.clip(jnp.searchsorted(support, liquid_grid, side="right") - 1, 0, last)
    upper = jnp.minimum(lower + 1, last)
    exact = liquid_grid == support[lower]
    above = (count > 0) & (liquid_grid >= support[last])
    adjacent = (
        indices[upper] == indices[lower] + 1
        if logical_rank is None
        else logical_rank[indices[upper]] == logical_rank[indices[lower]] + 1
    )
    interior_supported = (
        (count > 0) & ordered & (liquid_grid >= support[0]) & (exact | above | adjacent)
    )
    at_endpoint = exact | above
    selected_original = indices[lower]
    interior_value = jnp.where(
        at_endpoint,
        value_endog[selected_original],
        jnp.interp(liquid_grid, support, value_endog[indices]),
    )
    interior_consumption = jnp.where(
        at_endpoint,
        consumption[selected_original],
        jnp.interp(liquid_grid, support, consumption[indices]),
    )
    interior_savings = jnp.where(
        at_endpoint,
        savings_grid[selected_original],
        jnp.interp(liquid_grid, support, savings_grid[indices]),
    )

    floor = savings_grid[0] if savings_support is None else savings_support.floor
    domain = liquid_grid > floor
    corner_consumption = jnp.where(domain, liquid_grid - floor, 1.0)
    corner_value = (
        preferences.utility(corner_consumption)
        + discount_factor * value_next[floor_index]
    )
    invalid |= jnp.any(jnp.isnan(corner_value) | jnp.isposinf(corner_value))
    corner_supported = (
        domain
        & jnp.isfinite(value_next[floor_index])
        & ((count == 0) | (liquid_grid < support[0]))
    )
    choose_corner = corner_supported & (
        ~interior_supported | (corner_value > interior_value)
    )
    supported = interior_supported | corner_supported
    value = jnp.where(choose_corner, corner_value, interior_value)
    consumption_on_grid = jnp.where(
        choose_corner, corner_consumption, interior_consumption
    )
    saving_on_grid = jnp.where(choose_corner, floor, interior_savings)
    value = jnp.where(domain & supported, value, -jnp.inf)
    marginal = jnp.where(
        domain & supported, preferences.marginal_utility(consumption_on_grid), 0.0
    )
    return RetiredEGMResult(
        value=jnp.where(invalid, jnp.nan, value),
        marginal=jnp.where(invalid, jnp.nan, marginal),
        consumption=consumption_on_grid,
        savings=saving_on_grid,
        corner_continuation=value_next[floor_index],
        coverage_missing=domain & ~supported,
        rescore_interior=(
            ~invalid
            & domain
            & interior_supported
            & ~choose_corner
            & ~exact
            & ~above
            & adjacent
            & jnp.isfinite(value)
            & jnp.isfinite(marginal)
            & jnp.isfinite(saving_on_grid)
        ),
    )


def _prepare_augmented_savings_support(
    *,
    savings_grid: Float1D,
    next_liquid: Float1D,
    marginal_return: Float1D,
    value_next: Float1D,
    marginal_next: Float1D,
    savings_support: _EGMSavingsSupport,
) -> tuple[Bool1D, Int1D]:
    """Rank represented canonical reads before removing economic missing nodes.

    Every record is read by the same canonical numeric child reader. An exact
    repeated saving can be omitted only when its complete evaluated law and child
    tuple agrees. Generated side indices describe construction, not child owners.
    Activity is independent of economic viability, so an absent slot never splits
    an edge and removing an economic original never creates an edge.
    """
    active = savings_support.active
    same_read = (
        active[1:]
        & active[:-1]
        & (savings_grid[1:] == savings_grid[:-1])
        & (next_liquid[1:] == next_liquid[:-1])
        & (marginal_return[1:] == marginal_return[:-1])
        & (value_next[1:] == value_next[:-1])
        & (marginal_next[1:] == marginal_next[:-1])
    )
    keep = active & ~jnp.concatenate((jnp.asarray([False]), same_read))
    logical_rank = jnp.cumsum(keep, dtype=jnp.int32) - 1
    return keep, logical_rank
