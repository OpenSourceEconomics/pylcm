"""Regime-level taste shocks on discrete actions."""

from dataclasses import dataclass

from beartype import beartype

from _lcm.beartype_conf import REGIME_CONF
from _lcm.logsum import EULER_GAMMA

__all__ = ["EULER_GAMMA", "ExtremeValueTasteShocks"]


@beartype(conf=REGIME_CONF)
@dataclass(frozen=True, kw_only=True)
class ExtremeValueTasteShocks:
    """IID extreme-value (EV1/Gumbel) taste shocks on discrete-action combinations.

    Declared on a `Regime` via `taste_shocks=ExtremeValueTasteShocks()`. One
    shock is drawn per combination of the regime's discrete actions; its scale
    is the runtime parameter `{"taste_shocks": {"scale": ...}}` in the regime's
    params and must be strictly positive — the hard maximum is the
    no-taste-shocks model, not a zero scale.

    The solve first maximizes over every continuous-action axis within
    each discrete-action combination, then replaces the hard maximum over the
    discrete-action axes with the smoothed expected maximum
    `scale * logsumexp(Qc / scale)`. Simulation draws the discrete action by
    adding mean-zero `scale * (Gumbel(0, 1) - EULER_GAMMA)` noise
    (`EULER_GAMMA` the Euler-Mascheroni constant, the mean of a standard
    Gumbel) to the per-discrete-action values before the argmax. For any fixed
    candidate values, centering makes the expected latent perturbed maximum
    equal their smoothed log-sum. The shock affects the choice; simulation
    publishes the selected unshocked value. DCEGM simulation is
    grid-restricted and need not reproduce its off-grid solve value or choice
    probabilities.

    At least one discrete action is required. The implemented solver routes are
    `GridSearch` and `DCEGM`. The declaration is rejected for `NEGM`, `NBEGM`,
    and `NNBEGM`; for collective regimes; on a source regime with a
    `Gate`; together with an IID state declared
    `fold=True`; and together with a nonlinear certainty equivalent. These
    combinations are rejected during `Regime` declaration or `Model`
    construction, before solve, rather than silently dropping the shocks.

    The shocks are mean-zero. A model written with location-0 (standard
    Gumbel) shocks, whose mean is `scale * EULER_GAMMA`, is reproduced by
    adding `scale * EULER_GAMMA` to the utility of every action in each regime
    that declares taste shocks, with `EULER_GAMMA` imported from `lcm`. The
    recipe is exact when the regime's Koopmans aggregator is additive in
    utility with unit slope, as `LinearAggregator` is. Every choice-specific
    value then moves by the same constant, so choice probabilities stay the
    same and the value function rises by the discounted sum of
    `scale * EULER_GAMMA` over the remaining taste-shock periods. The shock
    enters after aggregation, so under any other aggregator, such as
    `CESAggregator`, the constant moves each action's value by a different
    amount and changes the choice probabilities.
    """
