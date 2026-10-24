"""Regime-level taste shocks on discrete actions."""

from dataclasses import dataclass

from beartype import beartype

from _lcm.beartype_conf import REGIME_CONF


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

    With built-in current-period exits, Full appends one source taste component
    per exit in declaration order after the discrete-action combinations.
    Continuous optimization still precedes taste: continuous action points do
    not receive separate components. Full solve takes the logsum over admitted
    stay-combination maxima and admitted exit payoffs. Dissolution remains a
    fallback when its kept stay set is empty; strict Consent governs formation.
    Full simulation uses one
    draw at the source regime's existing taste address for that entire roster,
    mapping the selected component back to its stable stay action or exit ID.
    Publication remains the selected unshocked value. Destination taste is
    separately hidden-priced at its logsum and drawn at its own address only
    when acting there. Arbitrary custom Full selectors with source EV1 taste
    shocks are unsupported and explicitly refused.

    At least one discrete action is required. The implemented solver routes are
    `GridSearch` and `DCEGM`. The declaration is rejected for `NEGM`, `NBEGM`,
    and `NNBEGM`; for collective regimes; on a source regime with a
    `Gate`; together with an IID state declared
    `fold=True`; and together with a nonlinear certainty equivalent. These
    combinations are rejected during `Regime` declaration or `Model`
    construction, before solve, rather than silently dropping the shocks.

    The shocks are mean-zero. For location-0 shocks, add
    `scale * lcm.EULER_GAMMA` to the utility of every action in each regime
    that declares taste shocks, and keep constraints, transitions and other
    consumers of the original utility on an unshifted `base_utility`. This is
    exact when the Koopmans aggregator is additive in utility with unit slope,
    as `LinearAggregator` is, and is not exact under `CESAggregator` in
    general; see the taste-shock docs.
    """
