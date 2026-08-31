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

    At least one discrete action is required. The implemented solver routes are
    `GridSearch` and `DCEGM`. The declaration is rejected for `NEGM`, `NBEGM`,
    and `NNBEGM`; for collective regimes; on a source regime with a
    `ValueDependentTransition`; together with an IID state declared
    `fold=True`; and together with a nonlinear certainty equivalent. These
    combinations are rejected during `Regime` declaration or `Model`
    construction, before solve, rather than silently dropping the shocks.
    """
