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
    params. A scale of `0.0` recovers the hard maximum.

    The semantics are solver-independent: the solve replaces the hard maximum
    over discrete-action axes with the smoothed expected maximum
    `scale * logsumexp(Qc / scale)`, and the simulation draws the discrete
    action by adding `scale * Gumbel(0, 1)` noise to the per-discrete-action
    values before the argmax.
    """
