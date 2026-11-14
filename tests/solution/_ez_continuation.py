"""The joint Epstein-Zin continuation, composed from the production partials.

Production composes `ez_transform_partials` and `ez_invert_partials` itself, across
blocks. The unit tests check the partials through this one-shot composition.
"""

from _lcm.egm.ez_kernel import ez_invert_partials, ez_transform_partials
from lcm.typing import FloatND, ScalarFloat


def ez_continuation(
    *,
    child_values: FloatND,
    child_marginals: FloatND,
    weights: FloatND,
    risk_aversion: ScalarFloat | float,
) -> tuple[FloatND, FloatND]:
    """Aggregate the continuation certainty equivalent and its savings derivative.

    Reduces over the last axis (the continuation lottery — the joint stochastic
    node and target regime). The certainty equivalent is the power mean
    `nu = (E[V'^(1-gamma)])^(1/(1-gamma))`, evaluated in the log domain so it stays
    finite near the borrowing constraint. Its savings derivative reweights each
    child's marginal by the child's risk-transformed value share,
    `dnu/ds = sum_j w_j (nu/V_j')^gamma * dV_j'/ds = nu^gamma * E[V'^(-gamma) dV'/ds]`,
    computed through the value ratio to keep the powers near one.

    This is the exogenous-probability form: the transition weights do not depend
    on end-of-period savings (`dP/ds = 0`), which holds whenever next-period
    uncertainty is an exogenous shock or regime lottery. `risk_aversion = 0`
    recovers the linear pair `(E[V'], E[dV'/ds])`.

    Args:
        child_values: Strictly positive next-period values on the continuation
            lottery, reduced over the last axis.
        child_marginals: The next-period value derivatives `dV'/ds` on the same
            lottery axis.
        weights: Nonnegative lottery probabilities over the last axis.
        risk_aversion: The Epstein-Zin risk-aversion coefficient.

    Returns:
        Tuple of the certainty equivalent `nu` and its savings derivative
        `dnu/ds`, each reduced over the last axis.

    """
    certainty_equivalent, weight_sum, marginal_log_scale, marginal_mantissa = (
        ez_transform_partials(
            child_values=child_values,
            child_marginals=child_marginals,
            weights=weights,
            risk_aversion=risk_aversion,
        )
    )
    return ez_invert_partials(
        certainty_equivalent=certainty_equivalent,
        weight_sum=weight_sum,
        marginal_log_scale=marginal_log_scale,
        marginal_mantissa=marginal_mantissa,
        risk_aversion=risk_aversion,
    )
