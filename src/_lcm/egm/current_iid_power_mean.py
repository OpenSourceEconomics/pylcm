"""Private explicit economic directions of the genuine stable PowerMean.

This reducer consumes transient original child/coordinate records. Signed
statistics retain their scales through final multiplication by C. It neither
differentiates a stored level interpolant nor changes aggregate_scaled's AD
refusal. Complete-bank ownership/admission belongs to its actual producer.
"""

import jax.numpy as jnp

from _lcm.egm.current_iid_power_math import (
    _SignedLog,
)


def _join(*, parts: tuple[_SignedLog, ...]) -> _SignedLog:
    return _SignedLog(
        jnp.concatenate(tuple(jnp.ravel(part.logabs) for part in parts)),
        jnp.concatenate(tuple(jnp.ravel(part.sign) for part in parts)),
    )
