"""Engine-side certainty-equivalent helpers: power transform pair and resolver.

The public `lcm.certainty_equivalent` module imports `power_transform`,
`power_inverse`, and `CE_VALUE_ARG` from here to build its classes; the
solver seam (`_lcm.regime_building.Q_and_F`) imports
`resolve_certainty_equivalent` to wire a declared certainty equivalent into
the backward-induction closure.
"""

from collections.abc import Callable
from types import MappingProxyType
from typing import TYPE_CHECKING

from _lcm.utils.functools import get_union_of_args
from lcm.typing import FloatND

# `lcm.certainty_equivalent` imports the transform pair and `CE_VALUE_ARG` from
# this module at load time, so this module cannot import the CE classes at module
# scope without a cycle. `resolve_certainty_equivalent` imports the concrete class
# lazily inside its body; for the signature, the precise types are kept for ty and
# a wide `object` runtime form is used for beartype (which cannot resolve the
# would-be-circular forward references).
if TYPE_CHECKING:
    from lcm.certainty_equivalent import CertaintyEquivalent, QuasiArithmeticMean

    _CEInput = CertaintyEquivalent | None
    _ResolvedCE = QuasiArithmeticMean | None
else:
    _CEInput = object
    _ResolvedCE = object

# Reserved argument name through which transform callables receive values.
CE_VALUE_ARG = "value"


def power_transform(value: FloatND, risk_aversion: FloatND) -> FloatND:
    """Apply `g(v) = v^(1 - risk_aversion)`."""
    return value ** (1.0 - risk_aversion)


def power_inverse(value: FloatND, risk_aversion: FloatND) -> FloatND:
    """Apply `g^(-1)(v) = v^(1 / (1 - risk_aversion))`."""
    return value ** (1.0 / (1.0 - risk_aversion))


def resolve_certainty_equivalent(
    certainty_equivalent: _CEInput,
) -> tuple[
    _ResolvedCE,
    MappingProxyType[str, str],
    MappingProxyType[str, str],
]:
    """Narrow the certainty equivalent and map its args to flat param names.

    The runtime parameters live under the pseudo-function name
    `certainty_equivalent` in the regime's flat params
    (`certainty_equivalent__<arg>`); the returned mappings let the Q-and-F
    closure pull each callable's kwargs from `states_actions_params`.

    Returns:
        Tuple of the narrowed quasi-arithmetic-mean CE (or `None`), the
        transform's arg-to-flat-name mapping, and the inverse's
        arg-to-flat-name mapping.

    """
    # Imported here rather than at module scope to break the load-time cycle with
    # `lcm.certainty_equivalent` (which imports the transform pair from this module).
    from lcm.certainty_equivalent import QuasiArithmeticMean  # noqa: PLC0415

    if certainty_equivalent is None:
        return None, MappingProxyType({}), MappingProxyType({})
    if not isinstance(certainty_equivalent, QuasiArithmeticMean):
        msg = (
            "Only `QuasiArithmeticMean` certainty equivalents are "
            f"supported, got {type(certainty_equivalent).__name__}."
        )
        raise NotImplementedError(msg)

    def flat_names(func: Callable[..., FloatND]) -> MappingProxyType[str, str]:
        return MappingProxyType(
            {
                arg: f"certainty_equivalent__{arg}"
                for arg in get_union_of_args([func]) - {CE_VALUE_ARG}
            }
        )

    return (
        certainty_equivalent,
        flat_names(certainty_equivalent.transform),
        flat_names(certainty_equivalent.inverse),
    )
