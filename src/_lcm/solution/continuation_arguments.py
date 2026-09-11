"""Expose one independently owned marginal without duplicating its input locator."""

from collections.abc import Callable, Mapping
from dataclasses import dataclass, fields, replace
from types import MappingProxyType
from typing import cast

from _lcm.egm.carry import EGMCarry
from _lcm.execution.core_program import CoreArgumentBuilder, CoreBuildContext, ValueRead
from _lcm.typing import RegimeName
from lcm.typing import FloatND

# Reserved core operand containing exactly one original continuation leaf.
MARGINAL_ARGUMENT = "__lcm_continuation_marginal__"


@dataclass(frozen=True, kw_only=True)
class MarginalLeafArguments:
    """Move the self carry's marginal into its sole top-level input slot."""

    inner: CoreArgumentBuilder
    target: RegimeName

    def __call__(self, context: CoreBuildContext) -> Mapping[str, object]:
        arguments = dict(self.inner(context))
        carries = dict(
            cast("Mapping[str, object]", arguments["next_regime_to_continuation"])
        )
        carry = carries[self.target]
        if not isinstance(carry, EGMCarry):
            raise TypeError("A marginal-leaf program requires an EGMCarry input.")
        carries[self.target] = MappingProxyType(
            {
                field.name: getattr(carry, field.name)
                for field in fields(EGMCarry)
                if field.name != "marginal_utility"
            }
        )
        arguments["next_regime_to_continuation"] = MappingProxyType(carries)
        arguments[MARGINAL_ARGUMENT] = carry.marginal_utility
        return MappingProxyType(arguments)


@dataclass(frozen=True, kw_only=True, eq=False)
class MarginalLeafCore:
    """Rebuild the original carry inside tracing, without copying an array."""

    core: Callable
    target: RegimeName

    def __call__(self, **arguments: object) -> object:
        marginal = cast("FloatND", arguments.pop(MARGINAL_ARGUMENT))
        carries = dict(
            cast("Mapping[str, object]", arguments["next_regime_to_continuation"])
        )
        residual = cast("Mapping[str, object]", carries[self.target])
        carries[self.target] = EGMCarry(
            endog_grid=cast("FloatND", residual["endog_grid"]),
            value=cast("FloatND", residual["value"]),
            marginal_utility=marginal,
            taste_shock_scale=cast("FloatND", residual["taste_shock_scale"]),
            breakpoints=cast("FloatND | None", residual["breakpoints"]),
            policy=cast("FloatND | None", residual["policy"]),
        )
        arguments["next_regime_to_continuation"] = MappingProxyType(carries)
        return self.core(**arguments)


def marginal_leaf_reads(reads: tuple[ValueRead, ...]) -> tuple[ValueRead, ...]:
    """Re-address every published leaf to the adapter's actual argument tree."""
    return tuple(
        replace(
            read,
            source=replace(
                read.source,
                argument=(
                    MARGINAL_ARGUMENT
                    if read.target.leaf_path == ("marginal_utility",)
                    else "next_regime_to_continuation"
                ),
                path=()
                if read.target.leaf_path == ("marginal_utility",)
                else read.source.path,
            ),
        )
        for read in reads
    )
