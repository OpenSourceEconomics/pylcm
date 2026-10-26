"""Unresolved direct-state envelope requirements of private owned U generations."""

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import TYPE_CHECKING

from dags import get_ancestors

from _lcm.typing import EconFunctionsMapping
from lcm.exceptions import UnsupportedOperationError
from lcm.typing import FunctionName, ReferenceName, RegimeName, StateName, UserFunction

if TYPE_CHECKING:
    from _lcm.continuation import ProjectedProducerDate
    from _lcm.current_iid import CurrentIIDPlan
    from _lcm.egm.current_iid_negm_origin import DatedNEGMCaseUBranch
    from _lcm.solution.contract import SolverBuildContext
    from _lcm.solution.dcegm import _BoundDCEGM
    from lcm.regime import Regime as UserRegime
else:
    # Solver validation imports this record while the public regime is loading.
    type UserRegime = object
    type ProjectedProducerDate = object
    type CurrentIIDPlan = object
    type SolverBuildContext = object
    type _BoundDCEGM = object
    type DatedNEGMCaseUBranch = object


@dataclass(frozen=True, kw_only=True, eq=False)
class DatedStateUtilityObligation:
    """Original declaration bindings requiring an actual economic U envelope."""

    regime_name: RegimeName
    """Original source whose envelope remains unresolved."""
    user_regime: UserRegime
    """Retained declaration holding the actual validated solver."""
    state: StateName
    """Euler state whose direct utility derivative must be published."""
    functions: Mapping[FunctionName, UserFunction]
    """Original utility ancestry, excluding unrelated rebuilt helpers."""
    periods: tuple[int, ...]
    """Every demanded source date, before numerical preparation."""

    def require(
        self,
        *,
        regime_name: RegimeName,
        user_regime: UserRegime,
        functions: Mapping[FunctionName, UserFunction],
        periods: tuple[int, ...] | None = None,
    ) -> None:
        """Accept rebuilt wrappers only when their original economic bindings agree."""
        from _lcm.solution.dcegm import _BoundDCEGM  # noqa: PLC0415

        solver = user_regime.solver
        if (
            not isinstance(solver, _BoundDCEGM)
            or solver.continuous_state != self.state
            or regime_name != self.regime_name
            or user_regime.solver is not self.user_regime.solver
            or any(
                functions.get(name) is not function
                for name, function in self.functions.items()
            )
            or (periods is not None and periods != self.periods)
            or not self.periods
        ):
            raise UnsupportedOperationError(
                "Direct-state utility lost its original dated envelope obligation."
            )


type StateUtilityObligations = Mapping[RegimeName, DatedStateUtilityObligation]


def dated_state_utility_obligations(
    *,
    user_regimes: Mapping[RegimeName, UserRegime],
    periods: Mapping[RegimeName, tuple[int, ...]],
) -> StateUtilityObligations:
    """Record unresolved requirements after authenticating the retained constructor."""
    from _lcm.regime_building.phases import _resolve_solve_functions  # noqa: PLC0415
    from _lcm.solution.dcegm import _BoundDCEGM  # noqa: PLC0415

    obligations = {}
    for name, regime in user_regimes.items():
        solver = regime.solver
        if not isinstance(solver, _BoundDCEGM):
            continue
        functions = _resolve_solve_functions(user_regime=regime)
        if "utility" not in functions:
            continue
        ancestors = get_ancestors(functions, targets=["utility"], include_targets=False)
        if solver.continuous_state not in ancestors:
            continue
        obligations[name] = DatedStateUtilityObligation(
            regime_name=name,
            user_regime=regime,
            state=solver.continuous_state,
            functions=MappingProxyType(
                {
                    key: value
                    for key, value in functions.items()
                    if key in ancestors or key == "utility"
                }
            ),
            periods=periods.get(name, ()),
        )
    return MappingProxyType(obligations)


@dataclass(frozen=True, kw_only=True, eq=False)
class DatedCaseUStepAuthority:
    """Original U source bindings whose steps supply only native Euler pieces."""

    context: SolverBuildContext
    solver: _BoundDCEGM
    period: int
    date: ProjectedProducerDate
    plan: CurrentIIDPlan
    state_utility_obligation: DatedStateUtilityObligation | None
    nested_origin: DatedNEGMCaseUBranch | None = field(
        default=None, metadata={"fingerprint_omit_if_default": True}
    )
    """Exact original NEGM branch, never an ordinary-publication permission."""

    def require(
        self,
        *,
        context: SolverBuildContext,
        solver: _BoundDCEGM,
        period: int,
    ) -> None:
        """Authenticate exact economics, support, plan and original source date."""
        from _lcm.regime_building.phases import (  # noqa: PLC0415
            _resolve_solve_functions,
        )
        from _lcm.solution.current_iid_case_u_native import (  # noqa: PLC0415
            is_case_u_source,
        )

        if self.nested_origin is not None:
            if (
                self.context is not self.nested_origin.context
                or self.solver is not self.nested_origin.solver
                or self.date is not self.nested_origin.source.date
                or self.plan is not self.nested_origin.source.plan
                or self.period != self.nested_origin.source.period
                or self.state_utility_obligation is not None
            ):
                raise UnsupportedOperationError(
                    "Nested U pieces lost their original source/branch bindings."
                )
            self.nested_origin.require(context=context, solver=solver, period=period)
            return
        original = self.context
        name = original.regime_name
        spec = original.projected_continuation_declarations.get(name)
        if (
            solver is not self.solver
            or solver is not original.user_regimes[name].solver
            or context.regime_name != name
            or context.user_regimes is not original.user_regimes
            or context.functions.keys() != original.functions.keys()
            or any(
                context.functions.get(name) is not function
                for name, function in original.functions.items()
            )
            or context.laws is not original.laws
            or context.state_action_space is not original.state_action_space
            or context.ages is not original.ages
            or period != self.period
            or period not in original.regimes_to_active_periods[name]
            or original.current_iid_plans.get(period) is not self.plan
            or context.current_iid_plans.get(period) is not self.plan
            or spec is None
            or spec.date_at(period=period) is not self.date
            or self.date.authority.period != period
            or self.date.producer != name
            or self.date.liquid_state != solver.continuous_state
            or not is_case_u_source(context=original, solver=solver, period=period)
        ):
            raise UnsupportedOperationError(
                "Native U numerical pieces lost their original dated source authority."
            )
        if self.state_utility_obligation is not None:
            self.state_utility_obligation.require(
                regime_name=name,
                user_regime=original.user_regimes[name],
                functions=_resolve_solve_functions(
                    user_regime=original.user_regimes[name]
                ),
                periods=original.regimes_to_active_periods[name],
            )

    def utility_inputs(
        self,
        *,
        regime_name: RegimeName,
        solver: _BoundDCEGM,
        functions: EconFunctionsMapping,
        periods: tuple[int, ...],
    ) -> frozenset[ReferenceName]:
        """Allow only this dated plan's physical draws and owed state derivative."""
        from _lcm.regime_building.age_normalization import (  # noqa: PLC0415
            resolve_periodized_nodes,
        )

        self.require(context=self.context, solver=solver, period=self.period)
        expected = resolve_periodized_nodes(
            mapping=self.context.functions,
            period=self.period,
        )
        if (
            regime_name != self.context.regime_name
            or periods != (self.period,)
            or functions.keys() != expected.keys()
            or any(
                functions[name] is not function for name, function in expected.items()
            )
        ):
            raise UnsupportedOperationError(
                "Native U utility scope differs from its original dated function pool."
            )
        return self.plan.physical_names | (
            frozenset({self.solver.continuous_state})
            if self.state_utility_obligation is not None
            else frozenset()
        )
