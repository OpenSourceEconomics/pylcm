"""Dated original NEGM ownership of its canonically derived U inner branches."""

from dataclasses import dataclass, field, fields
from types import FunctionType
from typing import TYPE_CHECKING, cast

from _lcm.typing import EconFunctionsMapping
from lcm.exceptions import UnsupportedOperationError

if TYPE_CHECKING:
    from _lcm.continuation import ProjectedProducerDate
    from _lcm.current_iid import CurrentIIDPlan
    from _lcm.solution.contract import SolutionKernels, SolverBuildContext
    from _lcm.solution.dcegm import EGMStepBuild, _BoundDCEGM
    from _lcm.solution.negm import _BoundNEGM
else:
    # Original solver modules import this record before their classes are bound.
    type ProjectedProducerDate = object
    type CurrentIIDPlan = object
    type SolutionKernels = object
    type SolverBuildContext = object
    type EGMStepBuild = object
    type _BoundDCEGM = object
    type _BoundNEGM = object


@dataclass(frozen=True, kw_only=True, eq=False)
class DatedNEGMCaseUSource:
    """The actual outer solver, full source context, date and current law."""

    context: SolverBuildContext
    solver: _BoundNEGM
    period: int
    date: ProjectedProducerDate
    plan: CurrentIIDPlan

    def require(self, *, context: SolverBuildContext, solver: _BoundNEGM) -> None:
        """Refuse replacement of the original source before deriving a branch."""
        from _lcm.solution.negm import _BoundNEGM  # noqa: PLC0415

        name = self.context.regime_name
        spec = self.context.projected_continuation_declarations.get(name)
        if (
            context is not self.context
            or type(solver) is not _BoundNEGM
            or solver is not self.solver
            or solver is not context.user_regimes[name].solver
            or context.current_iid_plans.get(self.period) is not self.plan
            or self.period not in context.regimes_to_active_periods[name]
            or spec is None
            or spec.date_at(period=self.period) is not self.date
            or self.date.authority.period != self.period
            or self.date.producer != name
            or self.date.liquid_state != solver.inner.continuous_state
        ):
            raise UnsupportedOperationError(
                "Nested U pieces lost their original dated outer source."
            )

    def build_inner(
        self, *, context: SolverBuildContext, solver: _BoundDCEGM, keeper: bool
    ) -> SolutionKernels:
        """Prepare the actual branch at its original NEGM construction site."""
        from _lcm.egm.current_iid_envelope import (  # noqa: PLC0415
            DatedCaseUStepAuthority,
        )

        branch = DatedNEGMCaseUBranch(
            source=self, context=context, solver=solver, keeper=keeper
        )
        authority = DatedCaseUStepAuthority(
            context=context,
            solver=solver,
            period=self.period,
            date=self.date,
            plan=self.plan,
            state_utility_obligation=None,
            nested_origin=branch,
        )
        build = solver.prepare_period_steps(
            context=context, periods=(self.period,), case_u_source=authority
        )
        # Bind the exact result of the original preparer, once, before any
        # assembler may consume it. The frozen branch retains references only.
        object.__setattr__(branch, "_prepared_build", build)
        return solver.finish_nested_current_u_period_kernels(
            context=context, build=build, authority=authority
        )


@dataclass(frozen=True, kw_only=True, eq=False)
class DatedNEGMCaseUBranch:
    """An exact original adjuster or injected keeper pool under that source."""

    source: DatedNEGMCaseUSource
    context: SolverBuildContext
    solver: _BoundDCEGM
    keeper: bool
    _prepared_build: EGMStepBuild | None = field(
        init=False, default=None, repr=False, compare=False
    )
    """Exact original preparer result, bound once by the original branch builder."""

    def require_prepared_build(self, *, build: EGMStepBuild) -> None:
        """Never authorize replaced steps through copied source metadata."""
        if self._prepared_build is not build:
            raise UnsupportedOperationError(
                "Nested U assembly lost its original prepared build identity."
            )

    def require(
        self, *, context: SolverBuildContext, solver: _BoundDCEGM, period: int
    ) -> None:
        """Preserve source identity and only the original branch derivation."""
        from _lcm.regime_building.age_normalization import (  # noqa: PLC0415
            resolve_periodized_nodes,
        )
        from _lcm.solution.dcegm import _BoundDCEGM  # noqa: PLC0415
        from _lcm.solution.negm import (  # noqa: PLC0415
            _KeeperOuterPostDecision,
            _with_no_adjustment_outer_function,
        )

        source = self.source
        source.require(context=source.context, solver=source.solver)
        original = source.context
        outer = source.solver
        if (
            context is not self.context
            or solver is not self.solver
            or type(solver) is not _BoundDCEGM
            or period != source.period
            or solver._retain_child_feasibility is not True  # noqa: SLF001
            or any(
                getattr(solver, field.name) is not getattr(outer.inner, field.name)
                for field in fields(solver)
                if field.name != "_retain_child_feasibility"
            )
            or any(
                getattr(context, field.name) is not getattr(original, field.name)
                for field in fields(context)
                if field.name not in {"functions", "flat_param_names"}
            )
        ):
            raise UnsupportedOperationError(
                "Nested U pieces lost their original derived branch context."
            )
        if not self.keeper:
            expected = {
                name: function
                for name, function in original.functions.items()
                if name != outer.outer_post_decision
            }
            valid = (
                context.flat_param_names
                == original.flat_param_names | {outer.outer_post_decision}
                and context.functions.keys() == expected.keys()
                and all(
                    context.functions[name] is function
                    for name, function in expected.items()
                )
            )
        else:
            expected = cast(
                "EconFunctionsMapping",
                resolve_periodized_nodes(mapping=original.functions, period=period),
            )
            no_adjustment = (
                None
                if outer.outer_no_adjustment_candidate is None
                else expected[outer.outer_no_adjustment_candidate]
            )
            actual = context.functions.get(outer.outer_post_decision)
            raw = (
                vars(actual).get("__wrapped__")
                if isinstance(actual, FunctionType)
                else None
            )
            # The original helper supplies both its exact dags forwarding code
            # and the original immutable keeper callback. Merely advertising a
            # __wrapped__ object cannot authorize a different numerical body.
            reference = _with_no_adjustment_outer_function(
                functions=expected,
                durable_state=outer.outer_state,
                outer_post_decision=outer.outer_post_decision,
                no_adjustment_func=no_adjustment,
            )[outer.outer_post_decision]
            valid = (
                context.flat_param_names is original.flat_param_names
                and context.functions.keys() == expected.keys()
                and all(
                    context.functions[name] is function
                    for name, function in expected.items()
                    if name != outer.outer_post_decision
                )
                and isinstance(actual, FunctionType)
                and isinstance(reference, FunctionType)
                and actual.__code__ is reference.__code__
                and type(raw) is _KeeperOuterPostDecision
                and raw.durable_state == outer.outer_state
                and raw.arg_names == (outer.outer_state,)
                and raw.no_adjustment_func is no_adjustment
                and actual.__closure__ is not None
                and dict(
                    zip(actual.__code__.co_freevars, actual.__closure__, strict=True)
                )["func"].cell_contents
                is raw
            )
        if not valid:
            raise UnsupportedOperationError(
                "Nested U utility scope differs from its original branch function pool."
            )
