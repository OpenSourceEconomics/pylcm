"""Original runtime Euler support validated before private dated producers."""

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np

from _lcm.dtypes import canonical_float_dtype
from _lcm.egm.current_iid_envelope import StateUtilityObligations
from _lcm.execution.core_program import CoreBuildContext
from _lcm.grids import ContinuousGrid, Grid, IrregSpacedGrid
from _lcm.grids.continuous import _validate_irreg_spaced_grid
from _lcm.materialization import DatedGenerationMaterializer
from _lcm.reachability import PhaseReachability
from _lcm.regime_law import RegimeLaws
from _lcm.time import TimeAxis
from _lcm.typing import PytreeValue
from lcm.exceptions import UnsupportedOperationError
from lcm.typing import Float1D, Int1D, ReferenceName, RegimeName, StateName

if TYPE_CHECKING:
    from lcm.regime import Regime as UserRegime
else:
    # The public regime imports solver validation while loading its definition.
    type UserRegime = object


@dataclass(frozen=True, kw_only=True, eq=False)
class DatedEulerGridObligation:
    """The original runtime grid and solver requiring call-owned validation."""

    regime_name: RegimeName
    user_regime: UserRegime
    state: StateName
    grid: IrregSpacedGrid
    periods: tuple[int, ...]

    def require(self, *, regime_name: RegimeName, user_regime: UserRegime) -> None:
        """Reject bindings detached from the retained original declaration."""
        from _lcm.solution.dcegm import _BoundDCEGM  # noqa: PLC0415

        if (
            regime_name != self.regime_name
            or not isinstance(user_regime.solver, _BoundDCEGM)
            or user_regime.solver is not self.user_regime.solver
            or user_regime.solver.continuous_state != self.state
            or user_regime.states.get(self.state) is not self.grid
            or type(self.grid) is not IrregSpacedGrid
            or not self.grid.pass_points_at_runtime
            or not self.periods
        ):
            raise UnsupportedOperationError(
                "Runtime Euler validation lost its original dated grid/solver binding."
            )


type EulerGridObligations = Mapping[RegimeName, DatedEulerGridObligation]


def dated_euler_grid_obligations(
    *,
    user_regimes: Mapping[RegimeName, UserRegime],
    periods: Mapping[RegimeName, tuple[int, ...]],
) -> EulerGridObligations:
    """Record only the retained private constructor's actual Euler-grid requirement."""
    from _lcm.solution.dcegm import _BoundDCEGM  # noqa: PLC0415

    obligations = {}
    for name, regime in user_regimes.items():
        solver = regime.solver
        if not isinstance(solver, _BoundDCEGM):
            continue
        grid = regime.states.get(solver.continuous_state)
        if type(grid) is IrregSpacedGrid and grid.pass_points_at_runtime:
            obligation = DatedEulerGridObligation(
                regime_name=name,
                user_regime=regime,
                state=solver.continuous_state,
                grid=grid,
                periods=periods.get(name, ()),
            )
            obligation.require(regime_name=name, user_regime=regime)
            obligations[name] = obligation
    return MappingProxyType(obligations)


@dataclass(frozen=True, kw_only=True, eq=False)
class ResolvedDatedEulerGrid:
    """Actual context and existing node identity whose original checks succeeded."""

    obligation: DatedEulerGridObligation
    call: CoreBuildContext
    nodes: Float1D

    def require(self) -> None:
        """Keep date, geometry and node ownership attached to the actual call."""
        obligation = self.obligation
        obligation.require(
            regime_name=obligation.regime_name, user_regime=obligation.user_regime
        )
        if (
            self.call.period not in obligation.periods
            or self.call.state_action_space is None
            or self.call.state_action_space.states.get(obligation.state)
            is not self.nodes
            or self.nodes.shape != (obligation.grid.n_points,)
            or self.nodes.dtype != np.dtype(canonical_float_dtype())
        ):
            raise UnsupportedOperationError(
                "Runtime Euler validation lost its actual date/node geometry."
            )


@dataclass(frozen=True, kw_only=True, eq=False)
class DatedEulerGridMaterializer:
    """Borrow the original owner, substituting only one existing runtime grid."""

    resolved: ResolvedDatedEulerGrid
    owner: DatedGenerationMaterializer | None

    def __call__(
        self, *, value: np.ndarray | jax.Array, dtype: np.dtype, name: str
    ) -> jax.Array:
        if self.owner is not None:
            return self.owner(value=value, dtype=dtype, name=name)
        return jnp.asarray(value, dtype=dtype)

    def materialize_grid(
        self, *, spec: Grid, placeholder: bool = False
    ) -> Float1D | Int1D:
        if spec is self.resolved.obligation.grid:
            self.resolved.require()
            if placeholder:
                raise UnsupportedOperationError(
                    "Resolved Euler validation cannot substitute a placeholder."
                )
            return self.resolved.nodes
        if self.owner is not None:
            return self.owner.materialize_grid(spec=spec, placeholder=placeholder)
        if placeholder:
            raise UnsupportedOperationError(
                "Euler validation reads original concrete grid points only."
            )
        return spec.to_jax()

    def produce[T: PytreeValue](
        self,
        *,
        function: Callable[..., T],
        arguments: Mapping[ReferenceName, PytreeValue],
    ) -> T:
        if self.owner is not None:
            return self.owner.produce(function=function, arguments=arguments)
        return function(**arguments)

    def evaluate_validation[T: PytreeValue](
        self,
        *,
        function: Callable[..., T],
        arguments: Mapping[ReferenceName, PytreeValue],
    ) -> T:
        if self.owner is not None:
            return self.owner.evaluate_validation(
                function=function, arguments=arguments
            )
        return function(**arguments)


def materialize_step_euler_grid(
    *,
    spec: ContinuousGrid,
    resolved: ResolvedDatedEulerGrid | None,
    materializer: DatedGenerationMaterializer | None,
) -> Float1D:
    """Read original step geometry from its validated call-owned grid, if present."""
    if resolved is not None:
        if spec is not resolved.obligation.grid:
            raise TypeError("EGM pieces substituted their original Euler declaration.")
        return DatedEulerGridMaterializer(
            resolved=resolved, owner=materializer
        ).materialize_grid(spec=spec)
    return (
        spec.to_jax()
        if materializer is None
        else materializer.materialize_grid(spec=spec)
    )


def resolve_dated_euler_grid(
    *, obligation: DatedEulerGridObligation, call: CoreBuildContext
) -> ResolvedDatedEulerGrid:
    """Validate the original canonical nodes without a replacement grid or upload."""
    if call.state_action_space is None:
        raise UnsupportedOperationError(
            "Runtime Euler validation needs actual support."
        )
    nodes = call.state_action_space.states.get(obligation.state)
    if not isinstance(nodes, jax.Array):
        raise UnsupportedOperationError(
            "Runtime Euler support must be a canonical array."
        )
    if nodes.shape != (obligation.grid.n_points,) or nodes.dtype != np.dtype(
        canonical_float_dtype()
    ):
        raise UnsupportedOperationError(
            "Runtime Euler validation needs original extent and floating dtype."
        )
    resolved = ResolvedDatedEulerGrid(obligation=obligation, call=call, nodes=nodes)
    resolved.require()
    # The original irregular-grid geometry predicate runs on existing canonical
    # values. This host read creates no replacement device grid or numeric oracle.
    _validate_irreg_spaced_grid(nodes.tolist())
    return resolved


type EulerGridValidations = Mapping[tuple[RegimeName, int], ResolvedDatedEulerGrid]


def validate_dated_euler_grids(
    *,
    obligations: EulerGridObligations,
    user_regimes: Mapping[RegimeName, UserRegime],
    periods: Mapping[RegimeName, tuple[int, ...]],
    call_contexts: Mapping[tuple[RegimeName, int], CoreBuildContext],
    ages: TimeAxis,
    laws: RegimeLaws,
    solution_reachability: PhaseReachability,
    state_utility_obligations: StateUtilityObligations,
    materializer: DatedGenerationMaterializer | None,
) -> EulerGridValidations:
    """Run the full original validation before any dated numerical producer."""
    from _lcm.egm.validation import validate_dcegm_regime  # noqa: PLC0415
    from _lcm.solution.dcegm import _BoundDCEGM  # noqa: PLC0415

    required_names = {
        name
        for name, regime in user_regimes.items()
        if isinstance(regime.solver, _BoundDCEGM)
        and isinstance(
            grid := regime.states.get(regime.solver.continuous_state), IrregSpacedGrid
        )
        and grid.pass_points_at_runtime
    }
    if set(obligations) != required_names:
        raise UnsupportedOperationError(
            "Runtime Euler validation has an incomplete source roster."
        )
    # Authenticate the whole roster before running even the first predicate.
    for name, obligation in obligations.items():
        obligation.require(regime_name=name, user_regime=user_regimes[name])
        if obligation.periods != periods[name]:
            raise UnsupportedOperationError(
                "Runtime Euler validation changed its demanded dates."
            )
        for period in obligation.periods:
            call = call_contexts.get((name, period))
            if call is None or call.period != period or call.ages is not ages:
                raise UnsupportedOperationError(
                    "Runtime Euler validation lacks its original dated context."
                )
    resolved = {}
    for name, obligation in obligations.items():
        for period in obligation.periods:
            result = resolve_dated_euler_grid(
                obligation=obligation, call=call_contexts[name, period]
            )
            adapter = DatedEulerGridMaterializer(resolved=result, owner=materializer)
            validate_dcegm_regime(
                regime_name=name,
                user_regime=user_regimes[name],
                user_regimes=user_regimes,
                laws=laws,
                solution_reachability=solution_reachability,
                state_utility_obligation=state_utility_obligations.get(name),
                euler_grid_obligation=obligation,
                resolved_euler_grid=result,
                materializer=adapter,
            )
            resolved[name, period] = result
    return MappingProxyType(resolved)
