"""Lower a joint Gaussian into Cartesian coordinates and named DAG outputs."""

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import ClassVar, cast, no_type_check

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
from dags import with_signature
from jaxtyping import Bool, Float

from _lcm.dtypes import canonical_float_dtype
from _lcm.grids import Grid
from _lcm.processes.iid import NormalIIDProcess
from _lcm.processes.joint_normal import (
    JointNormalIIDProcess,
    compiled_covariance_factor,
)
from _lcm.typing import FlatRegimeParams, RegimeNamesToIds
from lcm.collective import CollectiveUtility
from lcm.exceptions import (
    GridInitializationError,
    InvalidInitialConditionsError,
    InvalidParamsError,
    RegimeInitializationError,
)
from lcm.phased import Phased
from lcm.transition import AgeSpecializedGrid
from lcm.typing import (
    FloatND,
    FunctionName,
    ProcessName,
    RegimeName,
    StateName,
    UserFunction,
    UserInitialConditions,
)

type HostFloatArray = Float[np.ndarray, "..."]
type HostBoolArray = Bool[np.ndarray, "..."]


type JointStateEntry = Grid | JointNormalIIDProcess | Phased | AgeSpecializedGrid | None
type JointFunctionEntry = UserFunction | Phased | CollectiveUtility | None


def lower_joint_normal_processes(
    *,
    states: Mapping[StateName, JointStateEntry],
    functions: Mapping[FunctionName, JointFunctionEntry],
    other_names: frozenset[str],
) -> tuple[
    Mapping[StateName, Grid | Phased | AgeSpecializedGrid | None],
    Mapping[FunctionName, JointFunctionEntry],
]:
    """Expand joint declarations without changing scalar-process regimes."""
    expanded_states = {}
    expanded_functions = dict(functions)
    occupied = set(states) | set(functions) | set(other_names)
    for group, declaration in states.items():
        if not isinstance(declaration, JointNormalIIDProcess):
            expanded_states[group] = declaration
            continue
        latent_names = coordinate_names(group=group, process=declaration)
        generated_names = set(declaration.states) | set(latent_names)
        if (
            group in set(functions) | set(other_names)
            or generated_names & occupied
            or len(generated_names) != len(declaration.states) + len(latent_names)
        ):
            raise RegimeInitializationError(
                f"Joint normal process '{group}' has names colliding "
                "with another declaration."
            )
        occupied |= generated_names
        for name, count in zip(latent_names, declaration.node_counts, strict=True):
            expanded_states[name] = _JointNormalCoordinate(
                n_points=count,
                gauss_hermite=declaration.gauss_hermite,
                mu=0.0,
                sigma=1.0,
                n_std=declaration.n_std,
                fold=declaration.fold,
                joint=declaration,
                group=group,
            )
        expanded_functions[group] = _joint_function(group=group, process=declaration)
        for dimension, name in enumerate(declaration.states):
            expanded_functions[name] = _output_function(
                group=group, dimension=dimension
            )
    return MappingProxyType(expanded_states), MappingProxyType(expanded_functions)


def coordinate_names(
    *, group: ProcessName, process: JointNormalIIDProcess
) -> tuple[StateName, ...]:
    """Name the independent internal axes deterministically."""
    return tuple(
        f"{group}_normal_{dimension}" for dimension in range(len(process.states))
    )


def joint_normal_declarations(
    grids: Mapping[StateName, Grid],
) -> MappingProxyType[ProcessName, JointNormalIIDProcess]:
    """Collect the joint laws carried by scalar internal coordinate grids."""
    return MappingProxyType(
        {
            grid.group: grid.joint
            for grid in grids.values()
            if isinstance(grid, _JointNormalCoordinate)
        }
    )


def validate_joint_normal_params(
    *,
    grids: Mapping[StateName, Grid],
    params: FlatRegimeParams,
    regime_name: RegimeName,
) -> None:
    """Validate each joint covariance and mean at the canonical parameter boundary."""
    for group, process in joint_normal_declarations(grids).items():
        mean = (
            np.asarray(process.mean, dtype=canonical_float_dtype())
            if process.mean is not None
            else params[f"{group}__mean"]
        )
        cov = (
            np.asarray(process.cov, dtype=canonical_float_dtype())
            if process.cov is not None
            else params[f"{group}__cov"]
        )
        if not isinstance(mean, (jax.Array, np.ndarray)) or not isinstance(
            cov, (jax.Array, np.ndarray)
        ):
            raise InvalidParamsError(
                f"Joint normal process '{group}' in regime '{regime_name}': "
                "mean and cov must be numeric arrays."
            )
        try:
            process.validate_mean(mean)
            zero_pivots = process.validate_covariance(cov)
        except GridInitializationError as error:
            raise InvalidParamsError(
                f"Joint normal process '{group}' in regime '{regime_name}': {error}"
            ) from error
        # Only a fixed covariance carries its exact zero pivots into the factor.
        if process.cov is None and any(zero_pivots):
            raise GridInitializationError(
                f"Joint normal process '{group}' in regime '{regime_name}': the "
                "runtime cov is singular. A singular covariance must be supplied "
                "as the declaration's fixed cov."
            )


def lower_joint_normal_initial_conditions(
    *,
    initial_conditions: UserInitialConditions | pd.DataFrame,
    grids_by_regime: Mapping[RegimeName, Mapping[StateName, Grid]],
    params_by_regime: Mapping[RegimeName, FlatRegimeParams],
    regime_names_to_ids: RegimeNamesToIds,
    accept_coordinates: bool = False,
) -> UserInitialConditions | pd.DataFrame:
    """Convert named physical entry shocks to internal coordinates on the host.

    A coordinate spelling may also be an ordinary continuous state of another
    regime, so the shared column is carried as floats: joint rows receive their
    coordinates and the other regime's rows keep their own values.

    Args:
        initial_conditions: Per-subject inputs keyed by state name.
        grids_by_regime: Each regime's simulation grids.
        params_by_regime: Each regime's flat parameters, for runtime laws.
        regime_names_to_ids: Regime codes, for a mapping keyed by `regime_id`.
        accept_coordinates: Whether a joint process may instead be given by its
            complete set of internal coordinates. The inputs then describe the
            rows of one regime, so another regime's states play no part.

    Raises:
        InvalidInitialConditionsError: If a joint process is given neither by
            all of its physical shocks nor, where accepted, by all of its
            coordinates alone.
    """
    declarations = {
        name: joint_normal_declarations(grids)
        for name, grids in grids_by_regime.items()
    }
    if not any(declarations.values()):
        return initial_conditions
    ordinary_state_names = (
        set()
        if accept_coordinates
        else {
            name
            for grids in grids_by_regime.values()
            for name, grid in grids.items()
            if not isinstance(grid, _JointNormalCoordinate)
        }
    )
    is_frame = isinstance(initial_conditions, pd.DataFrame)
    regime_column = np.asarray(
        initial_conditions["regime_name" if is_frame else "regime_id"]
    )
    out = initial_conditions.copy() if is_frame else dict(initial_conditions)
    consumed = set()
    for regime_name, groups in declarations.items():
        rows = regime_column == (
            regime_name if is_frame else int(regime_names_to_ids[regime_name])
        )
        for group, process in groups.items():
            names = coordinate_names(group=group, process=process)
            present = set(process.states) & set(initial_conditions)
            consumed |= present
            if not rows.any():
                continue
            if _is_given_as_coordinates(
                group=group,
                process=process,
                present=present,
                given_coordinates={
                    name
                    for name in names
                    if name in initial_conditions and name not in ordinary_state_names
                },
                accept_coordinates=accept_coordinates,
            ):
                continue
            params = params_by_regime[regime_name]
            mean = np.asarray(
                process.mean if process.mean is not None else params[f"{group}__mean"],
                dtype=canonical_float_dtype(),
            )
            cov = np.asarray(
                process.cov if process.cov is not None else params[f"{group}__cov"],
                dtype=canonical_float_dtype(),
            )
            physical = _physical_inputs(
                group=group,
                process=process,
                initial_conditions=initial_conditions,
                rows=rows,
            )
            standard = _physical_to_standard(
                group=group,
                physical=physical.astype(float),
                mean=mean.astype(float),
                factor=np.asarray(
                    compiled_covariance_factor(
                        cov=jnp.asarray(cov), zero_pivots=process.fixed_zero_pivots()
                    ),
                    dtype=float,
                ),
                precision=float(np.finfo(cov.dtype).eps),
            )
            for dimension, name in enumerate(names):
                values = (
                    np.array(out[name], dtype=float)
                    if name in out
                    else np.full(len(regime_column), np.nan)
                )
                values[rows] = standard[:, dimension]
                out[name] = values
    # A physical name may also be an ordinary state in another regime.
    retained = set().union(*(set(grids) for grids in grids_by_regime.values()))
    if isinstance(out, pd.DataFrame):
        return out.drop(columns=list(consumed - retained))
    for name in consumed - retained:
        out.pop(name)
    return MappingProxyType(out)


def _is_given_as_coordinates(
    *,
    group: ProcessName,
    process: JointNormalIIDProcess,
    present: set[str],
    given_coordinates: set[str],
    accept_coordinates: bool,
) -> bool:
    """Classify which complete input form a joint process arrives in."""
    names = coordinate_names(group=group, process=process)
    if accept_coordinates and not present and given_coordinates == set(names):
        return True
    if accept_coordinates and (given_coordinates or present != set(process.states)):
        raise InvalidInitialConditionsError(
            f"Joint normal process '{group}' takes either its physical "
            f"shocks {process.states} or its internal coordinates "
            f"{names}, each complete and alone; got "
            f"{sorted(present | given_coordinates)}."
        )
    if given_coordinates:
        raise InvalidInitialConditionsError(
            f"Joint normal process '{group}' requires physical shocks "
            f"{process.states}, not internal coordinates."
        )
    if present != set(process.states):
        raise InvalidInitialConditionsError(
            f"Joint normal process '{group}' requires initial shocks {process.states}."
        )
    return False


def _physical_inputs(
    *,
    group: ProcessName,
    process: JointNormalIIDProcess,
    initial_conditions: UserInitialConditions | pd.DataFrame,
    rows: HostBoolArray,
) -> HostFloatArray:
    """Validate physical shock columns before selecting subjects."""
    columns = [np.asarray(initial_conditions[state]) for state in process.states]
    if rows.ndim != 1 or any(column.shape != rows.shape for column in columns):
        raise InvalidInitialConditionsError(
            f"Joint normal process '{group}' initial shocks must be "
            "one-dimensional arrays matching the regime values."
        )
    return np.stack([column[rows] for column in columns], axis=-1)


def _physical_to_standard(
    *,
    group: ProcessName,
    physical: HostFloatArray,
    mean: HostFloatArray,
    factor: HostFloatArray,
    precision: float,
) -> HostFloatArray:
    """Invert the triangular Gaussian map, including zero-variance coordinates.

    `factor` is the active-precision factor the forward map uses, so both
    directions share one factor and one rank.
    """
    if not np.isfinite(physical).all():
        raise InvalidInitialConditionsError(
            f"Joint normal process '{group}' initial shocks must be finite."
        )
    standard = np.zeros_like(physical)
    for dimension in range(len(mean)):
        residual = (
            physical[:, dimension]
            - mean[dimension]
            - standard[:, :dimension] @ factor[dimension, :dimension]
        )
        diagonal = factor[dimension, dimension]
        if diagonal > 0:
            standard[:, dimension] = residual / diagonal
        # A forward draw has rounded products, sums and physical inputs at the
        # active precision. Admit only that arithmetic envelope at a zero pivot.
        elif np.any(
            np.abs(residual)
            > precision
            * (len(mean) + 1)
            * (
                np.abs(physical[:, dimension])
                + abs(mean[dimension])
                + np.abs(standard[:, :dimension])
                @ np.abs(factor[dimension, :dimension])
            )
        ):
            raise InvalidInitialConditionsError(
                f"Joint normal process '{group}' initial shocks lie outside "
                "its singular covariance support."
            )
    return standard


_LAW_ARGUMENTS = MappingProxyType({"mean": "_law_mean", "cov": "_law_cov"})


@dataclass(frozen=True, kw_only=True)
class _JointNormalCoordinate(NormalIIDProcess):
    """One independent standard-normal axis belonging to a joint declaration."""

    joint: JointNormalIIDProcess
    """The physical joint distribution this axis parametrizes."""
    group: ProcessName
    """Parameter namespace and declaration key."""
    _NON_PARAM_FIELDS: ClassVar[frozenset[str]] = NormalIIDProcess._NON_PARAM_FIELDS | {  # noqa: SLF001 - extending the inherited process contract
        "joint",
        "group",
    }


def _joint_function(
    *, group: ProcessName, process: JointNormalIIDProcess
) -> UserFunction:
    names = coordinate_names(group=group, process=process)
    # Runtime law fields are owned by `(regime, group, field)`. Their arguments
    # start with `_`, which no state, action, function or parameter may, so no
    # economic name can capture them; `_lcm_law_arguments` binds each public
    # field to its argument for parameter discovery and renaming.
    law_arguments = {
        field: argument
        for field, argument in _LAW_ARGUMENTS.items()
        if getattr(process, field) is None
    }
    arguments = dict.fromkeys(names, "ContinuousState")
    arguments.update(dict.fromkeys(law_arguments.values(), "FloatND"))

    # The DAG publishes the precise scalar-coordinate signature on this variadic body.
    @no_type_check
    @with_signature(args=arguments, return_annotation="FloatND")
    def joint_function(**kwargs: FloatND) -> FloatND:
        return process.transform(
            standard=jnp.stack([kwargs[name] for name in names]),
            mean=kwargs.get(_LAW_ARGUMENTS["mean"]),
            cov=kwargs.get(_LAW_ARGUMENTS["cov"]),
        )

    setattr(joint_function, "_lcm_joint_normal_output", True)  # noqa: B010 - function metadata marker
    setattr(joint_function, "_lcm_law_arguments", MappingProxyType(law_arguments))  # noqa: B010 - parameter binding metadata
    return cast("UserFunction", joint_function)


def _output_function(*, group: ProcessName, dimension: int) -> UserFunction:
    # Each user-facing shock is a scalar, with the same annotation as an ordinary state.
    @no_type_check
    @with_signature(args={group: "FloatND"}, return_annotation="ContinuousState")
    def output_function(**kwargs: FloatND) -> FloatND:
        return kwargs[group][dimension]

    setattr(output_function, "_lcm_joint_normal_output", True)  # noqa: B010 - physical state output marker
    return cast("UserFunction", output_function)
