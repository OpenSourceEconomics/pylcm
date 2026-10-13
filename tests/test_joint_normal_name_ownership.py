"""Joint-normal names keep their owners through lowering, solving and simulation.

A runtime law field belongs to its `(regime, process group, field)`: an economic
state, action or function spelled `mean` or `cov` neither removes nor captures it.
Generated normal coordinates belong to the regime declaring the joint process: an
ordinary state of the same spelling in another regime keeps its own rows, and joint
rows always derive their coordinates from the physical shocks. Expected values
follow directly from each authored utility and the declared mean and covariance.
"""

from dataclasses import dataclass
from typing import cast, no_type_check

import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest
from dags import with_signature

import lcm
import lcm.typing
from lcm.exceptions import InvalidInitialConditionsError
from lcm.solver_api import SolutionResult
from lcm.typing import UserFunction


@lcm.categorical(ordered=False)
class OwnershipRegimeId:
    working: lcm.typing.ScalarInt
    done: lcm.typing.ScalarInt


def _two_period_model(
    *,
    states: dict[str, lcm.JointNormalIIDProcess | lcm.LinSpacedGrid],
    functions: dict[str, UserFunction],
    carried_name: str | None,
) -> lcm.Model:
    return lcm.Model(
        regimes={
            "working": lcm.Regime(
                states=states,
                functions=functions,
                state_transitions={}
                if carried_name is None
                else {carried_name: lcm.fixed_transition(carried_name)},
            ),
            "done": lcm.Regime(states=states, functions=functions),
        },
        n_periods=2,
        regime_id_class=OwnershipRegimeId,
        initial_nodes=lcm.InitialNodes(by_period={0: "working"}),
        edges={"working": {"done": lcm.Periods(values=(0,))}},
        fixed_params={"working": {"koopmans_aggregator": {"discount_factor": 0.9}}},
        execution_config=lcm.ExecutionConfig(device_memory_bytes=None),
    )


def _utility_with_mean_state(
    *,
    x: lcm.typing.ContinuousState,
    y: lcm.typing.ContinuousState,
    mean: lcm.typing.ContinuousState,
) -> lcm.typing.FloatND:
    return x + y + mean


def test_runtime_joint_mean_is_not_captured_by_an_ordinary_mean_state() -> None:
    """The ordinary state `mean` and the law field `joint.mean` stay distinct."""
    process = lcm.JointNormalIIDProcess(
        states=("x", "y"),
        n_points=3,
        gauss_hermite=True,
        mean=None,
        cov=((1.0, 0.0), (0.0, 1.0)),
    )
    model = _two_period_model(
        states={
            "mean": lcm.LinSpacedGrid(start=0.0, stop=1.0, n_points=2),
            "joint": process,
        },
        functions={"utility": _utility_with_mean_state},
        carried_name="mean",
    )
    params = {
        name: {"joint": {"mean": jnp.asarray((2.0, -1.0))}}
        for name in ("working", "done")
    }
    solution = model.solve(params=params, runtime_checks=True, log_level="off")
    result = model.lookup_policy(
        params=params,
        solution=solution,
        period=0,
        regime_name="working",
        states={
            "x": jnp.asarray((2.0, 2.0)),
            "y": jnp.asarray((-1.0, -1.0)),
            "mean": jnp.asarray((0.0, 1.0)),
        },
        keep_intermediates=("Q", "F", "U", "CE"),
    )
    # The ordinary state is carried unchanged and the law mean gives E[x+y]=1,
    # so a law that ignored its mean would fail as well as one that lost it.
    assert result.U is not None
    assert result.CE is not None
    tolerance = 64 * np.finfo(np.asarray(result.value).dtype).eps
    np.testing.assert_allclose(
        np.concatenate(
            (np.asarray(result.U), np.asarray(result.CE), np.asarray(result.value))
        ),
        (1.0, 2.0, 1.0, 2.0, 1.9, 3.8),
        atol=tolerance,
        rtol=0,
    )


def _polynomial(
    *, physical_names: tuple[str, str], extra_name: str | None
) -> UserFunction:
    arguments = dict.fromkeys(physical_names, "ContinuousState")
    if extra_name is not None:
        arguments[extra_name] = "FloatND"

    @no_type_check
    @with_signature(args=arguments, return_annotation="FloatND")
    def utility(**kwargs: lcm.typing.FloatND) -> lcm.typing.FloatND:
        left, right = (kwargs[name] for name in physical_names)
        value = left * left + left * right + 2 * right * right
        return value if extra_name is None else value + kwargs[extra_name]

    return cast("UserFunction", utility)


def _scalar_helper(*, scalar_level: lcm.typing.ContinuousState) -> lcm.typing.FloatND:
    return scalar_level + 0.25


_MEAN_0 = (2.0, -1.0)
_COV_0 = ((4.0, 1.0), (1.0, 2.0))
_MEAN_1 = (-1.0, 2.0)
_COV_1 = ((2.0, -0.5), (-0.5, 3.0))


_OWNER_CASES = (
    pytest.param("joint", ("x", "y"), "mean", "state", ("mean",), id="mean-state"),
    pytest.param("joint", ("x", "y"), "cov", "state", ("cov",), id="cov-state"),
    pytest.param("joint", ("x", "y"), "mean", "function", ("mean",), id="mean-helper"),
    pytest.param("joint", ("x", "y"), "cov", "function", ("cov",), id="cov-helper"),
    pytest.param("mean", ("x", "y"), None, "none", ("mean", "cov"), id="mean-group"),
    pytest.param("cov", ("x", "y"), None, "none", ("mean", "cov"), id="cov-group"),
    pytest.param(
        "joint", ("mean", "y"), None, "none", ("mean", "cov"), id="mean-physical"
    ),
    pytest.param(
        "joint", ("x", "cov"), None, "none", ("mean", "cov"), id="cov-physical"
    ),
    pytest.param(
        "innovations",
        ("left_shock", "right_shock"),
        None,
        "none",
        ("mean", "cov"),
        id="neutral-control",
    ),
)


def _owner_model(
    *,
    group: str,
    physical_names: tuple[str, str],
    collision_name: str | None,
    owner_kind: str,
    runtime_fields: tuple[str, ...],
) -> tuple[lcm.Model, str | None]:
    process = lcm.JointNormalIIDProcess(
        states=physical_names,
        n_points=3,
        gauss_hermite=True,
        mean=None if "mean" in runtime_fields else _MEAN_0,
        cov=None if "cov" in runtime_fields else _COV_0,
    )
    states: dict[str, lcm.JointNormalIIDProcess | lcm.LinSpacedGrid] = {group: process}
    functions = {
        "utility": _polynomial(physical_names=physical_names, extra_name=collision_name)
    }
    carried_name = None
    if owner_kind == "state":
        carried_name = collision_name
    elif owner_kind == "function":
        carried_name = "scalar_level"
        functions[cast("str", collision_name)] = _scalar_helper
    if carried_name is not None:
        states[carried_name] = lcm.LinSpacedGrid(start=0.0, stop=1.0, n_points=2)
    model = _two_period_model(
        states=states, functions=functions, carried_name=carried_name
    )
    return model, carried_name


@pytest.mark.slow
@pytest.mark.parametrize(
    ("group", "physical_names", "collision_name", "owner_kind", "runtime_fields"),
    _OWNER_CASES,
)
def test_process_parameter_ownership_survives_legal_identifier_changes(
    *,
    group: str,
    physical_names: tuple[str, str],
    collision_name: str | None,
    owner_kind: str,
    runtime_fields: tuple[str, ...],
) -> None:
    """Names cannot remove a law slot, capture its value, or keep a stale law."""
    model, carried_name = _owner_model(
        group=group,
        physical_names=physical_names,
        collision_name=collision_name,
        owner_kind=owner_kind,
        runtime_fields=runtime_fields,
    )
    template = model.get_params_template()
    slots = {
        regime_name: set(template[regime_name][group])
        for regime_name in ("working", "done")
    }
    assert slots == {"working": set(runtime_fields), "done": set(runtime_fields)}
    observed = []
    expected = []
    for trial in (0, 1):
        mean = _MEAN_1 if trial and "mean" in runtime_fields else _MEAN_0
        cov = _COV_1 if trial and "cov" in runtime_fields else _COV_0
        law = {"mean": jnp.asarray(mean), "cov": jnp.asarray(cov)}
        params = {
            regime_name: {group: {name: law[name] for name in runtime_fields}}
            for regime_name in ("working", "done")
        }
        solution = model.solve(params=params, runtime_checks=True, log_level="off")
        physical = {
            name: jnp.full(2, mean[index]) for index, name in enumerate(physical_names)
        }
        extra = np.zeros(2)
        if carried_name is not None:
            physical[carried_name] = jnp.asarray((0.0, 1.0))
            extra = np.asarray((0.0, 1.0)) + (0.25 if owner_kind == "function" else 0.0)
        result = model.lookup_policy(
            params=params,
            solution=solution,
            period=0,
            regime_name="working",
            states=physical,
            keep_intermediates=("Q", "F", "U", "CE"),
        )
        # E[X²+XY+2Y²] follows from the mean and covariance alone.
        at_mean = mean[0] ** 2 + mean[0] * mean[1] + 2 * mean[1] ** 2
        expectation = at_mean + cov[0][0] + cov[0][1] + 2 * cov[1][1]
        expected_u, expected_ce = at_mean + extra, expectation + extra
        assert result.U is not None
        assert result.CE is not None
        observed.append(
            np.concatenate(
                (np.asarray(result.U), np.asarray(result.CE), np.asarray(result.value))
            )
        )
        expected.append(
            np.concatenate((expected_u, expected_ce, expected_u + 0.9 * expected_ce))
        )
    tolerance = 512 * np.finfo(observed[0].dtype).eps
    np.testing.assert_allclose(
        np.stack(observed), np.stack(expected), atol=tolerance, rtol=0
    )


def _two_group_utility(
    *,
    x: lcm.typing.ContinuousState,
    y: lcm.typing.ContinuousState,
    u: lcm.typing.ContinuousState,
    v: lcm.typing.ContinuousState,
) -> lcm.typing.FloatND:
    return x * x + x * y + 2 * y * y + u * u + u * v + 2 * v * v


@pytest.mark.slow
def test_two_process_groups_do_not_share_their_runtime_law() -> None:
    """Identical field spellings in two process groups bind different laws."""
    states = {
        group: lcm.JointNormalIIDProcess(
            states=names, n_points=3, gauss_hermite=True, mean=None, cov=None
        )
        for group, names in (("wages", ("x", "y")), ("benefits", ("u", "v")))
    }
    regime = lcm.Regime(states=states, functions={"utility": _two_group_utility})
    model = lcm.Model(
        regimes={"working": regime, "done": regime},
        n_periods=2,
        regime_id_class=OwnershipRegimeId,
        initial_nodes=lcm.InitialNodes(by_period={0: "working"}),
        edges={"working": {"done": lcm.Periods(values=(0,))}},
        fixed_params={"working": {"koopmans_aggregator": {"discount_factor": 0.9}}},
        execution_config=lcm.ExecutionConfig(device_memory_bytes=None),
    )
    params = {
        name: {
            "wages": {"mean": jnp.asarray(_MEAN_0), "cov": jnp.asarray(_COV_0)},
            "benefits": {"mean": jnp.asarray(_MEAN_1), "cov": jnp.asarray(_COV_1)},
        }
        for name in ("working", "done")
    }
    solution = model.solve(params=params, runtime_checks=True, log_level="off")
    result = model.lookup_policy(
        params=params,
        solution=solution,
        period=0,
        regime_name="working",
        states={
            name: jnp.asarray((value,))
            for name, value in zip(
                ("x", "y", "u", "v"), (*_MEAN_0, *_MEAN_1), strict=True
            )
        },
        keep_intermediates=("Q", "F", "U", "CE"),
    )
    assert result.U is not None
    assert result.CE is not None
    tolerance = 1024 * np.finfo(np.asarray(result.value).dtype).eps
    np.testing.assert_allclose(
        np.concatenate(
            (np.asarray(result.U), np.asarray(result.CE), np.asarray(result.value))
        ),
        (11.0, 27.5, 35.75),
        atol=tolerance,
        rtol=0,
    )


@lcm.categorical(ordered=False)
class BoundaryRegimeId:
    A: lcm.typing.ScalarInt
    B: lcm.typing.ScalarInt


def _joint_utility(
    *, x: lcm.typing.ContinuousState, y: lcm.typing.ContinuousState
) -> lcm.typing.FloatND:
    return x + y


def _ordinary_utility(
    *, wages_normal_0: lcm.typing.ContinuousState
) -> lcm.typing.FloatND:
    return wages_normal_0


@pytest.fixture(scope="module")
def cross_regime_model() -> lcm.Model:
    """Two terminal start regimes: A declares joint `wages`, B an ordinary state."""
    return lcm.Model(
        regimes={
            "A": lcm.Regime(
                states={
                    "wages": lcm.JointNormalIIDProcess(
                        states=("x", "y"),
                        n_points=3,
                        gauss_hermite=True,
                        mean=(0.0, 0.0),
                        cov=((1.0, 0.0), (0.0, 1.0)),
                    )
                },
                functions={"utility": _joint_utility},
            ),
            "B": lcm.Regime(
                states={
                    "wages_normal_0": lcm.NormalIIDProcess(
                        n_points=3, gauss_hermite=True, mu=0.0, sigma=1.0
                    )
                },
                functions={"utility": _ordinary_utility},
            ),
        },
        n_periods=1,
        regime_id_class=BoundaryRegimeId,
        initial_nodes=lcm.InitialNodes(by_period={0: ("A", "B")}),
        edges={},
        execution_config=lcm.ExecutionConfig(device_memory_bytes=None),
    )


@pytest.fixture(scope="module")
def cross_regime_solution(cross_regime_model: lcm.Model) -> SolutionResult:
    return cross_regime_model.solve(params={}, log_level="off")


def _public_input(
    *,
    model: lcm.Model,
    rows: tuple[str, ...],
    states: dict[str, np.ndarray],
    as_frame: bool,
) -> dict[str, np.ndarray] | pd.DataFrame:
    values = {"period": np.zeros(len(rows), dtype=np.int32), **states}
    if as_frame:
        return pd.DataFrame({"regime_name": rows, **values})
    return {
        "regime_id": np.array(
            [int(model.regime_names_to_ids[name]) for name in rows], dtype=np.int32
        ),
        **values,
    }


_CROSS_REGIME_CASES = {
    "B_only": (("B",), {"wages_normal_0": np.array([0.5])}),
    "A_only": (("A",), {"x": np.array([0.25]), "y": np.array([-0.5])}),
    # The shared column has an ordinary owner in B; the A row derives its
    # coordinate from the physical shocks instead of this placeholder.
    "A_only_shared": (
        ("A",),
        {
            "x": np.array([0.25]),
            "y": np.array([-0.5]),
            "wages_normal_0": np.array([999.0]),
        },
    ),
    "mixed": (
        ("A", "B", "A"),
        {
            "x": np.array([0.25, np.nan, -0.75]),
            "y": np.array([-0.5, np.nan, 0.125]),
            "wages_normal_0": np.array([999.0, 0.5, -999.0]),
        },
    ),
}

_CROSS_REGIME_VALUES = {
    "B_only": (0.5,),
    "A_only": (-0.25,),
    "A_only_shared": (-0.25,),
    "mixed": (-0.25, 0.5, -0.625),
}


@pytest.mark.parametrize("as_frame", [False, True])
@pytest.mark.parametrize("case", ["B_only", "A_only", "A_only_shared", "mixed"])
def test_shared_coordinate_spelling_keeps_each_regimes_rows(
    *,
    cross_regime_model: lcm.Model,
    cross_regime_solution: SolutionResult,
    case: str,
    as_frame: bool,
) -> None:
    """Each row is valued from the inputs its own regime declares."""
    rows, states = _CROSS_REGIME_CASES[case]
    frame = (
        cross_regime_model.simulate(
            params={},
            solution=cross_regime_solution,
            initial_conditions=_public_input(
                model=cross_regime_model, rows=rows, states=states, as_frame=as_frame
            ),
            seed=521,
            runtime_checks=True,
            log_level="off",
        )
        .to_dataframe(use_labels=False)
        .sort_values(["subject_id", "period"])
    )
    np.testing.assert_allclose(
        frame["value"], _CROSS_REGIME_VALUES[case], atol=1e-6, rtol=1e-6
    )


@pytest.mark.parametrize("as_frame", [False, True])
def test_mixed_rows_publish_their_own_physical_states(
    *,
    cross_regime_model: lcm.Model,
    cross_regime_solution: SolutionResult,
    as_frame: bool,
) -> None:
    """Joint rows publish their physical shocks and the ordinary row its state."""
    rows, states = _CROSS_REGIME_CASES["mixed"]
    frame = (
        cross_regime_model.simulate(
            params={},
            solution=cross_regime_solution,
            initial_conditions=_public_input(
                model=cross_regime_model, rows=rows, states=states, as_frame=as_frame
            ),
            seed=521,
            runtime_checks=True,
            log_level="off",
        )
        .to_dataframe(use_labels=False)
        .sort_values(["subject_id", "period"])
    )
    published = np.concatenate(
        (
            frame.iloc[[0, 2]]["x"].to_numpy(),
            frame.iloc[[0, 2]]["y"].to_numpy(),
            frame.iloc[[1]]["wages_normal_0"].to_numpy(),
        )
    )
    np.testing.assert_allclose(
        published, (0.25, -0.75, -0.5, 0.125, 0.5), atol=1e-6, rtol=1e-6
    )


@pytest.mark.parametrize("as_frame", [False, True])
@pytest.mark.parametrize("kind", ["shared_only", "both_latents", "extra_latent"])
def test_latent_columns_cannot_replace_required_physical_shocks(
    *,
    cross_regime_model: lcm.Model,
    cross_regime_solution: SolutionResult,
    kind: str,
    as_frame: bool,
) -> None:
    """Joint rows still require physical shocks and refuse unowned coordinates."""
    if kind == "extra_latent":
        states = {
            "x": np.array([0.25]),
            "y": np.array([-0.5]),
            "wages_normal_1": np.array([0.0]),
        }
    else:
        states = {"wages_normal_0": np.array([0.25])}
        if kind == "both_latents":
            states["wages_normal_1"] = np.array([-0.5])
    initial = _public_input(
        model=cross_regime_model, rows=("A",), states=states, as_frame=as_frame
    )
    with pytest.raises(
        InvalidInitialConditionsError, match=r"requires (physical|initial) shocks"
    ):
        cross_regime_model.simulate(
            initial_conditions=initial,
            params={},
            solution=cross_regime_solution,
            seed=521,
            runtime_checks=True,
            log_level="off",
        )


def _sum_named(*, names: tuple[str, ...]) -> UserFunction:
    @no_type_check
    @with_signature(
        args=dict.fromkeys(names, "ContinuousState"), return_annotation="FloatND"
    )
    def utility(**kwargs: lcm.typing.FloatND) -> lcm.typing.FloatND:
        value = kwargs[names[0]]
        for name in names[1:]:
            value = value + kwargs[name]
        return value

    return cast("UserFunction", utility)


@dataclass(frozen=True)
class NamespaceBoundary:
    model: lcm.Model
    """One-period model whose two start regimes share a coordinate spelling."""
    solution: SolutionResult
    """The model's solution."""
    ordinary_name: str
    """The ordinary state in `done` spelled like a `working` coordinate."""
    other_coordinate: str
    """The `working` coordinate no regime declares as an ordinary state."""
    physical_names: tuple[str, str]
    """The physical shocks of the joint process in `working`."""


_BOUNDARY_CASES = tuple(
    pytest.param(
        (group, physical_names, axis, fold),
        id=f"{group}-axis{axis}-{'folded' if fold else 'unfolded'}",
    )
    for group, physical_names in (
        ("wages", ("x", "y")),
        ("labor_innovations", ("physical_left", "physical_right")),
    )
    for axis in (0, 1)
    for fold in (False, True)
)


@pytest.fixture(scope="module", params=_BOUNDARY_CASES)
def boundary(request: pytest.FixtureRequest) -> NamespaceBoundary:
    group, physical_names, axis, fold = request.param
    ordinary_name = f"{group}_normal_{axis}"
    process = lcm.JointNormalIIDProcess(
        states=physical_names,
        n_points=3,
        gauss_hermite=True,
        mean=(1.0, -2.0),
        cov=((1.0, 0.5), (0.5, 2.0)),
        fold=fold,
    )
    model = lcm.Model(
        regimes={
            "working": lcm.Regime(
                states={group: process},
                functions={"utility": _sum_named(names=physical_names)},
            ),
            "done": lcm.Regime(
                states={
                    ordinary_name: lcm.NormalIIDProcess(
                        n_points=3, gauss_hermite=True, mu=0.0, sigma=1.0
                    )
                },
                functions={"utility": _sum_named(names=(ordinary_name,))},
            ),
        },
        n_periods=1,
        regime_id_class=OwnershipRegimeId,
        initial_nodes=lcm.InitialNodes(by_period={0: ("working", "done")}),
        edges={},
        execution_config=lcm.ExecutionConfig(device_memory_bytes=None),
    )
    return NamespaceBoundary(
        model=model,
        solution=model.solve(params={}, runtime_checks=True, log_level="off"),
        ordinary_name=ordinary_name,
        other_coordinate=f"{group}_normal_{1 - axis}",
        physical_names=physical_names,
    )


@pytest.mark.slow
@pytest.mark.parametrize("as_frame", [False, True])
@pytest.mark.parametrize(
    "population", ["joint_only_shared", "ordinary_only", "mixed", "mixed_reversed"]
)
def test_cross_regime_input_and_publication_resolve_names_per_row(
    *, boundary: NamespaceBoundary, as_frame: bool, population: str
) -> None:
    """Generated labels neither reject nor overwrite another regime's state."""
    first, second = boundary.physical_names
    if population == "ordinary_only":
        rows = np.asarray(("done",))
        states = {boundary.ordinary_name: np.asarray((0.5,))}
        expected = np.asarray((0.5,))
    elif population == "joint_only_shared":
        rows = np.asarray(("working",))
        states = {
            first: np.asarray((1.25,)),
            second: np.asarray((-1.5,)),
            boundary.ordinary_name: np.asarray((999.0,)),
        }
        expected = np.asarray((-0.25,))
    else:
        rows = np.asarray(("done", "working", "done", "working"))
        states = {
            first: np.asarray((np.nan, 1.25, np.nan, 0.25)),
            second: np.asarray((np.nan, -1.5, np.nan, -2.75)),
            boundary.ordinary_name: np.asarray((0.5, 999.0, -0.25, -999.0)),
        }
        expected = np.asarray((0.5, -0.25, -0.25, -2.5))
        if population == "mixed_reversed":
            rows, expected = rows[::-1], expected[::-1]
            states = {name: value[::-1] for name, value in states.items()}
    initial: dict[str, np.ndarray] | pd.DataFrame = {
        "period": np.zeros(len(rows), dtype=np.int32),
        **states,
    }
    if as_frame:
        initial = pd.DataFrame({"regime_name": rows, **initial})
    else:
        initial["regime_id"] = np.asarray(
            [int(boundary.model.regime_names_to_ids[name]) for name in rows],
            dtype=np.int32,
        )
    frame = (
        boundary.model.simulate(
            params={},
            solution=boundary.solution,
            initial_conditions=initial,
            seed=521,
            runtime_checks=True,
            log_level="off",
        )
        .to_dataframe(use_labels=False)
        .sort_values(["subject_id", "period"])
    )
    tolerance = 128 * np.finfo(frame["value"].to_numpy().dtype).eps
    observed = [frame["value"].to_numpy()]
    wanted = [expected]
    assert boundary.other_coordinate not in frame.columns
    for regime_name in ("working", "done"):
        selected = rows == regime_name
        if not selected.any():
            continue
        names = (
            boundary.physical_names
            if regime_name == "working"
            else (boundary.ordinary_name,)
        )
        lookup = boundary.model.lookup_policy(
            params={},
            solution=boundary.solution,
            period=0,
            regime_name=regime_name,
            states={name: jnp.asarray(states[name][selected]) for name in names},
        )
        observed.append(np.asarray(lookup.value))
        wanted.append(expected[selected])
        for name in names:
            observed.append(frame.loc[selected, name].to_numpy())
            wanted.append(states[name][selected])
    np.testing.assert_allclose(
        np.concatenate(observed), np.concatenate(wanted), atol=tolerance, rtol=0
    )
