"""A colliding participation read retains its actual persistent-process code reader.

The only code consumer is the participation predicate: its explicit read ``z`` is 10,
while landed physical z=0 has ordinal 1 on [-1, 0, 1]. Target utility is 6 and
fallback utility is 10, so the literal source value is 3 at beta=1/2. A merged
read/physical channel changes that choice. Off-node entries and paths must use
existing process-code admission, and both phase bindings must stand on nodes.
"""

import cloudpickle
import jax.numpy as jnp
import numpy as np
import pytest

from lcm import (
    AgeGrid,
    ByAge,
    Model,
    ParticipationCheck,
    ParticipationRead,
    Phased,
    ProjectedRegimeValue,
    Regime,
    StakeholderRoute,
    TauchenAR1Process,
    Transition,
    categorical,
)
from lcm.exceptions import (
    InvalidSimulationInputError,
    ModelInitializationError,
    RegimeInitializationError,
)
from lcm.typing import BoolND, ContinuousState, DiscreteState, FloatND, ScalarInt


@categorical(ordered=False)
class _RegimeId:
    source: ScalarInt
    target: ScalarInt
    fallback: ScalarInt


def _zero() -> FloatND:
    return jnp.asarray(0.0)


def _six(z: ContinuousState) -> FloatND:
    return 6.0 + 0.0 * z


def _ten() -> FloatND:
    return jnp.asarray(10.0)


def _entry_zero() -> ContinuousState:
    return jnp.asarray(0.0)


def _entry_quarter() -> ContinuousState:
    return jnp.asarray(0.25)


def _predicate(*, z: FloatND, z_code: DiscreteState) -> BoolND:
    return (z > 5.0) & (z_code == 1)


def _process(*, continuous: bool):
    return TauchenAR1Process.on_nodes(
        nodes=np.array([-1.0, 0.0, 1.0]),
        rho=0.5,
        sigma=1.0,
        simulate_draws="continuous" if continuous else "nodes",
    )


def _model(*, off_node: bool = False, continuous_phase: str | None = None) -> Model:
    process = _process(continuous=False)
    if continuous_phase is not None:
        process = Phased(
            solve=_process(continuous=continuous_phase == "solve"),
            simulate=_process(continuous=continuous_phase == "simulate"),
        )
    return Model(
        regimes={
            "source": Regime(
                functions={"utility": _zero},
                state_transitions={
                    "z": {"target": _entry_quarter if off_node else _entry_zero}
                },
            ),
            "target": Regime(states={"z": process}, functions={"utility": _six}),
            "fallback": Regime(functions={"utility": _ten}),
        },
        edges={
            "source": Transition(
                targets={"target": 0, "fallback": 0},
                law=ByAge(cases={0: "target"}),
                participation={
                    "target": ParticipationCheck(
                        predicate=_predicate,
                        reads={
                            "z": ParticipationRead(
                                kind="Raw", regime="fallback", projection={}
                            )
                        },
                        routes={
                            "only": StakeholderRoute(
                                fallback=ProjectedRegimeValue(
                                    regime="fallback", projection={}
                                )
                            )
                        },
                    )
                },
            )
        },
        ages=AgeGrid(exact_values=(0, 1)),
        initial_nodes={0: "source"},
        regime_id_class=_RegimeId,
        enable_jit=False,
    )


def test_participation_read_alias_keeps_the_landed_process_node_code() -> None:
    model = _model()
    assert "edges" not in model.get_params_template()
    params = {"discount_factor": 0.5}
    solution = model.solve(params=params, log_level="off")
    np.testing.assert_array_equal(solution.values[0]["source"], 3.0)
    frame = model.simulate(
        params=params,
        solution=solution,
        initial_conditions={
            "age": jnp.array([0.0]),
            "regime_id": jnp.array([_RegimeId.source], dtype=jnp.int32),
        },
        shock_paths={"z": np.array([0.0, 0.0])},
        seed=0,
        log_level="off",
    ).to_dataframe(use_categorical_codes=True)
    landed = frame.loc[frame["period"] == 1]
    assert landed["regime_name"].tolist() == ["target"]
    np.testing.assert_array_equal(landed["z"], [0.0])
    np.testing.assert_array_equal(landed["value"], [6.0])


def test_participation_read_alias_preserves_fixed_entry_process_node_admission() -> (
    None
):
    with pytest.raises(
        ModelInitializationError,
        match=r"entry law of 'z'.*'source'.*'target'.*off the process's nodes",
    ):
        _model(off_node=True)


def test_participation_read_alias_model_pickle_replays_the_original_solution() -> None:
    model = _model()
    params = {"discount_factor": 0.5}
    solution = model.solve(params=params, log_level="off")
    restored = cloudpickle.loads(cloudpickle.dumps(model))
    assert isinstance(restored, Model)
    assert (
        restored._solution_model_instance_id
        == model._solution_model_instance_id
        == solution.metadata.model_instance_id
    )
    assert restored._model_structure_fingerprint == model._model_structure_fingerprint
    frame = restored.simulate(
        params=params,
        solution=solution,
        initial_conditions={
            "age": jnp.array([0.0]),
            "regime_id": jnp.array([_RegimeId.source], dtype=jnp.int32),
        },
        shock_paths={"z": np.array([0.0, 0.0])},
        seed=0,
        log_level="off",
    ).to_dataframe(use_categorical_codes=True)
    source = frame.loc[frame["period"] == 0]
    np.testing.assert_array_equal(source["subject_id"], [0])
    np.testing.assert_array_equal(source["value"], [3.0])
    landed = frame.loc[frame["period"] == 1]
    np.testing.assert_array_equal(landed["subject_id"], [0])
    assert landed["regime_name"].tolist() == ["target"]
    np.testing.assert_array_equal(landed["z"], [0.0])
    np.testing.assert_array_equal(landed["value"], [6.0])


def test_participation_read_alias_preserves_process_path_node_admission() -> None:
    model = _model()
    params = {"discount_factor": 0.5}
    solution = model.solve(params=params, log_level="off")
    with pytest.raises(
        InvalidSimulationInputError,
        match=r"shock_paths 'z'.*period 1.*'target'.*off the process's nodes",
    ):
        model.simulate(
            params=params,
            solution=solution,
            initial_conditions={
                "age": jnp.array([0.0]),
                "regime_id": jnp.array([_RegimeId.source], dtype=jnp.int32),
            },
            shock_paths={"z": np.array([0.0, 0.25])},
            seed=0,
            log_level="off",
        )


@pytest.mark.parametrize("phase", ["solve", "simulate"])
def test_participation_read_alias_uses_both_canonical_process_phase_bindings(
    phase: str,
) -> None:
    with pytest.raises(
        RegimeInitializationError, match=r"z_code.*simulate_draws='nodes'"
    ):
        _model(continuous_phase=phase)
