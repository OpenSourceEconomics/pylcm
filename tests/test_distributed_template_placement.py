"""Solver-owned continuation templates use their regime's placement."""

import jax

try:
    jax.config.update("jax_num_cpu_devices", 4)
    jax.config.update("jax_platform_name", "cpu")
    _BACKEND_ALREADY_INITIALIZED = False
except RuntimeError:
    _BACKEND_ALREADY_INITIALIZED = True

import dataclasses
import math

import jax.numpy as jnp
import numpy as np
import pytest

from lcm.exceptions import ExecutionPlanningError
from lcm.solver_api import ArtifactKey
from lcm.solvers import (
    ContinuationSpec,
    GridSearch,
    SolutionKernels,
    SolverBuildContext,
)
from lcm.typing import FloatND
from tests.test_distributed_placement import _make_three_type_model

pytestmark = pytest.mark.skipif(
    _BACKEND_ALREADY_INITIALIZED,
    reason="The four-device topology requires its own fresh process.",
)

_KEY = ArtifactKey(type_id="tests.template_placement", schema_version=1)


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True, kw_only=True)
class _Template:
    state_values: FloatND
    scalar: FloatND
    auxiliary: FloatND

    @property
    def artifact_key(self) -> ArtifactKey:
        return _KEY


@dataclasses.dataclass(frozen=True, kw_only=True)
class _TemplateSolver(GridSearch):
    misplaced_leaf: str | None = None
    misplaced_device: int = 3

    def build_period_kernels(self, *, context: SolverBuildContext) -> SolutionKernels:
        kernels = super().build_period_kernels(context=context)
        state_shape = tuple(
            value.size for value in context.state_action_space.states.values()
        )
        template = _Template(
            state_values=jnp.arange(math.prod(state_shape), dtype=float).reshape(
                state_shape
            ),
            scalar=jnp.asarray(7.0),
            auxiliary=jnp.arange(5, dtype=float),
        )
        template = context.place_on_regime_devices(template=template)
        if self.misplaced_leaf is not None:
            wrong = jax.device_put(
                getattr(template, self.misplaced_leaf),
                jax.devices()[self.misplaced_device],
            )
            template = dataclasses.replace(template, **{self.misplaced_leaf: wrong})
        return dataclasses.replace(
            kernels,
            continuation_spec=ContinuationSpec(template=template, artifact_key=_KEY),
        )


def _template(*, sharded: bool) -> _Template:
    model = _make_three_type_model(
        distributed=False,
        sharded=("type1",) if sharded else (),
        devices=None if sharded else (2, 3),
        solver=_TemplateSolver(),
    )
    template = model._regimes["working"].solution.continuation_template
    assert isinstance(template, _Template)
    return template


def test_four_devices_are_visible() -> None:
    assert len(jax.devices()) == 4


@pytest.mark.parametrize("leaf", ["state_values", "scalar", "auxiliary"])
@pytest.mark.parametrize("sharded", [False, True])
def test_every_template_leaf_uses_the_regimes_own_devices(
    *, leaf: str, sharded: bool
) -> None:
    value = getattr(_template(sharded=sharded), leaf)
    expected = (0, 1, 2) if sharded else (2,)
    assert tuple(sorted(device.id for device in value.sharding.device_set)) == expected


@pytest.mark.parametrize("leaf", ["state_values", "scalar", "auxiliary"])
def test_only_state_shaped_leaves_are_partitioned(*, leaf: str) -> None:
    value = getattr(_template(sharded=True), leaf)
    assert value.sharding.is_fully_replicated is (leaf != "state_values")


def test_template_placement_preserves_values() -> None:
    value = _template(sharded=True).state_values
    np.testing.assert_array_equal(np.asarray(value).ravel(), np.arange(value.size))


@pytest.mark.parametrize("leaf", ["state_values", "scalar", "auxiliary"])
@pytest.mark.parametrize("device", [0, 3])
def test_a_single_misplaced_leaf_is_refused_at_model_build(
    *, leaf: str, device: int
) -> None:
    with pytest.raises(ExecutionPlanningError, match="place_on_regime_devices"):
        _make_three_type_model(
            distributed=False,
            sharded=("type1",),
            solver=_TemplateSolver(misplaced_leaf=leaf, misplaced_device=device),
        )
