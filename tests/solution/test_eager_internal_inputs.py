"""Public eager producer-consumer graphs retain their complete input contract."""

import gc
import weakref
from functools import partial
from typing import Any

import jax
import numpy as np
import pytest

from _lcm.execution.output_layout import PlannedCore
from _lcm.solution.negm import _KEEPER_CARRY, _KEEPER_VALUE
from lcm import Model
from tests.conftest import assert_agrees_to_ulp
from tests.test_models import n_nbegm_toy as toy


def test_public_eager_negm_consumes_actual_keeper_outputs(
    *, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The sweep reads actual keeper owners and publishes the compiled solve's V."""
    with monkeypatch.context() as builder:
        builder.setattr(toy, "Model", partial(Model, enable_jit=False))
        model = toy.build_model(variant="negm", n_periods=2)
    keeper_refs: list[weakref.ReferenceType[jax.Array]] = []
    keeper_copies: list[np.ndarray] = []
    handoffs: list[int] = []
    original = PlannedCore.__call__

    def observe(core: PlannedCore, *args: object, **kwargs: Any) -> object:
        if core.name == "outer_sweep":
            leaves = jax.tree.leaves((kwargs[_KEEPER_VALUE], kwargs[_KEEPER_CARRY]))
            assert len(leaves) == len(keeper_refs) > 1
            for leaf, reference, expected in zip(
                leaves, keeper_refs, keeper_copies, strict=True
            ):
                assert leaf is reference()
                assert not leaf.is_deleted()
                np.testing.assert_array_equal(leaf, expected)
            handoffs.append(len(leaves))
        output = original(core, *args, **kwargs)
        if core.name == "keeper":
            assert isinstance(output, tuple)
            leaves = jax.tree.leaves((output[0], output[1]))
            keeper_refs[:] = [weakref.ref(leaf) for leaf in leaves]
            keeper_copies[:] = [np.asarray(leaf).copy() for leaf in leaves]
        return output

    with monkeypatch.context() as probe:
        probe.setattr(PlannedCore, "__call__", observe)
        solution = model.solve(params={"discount_factor": 0.95}, log_level="off")
    assert len(handoffs) == 1
    gc.collect()
    assert all((leaf := ref()) is None or leaf.is_deleted() for ref in keeper_refs)
    compiled = toy.build_model(variant="negm", n_periods=2).solve(
        params={"discount_factor": 0.95}, log_level="off"
    )
    assert solution.values.keys() == compiled.values.keys()
    for period, values in solution.values.items():
        assert values.keys() == compiled.values[period].keys()
        for regime, value in values.items():
            assert np.isfinite(np.asarray(value)).all()
            # Existing NEGM sweep/per-node code-generation contract, unchanged.
            assert_agrees_to_ulp(
                got=value, expected=compiled.values[period][regime], n_ulp=16
            )
