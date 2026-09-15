"""Labeled parameter uploads share the numeric entry admission boundary."""

from functools import partial
from typing import Literal, cast

import jax
import numpy as np
import pandas as pd
import pytest

from _lcm import pandas_utils
from lcm import ExecutionConfig
from lcm.exceptions import ExecutionPlanningError
from lcm.params import UserMappingLeaf, UserSequenceLeaf
from tests.simulation.test_entry_allocations import _owner
from tests.simulation.test_pandas_entry_admission import _NoDirectPandasUploads
from tests.solution.test_solution_result import _small_grid_search_inputs
from tests.test_models.stochastic import get_model
from tests.test_pandas_utils import _build_partner_probs_series


@pytest.mark.parametrize("fits", [True, False])
@pytest.mark.parametrize("form", ["scalar", "indexed", "empty", "mapping", "sequence"])
def test_series_uploads_use_the_entry_writer(
    *,
    form: Literal["scalar", "indexed", "empty", "mapping", "sequence"],
    fits: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Index scattering and wrapper traversal admit every completed array."""
    model = get_model(3)
    series = _build_partner_probs_series(model)
    key = "next_partner__probs_array"
    value: object = series
    if form == "scalar":
        key = "labor_income__wage"
        value = pd.Series([10.0])
    elif form == "empty":
        value = series.iloc[:0]
    elif form == "mapping":
        value = UserMappingLeaf({"inner": UserMappingLeaf({"table": series})})
    elif form == "sequence":
        value = UserSequenceLeaf((series, series.copy()))
    convert = partial(
        pandas_utils.convert_series_in_params,
        flat_params={"working_life": {key: value}},
        user_regimes=model.user_regimes,
        ages=model.ages,
        regime_names_to_ids=model.regime_names_to_ids,
    )
    expected = convert()
    owner = _owner(budget=2**20 if fits else 1)
    monkeypatch.setattr(
        pandas_utils, "jnp", _NoDirectPandasUploads(original=pandas_utils.jnp)
    )
    try:
        if fits:
            actual = convert(array_writer=owner)
            expected_leaves, expected_tree = jax.tree.flatten(expected)
            actual_leaves, actual_tree = jax.tree.flatten(actual)
            assert actual_tree == expected_tree
            for actual_leaf, expected_leaf in zip(
                actual_leaves, expected_leaves, strict=True
            ):
                np.testing.assert_array_equal(actual_leaf, expected_leaf)
        else:
            with pytest.raises(ExecutionPlanningError, match="budget"):
                convert(array_writer=owner)
    finally:
        owner.close()


def test_model_parameter_conversion_forwards_the_entry_writer(
    *, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The model's parameter boundary admits Series before canonical dtype casts."""
    model, params, _ = _small_grid_search_inputs(
        execution_config=ExecutionConfig(device_memory_bytes=2**28)
    )
    params = dict(params) | {"discount_factor": pd.Series([0.95])}
    owner = _owner(budget=2**20)
    monkeypatch.setattr(
        pandas_utils, "jnp", _NoDirectPandasUploads(original=pandas_utils.jnp)
    )
    try:
        actual = model._process_params(params, array_writer=owner)
        discount = cast(
            "jax.Array", actual["working_life"]["koopmans_aggregator__discount_factor"]
        )
        np.testing.assert_array_equal(
            discount, np.asarray([0.95], dtype=discount.dtype)
        )
    finally:
        owner.close()
