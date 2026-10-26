"""Admit original EGM preparation stages through the existing entry owner."""

from functools import partial

import numpy as np

from _lcm.dtypes import canonical_float_dtype
from _lcm.egm.carry import EGMCarry, _build_template_egm_carry_at_dtype
from _lcm.egm.query_data import (
    EGMQueryData,
    _build_template_egm_query_data_at_dtype,
    _template_query_shapes,
)
from _lcm.materialization import DatedGenerationMaterializer


def prepare_egm_carry(
    *,
    n_rows: int,
    leading_shape: tuple[int, ...],
    query_data: EGMQueryData | None,
    materializer: DatedGenerationMaterializer,
) -> EGMCarry:
    """Replay original linspace/broadcast/zero stages with explicit query roots."""
    return materializer.evaluate_validation(
        function=partial(
            _build_template_egm_carry_at_dtype,
            n_rows=n_rows,
            leading_shape=leading_shape,
            dtype=np.dtype(canonical_float_dtype()),
        ),
        arguments={"query_data": query_data},
    )


def prepare_egm_query_data(
    *,
    state_shape: tuple[int, ...],
    action_shape: tuple[int, ...],
    row_capacity: int,
    interval_capacity: int,
    liquid_corner: bool,
    materializer: DatedGenerationMaterializer,
) -> EGMQueryData:
    """Admit the original empty query/status records, preserving their stages."""
    rows, row_shape, corner_shape, liquid_shape = _template_query_shapes(
        state_shape=state_shape,
        action_shape=action_shape,
        row_capacity=row_capacity,
        interval_capacity=interval_capacity,
        outer_capacity=0,
    )
    return materializer.evaluate_validation(
        function=partial(
            _build_template_egm_query_data_at_dtype,
            state_shape=state_shape,
            rows=rows,
            row_shape=row_shape,
            corner_shape=corner_shape,
            liquid_shape=liquid_shape,
            row_capacity=row_capacity,
            interval_capacity=interval_capacity,
            liquid_corner=liquid_corner,
            dtype=np.dtype(canonical_float_dtype()),
        ),
        arguments={},
    )
