"""Public result and store constructors admit a mapping through one exact traversal.

`ValueStore`, `ArtifactStore`, and `SolutionResult` accept plain mappings at the public
boundary. A mapping can present coordinates that compare equal without being the same
address (`1` and `True`, `0` and `False`), can emit one logical address twice, or can
expose a key view that disagrees with its item traversal. Each raw coordinate is checked
exactly and for uniqueness before it is inserted anywhere, from the one item traversal
the constructor consumes, so no equality-based contraction can erase the evidence.
"""

from collections.abc import Iterator, Mapping
from typing import cast

import jax.numpy as jnp
import numpy as np
import pytest
from pandas.testing import assert_frame_equal

from lcm.persistence import load_solution
from lcm.solver_api import (
    SIMULATION_POLICY,
    ArtifactKey,
    ArtifactRef,
    ArtifactStore,
    OmissionReason,
    ResultRetention,
    SolutionMetadata,
    SolutionResult,
    ValueArraySchema,
    ValueStore,
)
from lcm.typing import FloatND
from tests.solution.test_solution_result import _small_grid_search_inputs

_REGIME = "alive"
_OTHER_REGIME = "retired"


class _ItemStream(Mapping[object, object]):
    """Schema-shaped mapping whose item traversal is its only authoritative view.

    `keys` optionally substitutes a different key view so a test can show that the
    constructor derives everything from the items it traverses. Passing
    `keys=None` makes the key view raise, proving the constructor never consults it.
    """

    def __init__(
        self,
        *,
        items: list[tuple[object, object]],
        keys: list[object] | None = None,
        max_traversals: int | None = None,
    ) -> None:
        self._items = tuple(items)
        self._keys = keys
        self._max_traversals = max_traversals
        self.traversals = 0

    def __getitem__(self, key: object) -> object:
        for candidate, value in self._items:
            if candidate == key:
                return value
        raise KeyError(key)

    def __iter__(self) -> Iterator[object]:
        if self._keys is None:
            raise RuntimeError("key view consulted")
        return iter(self._keys)

    def __len__(self) -> int:
        return len(self._items)

    def items(self) -> list[tuple[object, object]]:  # ty: ignore[invalid-method-override]
        self.traversals += 1
        if self._max_traversals is not None and self.traversals > self._max_traversals:
            raise RuntimeError("item traversal repeated")
        return list(self._items)


def _payload(value: float) -> FloatND:
    return jnp.asarray([value])


def _metadata(*, n_periods: int = 2) -> SolutionMetadata:
    return SolutionMetadata(
        retention=ResultRetention.VALUES,
        n_periods=n_periods,
        regime_names=(_REGIME, _OTHER_REGIME),
        solver_types={_REGIME: "example.Grid", _OTHER_REGIME: "example.Grid"},
        model_instance_id="model-1",
        params_fingerprint="0" * 64,
        value_schemas={
            (period, regime): ValueArraySchema(
                shape=(1,), dtype=str(_payload(0.0).dtype), axis_names=("wealth",)
            )
            for period in range(n_periods)
            for regime in (_REGIME, _OTHER_REGIME)
        },
    )


@pytest.mark.parametrize(
    ("exact", "alias"), [(0, False), (1, True)], ids=["zero-False", "one-True"]
)
@pytest.mark.parametrize(
    "alias_first", [False, True], ids=["exact-first", "alias-first"]
)
@pytest.mark.parametrize("form", ["flat", "nested"])
def test_value_store_rejects_boolean_period_alias_in_either_order(
    *, exact: int, alias: bool, alias_first: bool, form: str
) -> None:
    """A Boolean period alias is refused whether it arrives before or after its
    equal exact integer, in flat and nested form."""
    coordinates = [(alias, _payload(2.0)), (exact, _payload(1.0))]
    if not alias_first:
        coordinates.reverse()
    if form == "flat":
        stream = _ItemStream(
            items=[((period, _REGIME), value) for period, value in coordinates]
        )
    else:
        stream = _ItemStream(
            items=[(period, {_REGIME: value}) for period, value in coordinates]
        )

    with pytest.raises(TypeError, match="exact ints"):
        ValueStore(stream)


@pytest.mark.parametrize("form", ["flat", "nested-period", "nested-regime"])
def test_value_store_rejects_a_repeated_logical_coordinate(*, form: str) -> None:
    """One logical address emitted twice is refused instead of silently contracted."""
    first, second = _payload(1.0), _payload(2.0)
    if form == "flat":
        stream = _ItemStream(items=[((0, _REGIME), first), ((0, _REGIME), second)])
    elif form == "nested-period":
        stream = _ItemStream(items=[(0, {_REGIME: first}), (0, {_REGIME: second})])
    else:
        stream = _ItemStream(
            items=[(0, _ItemStream(items=[(_REGIME, first), (_REGIME, second)]))]
        )

    with pytest.raises(ValueError, match="twice"):
        ValueStore(stream)


def test_value_store_rejects_mixed_flat_and_nested_keys() -> None:
    """A mapping is either all `(period, regime)` tuples or all periods."""
    stream = _ItemStream(
        items=[((0, _REGIME), _payload(1.0)), (1, {_REGIME: _payload(2.0)})]
    )

    with pytest.raises(TypeError, match="either"):
        ValueStore(stream)


def test_value_store_rejects_a_nested_period_whose_value_is_not_a_mapping() -> None:
    stream = _ItemStream(items=[(0, _payload(1.0))])

    with pytest.raises(TypeError, match="mapping of regime names"):
        ValueStore(stream)


@pytest.mark.parametrize("form", ["flat", "nested"])
def test_value_store_reads_its_form_and_coordinates_from_the_item_traversal(
    *, form: str
) -> None:
    """Flat-versus-nested classification and every coordinate come from the items;
    a key view that disagrees, or cannot be read at all, is never consulted."""
    payload = _payload(3.0)
    if form == "flat":
        stream = _ItemStream(items=[((0, _REGIME), payload)], keys=[0])
    else:
        stream = _ItemStream(items=[(0, {_REGIME: payload})], keys=[(0, _REGIME)])

    store = ValueStore(stream)

    assert list(store) == [0]
    assert list(store[0]) == [_REGIME]
    np.testing.assert_array_equal(store[0][_REGIME], payload)


def test_value_store_consumes_exactly_one_item_traversal() -> None:
    stream = _ItemStream(items=[((0, _REGIME), _payload(1.0))], max_traversals=1)

    store = ValueStore(stream)

    assert stream.traversals == 1
    assert (0, _REGIME) in {
        (period, regime) for period in store for regime in store[period]
    }


@pytest.mark.parametrize("form", ["flat", "nested"])
def test_value_store_preserves_every_addressed_payload_of_a_valid_mapping(
    *, form: str
) -> None:
    """Valid flat and nested public mappings keep each payload at its address."""
    expected = {
        (period, regime): _payload(float(10 * period + index))
        for period in range(2)
        for index, regime in enumerate((_REGIME, _OTHER_REGIME))
    }
    if form == "flat":
        source = cast("Mapping[object, object]", dict(expected))
    else:
        source = cast(
            "Mapping[object, object]",
            {
                period: {
                    regime: expected[(period, regime)]
                    for regime in (_REGIME, _OTHER_REGIME)
                }
                for period in range(2)
            },
        )

    store = ValueStore(source)

    materialized = {
        (period, regime): value
        for period, regime_to_value in store.materialize().items()
        for regime, value in regime_to_value.items()
    }
    assert set(materialized) == set(expected)
    for coordinate, value in expected.items():
        np.testing.assert_array_equal(materialized[coordinate], value)


def test_solution_result_rejects_an_aliased_value_mapping_at_construction() -> None:
    """The public result constructor fails closed before any snapshot or save."""
    stream = _ItemStream(
        items=[((1, _REGIME), _payload(1.0)), ((True, _REGIME), _payload(2.0))]
    )

    with pytest.raises(TypeError, match="exact ints"):
        SolutionResult(
            values=cast("Mapping[int, Mapping[str, FloatND]]", stream),
            metadata=_metadata(),
        )


def test_public_value_mapping_survives_save_load_and_replay(tmp_path) -> None:
    """A result rebuilt from a valid nested public mapping persists, restores, and
    replays exactly like the solve that produced it."""
    model, params, initial_conditions = _small_grid_search_inputs()
    solved = model.solve(params=params, log_level="off")
    materialized = cast("ValueStore", solved.values).materialize()
    nested = {
        period: dict(regime_to_value)
        for period, regime_to_value in materialized.items()
    }
    rebuilt = SolutionResult(
        values=nested,
        metadata=solved.metadata,
        retained_continuations=solved.retained_continuations,
        replay_artifacts=solved.replay_artifacts,
        auxiliary_artifacts=solved.auxiliary_artifacts,
        omissions=dict(solved.omissions),
        diagnostics=solved.diagnostics,
    )
    object.__setattr__(rebuilt, "_artifact_authority", solved._artifact_authority)

    restored = load_solution(path=rebuilt.save(path=tmp_path / "rebuilt.lcm"))

    for period, regime_to_value in (
        cast("ValueStore", solved.values).materialize().items()
    ):
        for regime, value in regime_to_value.items():
            np.testing.assert_array_equal(
                restored.value(period=period, regime=regime), value
            )
    expected = model.simulate(
        params=params,
        initial_conditions=initial_conditions,
        solution=solved,
        log_level="off",
    ).to_dataframe()
    replayed = model.simulate(
        params=params,
        initial_conditions=initial_conditions,
        solution=restored,
        log_level="off",
    ).to_dataframe()
    assert_frame_equal(replayed, expected)


def _ref(*, period: int, regime: str = _REGIME) -> ArtifactRef:
    return ArtifactRef(period=period, regime=regime, key=SIMULATION_POLICY)


def test_artifact_store_rejects_a_repeated_address_at_construction() -> None:
    stream = _ItemStream(items=[(_ref(period=0), object()), (_ref(period=0), object())])

    with pytest.raises(ValueError, match="twice"):
        ArtifactStore(cast("Mapping[ArtifactRef, object]", stream))


def test_artifact_store_rejects_a_nonexact_address_at_construction() -> None:
    """A tuple that spells an address is not an `ArtifactRef` and never enters."""
    stream = _ItemStream(items=[((0, _REGIME, ArtifactKey(type_id="x")), object())])

    with pytest.raises(TypeError, match="exact ArtifactRef"):
        ArtifactStore(cast("Mapping[ArtifactRef, object]", stream))


def test_artifact_store_consumes_exactly_one_item_traversal() -> None:
    stream = _ItemStream(
        items=[(_ref(period=0), object()), (_ref(period=1), object())], max_traversals=1
    )

    store = ArtifactStore(cast("Mapping[ArtifactRef, object]", stream))

    assert stream.traversals == 1
    assert set(store) == {_ref(period=0), _ref(period=1)}


def test_solution_result_rejects_a_nonexact_omission_address_at_construction() -> None:
    values = ValueStore({(0, _REGIME): _payload(1.0)})

    with pytest.raises(TypeError, match="exact ArtifactRef"):
        SolutionResult(
            values=values,
            metadata=_metadata(),
            omissions=cast(
                "Mapping[ArtifactRef, OmissionReason]",
                {(0, _REGIME, SIMULATION_POLICY): OmissionReason.NOT_REQUESTED},
            ),
        )


def test_solution_result_rejects_a_repeated_omission_address_at_construction() -> None:
    values = ValueStore({(0, _REGIME): _payload(1.0)})
    stream = _ItemStream(
        items=[
            (_ref(period=0), OmissionReason.NOT_REQUESTED),
            (_ref(period=0), OmissionReason.NOT_PERSISTED),
        ]
    )

    with pytest.raises(ValueError, match="twice"):
        SolutionResult(
            values=values,
            metadata=_metadata(),
            omissions=cast("Mapping[ArtifactRef, OmissionReason]", stream),
        )


def test_solution_result_rejects_a_nonexact_omission_reason_at_construction() -> None:
    values = ValueStore({(0, _REGIME): _payload(1.0)})

    with pytest.raises(TypeError, match="exact OmissionReason"):
        SolutionResult(
            values=values,
            metadata=_metadata(),
            omissions=cast(
                "Mapping[ArtifactRef, OmissionReason]",
                {_ref(period=0): OmissionReason.NOT_REQUESTED.value},
            ),
        )
