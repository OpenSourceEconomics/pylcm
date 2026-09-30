import itertools
from types import MappingProxyType

import jax
import jax.numpy as jnp
import pytest
from beartype.roar import BeartypeCallHintViolation
from numpy.testing import assert_array_almost_equal as aaae

from _lcm.utils.dispatchers import (
    map_over_leading_axis,
    productmap,
    simulation_spacemap,
    vmap_1d,
)
from _lcm.utils.functools import allow_args
from lcm.exceptions import FunctionDispatchError


@pytest.mark.parametrize(("batch_size", "expected"), [(0, "vmap"), (2, "lax")])
def test_map_over_leading_axis_selects_the_declared_execution_window(
    *, monkeypatch: pytest.MonkeyPatch, batch_size: int, expected: str
) -> None:
    """A positive batch size reaches ``lax.map`` as its actual static window."""
    selected: list[tuple[str, int | None]] = []
    original_lax_map = jax.lax.map
    original_vmap = jax.vmap

    # keyword-only-exempt: library-callback=_lcm.utils.dispatchers.map_over_leading_axis
    def lax_spy(func, xs, *, batch_size=None):
        selected.append(("lax", batch_size))
        return original_lax_map(func, xs, batch_size=batch_size)

    def vmap_spy(func, *args, **kwargs):
        selected.append(("vmap", None))
        return original_vmap(func, *args, **kwargs)

    monkeypatch.setattr(jax.lax, "map", lax_spy)
    monkeypatch.setattr(jax, "vmap", vmap_spy)
    result = map_over_leading_axis(
        func=lambda row: row + 1,
        xs=jnp.arange(4),
        batch_size=batch_size,
    )

    assert result.tolist() == [1, 2, 3, 4]
    assert selected == [(expected, batch_size if expected == "lax" else None)]


def _scan_body(row: dict[str, jax.Array]) -> dict[str, jax.Array]:
    """Row body with a scan inside, so the trace is not a single primitive."""

    # keyword-only-exempt: library-callback=jax.lax.scan
    def step(carry, value):
        carry = carry * row["scale"] + value
        return carry, carry

    total, path = jax.lax.scan(step, jnp.float32(0.0), row["values"])
    return {"total": total, "path": path, "index": row["index"] * 2}


def _scan_tree(n_rows: int) -> dict[str, jax.Array]:
    key = jax.random.key(n_rows)
    return {
        "values": jax.random.normal(key, (n_rows, 3), dtype=jnp.float32),
        "scale": jnp.linspace(0.5, 1.5, n_rows, dtype=jnp.float32),
        "index": jnp.arange(n_rows, dtype=jnp.int32),
    }


@pytest.mark.parametrize(("n_rows", "batch_size"), [(1016, 64), (1000, 64), (5, 4)])
def test_map_over_leading_axis_matches_unpadded_lax_map_bitwise(
    *, n_rows: int, batch_size: int
) -> None:
    """Every real row equals what ``jax.lax.map`` computes on the unpadded input."""
    xs = _scan_tree(n_rows)
    expected = jax.lax.map(_scan_body, xs, batch_size=batch_size)
    result = map_over_leading_axis(func=_scan_body, xs=xs, batch_size=batch_size)
    for name in expected:
        assert jnp.array_equal(result[name], expected[name]).item(), name


@pytest.mark.parametrize(
    ("n_rows", "batch_size", "expected_traces"),
    [(1016, 64, 1), (1000, 64, 2), (5, 4, 2)],
)
def test_map_over_leading_axis_traces_the_body_once_when_padding_is_cheap(
    *, n_rows: int, batch_size: int, expected_traces: int
) -> None:
    """A remainder batch costs a second trace unless padding it away is cheap.

    Padding to the next multiple of the batch size adds rows that are also
    evaluated; the body is padded away only when they are at most a small share
    of the axis, otherwise the remainder batch keeps its separate trace.
    """
    traces: list[int] = []

    def counting_body(row: dict[str, jax.Array]) -> dict[str, jax.Array]:
        traces.append(1)
        return _scan_body(row)

    jax.make_jaxpr(
        lambda xs: map_over_leading_axis(
            func=counting_body, xs=xs, batch_size=batch_size
        )
    )(_scan_tree(n_rows))
    assert len(traces) == expected_traces


@pytest.mark.parametrize(("n_rows", "batch_size"), [(0, 4), (4, 4), (4, 8), (4, 0)])
def test_map_over_leading_axis_leaves_covered_or_empty_axes_alone(
    *, n_rows: int, batch_size: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An empty axis or a window covering it takes the vectorized route unpadded."""
    seen: list[tuple[int, ...]] = []
    original_vmap = jax.vmap

    def vmap_spy(func, *args, **kwargs):
        def record(xs):
            seen.append(tuple(leaf.shape for leaf in jax.tree.leaves(xs)))
            return original_vmap(func, *args, **kwargs)(xs)

        return record

    monkeypatch.setattr(jax, "vmap", vmap_spy)
    result = map_over_leading_axis(
        func=lambda row: row + 1, xs=jnp.arange(n_rows), batch_size=batch_size
    )
    assert result.shape == (n_rows,)
    assert seen == [((n_rows,),)]


@pytest.mark.parametrize(("n_rows", "batch_size"), [(1016, 64), (999, 8)])
def test_map_over_leading_axis_padded_rows_never_reach_the_outputs(
    *, n_rows: int, batch_size: int
) -> None:
    """Every output leaf keeps exactly ``n_rows`` on its leading axis."""
    result = map_over_leading_axis(
        func=_scan_body, xs=_scan_tree(n_rows), batch_size=batch_size
    )
    assert {name: leaf.shape[0] for name, leaf in result.items()} == {
        "total": n_rows,
        "path": n_rows,
        "index": n_rows,
    }


# keyword-only-exempt: library-callback=_lcm.utils.dispatchers.productmap
def f(a, *, b, c):
    """Tests that dispatchers can handle standard arguments and keyword-only arguments.

    a is positional-or-keyword, b and c are keyword-only
    """
    return jnp.sin(a) + jnp.cos(b) + jnp.tan(c)


@pytest.fixture
def setup_productmap_f():
    return {
        "a": jnp.linspace(-5, 5, 10),
        "b": jnp.linspace(0, 3, 7),
        "c": jnp.linspace(1, 5, 5),
    }


@pytest.fixture
def expected_productmap_f():
    grids = {
        "a": jnp.linspace(-5, 5, 10),
        "b": jnp.linspace(0, 3, 7),
        "c": jnp.linspace(1, 5, 5),
    }

    helper = jnp.array(list(itertools.product(*grids.values()))).T
    return allow_args(f)(*helper).reshape(10, 7, 5)


@pytest.mark.parametrize(
    ("func", "args", "grids", "expected"),
    [
        (f, ["a", "b", "c"], "setup_productmap_f", "expected_productmap_f"),
    ],
)
def test_productmap_with_all_arguments_mapped(*, func, args, grids, expected, request):
    grids = request.getfixturevalue(grids)
    expected = request.getfixturevalue(expected)

    variables = tuple(args)
    decorated = productmap(
        func=func, variables=variables, batch_sizes=dict.fromkeys(variables, 0)
    )

    calculated = decorated(**grids)
    aaae(calculated, expected)


def test_productmap_with_positional_args(setup_productmap_f):
    decorated = productmap(
        func=f, variables=("a", "b", "c"), batch_sizes=dict.fromkeys(("a", "b", "c"), 0)
    )
    match = (
        "This function has been decorated so that it allows only kwargs, but was "
        "called with positional arguments."
    )
    with pytest.raises(ValueError, match=match):
        decorated(*setup_productmap_f.values())  # ty: ignore[missing-argument]


def test_productmap_change_arg_order(*, setup_productmap_f, expected_productmap_f):
    expected = jnp.transpose(expected_productmap_f, (1, 0, 2))

    decorated = productmap(
        func=f, variables=("b", "a", "c"), batch_sizes=dict.fromkeys(("b", "a", "c"), 0)
    )
    calculated = decorated(**setup_productmap_f)

    aaae(calculated, expected)


def test_productmap_with_all_arguments_mapped_some_len_one():
    grids = {
        "a": jnp.array([1]),
        "b": jnp.array([2]),
        "c": jnp.linspace(1, 5, 5),
    }

    helper = jnp.array(list(itertools.product(*grids.values()))).T

    expected = allow_args(f)(*helper).reshape(1, 1, 5)

    decorated = productmap(
        func=f, variables=("a", "b", "c"), batch_sizes=dict.fromkeys(("a", "b", "c"), 0)
    )
    calculated = decorated(**grids)
    aaae(calculated, expected)


def test_productmap_with_some_arguments_mapped():
    grids = {
        "a": jnp.linspace(-5, 5, 10),
        "b": 1,
        "c": jnp.linspace(1, 5, 5),
    }

    helper = jnp.array(list(itertools.product(grids["a"], [grids["b"]], grids["c"]))).T

    expected = allow_args(f)(*helper).reshape(10, 5)

    decorated = productmap(
        func=f, variables=("a", "c"), batch_sizes=dict.fromkeys(("a", "c"), 0)
    )
    calculated = decorated(**grids)
    aaae(calculated, expected)


@pytest.mark.parametrize("batch_size", [0, 1, 2])
def test_productmap_batch_size_produces_same_result(batch_size):
    grids = {
        "a": jnp.linspace(-5, 5, 4),
        "b": jnp.linspace(1, 5, 3),
    }

    def h(*, a, b):
        return a**2 + b

    reference = productmap(
        func=h, variables=("a", "b"), batch_sizes=dict.fromkeys(("a", "b"), 0)
    )(**grids)
    batched = productmap(
        func=h,
        variables=("a", "b"),
        batch_sizes={"a": batch_size, "b": batch_size},
    )(**grids)
    aaae(batched, reference)


def test_productmap_with_some_argument_mapped_twice():
    error_msg = "Same argument provided more than once."
    with pytest.raises(ValueError, match=error_msg):
        productmap(
            func=f,
            variables=("a", "a", "c"),
            batch_sizes=dict.fromkeys(("a", "a", "c"), 0),
        )


def test_productmap_rejects_positional_only():
    # keyword-only-exempt: library-callback=_lcm.utils.dispatchers.productmap
    def h(a, /, *, b):
        return a + b

    with pytest.raises(FunctionDispatchError, match="POSITIONAL_ONLY"):
        productmap(func=h, variables=("a", "b"), batch_sizes={"a": 0, "b": 0})


@pytest.fixture
def setup_spacemap():
    value_grid = {
        "a": jnp.array([1.0, 2, 3]),
        "b": jnp.array([3.0, 4]),
    }

    combination_values = {
        "c": jnp.array([7.0, 8, 9, 10]),
    }

    helper = jnp.array(list(itertools.product(*combination_values.values()))).T

    combination_grid = {
        "c": helper[0],
    }
    return value_grid, combination_grid


@pytest.fixture
def expected_spacemap():
    value_grid = {
        "a": jnp.array([1.0, 2, 3]),
        "b": jnp.array([3.0, 4]),
    }

    combination_grid = {
        "c": jnp.array([7.0, 8, 9, 10]),
    }

    all_grids = {**value_grid, **combination_grid}
    helper = jnp.array(list(itertools.product(*all_grids.values()))).T

    return allow_args(f)(*helper).reshape(3, 2, 4)


def test_spacemap_all_arguments_mapped(
    *,
    setup_spacemap,
    expected_spacemap,
):
    product_vars, combination_vars = setup_spacemap

    decorated = simulation_spacemap(
        func=f,
        action_names=tuple(product_vars),
        state_names=tuple(combination_vars),
    )
    calculated = decorated(**product_vars, **combination_vars)

    aaae(calculated, jnp.transpose(expected_spacemap, axes=(2, 0, 1)))


@pytest.mark.parametrize(
    ("error_msg", "product_vars", "combination_vars"),
    [
        (
            "Same argument provided more than once in actions or states variables",
            ("a", "b"),
            ("a", "c", "d"),
        ),
        (
            "Same argument provided more than once in actions or states variables",
            ("a", "a", "b"),
            ("c", "d"),
        ),
    ],
)
def test_spacemap_arguments_overlap(*, error_msg, product_vars, combination_vars):
    with pytest.raises(ValueError, match=error_msg):
        simulation_spacemap(
            func=f, action_names=product_vars, state_names=combination_vars
        )


def test_vmap_1d():
    def func(*, a, b, c):
        return c * (a + b)

    vmapped = vmap_1d(func=func, variables=("a", "b"))
    a = jnp.array([1, 2])
    got = vmapped(a=a, b=a, c=-1)
    exp = jnp.array([-2, -4])

    aaae(got, exp)


def test_vmap_1d_co_maps_pytree_argument_leading_axis_with_variable():
    """A co-mapped pytree argument is sliced along each leaf's leading axis in
    lockstep with the mapped variable, so the body sees only the matching slice."""
    table = MappingProxyType({"r": jnp.arange(6).reshape(3, 2)})

    def func(*, idx, table):  # noqa: ARG001
        # `table["r"]` is already the per-idx slice (shape (2,)); a leftover
        # leading axis would make this the full (3, 2) and break the stack.
        return table["r"]

    mapped = vmap_1d(
        func=func,
        variables=("idx",),
        co_mapped_in_axes=MappingProxyType({"table": 0}),
    )
    got = mapped(idx=jnp.arange(3), table=table)

    aaae(got, table["r"])


def test_vmap_1d_error():
    def func(a):
        return a

    with pytest.raises(ValueError, match=r"Same argument provided more than once."):
        vmap_1d(func=func, variables=("a", "a"))


def test_vmap_1d_callable_with_only_args():
    def func(a):
        return a

    vmapped = vmap_1d(func=func, variables=("a",), callable_with="only_args")
    a = jnp.array([1, 2])
    # check that the function works with positional arguments
    aaae(vmapped(a), a)
    # check that the function fails with keyword arguments
    with pytest.raises(
        ValueError,
        match="vmap in_axes must be an int, None, or a tuple of entries corresponding",
    ):
        vmapped(a=1)


def test_nested_vmap_1d_callable_with_only_args_preserves_positional_origin():
    def func(*, a, b):
        return 10 * a + b

    positional_func = allow_args(func)
    inner = vmap_1d(func=positional_func, variables=("b",), callable_with="only_args")
    outer = vmap_1d(func=inner, variables=("a",), callable_with="only_args")

    got = outer(jnp.array([1, 2]), jnp.array([3, 4]))

    aaae(got, jnp.array([[13, 14], [23, 24]]))


def test_vmap_1d_callable_with_only_kwargs():
    def func(a):
        return a

    vmapped = vmap_1d(func=func, variables=("a",), callable_with="only_kwargs")
    a = jnp.array([1, 2])
    # check that the function works with keyword arguments
    aaae(vmapped(a=a), a)
    # check that the function fails with positional arguments
    with pytest.raises(
        ValueError,
        match="This function has been decorated so that it allows only kwargs, but was",
    ):
        vmapped(a)


def test_vmap_1d_callable_with_invalid():
    """`callable_with` rejects anything outside the documented literal options."""

    def func(a):
        return a

    with pytest.raises(BeartypeCallHintViolation):
        vmap_1d(func=func, variables=("a",), callable_with="invalid")  # ty: ignore[invalid-argument-type]


def _identity_utility(consumption):
    return consumption


def test_productmap_publishes_the_name_of_the_mapped_function():
    """The product map carries the mapped function's name, as `wraps` would."""
    mapped = productmap(
        func=_identity_utility,
        variables=("consumption",),
        batch_sizes={"consumption": 0},
    )

    assert mapped.__name__ == "_identity_utility"


def test_productmap_accepts_jax_instrumentation_that_reads_the_name():
    """`jax.named_call` reads the callable's name and then runs the product map."""
    mapped = productmap(
        func=_identity_utility,
        variables=("consumption",),
        batch_sizes={"consumption": 0},
    )

    got = jax.named_call(mapped)(consumption=jnp.array([1.0, 2.0]))

    aaae(got, jnp.array([1.0, 2.0]))
