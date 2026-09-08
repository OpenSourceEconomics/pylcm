"""Static closure checks for the streamed interval-envelope repair.

This file deliberately imports neither :mod:`lcm` nor :mod:`_lcm`. It makes the
cross-language and public-path obligations executable where the project runtime
and rebuilt native payload are unavailable.
"""

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
QUERY = ROOT / "src/_lcm/egm/upper_envelope/query.py"
STEP = ROOT / "src/_lcm/egm/nbegm_step.py"
SOLUTION = ROOT / "src/_lcm/solution/nbegm.py"
FFI = ROOT / "src/_lcm/egm/upper_envelope/_exact_affine/ffi.py"
CORE = ROOT / "src/_lcm/egm/upper_envelope/_exact_affine/exact_affine_core.h"
CPU = ROOT / "src/_lcm/egm/upper_envelope/_exact_affine/certified_affine_ffi_cpu.cc"
CUDA = ROOT / "src/_lcm/egm/upper_envelope/_exact_affine/certified_affine_ffi_cuda.cu"
LEDGER = ROOT / "docs/development/architecture_transition_ledger.md"
SOLVER_DOCS = ROOT / "docs/reference/solvers.md"
INTEGRATION = ROOT / "tests/solution/test_nbegm_streamed_interval_envelope.py"


def _text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _tree(path: Path) -> ast.Module:
    return ast.parse(_text(path), filename=str(path))


def _definitions(*, tree: ast.Module, name: str) -> list[ast.FunctionDef]:
    return [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == name
    ]


def _argument_names(node: ast.FunctionDef) -> list[str]:
    positional = [*node.args.posonlyargs, *node.args.args]
    return [arg.arg for arg in [*positional, *node.args.kwonlyargs]]


def _function_source(*, path: Path, name: str) -> str:
    text = _text(path)
    nodes = _definitions(tree=ast.parse(text, filename=str(path)), name=name)
    assert nodes, f"missing {name} in {path}"
    segment = ast.get_source_segment(text, nodes[-1])
    assert segment is not None
    return segment


def _class_source(*, path: Path, name: str) -> str:
    text = _text(path)
    nodes = [
        node
        for node in ast.parse(text, filename=str(path)).body
        if isinstance(node, ast.ClassDef) and node.name == name
    ]
    assert nodes, f"missing class {name} in {path}"
    segment = ast.get_source_segment(text, nodes[-1])
    assert segment is not None
    return segment


def test_ordinary_order_uses_the_global_index_as_its_last_field() -> None:
    tree = _tree(QUERY)
    tie_class = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "_TieBreakKey"
    )
    fields = [
        node.target.id
        for node in tie_class.body
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name)
    ]
    assert fields == ["right_available", "slope_high", "slope_low", "stable_index"]

    dense = _function_source(path=QUERY, name="_lexicographic_argmax")
    blocked = _function_source(path=QUERY, name="_outranks")
    assert "jnp.min" in dense
    assert "key.stable_index" in dense
    assert "challenger.stable_index < held.stable_index" in blocked
    assert "merge_envelope_winner" in _text(QUERY)
    assert "stable_index=identities" in _text(QUERY)


def test_python_exact_winner_abi_requires_an_explicit_stable_index_operand() -> None:
    tree = _tree(FFI)
    for name in ("exact_query_winner", "exact_query_winner_batched"):
        definitions = _definitions(tree=tree, name=name)
        assert definitions, name
        assert all("stable_index" in _argument_names(node) for node in definitions)
        body = _function_source(path=FFI, name=name)
        assert "stable_index" in body
        assert "segment shape" in body

    ffi_text = _text(FFI)
    assert "stable_index=stopped[5]" in ffi_text
    assert "operand=stable_index" in ffi_text


def test_cpu_cuda_and_core_share_the_same_stable_index_abi() -> None:
    core = _text(CORE)
    assert "const int32_t* stable_index" in core
    assert "stable_index[index] < stable_index[held_index]" in core

    for path in (CPU, CUDA):
        native = _text(path)
        assert "ffi::Buffer<ffi::S32> stable_index" in native
        assert "stable_index.element_count()" in native
        assert "stable_index.typed_data()" in native
        assert "ExactQueryWinner" in native


def test_positive_interval_width_streams_the_read_and_fold_together() -> None:
    entry = _function_source(
        path=STEP, name="nbegm_per_interval_continuation_step_savings"
    )
    assert "interval_block_reader" in entry
    assert "interval_batch_size" in entry
    assert entry.index("if interval_block_reader is not None") < entry.index(
        "if cont_value is None or cont_marginal is None"
    )

    stream = _function_source(
        path=STEP, name="_streamed_interval_continuation_envelope"
    )
    assert "_IntervalWinnerStep(" in stream
    assert "jax.lax.scan" in stream
    assert "finish_envelope_winner" in stream

    # The scan step is a module-level callable, so the block read and the fold
    # live in its `__call__` rather than in the envelope function itself.
    step = _class_source(path=STEP, name="_IntervalWinnerStep")
    assert "rows = self.interval_block_reader(indices)" in step
    assert "merge_envelope_winner" in step
    assert "_streamed_block_candidate_positions" in step

    solution = _text(SOLUTION)
    assert "class _NBEGMIntervalContinuation" in solution
    assert (
        "statics.continuation_reads_liquid and statics.interval_batch_size > 0"
        in solution
    )
    assert "selected_midpoints = midpoints[interval_indices]" in solution
    reader = solution.split("def _bind_cell_interval_reader_for_pool", 1)[1].split(
        "def _cell_rows_for_pool", 1
    )[0]
    assert "_map_ride_partitioned" not in reader
    assert "return read" in reader


def test_docs_retire_the_transition_and_define_zero_versus_positive_width() -> None:
    ledger = _text(LEDGER)
    retired = "NB-EGM's per-interval merge stacks every interval's candidate families"
    assert retired not in ledger

    docs = _text(SOLVER_DOCS)
    assert "`interval_batch_size` streams the continuation read" in docs
    assert "global stored-link index" in docs
    assert "decides ownership by the same total order" in docs
    assert "return_owner=True" in docs
    assert "units in the last place" in docs
    assert "`interval_batch_size=0`" in docs


def test_maintainer_suite_covers_channels_profiles_arithmetics_and_partitions() -> None:
    integration = _text(INTEGRATION)
    for token in (
        '"ordinary"',
        '"certified"',
        "interval_batch_size",
        "cont_value=None",
        "cont_marginal=None",
        "assert_agrees_to_ulp",
        "np.isfinite",
        "NO_OWNER",
        "return_owner=True",
        "(1, 2, 4, 7)",
    ):
        assert token in integration
