"""Paper-independent packed NumPy ownership regression and refusal controls."""

import itertools
import struct
from pathlib import Path
import os
import subprocess
import textwrap

import numpy as np
import pytest

from _lcm_regime_finder.structural_arrays import NativeWriter
from _lcm_regime_finder.structural_check import Reader
from _lcm_regime_finder.structural_events import compare_source, source_anchor


@pytest.mark.parametrize("axes", list(itertools.permutations(range(3))))
def test_owned_axis_permutation_preserves_values_layout_and_aliases(
    axes: tuple[int, ...],
) -> None:
    source = np.arange(24, dtype="<f8").reshape(2, 3, 4).transpose(axes).copy(order="K")
    assert source.base is None and source.flags.owndata
    view = source[::-1]
    roots = [source, view, source]
    anchor = source_anchor(
        roots,
        source_context={},
        transaction={"kind": "exclusive-owner", "authority": "test-owned arrays"},
    )
    bundle = NativeWriter().finish(roots, anchor)
    reader = Reader(bundle)
    result = reader.decode()
    assert result[0] is result[2]
    assert result[0].buffer is result[1].buffer
    assert result[0].shape == source.shape
    assert result[0].strides == source.strides
    assert result[1].strides == view.strides
    for original, decoded in zip((source, view), result[:2], strict=True):
        raw = b"".join(decoded.buffer.iter_bytes())
        for index in np.ndindex(original.shape):
            offset = decoded.offset + sum(
                i * stride for i, stride in zip(index, decoded.strides, strict=True)
            )
            assert struct.unpack_from("<d", raw, offset)[0] == float(original[index])
    compare_source(reader, anchor)


@pytest.mark.parametrize("strides", [(16, 0), (8, 8), (0, 24)])
def test_owned_holey_or_overlapping_storage_still_refuses(
    strides: tuple[int, int],
) -> None:
    source = np.ndarray((2, 2), dtype="<f8", strides=strides)
    assert source.base is None and source.flags.owndata
    with pytest.raises(ValueError, match="unsupported native array storage"):
        NativeWriter().finish(source, {})
    with pytest.raises(ValueError, match="Unsupported source buffer ownership"):
        source_anchor(source, source_context={}, transaction={})


def test_external_storage_still_refuses() -> None:
    source = np.frombuffer(bytearray(32), dtype="<f8")
    with pytest.raises(ValueError, match="unsupported native array storage"):
        NativeWriter().finish(source, {})
    with pytest.raises(ValueError, match="Unsupported source buffer ownership"):
        source_anchor(source, source_context={}, transaction={})


def test_reinterpreted_view_uses_the_owners_itemsize() -> None:
    owner = (
        np.arange(24, dtype="<f8").reshape(2, 3, 4).transpose(1, 0, 2).copy(order="K")
    )
    view = owner.view("<u4")
    roots = [owner, view]
    anchor = source_anchor(roots, source_context={}, transaction={})
    reader = Reader(NativeWriter().finish(roots, anchor))
    result = reader.decode()
    assert result[0].buffer is result[1].buffer
    raw = b"".join(result[1].buffer.iter_bytes())
    for index in np.ndindex(view.shape):
        offset = result[1].offset + sum(
            i * stride for i, stride in zip(index, result[1].strides, strict=True)
        )
        assert struct.unpack_from("<I", raw, offset)[0] == int(view[index])
    compare_source(reader, anchor)


def test_owned_permutation_mutation_still_refuses() -> None:
    source = (
        np.arange(24, dtype="<f8").reshape(2, 3, 4).transpose(1, 2, 0).copy(order="K")
    )

    def mutate() -> None:
        source[0, 0, 0] = -1

    with pytest.raises(ValueError, match="changed"):
        NativeWriter().finish(source, {}, mutate)


def test_native_capture_owned_packed_numpy_fixed_params(tmp_path: Path) -> None:
    script = textwrap.dedent(
        """
        import json
        from pathlib import Path
        import sys
        import numpy as np
        from lcm import DiscreteGrid, InitialNodes, Model, Regime, categorical
        from lcm.typing import DiscreteState, FloatND, ScalarInt
        from _lcm_regime_finder.callable_identity import capture_session
        from _lcm_regime_finder.model_snapshot import normalize, read_model
        from _lcm_regime_finder.structural_arrays import NativeWriter
        from _lcm_regime_finder.structural_store import CaptureStore

        @categorical(ordered=False)
        class Binary:
            low: ScalarInt
            high: ScalarInt

        @categorical(ordered=False)
        class Regimes:
            terminal: ScalarInt

        def utility(x: DiscreteState, table: FloatND) -> FloatND:
            raise AssertionError("Economic callback executed")

        table = np.arange(24, dtype=np.float64).reshape(2, 3, 4).transpose(1, 0, 2).copy(order="K")
        assert table.base is None and table.flags.owndata
        assert not table.flags.c_contiguous and not table.flags.f_contiguous
        destination = Path(sys.argv[1])
        destination.mkdir()
        model = Model(
            n_periods=1,
            regime_id_class=Regimes,
            regimes={"terminal": Regime(
                states={"x": DiscreteGrid(Binary)}, functions={"utility": utility},
            )},
            initial_nodes=InitialNodes(by_period={0: "terminal"}),
            edges={}, fixed_params={"table": table}, enable_jit=False,
        )
        stored = model.fixed_params["table"]
        stored_flavor = NativeWriter.flavor(stored)
        assert stored_flavor in {"numpy", "jax-value"}
        transaction = {"kind": "exclusive-owner", "authority": "test owns synthetic Model and table; no concurrent writes"}
        model_store = CaptureStore(
            destination / "model.json", check=lambda: None,
            source_context={"synthetic": "owned-packed-axis-permutation"},
            transaction=transaction,
        )
        captured = read_model(model, structural_store=model_store)
        assert captured["capture_report"]["status"] == "COMPLETE", captured["capture_report"]
        header = captured["structural_store"]
        assert header["source_comparison"] == "PASS"
        assert header["manifest"]["source_consistency"] == "CALLER_QUIESCENCE_CONTRACT"
        model_ref = captured["fixed_params"]["table"]["structural_ref"]
        model_array = model_store.reader.decode()[model_ref["root"]]
        assert model_array.flavor == stored_flavor
        assert model_array.shape == tuple(stored.shape)
        assert model_array.dtype == np.dtype(stored.dtype).str
        if stored_flavor == "numpy":
            assert model_array.strides == tuple(stored.strides)

        raw_store = CaptureStore(
            destination / "original-array.json", check=lambda: None,
            source_context={"synthetic": "same-original-array-object"},
            transaction=transaction,
        )
        with capture_session() as session:
            session.structural_store = raw_store
            raw_store.check = session.check
            raw_store.writer.check = session.check
            raw = normalize(table)
        assert session.report()["status"] == "COMPLETE", session.report()
        raw_header = raw_store.header
        assert raw_header["source_comparison"] == "PASS"
        raw_array = raw_store.reader.decode()[raw["structural_ref"]["root"]]
        assert raw_array.flavor == "numpy"
        assert raw_array.shape == tuple(table.shape)
        assert raw_array.strides == tuple(table.strides)
        receipt = {
            "capture_status": captured["capture_report"]["status"],
            "model_leaf_flavor": stored_flavor,
            "model_leaf_is_original_object": stored is table,
            "model_source_comparison": header["source_comparison"],
            "model_source_anchor": header["manifest"]["source_anchor"]["logical_sha256"],
            "original_array_flavor": raw_array.flavor,
            "original_array_source_comparison": raw_header["source_comparison"],
            "original_array_source_anchor": raw_header["manifest"]["source_anchor"]["logical_sha256"],
            "original_owned": bool(table.flags.owndata),
            "original_c_contiguous": bool(table.flags.c_contiguous),
            "original_f_contiguous": bool(table.flags.f_contiguous),
            "original_shape": list(table.shape),
            "original_strides": list(table.strides),
            "scope": "Synthetic public Model construction and structural capture only; no economic callback, solve or simulation.",
        }
        (destination / "CAPABILITY.json").write_text(json.dumps(receipt, indent=2) + "\\n")
        """
    )
    command = [
        "pixi",
        "run",
        "--as-is",
        "--manifest-path",
        str(Path(__file__).resolve().parents[2] / "pyproject.toml"),
        "-e",
        os.environ.get("PIXI_ENVIRONMENT_NAME", "type-checking"),
        "python",
        "-c",
        script,
        str(tmp_path / "capture-proof"),
    ]
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert (tmp_path / "capture-proof" / "CAPABILITY.json").is_file()
