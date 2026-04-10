"""Parallel lowering and compilation via subprocesses.

JAX tracing (lowering) is GIL-bound and takes minutes per function for large
models. This module parallelizes lowering across subprocesses: each subprocess
rebuilds the Model from cloudpickled Regimes, lowers its assigned functions,
serializes the HLO bytes + metadata, and sends them back. The main process
reconstructs the compiled executable from HLO bytes in milliseconds (no
re-tracing).
"""

import pickle
from collections.abc import Callable
from typing import Any

import jax
import jax.tree_util
import jaxlib.mlir.ir as mlir_ir
from jax._src import xla_bridge
from jax._src.interpreters import mlir as jax_mlir
from jax._src.interpreters import pxla

_NON_PICKLABLE_COMPILE_ARG_KEYS = frozenset(
    {"backend", "pgle_profiler", "context_mesh"}
)


def lower_functions_in_subprocess(
    pickled_model_and_assignments: bytes,
    enable_x64: bool,
) -> bytes:
    """Rebuild Model in subprocess, lower assigned functions, return HLO.

    Each subprocess rebuilds the full Model (to get all internal regimes),
    then lowers only its assigned (regime_name, period) functions.

    Args:
        pickled_model_and_assignments: cloudpickled tuple of
            (regimes, regime_id_class, ages, fixed_params, enable_jit,
             internal_params, next_regime_to_V_arr,
             assignments: list[tuple[regime_name, period]]).
        enable_x64: Whether 64-bit mode is enabled in the main process.

    Returns:
        cloudpickled list of (regime_name, period, hlo_bytes, metadata) tuples.

    """
    import jax  # noqa: PLC0415

    jax.config.update("jax_enable_x64", enable_x64)

    (
        regimes,
        regime_id_class,
        ages,
        fixed_params,
        enable_jit,
        internal_params,
        next_regime_to_V_arr,
        assignments,
    ) = pickle.loads(pickled_model_and_assignments)  # noqa: S301

    from lcm.model import Model  # noqa: PLC0415

    model = Model(
        regimes=regimes,
        regime_id_class=regime_id_class,
        ages=ages,
        fixed_params=fixed_params,
        enable_jit=enable_jit,
    )

    import cloudpickle  # noqa: PLC0415

    results = []
    for regime_name, period in assignments:
        func = model.internal_regimes[regime_name].solve_functions.max_Q_over_a[period]
        state_action_space = model.internal_regimes[regime_name].state_action_space(
            regime_params=internal_params[regime_name],
        )
        lower_args = {
            **dict(state_action_space.states),
            **dict(state_action_space.actions),
            "next_regime_to_V_arr": next_regime_to_V_arr,
            **dict(internal_params[regime_name]),
            "period": period,
            "age": ages.values[period],
        }
        lowered = func.lower(**lower_args)  # ty: ignore[unresolved-attribute]
        computation = lowered._lowering

        hlo_bytes = computation.stablehlo().operation.get_asm(binary=True)
        metadata: dict[str, Any] = {
            "name": computation._name,
            "const_args": computation.const_args,
            "donated_invars": computation._donated_invars,
            "platforms": computation._platforms,
            "compiler_options_kvs": computation._compiler_options_kvs,
            "compile_args": {
                k: v
                for k, v in computation.compile_args.items()
                if k not in _NON_PICKLABLE_COMPILE_ARG_KEYS
            },
            "out_tree": lowered.out_tree,
        }
        results.append((regime_name, period, hlo_bytes, metadata))

    return cloudpickle.dumps(results)


def reconstruct_and_compile(
    hlo_bytes: bytes,
    metadata: dict[str, Any],
) -> Callable:
    """Reconstruct a compiled executable from serialized HLO bytes.

    Parse the HLO bytecode (milliseconds, no re-tracing) and compile it.

    Args:
        hlo_bytes: StableHLO bytecode from the lowered computation.
        metadata: Metadata dict with compile_args, tree info, etc.

    Returns:
        Callable wrapping the compiled executable.

    """
    ctx = jax_mlir.make_ir_context()
    module = mlir_ir.Module.parse(hlo_bytes, context=ctx)

    backend = xla_bridge.get_backend()
    compile_args = metadata["compile_args"]
    compile_args["backend"] = backend

    mesh_computation = pxla.MeshComputation(
        metadata["name"],
        module,
        metadata["const_args"],
        metadata["donated_invars"],
        metadata["platforms"],
        metadata["compiler_options_kvs"],
        tuple(backend.devices()),
        **compile_args,
    )
    mesh_executable = mesh_computation.compile()
    out_tree = metadata["out_tree"]

    def call_compiled(**kwargs: Any) -> Any:
        flat_args = jax.tree_util.tree_leaves(({}, kwargs))
        flat_out = mesh_executable.call(*flat_args)
        return jax.tree_util.tree_unflatten(out_tree, flat_out)

    return call_compiled
