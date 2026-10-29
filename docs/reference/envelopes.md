---
title: Upper envelopes
---

# Upper envelopes

`DCEGM` produces competing value branches and needs an upper-envelope configuration.
Pass a typed configuration through `DCEGM(envelope=...)`:

```python
from lcm.solvers import DCEGM, LTMEnvelope

solver = DCEGM(savings_grid=..., envelope=LTMEnvelope())
```

| Configuration   | Contract                                                                               | Main controls                     |
| --------------- | -------------------------------------------------------------------------------------- | --------------------------------- |
| `ExactEnvelope` | Certified finite-candidate ownership using pylcm's packaged native exact-affine kernel | `max_runs`                        |
| `FUESEnvelope`  | Fast upper-envelope scan approximation                                                 | `jump_thresh`, `n_points_to_scan` |
| `RFCEnvelope`   | Roof-cutting approximation                                                             | `jump_thresh`, `search_radius`    |
| `LTMEnvelope`   | Query-side line/segment evaluation                                                     | none                              |

These four typed objects are the supported strategies. `EnvelopeConfig` is their union;
string selectors are invalid.

pylcm no longer ships an MSS (HARK-style multi-segment scan) envelope: a timing study
found it dominated by `ExactEnvelope`. An interface for envelopes maintained outside
pylcm, which would let benchmark suites keep MSS, is planned in
[issue #632](https://github.com/OpenSourceEconomics/pylcm/issues/632).

(api-exact-envelope-availability)=

## Exact envelope availability

The certified exact-affine read is **forward-mode differentiable only**. It carries a
custom JVP, so `jax.jvp` and `jax.jacfwd` work and carry the exact slope, while
`jax.grad` and `jax.vjp` raise: the rule inspects tangent finiteness so it can fail
closed on a non-finite direction, which leaves JAX unable to transpose it, and no
reverse rule is registered. This reaches ordinary models, because `ExactEnvelope` is the
DCEGM default; NBEGM reaches it only with `envelope_arithmetic="certified"`.

`ExactEnvelope` is the default. Its ownership decision relies on the exact-affine native
payload installed as part of pylcm. pylcm neither downloads nor discovers an unrelated
shared library at runtime.

A binary wheel contains the payload built for that wheel's platform and toolchain. A
source or editable install runs pylcm's build hook locally: it builds the CPU library
with the available C++ compiler and also builds the CUDA library when `nvcc` is
available. A CPU payload does not provide a CUDA capability, and a payload built for a
different platform, ABI, toolchain, or JAX backend is not interchangeable.

If the selected backend has no compatible loadable payload, model construction raises
rather than falling back to ordinary floating-point comparisons. Reinstall pylcm in the
target environment after supplying the required compiler; source-install details and the
explicit no-kernel installation option are in
[Installation](../getting_started/installation.md#the-compiled-kernel-and-installing-without-a-c-compiler).

`max_runs` bounds supported envelope topology. The execution plan sizes the exact
envelope's independent node-cell loop through `ENVELOPE_CELL_AXIS` (`"envelope_cell"`,
exported by `lcm.solvers`). For example,
`Model(..., execution_config=ExecutionConfig(axis_widths={"envelope_cell": 3}))`
resolves three adjacent-candidate cells together. These are cells of the endogenous
resource grid, distinct from the regime's output-state `cell` axis. Width changes the
working set while preserving candidate ownership and published quantities.

FUES uses a fixed scan unroll factor of one. This compiler choice is recorded in
compilation identity and is not a field on `FUESEnvelope`. A model using only FUES has
no `envelope_cell` loop and rejects that axis name.

(api-approximate-envelope-backends)=

## Approximate backends

FUES, RFC, and LTM make different topology and execution trade-offs. FUES is
scan-shaped; LTM evaluates candidate segments at query points and is usually more
accelerator-friendly. Thresholds such as `jump_thresh` are algorithmic approximation
parameters, not generic tolerances.

Switch backends only with model-specific validation:

- compare values and discrete ownership;
- inspect crossings and borrowing corners;
- repeat at both numerical precisions;
- measure cold compile, warm execution, and peak memory;
- record the typed configuration with benchmark results.

The method background is in
[Discrete choice and upper envelopes](../explanations/iskhakov_et_al_2017.ipynb) and
[Scaling, memory, and hardware](../methods/performance_scaling.md).
