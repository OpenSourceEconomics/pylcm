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

| Configuration   | Contract                                                                               | Main controls                                    |
| --------------- | -------------------------------------------------------------------------------------- | ------------------------------------------------ |
| `ExactEnvelope` | Certified finite-candidate ownership using pylcm's packaged native exact-affine kernel | `max_runs`, `cell_batch_size`                    |
| `FUESEnvelope`  | Fast upper-envelope scan approximation                                                 | `jump_thresh`, `n_points_to_scan`, `scan_unroll` |
| `RFCEnvelope`   | Roof-cutting approximation                                                             | `jump_thresh`, `search_radius`                   |
| `LTMEnvelope`   | Query-side line/segment evaluation                                                     | none                                             |
| `MSSEnvelope`   | Multi-segment scan with a selectable comparison arithmetic                             | `arithmetic`                                     |

These five typed objects are the supported strategies. `EnvelopeConfig` is their union;
string selectors are invalid.

(api-exact-envelope-availability)=

## Exact envelope availability

The certified exact-affine read is **forward-mode differentiable only**. It carries a
custom JVP, so `jax.jvp` and `jax.jacfwd` work and carry the exact slope, while
`jax.grad` and `jax.vjp` raise: the rule inspects tangent finiteness so it can fail
closed on a non-finite direction, which leaves JAX unable to transpose it, and no
reverse rule is registered. This reaches ordinary models, because `ExactEnvelope` is the
DCEGM default and `envelope_arithmetic="certified"` is NBEGM's.

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

`max_runs` bounds supported envelope topology. `cell_batch_size` sets how many
independent state cells the exact native operation resolves in parallel; `None` selects
serial resolution.

(api-approximate-envelope-backends)=

## Approximate backends

FUES, RFC, LTM, and MSS make different topology and execution trade-offs. FUES and MSS
are scan-shaped; LTM evaluates candidate segments at query points and is usually more
accelerator-friendly. Thresholds such as `jump_thresh` are algorithmic approximation
parameters, not generic tolerances.

`MSSEnvelope(arithmetic=...)` selects how a comparison between two candidate chords is
settled. The geometry is the same either way — which stored piece covers an interval,
which node owns a query, and where two branches hand over:

- `"certified"`, the default, decides on the stored operands, so an ordering the working
  format cannot separate is still settled and a comparison the arithmetic cannot decide
  publishes `NaN`. Like `ExactEnvelope`, it needs the installed exact-affine payload for
  the active backend, and a regime selecting it is refused during `Model(...)` when that
  payload is absent rather than falling back.
- `"ordinary"` compares two readings formed in the working floating format. Each reading
  is a slope and then an affine step, two rounded operations rather than one correctly
  rounded value, so it carries no bound in units of the format's spacing: cancellation
  between a chord's endpoints, or a large common level under both chords, can move a
  reading by many representable steps and reverse an ordering the correctly rounded
  values would separate. Candidates whose readings coincide are separated by the
  declared tie order — greatest reading, then reaching strictly right of the query, then
  steeper, then the earliest stored link. It reaches no native kernel, so it is the
  route available where that payload is absent. Its published values, owners and
  crossings are the caller's to validate for the intended model at every precision,
  backend and transformation in use, checking cancellation and dynamic-range sensitivity
  as well as near-ties; nothing about the selection discharges that.

Switch backends only with model-specific validation:

- compare values and discrete ownership;
- inspect crossings and borrowing corners;
- repeat at both numerical precisions;
- measure cold compile, warm execution, and peak memory;
- record the typed configuration with benchmark results.

The method background is in
[Discrete choice and upper envelopes](../explanations/iskhakov_et_al_2017.ipynb) and
[Scaling, memory, and hardware](../methods/performance_scaling.md).
