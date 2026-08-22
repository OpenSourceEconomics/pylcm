---
title: Upper envelopes
---

# Upper envelopes

`DCEGM` produces competing value branches and needs an upper-envelope configuration.
Pass a typed configuration through `DCEGM(envelope=...)`.

| Configuration   | Contract                                                                               | Main controls                                    |
| --------------- | -------------------------------------------------------------------------------------- | ------------------------------------------------ |
| `ExactEnvelope` | Certified finite-candidate ownership using pylcm's packaged native exact-affine kernel | `max_runs`, `cell_batch_size`                    |
| `FUESEnvelope`  | Fast upper-envelope scan approximation                                                 | `jump_thresh`, `n_points_to_scan`, `scan_unroll` |
| `RFCEnvelope`   | Roof-cutting approximation                                                             | `jump_thresh`, `search_radius`                   |
| `LTMEnvelope`   | Query-side line/segment evaluation                                                     | none                                             |
| `MSSEnvelope`   | Multi-segment scan approximation                                                       | none                                             |

`EnvelopeConfig` is the union accepted by `DCEGM`.

## Exact envelope availability

`ExactEnvelope` is the default. Its certified ownership decision relies on the native
exact-affine extension built and packaged with pylcm. It is not an unrelated library
downloaded at runtime. If that extension is missing or cannot be loaded, model
construction raises rather than falling back to ordinary floating-point comparisons that
cannot provide the same guarantee.

`max_runs` bounds supported envelope topology. `cell_batch_size` streams independent
state cells to reduce retained memory.

## Approximate backends

FUES, RFC, LTM, and MSS make different topology and execution trade-offs. FUES and MSS
are scan-shaped; LTM evaluates candidate segments at query points and is usually more
accelerator-friendly. Thresholds such as `jump_thresh` are algorithmic approximation
parameters, not generic tolerances.

Switch backends only with model-specific validation:

- compare values and discrete ownership;
- inspect crossings and borrowing corners;
- repeat at both numerical precisions;
- measure cold compile, warm execution, and peak memory;
- record the configuration with benchmark results.

The method background is in
[Discrete choice and upper envelopes](../explanations/iskhakov_et_al_2017.ipynb) and
[Scaling, memory, and hardware](../methods/performance_scaling.md).
