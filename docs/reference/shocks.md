---
title: IID shock declarations
---

# IID shock declarations

`lcm.shocks` separates a probability law from its numerical integration rule. A group
owns named base innovations and maps them into physical values at the current state. The
declarations and numerical helpers are independent of model consumers.

```python
from lcm import shocks as sh

wage = sh.ShockGroup(
    shocks={"eps": sh.Normal(mu=0.0, sigma=None)},
    nodes=sh.Product(eps=sh.GaussHermite(5)),
)
nodes = wage.resolve_nodes(group_name="wage", age=40)
realised = wage.realise(nodes=nodes, params={"eps__sigma": 0.2})
realised.raise_if_invalid()
```

`params` here is the group-local namespace. A model registry supplies the enclosing
group namespace, giving `params["wage"]["eps__sigma"]`. A fixed parameter is numeric;
`None` requests a runtime parameter. A callable receives named current-state, time and
parameter inputs. `validate_dependencies` walks the caller's function DAG and refuses an
action, realised shock, next-state, choice or downstream value dependency, including
indirect dependencies. The caller supplies every group's shock names and downstream node
vocabulary so an unknown dependency cannot conceal one of these roles.

`StateConditioned(on=..., by=...)` is a category lookup for any law parameter. Bind it
with `bind_conditioning(state_grids=...)` before evaluation; lookup order follows the
conditioning grid's category codes. The mapping must name exactly those categories.

## Laws and base coordinates

- `Normal(mu, sigma)` uses one standard-normal base coordinate. Scale is nonnegative;
  zero gives a deterministic law.
- `LogNormal(mu, sigma)` exponentiates a normal variable. Its parameters describe the
  logarithm, and its scale may be zero. A nonfinite physical value is invalid.
- `Uniform(start, stop)` transforms a unit-uniform coordinate affinely. Equal endpoints
  are deterministic; reversed endpoints are refused.
- `NormalMixture(p1, mu1, sigma1, mu2, sigma2)` maps one normal-base coordinate through
  the mixture quantile. The first component has probability `p1`. Active components
  require positive scales; an inactive component may have zero scale. This contract
  excludes active point masses.
- `Discrete(categories, probs)` uses a pylcm categorical class. Fixed probabilities are
  a vector in category-code order; runtime or callable probabilities may have leading
  cell axes. Solve enumeration retains every category and multiplies its probability
  into the joint weights. A realised uniform uses half-open CDF cells and skips zero
  probabilities, including exact cutpoints and the terminal zero plateau.

Every law parameter is finite in the evaluated cell. Resolved parameter shapes must
broadcast to that cell's physical values. Probability vectors are nonnegative, have the
category count on the final axis, and sum to unity within the active dtype tolerance.
Validation never silently normalizes a malformed vector.

The mixture solver brackets a quantile between the component quantiles at its common
base probability. Log-CDF and log-survival equations retain both tails. A capacity or
floating-point failure produces an invalid status rather than an unchecked quantile.
Smooth first derivatives follow the implicit CDF equation. Differentiating through an
inactive zero-scale component is outside that smooth derivative contract: derivative
consumers must additionally reject nonfinite tangents or marginals. Primal validity
alone is insufficient to admit a derivative consumer.

Omitted `correlation` means independent base coordinates. Explicit `correlation=None`
requests runtime `corr` in the group namespace. A matrix fixes Gaussian correlation in
shock order. It is restricted to normal-base continuous shocks; there is no copula or
cross-group realised-draw conditioning declaration. Exact represented PSD validation
retains singular support. For a compiled runtime-correlation call, first use
`bind_correlation` to prepare its exact pivot support. This binds the support, retaining
matrix values as traced runtime parameters and preserving their interior derivatives.
The host boundary must call `validate_runtime_correlation` for each invocation; support
changes require rebinding. An unprepared traced matrix is refused. Rounded Schur
residuals cannot certify an exact zero pivot, so the exact host check owns singular
support. The rounded matrix is also checked at the active precision before its factor is
formed.

## Integration rules

`Product(**rules)` supplies one compatible rule per continuous shock. Shock declaration
order determines columns; the last coordinate varies fastest in the tensor product.
Discrete variables are independently enumerated exactly beside the continuous rule.

- `GaussHermite(n)` integrates the standard-normal measure. Hermite coordinates are
  multiplied by `sqrt(2)` and weights divided by `sqrt(pi)`; its analytic polynomial
  exactness is through degree `2*n-1`, subject to floating-point representation.
- `EqualSpaced(n, n_std)` has normal-base support `[-n_std, n_std]`, midpoint CDF bins
  and edge bins containing both tails. It requires adjacent endpoints.
- `EqualSpacedUniform(n)` uses inclusive unit endpoints with equal weights. A singleton
  uses the midpoint.
- `MonteCarlo(n, seed)` uses equal-weight PCG64 uniforms.
- `QuasiMonteCarlo(n, seed, sequence="sobol")` uses scrambled Sobol LMS+shift points
  with a fixed bit budget and a power-of-two sample count. It retains every point in
  sequence order. Unit endpoints are opened before inverse-normal transformation.
- `Smolyak(level, signed=False)` uses zero-based difference levels with total component
  level at most `level`. Its normal family is the bounded Genz-Keister-19 prefix: levels
  zero through three have orders one, three, nine and nineteen; higher normal levels are
  refused. The uniform family is Clenshaw-Curtis on the unit interval: level zero is the
  midpoint, and subsequent orders are `2**level+1`. Shared nodes are merged by
  master-table identifiers or reduced dyadic angles, never by approximate equality. The
  rule validates its actual combined weights.
- `UserNodes(nodes, weights, signed=False)` owns a finite `(K,d)` base-coordinate matrix
  and `(K,)` weights. It preserves point order and copies mutable input arrays. Uniform
  coordinates lie in the unit interval. It cannot replace exact category enumeration.

MC and QMC regenerate per exact age or explicitly labelled period from a canonical
versioned seed, time-coordinate and group-name encoding. Parameters do not enter that
encoding. Their solve seed is independent of simulation randomness. Fingerprints retain
rule declarations, group and column order, exact time, numerical algorithm and library
versions, endpoint handling and the resolved point/weight bytes. Physical-law identity
is separate from node-rule identity: `has_same_laws_as` ignores the node declaration.
Joint-law comparison conservatively retains named column order and fingerprints callable
captured semantics; a reorder cannot reinterpret correlation rows silently.

## Validity and signed provenance

`Realisation` is a JAX-compatible tuple of physical values, joint weights and a scalar
`valid` channel. Every consumer must enforce that status at its runtime boundary.
`raise_if_invalid()` performs this check eagerly. Compiled callers carry the channel
into their runtime checks; they must never use an invalid value or silently replace it
with a finite placeholder.

`NodeSet.sum_abs_weights` records the conditioning of the continuous numerical rule. For
exact categorical enumeration, `NodeSet.weights` repeats continuous weights; `realise`
multiplies the cell's categorical probabilities before integration. The completed
`Realisation.weights` is the joint integration operator.

Negative concrete weights require `signed=True`. Signed provenance persists even when
one concrete realization happens to have nonnegative weights.
`NodeSet.admit(reader=...)` accepts only `ShockReader.LINEAR_EXPECTATION` for a signed
producer. Discrete laws, constraints, regime transitions, exits, availability, gates,
simulation draws, nonlinear certainty equivalents, inverse transforms, Euler inversion
and root steps are refused. The producer and every derived stored channel must retain
this provenance, so an indirect nonlinear reader is subject to the same refusal.

A signed linear integral may lie outside the range of its node values. That is a
property of the numerical operator. Its weights are not a probability law.

## Model registry and compatibility

`Model(shocks={"wage": wage})` snapshots named groups as the immutable `model.shocks`
mapping. `Regime(shocks=("wage",))` references that shared declaration. Group namespaces
cannot alias regime or edge parameter roots. State, action and function names live in
separate namespaces and can also name a group. Physical shock names cannot collide with
model inputs or be redeclared by competing groups. Base address identity is the
immutable physical shock name and component, independent of group membership and node
rules. Physical values, category codes and local quadrature indices do not identify it.

The registry records each consumer phase's current-state law dependencies and domains
before broadcast pruning. Runtime slots such as `eps__sigma`, callable parameters and
`corr` remain group-local metadata; they are not added to `Model.get_params_template()`
as ordinary regime parameters. Integrated execution must explicitly project those slots
from `params[group]`, bind correlation support and enforce per-invocation exact host
correlation admission, numerical validity and signed ancestry.

Registry declarations can be inspected and persisted. Execution of a regime referencing
a group, or construction of an ordinary function reading an unlowered physical shock, is
explicitly refused until integrated current-IID kernels are available. This boundary
prevents a physical draw from silently becoming a free parameter.

Legacy IID state processes retain their public classes, state layouts, phase-specific
laws, transition timing, fold flags and simulation draw modes. Compatibility records
retain the original joint declaration and physical-grid rule. In particular, legacy
normal-mixture CDF bins on a physical equally spaced grid are not a standard-normal base
rule transformed through mixture quantiles. Persistent processes and transition-local
state laws keep their existing consumers. An empty registry and empty regime references
leave feature-free durable fingerprints unchanged.
