
# A Conditional Theory of Solver Choice for Finite-Horizon Discrete-Continuous Dynamic Programs

**Expanded Version 3: revised after adversarial math/code audit; clarifies the interpolation target, topology-preserving publication, endpoint ownership, NaN-dead masks, validator scope, and the full BQSEGM cost inequality.**

**Date:** 2026-06-28
**Project context:** `pylcm` branch / PR adding DC-EGM, NEGM, 1-D and 2-D envelope backends, DS benchmark replications, query-side envelope machinery, numerical inverse EGM, and continuation-operator extensions.
**Purpose:** Provide a publication-quality decision and proof framework for when brute force, standard EGM, DC-EGM/FUES, query-side segmented EGM, BQSEGM, NEGM, G2EGM, RFC, and related pylcm implementations are possible, chunkable, GPU-friendly, and optimal.

---

## Table of contents

1. Scope and definitions
2. Hardware and cost model
3. Algorithm inventory in pylcm PR #390
4. Implementation-versus-paper ledger
5. Formal BQSEGM specification
6. BQSEGM correctness theorems
7. General solver-choice theorems
8. GPU and CPU consequences
9. ACA-style institutional models
10. Decision tables
11. Practical implementation requirements
12. Decisive experiments
13. Limitations and open problems
14. References and source notes

---

## 1. Scope and definitions

### 1.1 Problem class

We consider finite-horizon discrete-continuous dynamic programs. At period \(t\), an agent is in a regime \(r_t\), a finite state \(z_t\), and a continuous state \(x_t \in X_t \subset \mathbb R^{d_x}\). The agent chooses a finite action \(d_t \in \mathcal D_t\) and a continuous action \(y_t \in Y_t(x_t,z_t,d_t)\subset \mathbb R^{d_y}\). The Bellman equation is

\[
V_t(r,z,x)
=
\max_{d\in \mathcal D_t(r,z,x)}
\sup_{y\in Y_t(r,z,x,d)}
\left\{
u_t(r,z,x,d,y)
+
\beta_t
\mathcal C_t
\left[
V_{t+1}(r',z',x')
\mid r,z,x,d,y
\right]
\right\}.
\]

Here \(\mathcal C_t\) is the continuation operator. In expected-utility models it is a linear expectation. In recursive or Epstein--Zin-style models it may be a nonlinear transform over a joint next-period lottery.

The next state is

\[
(r',z',x') = F_t(r,z,x,d,y,\epsilon),
\]

with \(\epsilon\) approximated by finite quadrature or a finite Markov transition.

### 1.2 Numerical target

All algorithms compared here solve a **discretized numerical problem**. Let:

- \(N_X\): number of current continuous-state grid nodes;
- \(N_A\): number of brute-force continuous-action candidates;
- \(N_S\): number of EGM post-decision / savings-grid nodes;
- \(N_Q\): number of query nodes at which the solution is published;
- \(N_D\): number of discrete action alternatives;
- \(N_C\): number of BQSEGM case assignments or case combinations that are actually solved after any static pruning;
- \(N_{\mathrm{seg}}\): number of explicit one-dimensional monotone envelope segments after masks, folds, boundary splits, and case splitting;
- \(N_{\triangle}\): number of two-dimensional envelope triangles;
- \(N_{\mathrm{pad}}\): padded refined-envelope row length;
- \(N_Z\): number of stochastic quadrature nodes;
- \(N_R\): number of regimes or target regimes.

A solver is **correct for the discretized target** if it gives the value and policy obtained by exhaustive evaluation of all candidates and exact application of the declared interpolation / envelope convention.

A solver is **consistent for the continuous target** if, under stated regularity and mesh-refinement assumptions, the discretized solution converges to the continuous Bellman solution in the claimed norm.

### 1.3 Terminology

| Term | Meaning |
|---|---|
| **Regime** | pylcm dynamic-programming mode with its own value function, target transitions, and solver. |
| **Case boundary** | A Boolean DAG node defining a threshold surface, such as `assets < medicaid_asset_limit`, together with boundary/equality ownership metadata. |
| **Piece** | Alternative formula for a DAG output under a case-boundary predicate. |
| **Segment** | Solver-internal smooth one-dimensional EGM object or explicit envelope piece; in BQSEGM, segment identifiers are assigned per monotone feasible subsegment, not merely per case. |
| **Endpoint ownership** | Metadata saying whether a segment includes its left and right endpoints. This is needed at discontinuities and predicate equality surfaces. |
| **Branch** | Mathematical synonym for a smooth candidate function; avoid as a user-facing pylcm term because it can be confused with regime. |
| **Envelope** | Pointwise maximum over candidate branches, segments, or triangles. |
| **Asset-row mode** | A DC-EGM mode in which one EGM correspondence is solved per current Euler-state node because the Euler RHS depends on current state. |

---

## 2. Hardware and cost model

### 2.1 Device model

Let the hardware be described by:

- \(P\): number of independent lanes needed to saturate the device;
- \(F\): sustained floating-point throughput;
- \(B\): sustained memory bandwidth;
- \(M\): usable device memory;
- \(C\): relevant CPU cache capacity or GPU-local memory budget;
- \(\lambda\): sequential dependency / launch latency;
- \(\eta_F,\eta_B\in(0,1]\): utilization factors for compute and bandwidth.

For any implementation with arithmetic work \(W\), bytes transferred \(R\), resident memory \(m\), and dependency depth \(D\),

\[
T_{\mathrm{wall}}
\ge
\max
\left\{
\frac{W}{\eta_F F},
\frac{R}{\eta_B B},
D\lambda
\right\},
\qquad
m\le M.
\]

A GPU-friendly algorithm exposes \(\Omega(P)\) independent lanes, avoids long loop-carried dependencies, uses regular memory access, and can be chunked with associative reductions. A CPU-friendly algorithm may use dynamic data structures, branch-heavy scalar loops, host Delaunay, KD-trees, or cache-resident row scans.

### 2.2 Why solver speed is conditional, not universal

An algorithm with lower arithmetic complexity can be slower if:

- it materializes a larger transient;
- it has a long sequential scan;
- it has poor GPU utilization;
- it compiles many static-shape variants;
- it uses dynamic geometry on host;
- it reads/writes irregular memory.

Conversely, brute force can win on GPU because it is a dense map-reduce even if it evaluates more numerical candidates.

---

## 3. Algorithm inventory in pylcm PR #390

This section enumerates the algorithms and numerical components implemented or wired in the PR. The PR description says it adds DC-EGM behind the `Solver` ABC seam, with Euler inversion on an exogenous end-of-period grid, a FUES upper envelope, constrained segment, multi-target continuation, and asset-row mode. It also describes stochastic-node expectation in `lax.scan` blocks, modular splitting of the former monolithic EGM step, and streaming asset-row refine-to-query. The PR further lists NEGM, 1-D RFC/MSS/LTM envelope backends, a 2-D RFC/G2EGM foundation for the pension comparison, and DS-2026 Applications 1/2/3 reproductions.

### 3.1 GridSearch / brute-force solver

**Implemented concept.** The baseline `GridSearch` evaluates the Bellman objective over a finite grid of continuous actions and takes the maximum. It remains the default solver.

**Mathematical object.**

\[
V_h(x)=\max_{a\in A_h} Q_h(x,a).
\]

**Chunkability.** Exact, because max is associative.

**Difference from papers.** Brute force is a numerical baseline, not a paper algorithm. In pylcm it is a general finite-grid Bellman evaluator and may use the same model DAG and continuation machinery as other solvers. It can differ from paper baselines if the paper uses off-grid root finding, interpolation conventions, or optimized language-specific loops.

---

### 3.2 Standard 1-D EGM / OneAssetEGM

**Implemented concept.** The branch includes a prime-time one-asset EGM solver for smooth one-dimensional problems.

**Mathematical object.** For post-decision asset \(a'\),

\[
u'(c(a')) = \beta\,\mathbb E[V'_{t+1}(a')].
\]

If \(u'\) is invertible,

\[
c(a') = (u')^{-1}\left(\beta\,\mathbb E[V'_{t+1}(a')]\right),
\]

and current resources follow from the budget.

**Difference from Carroll.** Carroll’s EGM assumes a smooth concave one-state consumption-saving problem and avoids root finding through a closed-form or numerical inverse marginal utility. pylcm generalizes the same idea into its regime/solver framework and allows it to interoperate with DC-EGM carries, terminal carry producers, and generic continuation payloads. pylcm also supports a numerical inverse marginal utility mode, which is not Carroll’s original closed-form inverse setup.

---

### 3.3 One-dimensional DC-EGM core

**Implemented concept.** Conditional on discrete alternatives and target regimes, the solver inverts the Euler equation for the continuous choice, builds candidate correspondences, adds constrained candidates, selects an upper envelope, and publishes value, policy, and marginal rows.

**Mathematical object.**

For each branch \(b\),

\[
a'_j \mapsto
\left(
m_{b,j},
c_{b,j},
V_{b,j},
\mu_{b,j}
\right),
\]

then

\[
V(m)=\max_b I_b[V_b](m).
\]

**Key pylcm additions beyond the canonical DC-EGM paper:**

1. JAX fixed-shape implementation.
2. Multi-target continuation.
3. Passive-state interpolation.
4. Extreme-value taste-shock aggregation.
5. Terminal carry producers.
6. Asset-row mode.
7. Streaming refine-to-query.
8. Multiple envelope backends.

**Difference from Iskhakov et al. DC-EGM.** The paper’s algorithmic object is the discrete-continuous endogenous-grid method. pylcm’s implementation is a general JAX engine integration with additional constraints: static shapes, pytrees, solver dispatch, and GPU memory limits. The extra asset-row and refine-to-query paths are software representations not part of the original paper’s core presentation.

---

### 3.4 FUES upper-envelope backend

**Implemented concept.** The FUES backend scans an endogenous-grid candidate row and emits the refined upper envelope, including kink/crossing points.

**Mathematical object.**

Given candidate points \((m_i,c_i,V_i)\), FUES should return an ordered representation of the upper envelope

\[
V^{\star}(m)=\max_i I_i[V_i](m)
\]

under the method’s local segment assumptions.

**pylcm differences from the original FUES paper / source algorithm:**

1. Static-shape JAX arrays replace dynamic Python/Numba-style lists.
2. Output is padded to \(N_{\mathrm{pad}}\) in full-row mode.
3. A `scan_unroll` knob exists for the candidate scan.
4. A streamed `refine_to_bracket` / `refine_to_query` path exists for asset-row single-query use.
5. Exact-node crossing handling was repaired in the branch by preserving node-aligned crossings via a dual-policy post-pass.
6. Branch topology is increasingly externalized to a host oracle and explicit segment path rather than inferred entirely inside FUES.

**Theoretical note.** FUES is a topology-discovering scan. It is not inherently GPU-native when applied to many large rows because the scan carries sequential state.

---

### 3.5 1-D RFC backend

**Implemented concept.** A one-dimensional rooftop-cut style backend is available as `DCEGM(upper_envelope="rfc")`.

**Mathematical object.** RFC deletes candidate points dominated by neighbouring tangent planes subject to a jump test, then interpolates survivors.

**Implementation differences from Dobrescu--Shanker RFC.**

1. In one dimension, pylcm can exploit sorted order. The original multidimensional RFC uses local Euclidean neighbours in an irregular cloud.
2. pylcm’s 1-D backend is a special row backend, not a general multidimensional cloud RFC.
3. The JAX version must maintain static shapes and masks rather than dynamically deleting nodes.

**Consequence.** The 1-D RFC backend is useful as an alternative upper-envelope selection strategy, but it is not the same object as the 2-D combined-cloud RFC algorithm.

---

### 3.6 LTM backend

**Implemented concept.** LTM evaluates candidate line segments and takes a brute upper envelope over them.

**Mathematical object.**

Given explicit segments \(s\), evaluate

\[
V(q)=\max_{s:q\in[l_s,r_s]} I_s[V](q).
\]

**Difference from a naive line-through-neighbours interpretation.** Early LTM-style dense evaluation can be wrong if it treats unrelated adjacent candidate points as a segment. The branch later made LTM consume explicit branch topology. This is essential: LTM is correct as a segment evaluator, not as a topology inference algorithm.

---

### 3.7 MSS backend

**Implemented concept.** MSS is related to HARK-style upper-envelope logic for EGM candidate rows.

**Mathematical object.** It should evaluate and select among monotone smooth segments.

**Implementation caveat.** If branch boundaries are inferred from simple decreases in endogenous coordinates or values, MSS can form spurious bridges. The PR later made MSS consume explicit topology. This is not merely an implementation detail; it is mathematically required for general DC-EGM candidate sets.

---

### 3.8 Exact host envelope oracle

**Implemented concept.** A host oracle evaluates explicit branch topology and serves as the authoritative reference for 1-D envelope backends.

**Mathematical role.** It is not necessarily an envelope constructor for production; it is a pointwise or event-complete evaluator against which backends can be tested.

**Important distinction.** The oracle is only as good as its topology input. It verifies consequences of topology; it does not generate branch topology from arbitrary candidate rows unless explicitly designed to do so.

---

### 3.9 Query-side envelope backend

**Implemented concept.** The branch adds an exact query-side envelope backend that evaluates explicit segments at the query points and reduces by max. A later commit adds a blocked segment scan to reduce memory.

**Mathematical object.**

\[
V(q)=\max_{s\in\mathcal S(q)}I_s[V](q).
\]

**Difference from FUES.** FUES constructs the envelope row. Query-side evaluation assumes segment topology is already known and asks only for the envelope at required query points. This is a different representation of the same piecewise-linear envelope object.

**GPU significance.** The operation is a masked segment-by-query map-reduce, which is much more GPU-friendly and chunkable than a topology-discovering scan.

---

### 3.10 Asset-row mode

**Implemented concept.** Asset-row mode solves a separate EGM problem for each current Euler-state node when the Euler RHS depends on current state.

**Mathematical trigger.**

Standard EGM requires:

\[
u'(c)=\Phi(a')
\]

after conditioning on discrete states and smooth branches. Asset-row mode is needed when:

\[
u'(c)=\Phi(a';x).
\]

**Difference from canonical EGM.** It loses the main amortization of EGM. Standard EGM builds one curve and queries it at all current states; asset-row mode builds one curve per current-state node.

**pylcm-specific performance issue.** Asset-row mode originally materialized refined \(N_{\mathrm{pad}}\) envelope rows per node; refine-to-query and query-side methods remove that transient where only one query is needed.

---

### 3.11 Streaming refine-to-query

**Implemented concept.** Instead of building a full refined FUES row and then interpolating it at one query, the FUES scan directly tracks the bracket needed for that query.

**Mathematical equivalence.** If the scan emits the same ordered envelope points and the bracket state implements the same `searchsorted(side="right")` convention, then it is equivalent to full-row construction followed by interpolation at that query.

**Difference from the original FUES algorithm.** This is not a new economic method. It is a memory representation change for a static-shape JAX implementation.

---

### 3.12 NEGM

**Implemented concept.** NEGM nests an inner one-dimensional EGM consumption solve inside an outer deterministic search over a durable or illiquid post-state.

**Mathematical object.**

For outer node \(z'_k\),

\[
V_k(x,z)=\text{inner-EGM solution conditional on }z'_k,
\]

then

\[
V(x,z)=\max_k V_k(x,z).
\]

**pylcm implementation features.**

1. Dual-core period kernel: keeper and adjuster.
2. Keeper durable map via `outer_no_adjustment_candidate`.
3. Cash-on-hand outer-max carry envelope.
4. Chunked/folded outer search for memory.
5. Simulation support.

**Difference from Druedahl’s NEGM.**

Druedahl’s NEGM emphasizes continuation precomputation and loop reordering in a non-convex consumption-saving context. pylcm implements a general solver integration in which the inner solver is the existing DC-EGM kernel and the outer dimension is a deterministic search axis. pylcm’s cash-on-hand outer envelope is a representation detail needed for a finite-horizon carry-based engine.

---

### 3.13 TwoDimEGM / G2EGM foundation

**Implemented concept.** The branch implements a two-asset G2EGM foundation for the DS pension benchmark.

**Mathematical object.** For current states \((m,n)\), choices \((c,d)\), and post-decision states \((a,b)\), solve KKT-region-specific inverse Euler equations. Four regions are used:

1. `ucon`: \(a>0,d>0\);
2. `dcon`: \(a>0,d=0\);
3. `acon`: \(a=0,d>0\);
4. `con`: \(a=0,d=0\).

Each region produces a cloud/mesh of policies and endogenous current states. The G2EGM envelope:

1. triangulates each region mesh;
2. interpolates policies to a common state grid;
3. recomputes the Bellman objective;
4. takes a within-region envelope;
5. takes a cross-region envelope;
6. publishes value, policy, and region-aware gradients.

**Implementation differences from Druedahl--Jørgensen G2EGM.**

1. pylcm is specialized to a two-dimensional pension benchmark path rather than a general arbitrary-dimensional G2EGM engine.
2. It uses JAX static-shape segment meshes and masks.
3. It has an exhaustive all-simplex host oracle for first-envelope validation.
4. It includes direct-Bellman hole fill for no-candidate target nodes.
5. It publishes an `EGMCarry2D`-style regular-grid solution rather than dynamic mesh objects.
6. It includes a working-to-retired boundary adapter.

**Important paper-faithfulness point.** The reference G2EGM method interpolates policies and recomputes the objective. Transporting values directly is not equivalent and can choose different branches. pylcm’s later G2EGM path follows the policy-recompute design.

---

### 3.14 Two-dimensional RFC

**Implemented concept.** The branch adds:

1. host NumPy/SciPy reference for multidimensional RFC;
2. on-device RFC delete mask;
3. combined-cloud 2-D RFC publisher;
4. routing through `TwoDimEGM.upper_envelope`.

**Mathematical object.** RFC deletes dominated points from a combined irregular cloud using tangent-plane dominance and a jump observable. A combined survivor cloud is then published/interpolated once.

**Differences from Dobrescu--Shanker Box 1 RFC.**

1. The paper-style algorithm uses a combined irregular cloud and host-side Delaunay-style interpolation in the reference implementation.
2. pylcm uses an on-device JAX-compatible fixed-shape delete kernel and a barycentric combined-cloud publisher.
3. pylcm uses normalized-distance neighbourhoods, finite \(k\)-style or bounded candidate sets, and a most-local well-conditioned simplex selection rule.
4. pylcm must manage static shapes and memory; the original algorithm can rely on dynamic neighbour structures.

**Consequence.** pylcm’s 2-D RFC is best understood as a paper-inspired, validated JAX approximation to the RFC selection/publishing idea, not necessarily bit-equivalent to host Delaunay RFC at every finite grid.

---

### 3.15 Numerical inverse marginal utility / iEGM-style fallback

**Implemented concept.** When no analytic inverse marginal utility is supplied, pylcm can use a numerical inverse mode with safeguarded Newton / bisection and implicit-gradient correction.

**Mathematical object.** Solve

\[
u'(c)=\mu.
\]

If \(u'\) is strictly decreasing and positive on the bracket, and \(u''(c^*)\ne0\), the implicit derivative is:

\[
\frac{\partial c^*}{\partial \mu}
=
\frac{1}{u''(c^*)}.
\]

**Difference from Carroll and from “iEGM.”** Carroll’s EGM assumes an available inverse marginal utility or cheap inversion. Recent “iEGM” terminology can refer to interpolation or precomputation of inverse marginal utility, not necessarily a per-node root solve. pylcm’s implementation is a generic numerical inverse fallback and must be treated as a correctness fallback, not necessarily a high-throughput default.

---

### 3.16 Continuation operator / W4a

**Implemented concept.** pylcm adds a continuation operator represented by value-transform and inverse-transform functions at the `Q_and_F` seam.

**Mathematical object.** In the joint-lottery case:

\[
\mathcal C[V]
=
g^{-1}
\left(
\sum_{r,z}p(r,z)\,g(V_{r,z})
\right).
\]

**Difference from standard expected utility.** Expected utility is the special case \(g=\mathrm{id}\). Epstein--Zin or other recursive preferences require nonlinear transforms. pylcm’s design applies a joint unresolved-lottery transform; staged resolution would require a different operator.

---

### 3.17 Tax-kink / W5 handling

**Implemented conclusion.** App.3’s discontinuous tax jumps are not fixed by inserting ordinary grid nodes. Continuous kinks may warrant one-sided KKT candidates, but the branch did not build a general W5 feature after magnitude and residual analysis.

**Mathematical point.** A jump discontinuity cannot be fixed by continuous interpolation across it. One-sided candidates or branch pieces are required.

---

### 3.18 DS benchmark harnesses

The PR includes DS-2026 Applications 1/2/3 and DS-2024 housing/pension comparisons:

- App.1 retirement and upper-envelope backend comparisons.
- App.2 continuous-housing NEGM and EGM-FUES discrete-housing model.
- App.3 discrete housing with and without tax.
- DS-2024 pension G2EGM-vs-RFC comparison.
- DS-2024 housing RFC-vs-NEGM comparison.

These are benchmark applications of the algorithms above, not separate numerical algorithms.

---

## 4. Implementation-versus-paper ledger

This section lists the main differences between the pylcm implementation and original papers/method descriptions. Differences are not necessarily bugs; many are necessary for JAX, static shapes, or generic solver integration.

| Method | Original paper / reference | pylcm PR #390 implementation | Difference and consequence |
|---|---|---|---|
| Brute force | generic VFI / grid search | `GridSearch` solver behind ABC seam | No paper conformance claim; useful baseline. |
| Carroll EGM | smooth 1-D consumption-saving, invert Euler | OneAssetEGM / DCEGM core | Integrated into regime engine; supports terminal carries, JAX, optional numeric inverse. |
| IJRS DC-EGM | conditional EGM over discrete alternatives, upper envelope | DCEGM with multi-target continuation, taste shocks, passive states | Broader engine integration; additional static-shape and memory paths. |
| FUES | dynamic scan over candidate correspondence | static-shape padded row, optional `scan_unroll`, refine-to-query, exact-node crossing repair | Same mathematical target; different representation and memory profile. |
| RFC 1-D | rooftop-cut idea based on local dominance | sorted 1-D row backend | Specialized finite-row backend, not full multidimensional RFC. |
| LTM | line-through / brute segment evaluation | explicit-topology segment evaluator | Correct only when topology is supplied; no topology inference guarantee. |
| MSS | HARK-style monotone segment selection | topology-consuming backend after W2 | Earlier heuristic inference is not generally valid; explicit topology is required. |
| Query-side envelope | not a classical paper algorithm | explicit segment query evaluator, blocked scan | GPU-native representation of the same envelope object. |
| NEGM | inner EGM + outer choice, continuation precompute | inner pylcm DC-EGM + outer durable fold + cash-on-hand carry | Same nesting idea; pylcm-specific carry representation and chunking. |
| G2EGM | segment meshes, policy interpolation, recomputed objective, two envelopes | two-dimensional DS pension implementation with host oracle, masks, hole fill | Faithful at algorithmic level for target benchmark; static-shape specialization rather than general arbitrary-dimensional engine. |
| RFC 2-D | combined cloud delete + Delaunay survivors | host reference plus on-device delete and local simplex publisher | Approximate / JAX-friendly publisher; not bit-equivalent to host Delaunay by construction. |
| Numerical inverse | not original Carroll closed-form path | safeguarded Newton / bisection with IFT gradients | Useful fallback; slower and needs strict monotonic/positive conditions. |
| Continuation transform | expected utility in baseline papers | value-transform pair for joint lottery | Extends beyond standard EGM papers; semantics depend on information timing. |
| Tax-kink W5 | model-specific tax schedule handling | no general code after App.3 finding | Correct for discontinuous jumps; continuous-kink candidates remain a future optional tool. |

---

## 5. Formal BQSEGM specification

BQSEGM is best read as **case-piece query-side EGM**. The model-facing concepts are
**case boundaries** and **pieces**. The solver-facing objects are **segments** with
explicit topology.

BQSEGM is not implemented in PR #390 as a named solver. PR #390 supplies useful
building blocks, especially query-side envelope evaluation with `segment_id`, but a full
BQSEGM solver also needs endpoint ownership metadata, NaN-dead masking, and a
continuation object that preserves segment topology or an equivalent switch-refined grid.

### 5.1 DAG model

Let a model be represented by a directed acyclic graph \(G=(\mathcal N,\mathcal E)\).
Each node \(n\in\mathcal N\) is a named function:

\[
f_n:\prod_{p\in\mathrm{pa}(n)} \mathcal X_p\to \mathcal X_n.
\]

A subset of nodes are primitive state/action/parameter inputs. Other nodes compute derived
quantities such as income, resources, taxes, subsidies, premiums, utility, and transition
components.

### 5.2 Case-boundary predicates and endpoint ownership

A **case boundary** is a Boolean DAG node \(p\) with function

\[
p(x)=\mathbf 1\{g_p(x)\le0\}
\]

or a vectorized Boolean expression. It is decorated with boundary metadata

\[
\mathcal B_p=\{(h_{p,\ell}(x)=0,\; e_{p,\ell},\; k_{p,\ell}):\ell=1,\ldots,L_p\},
\]

where \(e_{p,\ell}\in\{\texttt{when},\texttt{otherwise}\}\) says which predicate side owns
equality at that surface, and \(k_{p,\ell}\) records the economic boundary type
(`continuous_kink`, `jump`, or `hard_constraint`).

For example,

\[
p_{\mathrm{medicaid}}(a,y)
=
\mathbf 1\{a<\bar a,\ y<\bar y\}
\]

has equality owned by the `otherwise` side at both surfaces:

\[
a=\bar a,\qquad y=\bar y.
\]

The equality owner is not a tie-breaking detail. It is part of the feasible-set definition.
If the left limit has higher value at \(a=\bar a\) but the model predicate excludes that
side at equality, the envelope must not select the left-limit policy at the exact boundary.

### 5.3 Pieces

A **piece** for output node \(o\) under predicate \(p\) is an alternative producer function
\(f_{o,p}^{+}\) valid when \(p=1\), or \(f_{o,p}^{-}\) valid when \(p=0\).

For each pair \((o,p)\), the minimal binary API requires exactly

\[
f_{o,p}^{+},\qquad f_{o,p}^{-}.
\]

This ensures coverage of both predicate sides. A later multiway-table API can lower a table
into many binary or multiway pieces, but the v1 correctness statements below assume finite,
explicit coverage.

### 5.4 Piece combinations

Let \(\mathcal P=\{p_1,\ldots,p_K\}\) be case-boundary predicates used by the solver. A
**case assignment** is

\[
\sigma\in\{0,1\}^K.
\]

Given \(\sigma\), BQSEGM constructs a specialized DAG \(G_\sigma\) by replacing each
piecewise output \(o\) with the producer corresponding to \(\sigma_p\).

A case assignment may be impossible. It is mathematically acceptable to enumerate and mask
impossible assignments, but implementation benchmarks and compile budgets must count the full
static shape that enumeration creates. Static pruning is a performance improvement, not an
assumption in the exactness theorem.

### 5.5 Smoothness condition for a specialized DAG

For each specialized DAG \(G_\sigma\), every **user-authored economic function** reachable
from the current-period resources, utility, feasibility, transition, or Euler residual in that
variant must be smooth on the interior of its declared domain. Formally, if \(H_\sigma\) is
the Euler residual function,

\[
H_\sigma(c,a';\theta)=u'_c(c;\theta)-\Phi_\sigma(a';\theta),
\]

then \(H_\sigma\) must be continuously differentiable in \(c\) on the interior, and
\(u'_c\) must be strictly monotone in \(c\).

This smoothness check is deliberately scoped. Solver-provided continuation reads, grid
location, envelope evaluation, and interpolation are trusted numerical infrastructure and
may lower to comparisons, `select`, `clamp`, or search primitives. Those primitives are not,
by themselves, evidence of hidden economic case logic. The forbidden-primitive validator
applies only to user-authored economic nodes inside a declared smooth piece, plus user helpers
reachable from those nodes unless explicitly reviewed as trusted smooth helpers.

Any hidden `where`, `if`, `clip`, `maximum`, `searchsorted`, lookup table, or other piecewise
logic in a user-authored smooth node violates the condition unless it is exposed as another
case boundary or admitted through a narrow trusted-helper mechanism with a stated domain proof.

### 5.6 BQSEGM candidate generation

For each assignment \(\sigma\), choose a post-decision grid
\(A_\sigma=\{a'_j\}_{j=1}^{N_S}\). Solve

\[
u'_c(c_{\sigma,j})
=
\Phi_\sigma(a'_j),
\]

where \(\Phi_\sigma\) is the case-specific continuation marginal evaluated through the
trusted continuation machinery.

Then recover current resources or current state:

\[
m_{\sigma,j}=R_\sigma(c_{\sigma,j},a'_j),
\]

and compute value:

\[
V_{\sigma,j}
=
u_\sigma(c_{\sigma,j})
+
\beta \mathcal C_\sigma[V_{t+1}](a'_j).
\]

Candidate generation must record provenance: case assignment, post-decision node, boundary
source if any, and enough local ordering information to split folds and masked holes into
monotone feasible subsegments.

### 5.7 Consistency masks and the absent-candidate convention

The candidate \((m_{\sigma,j},c_{\sigma,j},a'_j)\) is valid only if all predicate choices are
consistent with the equality-owner convention:

\[
p_k(m_{\sigma,j},\ldots)=\sigma_k,
\qquad k=1,\ldots,K.
\]

Thus define

\[
\mathrm{valid}_{\sigma,j}
=
\prod_{k=1}^K
\mathbf 1\{p_k(m_{\sigma,j},\ldots)=\sigma_k\}.
\]

Invalid candidates are removed before segment formation using the **NaN-dead convention**:

\[
(m,V,c,\mu)_{\sigma,j}=(\mathrm{NaN},\mathrm{NaN},\mathrm{NaN},\mathrm{NaN})
\quad\text{when }\mathrm{valid}_{\sigma,j}=0.
\]

The important point is that a finite abscissa with \(V=-\infty\) is not absent for the current
query-envelope primitive: it can still form a live link and can produce `NaN` through linear
interpolation at an endpoint. The pre-envelope absent form is therefore NaN-dead. A published
post-envelope value object may still use \(-\infty\) value and zero marginal for infeasible
choice rows if downstream aggregation requires that finite-probability convention.

### 5.8 Boundary candidates

For every declared boundary \(h_{p,\ell}(x)=0\), BQSEGM must either:

1. generate one-sided candidates adjacent to the boundary with explicit side labels;
2. insert the boundary into the query/published representation with endpoint ownership; or
3. prove that the boundary cannot be optimal for the continuous choice under the declared
   discretized target.

At discontinuous jumps, a single continuous node is insufficient. One-sided values must be
represented separately:

\[
V(x_0^-),\qquad V(x_0^+).
\]

Only the side that owns equality is eligible at the exact boundary query. The other one-sided
limit may be eligible for nearby off-boundary queries, but it must be open at \(x_0\). This is
why boundary handling and consistency masking are a coupled requirement: masking creates holes
at exactly the locations where boundary candidates and endpoint ownership are needed.

### 5.9 Segment construction

After masking and boundary insertion, each assignment \(\sigma\) produces monotone feasible
subsegments

\[
s=(\sigma,\kappa,j_L,j_R,e_L,e_R),
\]

where \(\kappa\) is a segment counter within the case, \((j_L,j_R)\) are adjacent live
candidate indices in that subsegment, and \(e_L,e_R\in\{\text{open},\text{closed}\}\) encode
endpoint ownership. The segment has interval

\[
[l_s,r_s]
=
[\min(m_{\sigma,j_L},m_{\sigma,j_R}),\max(m_{\sigma,j_L},m_{\sigma,j_R})]
\]

with interpolation rules \(I_s[V](q)\), \(I_s[c](q)\), and \(I_s[\mu](q)\).

If the endogenous grid folds, or if a consistency mask creates an interior hole, the path must
be split into multiple subsegments. A `segment_id` is therefore an identifier for one monotone
contiguous feasible subsegment, not merely for a case assignment. Assigning one id per case is
correct only after proving that each case creates exactly one monotone, hole-free segment.

### 5.10 Query-side envelope

For each query \(q\), define eligibility using both the interval and the endpoint flags:

\[
q\in_e [l_s,r_s]
\]

means \(l_s<q<r_s\), with equality at \(l_s\) allowed only if the left endpoint is closed and
equality at \(r_s\) allowed only if the right endpoint is closed. The envelope is

\[
V(q)=\max_{s:q\in_e[l_s,r_s]} I_s[V](q).
\]

The winning policy and marginal are selected from the same segment:

\[
c(q)=I_{s^\star}[c](q),\qquad
\mu(q)=I_{s^\star}[\mu](q).
\]

Ties are handled by one explicit convention, such as right-continuity or highest slope. The
convention must be consistent across full-row, query-side, simulation, and oracle paths.

The existing `upper_envelope/query.py:envelope_at_query` is a useful primitive for closed
segments with `segment_id` topology. A complete BQSEGM implementation must either extend that
primitive with endpoint flags or pre-process boundary queries so that open endpoints cannot win
at excluded equality points.

### 5.11 Published continuation object

BQSEGM may publish value, policy, and marginal on a regular query grid, but exact backward
induction additionally requires one of the following two representation guarantees.

**Topology-preserving payload.** Publish a continuation payload carrying segment ids, endpoint
ownership, boundary/switch flags, and any top-two records needed for one-sided reads. A parent
then evaluates continuation by applying the same segment-aware query convention.

**Switch-refined aggregate grid.** Publish an ordinary aggregate grid only after inserting every
switch, cliff, and boundary node needed to make interpolation of the aggregate value equivalent
to segment-aware interpolation under the declared convention.

A plain aggregate `EGMCarry` containing only maxed values, policies, and marginals on a coarse
query grid is not sufficient in general. It can bridge across discrete-choice or case switches
and overstate continuation values.

If previous-period Euler equations depend on one-sided derivatives at switches, a single
averaged marginal is also insufficient; the representation must preserve the relevant side.

### 5.12 BQSEGM algorithm

For each period \(t\) and regime \(r\):

1. collect case-boundary predicates, endpoint/equality metadata, and piece outputs relevant to
   resources, utility, transitions, feasibility, and the Euler equation;
2. enumerate or statically prune feasible case assignments \(\sigma\);
3. build specialized smooth DAG \(G_\sigma\);
4. validate only the user-authored smooth economic sub-DAG, leaving trusted solver
   interpolation and continuation outside the forbidden-primitive scope;
5. solve EGM on \(A_\sigma\);
6. apply NaN-dead consistency masks;
7. add one-sided boundary candidates with endpoint ownership;
8. split each case into monotone feasible subsegments and assign `segment_id` per subsegment;
9. evaluate the query-side upper envelope in blocks using the endpoint-aware segment convention;
10. publish value, policy, marginal, and either topology-preserving continuation metadata or a
    switch-refined aggregate grid with a proof that ordinary interpolation is convention-exact.

---

## 6. BQSEGM correctness theorems

### Theorem 1: Correctness of case enumeration for the declared finite interpolation target

**Statement.** Fix a finite discretized Bellman target together with its interpolation,
endpoint-ownership, and envelope convention \(\mathcal K_h\). Let \(\mathcal S_h\) be the
finite set of convention-level candidates: one-sided boundary records and monotone segments
eligible under \(\mathcal K_h\). Suppose the feasible candidate set is covered by case
assignments

\[
\mathcal S_h=\bigcup_{\sigma\in\Sigma}\mathcal S_{\sigma,h}.
\]

Suppose further that for each \(\sigma\), the BQSEGM generator produces exactly the segments
and boundary records in \(\mathcal S_{\sigma,h}\), uses NaN-dead masks for all invalid
pre-segment candidates, assigns endpoint ownership correctly, and the final envelope takes the
maximum over the union of all eligible generated records. Then BQSEGM returns the exact value
and selected policy for the declared discretized interpolation target:

\[
V_h(q)=\max_{s\in\mathcal S_h(q)} I_s[Q_h](q),
\]

where \(\mathcal S_h(q)\) includes exactly the segments whose intervals contain \(q\) under the
open/closed endpoint convention.

This theorem is not a claim of equality to an independent brute-force action-grid solve using
a different candidate grid or interpolation rule. A brute-force comparison is exact only when
it is defined over the same finite candidate records and the same convention \(\mathcal K_h\),
or asymptotic/approximate under a separate refinement theorem.

**Proof.** By the cover,

\[
\max_{s\in\mathcal S_h(q)} I_s[Q_h](q)
=
\max_{\sigma\in\Sigma}
\max_{s\in\mathcal S_{\sigma,h}(q)} I_s[Q_h](q).
\]

By candidate completeness, NaN-dead masking, and endpoint-correct boundary generation, the
generated candidate records for \(\sigma\) are exactly \(\mathcal S_{\sigma,h}\) under
\(\mathcal K_h\). Taking the maximum over all eligible generated records is therefore equal to
the right-hand side. The policy and marginal are selected from the same winning record by the
same convention. \(\square\)

---

### Theorem 2: BQSEGM restores standard-EGM amortization when case conditioning removes current-state dependence

**Statement.** Suppose that after fixing a case assignment \(\sigma\), the Euler equation has
the form

\[
u'_c(c)=\Phi_\sigma(a')
\]

and does not depend separately on current resources \(m\). Then one EGM correspondence per
\(\sigma\) serves all current-resource queries under the declared segment/envelope convention.
If no finite case assignment removes the dependence and the RHS is \(\Phi(a';m)\) with distinct
values across current grid nodes on a set of savings nodes, then an exact shared-curve EGM
correspondence cannot satisfy the Euler equation at all those current nodes; asset-row
replication or a different exact representation is required.

**Proof.** For fixed \(\sigma\), invert

\[
c_\sigma(a')=(u'_c)^{-1}(\Phi_\sigma(a')).
\]

This is independent of \(m\), so a single correspondence
\(a'\mapsto(m_\sigma(a'),c_\sigma(a'))\), split into eligible segments if needed, can be
queried at all \(m\). This proves the first claim.

For the converse, suppose a single exact correspondence \(c(a')\) served two current states
\(m_1,m_2\) at a savings node where the right-hand sides differ. Then

\[
u'_c(c(a'))=\Phi(a';m_1)=\Phi(a';m_2),
\]

contradicting the distinct RHS values when \(u'_c\) is single-valued. \(\square\)

---

### Theorem 3: Hidden piecewise logic makes exact BQSEGM treatment impossible from finite black-box samples

**Statement.** No solver that only receives black-box evaluations of a piecewise function can
infer all threshold locations exactly from finitely many evaluations.

**Proof.** Let \(S=\{x_1,\ldots,x_n\}\) be any finite sample set in \([0,1]\). Choose
\(\delta>0\) such that no sample lies in \((1/2,1/2+\delta)\). Define

\[
f_1(x)=\mathbf 1\{x\ge1/2\},
\qquad
f_2(x)=\mathbf 1\{x\ge1/2+\delta\}.
\]

Then \(f_1(x_i)=f_2(x_i)\) for all samples, but the boundary locations differ. Any solver using
only the samples cannot distinguish the two functions and cannot know which boundary
candidates to insert. \(\square\)

**Corollary.** BQSEGM requires exposed `case_boundary` metadata for exact threshold handling.

---

### Theorem 4: Query-side envelope evaluation is exactly chunkable

**Statement.** Let \(\mathcal S=\cup_{b=1}^B\mathcal S_b\) be a partition of explicit segments.
For fixed query \(q\),

\[
V(q)=\max_{s\in\mathcal S(q)}I_s[V](q)
=
\max_{b=1,\ldots,B}
\max_{s\in\mathcal S_b(q)}I_s[V](q).
\]

Thus segment blocks can be scanned with a running maximum without changing the result.

**Proof.** The maximum over a finite set is associative and commutative. Endpoint eligibility
is evaluated segment by segment before the maximum, so blocking does not alter the feasible
set. \(\square\)

---

### Theorem 5: A continuous interpolant cannot uniformly approximate a jump discontinuity on a cell containing the jump

**Statement.** Let \(V\) have a jump discontinuity at \(x_0\). No sequence of continuous
interpolants on a fixed interval containing \(x_0\) can converge uniformly to \(V\).

**Proof.** Uniform limits of continuous functions are continuous. A jump-discontinuous \(V\)
is not continuous. Contradiction. \(\square\)

**Implication.** Discontinuous tax, subsidy, or benefit notches require one-sided
representation. Ordinary node insertion without side ownership is insufficient.

---

### Theorem 6: Segment-aware interpolation is less than or equal to aggregate-max interpolation

**Statement.** Let \(I\) be a positive linear interpolation operator and \(V_b\) branch- or
segment-specific values sampled at the same interpolation nodes. Then

\[
\max_b I[V_b](q)
\le
I[\max_b V_b](q).
\]

**Proof.** Since \(V_b(x_i)\le \max_j V_j(x_i)\) at every interpolation node \(x_i\), positive
linearity gives

\[
I[V_b](q)\le I[\max_j V_j](q)
\]

for every \(b\). Taking the maximum over \(b\) proves the claim. \(\square\)

**Implication.** Interpolating already-maximized aggregate values can bridge over discrete
choice, case, or segment switches. Choice-specific or segment-specific continuation carries are
necessary near switches unless the published grid is refined so that aggregate interpolation is
identical to the segment-aware convention.

---

### Theorem 7: BQSEGM dominates asset-row EGM only under an explicit cost inequality

**Statement.** Let asset-row EGM have total cost

\[
T_{\mathrm{row}}
=
N_X C_E^{\mathrm{row}}
+N_Q^{\mathrm{row}}N_{\mathrm{seg}}^{\mathrm{row}}C_I^{\mathrm{row}}
+C_{\mathrm{pub}}^{\mathrm{row}}
+C_{\mathrm{compile}}^{\mathrm{row}}(S_{\mathrm{row}})
+C_{\mathrm{mem}}^{\mathrm{row}}(S_{\mathrm{row}}),
\]

and let BQSEGM have total cost

\[
T_{\mathrm{BQ}}
=
N_C C_E^{\mathrm{case}}
+N_{\mathrm{bd}}C_{\mathrm{bd}}
+N_QN_{\mathrm{seg}}C_I
+C_{\mathrm{pub}}^{\mathrm{topo}}
+C_{\mathrm{compile}}^{\mathrm{BQ}}(S_{\mathrm{BQ}})
+C_{\mathrm{mem}}^{\mathrm{BQ}}(S_{\mathrm{BQ}}).
\]

Here \(N_{\mathrm{seg}}\) is the number of monotone feasible subsegments after case splitting,
fold splitting, masks, boundary candidates, and endpoint splits; \(S_{\mathrm{BQ}}\) denotes the
static shapes compiled by JAX; and \(C_{\mathrm{pub}}^{\mathrm{topo}}\) is the cost of publishing a
topology-preserving payload or a switch-refined aggregate grid.

BQSEGM is faster than asset-row EGM if and only if

\[
T_{\mathrm{BQ}} < T_{\mathrm{row}}.
\]

The simple condition \(N_C<N_X\) is only a heuristic corollary under additional assumptions:
per-correspondence costs are comparable, \(N_{\mathrm{seg}}\) is uniformly bounded or small enough
that \(N_QN_{\mathrm{seg}}C_I\) is lower order, boundary and publication costs are small, and
static compile/memory shapes do not dominate.

**Proof.** Direct comparison of the complete work, memory, and compile-shape cost expressions.
The stated heuristic follows only after imposing the listed bounds, because generally
\(N_{\mathrm{seg}}\) scales with the number of case assignments times the number of monotone
subsegments and boundary splits per case. When \(N_Q\) is comparable to \(N_X\), the
query-envelope term can be \(O(N_QN_C)\), so \(N_C<N_X\) is not by itself sufficient. \(\square\)

**Implication.** ACA-like models benefit from BQSEGM only if the full inequality is favorable:
institutional cases must remain few, segment counts and boundary splits must be bounded, and
topology-preserving publication must not create a compile or memory wall.

---

## 7. General solver-choice theorems

### 7.1 Brute-force streaming theorem

For a finite action grid partition \(A_h=\cup_k A_k\),

\[
\max_{a\in A_h}Q(x,a)
=
\max_k \max_{a\in A_k}Q(x,a).
\]

Therefore brute force can stream action chunks with resident memory proportional to the largest chunk rather than the full action grid.

### 7.2 Standard EGM amortization theorem

Standard EGM amortizes over all current-state queries if and only if the Euler RHS is independent of current resources after conditioning on exogenous states and smooth cases. If it is not, asset-row replication is required for exact EGM.

### 7.3 Brute versus EGM hardware theorem

Let brute and EGM times be:

\[
T_B
=
\max
\left\{
\frac{N_XN_A C_Q}{\eta_B^F F},
\frac{N_XN_A R_Q}{\eta_B^M B}
\right\},
\]

\[
T_E
=
\max
\left\{
\frac{N_CN_S C_E+N_QN_{\mathrm{seg}}C_I}{\eta_E^F F},
\frac{R_E}{\eta_E^M B},
D_E\lambda
\right\}.
\]

Brute is faster if and only if \(T_B<T_E\). EGM is faster if and only if \(T_E<T_B\).

This tautological theorem is important because it prevents unconditional claims. The key empirical inputs are utilization and resident memory, not only nominal candidate counts.

### 7.4 Asymptotic action-dimension theorem

If brute requires:

\[
N_A(\varepsilon)=\Theta(\varepsilon^{-d_y/p_A})
\]

and EGM requires:

\[
N_S(\varepsilon)=\Theta(\varepsilon^{-d_s/p_S}),
\]

with \(d_y/p_A>d_s/p_S\), and EGM overhead does not add a term of equal or higher order, then EGM eventually dominates as \(\varepsilon\to0\).

### 7.5 NEGM dominance theorem

For an inner consumption grid \(N_S\), outer durable grid \(N_Z\), and brute consumption grid \(N_C\), NEGM work is lower than two-action brute if:

\[
N_ZN_SC_E + N_ZC_{\mathrm{outer}}
<
N_ZN_CC_Q.
\]

Equivalently:

\[
\frac{N_C}{N_S}
>
\frac{C_E}{C_Q}
+
\frac{C_{\mathrm{outer}}}{N_SC_Q}.
\]

### 7.6 Multidimensional method necessity theorem

If no one-dimensional nesting exists and the optimum is characterized by a coupled system of \(d_y>1\) first-order conditions, then a one-dimensional EGM correspondence cannot represent the solution. A method must either:

1. solve a multidimensional inverse system;
2. perform an outer search over \(d_y-1\) dimensions;
3. or revert to brute force / direct optimization.

This is a structural fact, not an implementation preference.

---

## 8. GPU and CPU consequences

### 8.1 Dense brute force

GPU strengths:

- dense action-state matrix evaluation;
- associative max reduction;
- static shapes;
- chunked memory.

CPU strengths:

- fine for small grids;
- no JIT compile overhead.

Weakness:

- curse of dimensionality in continuous action dimension.

### 8.2 FUES

CPU strengths:

- row scans can be cache-resident;
- branchy sequential control is tolerable.

GPU weaknesses:

- `lax.scan` dependency;
- padded envelope rows;
- batched row materialization;
- compile complexity.

### 8.3 Query-side segmented envelopes

GPU strengths:

- map-reduce over query and segment blocks;
- exact chunking by max associativity;
- memory controlled by block sizes.

Weakness:

- topology must already be correct;
- if \(N_QN_{\mathrm{seg}}\) is huge, dense query-side matrices must be blocked.

### 8.4 G2EGM

GPU strengths:

- if implemented as blocked triangle scans, it becomes a large map-reduce.

Weakness:

- geometry and hole fill can be irregular;
- static shape requires padding/masks;
- host Delaunay is not GPU-native.

### 8.5 RFC

GPU strengths:

- delete mask can be parallel over candidate pairs if memory permits.

Weakness:

- neighbour search and cloud publication are hard;
- all-pairs \(O(N^2)\) memory is impossible at scale;
- spatial hashes need static occupancy and overflow rules;
- publisher may differ from host Delaunay.

### 8.6 NEGM

GPU strengths:

- outer nodes can be blocked or vmapped;
- inner EGM is reusable.

Weakness:

- if the outer fold stores only sampled maxima, subgrid islands can be missed;
- correct segment-aware carry representation is needed.

---

## 9. ACA-style institutional models

### 9.1 Why ACA is hostile to ordinary EGM

ACA-like models contain:

- Medicaid asset/income eligibility;
- ACA subsidy brackets;
- premium default;
- consumption floors;
- insurance plan choices;
- medical expense cost-sharing;
- borrowing-rate kinks;
- tax/transfer cliffs.

If these remain opaque functions of current assets, the Euler RHS may depend on current assets after savings is fixed:

\[
u'(c)=\Phi(a';x).
\]

Then exact shared-curve EGM is impossible by Theorem 2.

### 9.2 BQSEGM path

Expose rules as case pieces:

```python
@lcm.case_boundary(
    lcm.boundary("assets", "asset_limit", equality="otherwise", kind="jump")
)
def medicaid_eligible(assets, asset_limit):
    return assets < asset_limit

@lcm.piece("oop", when=medicaid_eligible)
def oop_medicaid(...): ...

@lcm.piece("oop", otherwise=medicaid_eligible)
def oop_private(...): ...
```

Within each piece, run EGM; after inversion, mask inconsistent points with the NaN-dead convention; add side-aware boundary candidates; split into monotone feasible subsegments; upper-envelope across all eligible segments.

### 9.3 When BQSEGM fails for ACA

BQSEGM fails if, even after all finite cases are fixed,

\[
\Phi(a';x)
\]

still depends continuously and nonseparably on \(x\).

It also becomes unattractive if case combinations explode:

\[
N_C \approx \prod_{\ell}K_\ell
\]

or when the resulting segment, boundary, publication, compile, and memory terms make the full Theorem 7 inequality fail. The relevant comparison is not only the case product versus \(N_X\), because \(N_QN_{\mathrm{seg}}\) and static-shape memory can dominate.

### 9.4 Critical validator for ACA-style BQSEGM

A BQSEGM implementation must reject hidden economic case logic inside user-authored smooth
pieces. AST plus JAXPR validation is required, but the scope is essential:

- reject Python `if`, `match`, and conditional expressions in smooth user economic nodes;
- reject undeclared comparisons, Boolean case logic, `jnp.where`, `clip`, `maximum`, `minimum`,
  `searchsorted`, and lookup-table primitives inside smooth user economic nodes;
- inspect user helpers reachable from those nodes, including nested JAXPRs;
- allow case-boundary functions to contain vectorized comparisons and Boolean logic;
- exclude trusted solver infrastructure: continuation interpolation, grid location, envelope
  evaluation, and the EGM kernel itself;
- treat the existing numerical cliff checker as a savings-stage/node-resolution diagnostic, not
  as a complete substitute for BQSEGM case-piece validation.

Otherwise the solver cannot know where to insert boundary candidates, and a global primitive
ban would reject every realistic EGM model because continuation reads necessarily use grid
interpolation.

---

## 10. Decision tables

### 10.1 Solver by model feature

| Feature | Recommended solver |
|---|---|
| Smooth one-asset problem | Standard EGM or brute after benchmark |
| Smooth one-asset problem with modest action grid | Brute may win on GPU |
| One-dimensional discrete-continuous non-concavity | DC-EGM/FUES or query-side segmented EGM |
| Huge memory wall from padded envelope | Query-side segmented EGM |
| Finite institutional thresholds | BQSEGM |
| Current-state dependence remains after cases | Brute or asset-row EGM |
| Two continuous actions with clean inner nest | NEGM |
| Genuinely coupled 2-D FOC system | G2EGM |
| Paper RFC comparison required | RFC plus G2EGM/brute oracle |
| Discontinuous jumps/notches | Case pieces and one-sided candidates |
| Dense, regular, low-dimensional action grid | Brute force |

### 10.2 Solver by hardware

| Hardware condition | Solver tendency |
|---|---|
| GPU, high memory bandwidth, moderate action grid | Brute |
| GPU, memory-bound full-row envelopes | Query-side segmented EGM |
| GPU, high action dimension | EGM/NEGM/G2EGM if topology works |
| CPU, small rows, branchy envelopes | FUES/LTM/MSS/G2EGM viable |
| Host geometry available and model small | RFC/G2EGM host routines viable |
| Many static shapes / age variants | Brute or simplified EGM to avoid compile explosion |

---

## 11. Practical implementation requirements

### 11.1 For all EGM-like solvers

1. Explicit state/action support and boundary conditions.
2. Valid inverse Euler equation or numerical inverse with bracket proof.
3. Complete constrained-region candidate generation.
4. Explicit branch/segment topology.
5. Correct upper-envelope selection.
6. One-sided handling of jumps and ties.
7. Independent brute or host oracle on small grids.
8. Accuracy metrics beyond Euler residuals.

### 11.2 For BQSEGM specifically

1. Decorated `case_boundary` predicates with equality-owner and boundary-type metadata.
2. Decorated `piece` formula alternatives.
3. Complete true/false coverage for every binary predicate/output pair.
4. Scoped AST and JAXPR validators for user-authored smooth economic nodes only.
5. DAG lowering into smooth variants.
6. Post-EGM consistency masks using the NaN-dead pre-envelope convention.
7. One-sided boundary candidate logic with open/closed endpoint ownership.
8. Query-side segmented envelope with segment ids per monotone feasible subsegment.
9. Topology-preserving continuation payload or a switch-refined aggregate grid with an exactness proof.
10. Case-combination pruning or compile/memory diagnostics.
11. Brute-force toy oracle under the same interpolation/envelope convention.

### 11.3 For query-side envelopes

1. Explicit segment links.
2. Valid bracketing predicate.
3. Tie convention.
4. Blocked implementation.
5. Full-row equivalence tests where applicable.
6. Exact-crossing tests.

### 11.4 For G2EGM

1. Region masks enforcing KKT conditions.
2. Nondegenerate simplices.
3. Policy interpolation, not value transport.
4. Objective recomputation.
5. First and second envelope.
6. Direct-Bellman hole fill with policy and value.
7. Region-aware gradients.
8. Switch masks.
9. Host all-simplex oracle.

### 11.5 For RFC

1. Combined cloud.
2. Well-defined distance metric.
3. Jump observable tied to model economics.
4. Radius / neighbourhood semantics.
5. Blocked delete mask.
6. Combined-cloud publisher.
7. Host reference.
8. Hybrid fallback diagnostics.

---

## 12. Decisive experiments

### 12.1 For BQSEGM

- One-predicate Medicaid toy.
- Two-predicate Medicaid plus premium-default toy.
- ACA-like subsidy table toy.
- Compare to brute on dense grids using the same finite interpolation/envelope convention.
- Compare memory and compile time to asset-row EGM.
- Test hidden `where` rejection in a user piece without rejecting trusted continuation interpolation.
- Test NaN-dead invalid masks; a finite `x` with `value=-inf` must not be used as a pre-envelope absent candidate.
- Test open/closed boundary ownership at exact threshold queries.
- Test segment ids per monotone subsegment, including a folded or hole-split case.
- Test topology-preserving continuation against an aggregate-grid bridge counterexample.

### 12.2 For brute versus EGM

For each model:

1. run brute at increasing action-grid density;
2. run EGM/BQSEGM/NEGM at increasing post-decision grid density;
3. compare value error, policy objective loss, Euler residual, moments;
4. report compile time, execution time, and peak memory.

### 12.3 For query-side envelope

- exact crossing;
- near crossing;
- multiple branches;
- disconnected segments;
- constrained branch;
- value/policy/marginal read equivalence;
- blocked versus dense equivalence.

### 12.4 For G2EGM/RFC

- tiny exhaustive all-simplex oracle;
- host Delaunay / RFC reference;
- off-grid Bellman objective gap;
- representation consistency;
- switch-local accuracy;
- hole frequency;
- grid-refinement convergence.

---

## 13. Limitations and open problems

1. **Case and segment explosion.** BQSEGM can become as bad as asset-row mode if the case product or the monotone-subsegment count is large.
2. **Automatic branch discovery.** Exact discovery from black-box code is impossible.
3. **Switch-cell derivatives.** One regular-grid gradient per node is insufficient near branch switches unless side information is preserved.
4. **Topology publication.** A plain aggregate carry can bridge across switches; BQSEGM needs a topology-preserving payload or a switch-refined aggregate grid.
5. **Endpoint ownership.** Closed-interval envelope primitives are insufficient at discontinuous predicate boundaries unless open/closed endpoint metadata is enforced.
6. **High-dimensional continuous states.** G2EGM and RFC do not scale simply beyond two dimensions.
7. **Nonlinear continuation operators.** Epstein--Zin and other risk transforms require explicit timing assumptions.
8. **Simulation moments.** A small value error near a switch can cause large policy/moment changes.
9. **GPU geometry.** Dynamic Delaunay and KD-tree methods are not natural inside JIT.
10. **Papers versus implementations.** CPU reference algorithms often use dynamic memory and scalar loops; JAX implementations may need different but mathematically equivalent representations.

---

## 14. Source notes and references

### 14.1 pylcm PR #390

The PR is the primary implementation inventory for this document. Its public description states that it adds the DC-EGM solver behind the `Solver` ABC seam, using Euler inversion and FUES, and later builds NEGM, 1-D RFC/MSS/LTM, 2-D RFC/G2EGM, DS replications, query-side envelopes, numerical inverse, and continuation-operator work.

Source: <https://github.com/OpenSourceEconomics/pylcm/pull/390>

### 14.2 Method references

- Carroll, C. D. (2006). *The method of endogenous gridpoints for solving dynamic stochastic optimization problems*. Economics Letters.
- Iskhakov, F., Jørgensen, T. H., Rust, J., & Schjerning, B. (2017). *The endogenous grid method for discrete-continuous dynamic choice models with (or without) taste shocks*. Quantitative Economics.
- Druedahl, J. (2021). *A guide on solving non-convex consumption-saving models*. Computational Economics.
- Druedahl, J., & Jørgensen, T. H. (2017). *A general endogenous grid method for multi-dimensional models with non-convexities and constraints*. Journal of Economic Dynamics and Control.
- Dobrescu--Shanker / InverseDCDP RFC and DS benchmark materials.
- JAX documentation for `vmap` and `lax.scan`.

---

## 15. Final guidance

The best default is still brute force until a structure-specific solver proves a matched-accuracy advantage. The best EGM-like production direction for GPUs is not “more FUES,” but:

\[
\textbf{explicit topology}
+
\textbf{case-piece EGM}
+
\textbf{query-side blocked envelopes}.
\]

For ACA-like models, the core question is economic and structural:

> Can the institutional rules be exposed as finitely many smooth pieces so that current-state dependence disappears within each piece?

If yes, BQSEGM is the natural solver. If no, brute force or asset-row EGM remains the honest choice.
