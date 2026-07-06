# NB-EGM: An Endogenous-Grid Method for Lifecycle Models with Declared Institutional Kinks and Cliffs

**Draft — computational-economics methods paper.**

---

## Abstract

Institutional rules — asset tests, subsidy brackets, benefit notches, consumption
floors — break the smoothness that the endogenous-grid method (EGM) requires: the
budget constraint becomes piecewise, the continuation value acquires kinks and jump
discontinuities, and the Euler equation's right-hand side may depend on the current
state even after savings is fixed. In practice these models are solved by brute-force
grid search, which is exact for its discretized target, embarrassingly parallel, and —
on modern accelerators — often faster than nominally cheaper methods. We present
NB-EGM, an endogenous-grid solver for finite-horizon discrete–continuous dynamic
programs whose budget is split by *declared* breakpoints (continuous kinks, jump
cliffs, and hard-constraint floors) on monotone derived income variables. The model
author exposes each institutional boundary as metadata; the solver partitions the
liquid-asset axis into intervals on which the budget is affine, runs one EGM inversion
per interval, generates an explicit candidate system (Euler paths, borrowing and
saturation corners, boundary-targeting and save-to-cliff candidates, and a dense
savings-node floor), removes inconsistent candidates by NaN-dead masking, and merges
everything with a branch-aware query-side upper envelope. A discrete action is handled
by a second, outer envelope over per-branch continuous solves streamed through a
compiled batch axis. Cross-regime continuation values are read in one of two cliff
modes: an exact one-sided mode that publishes every jump preimage as a duplicated
abscissa carrying one-sided limits, and a cheaper bridged mode that tolerates
finite-grid cliff error. We give correctness results for the discretized target, an
impossibility result showing that declared breakpoints are necessary (no finite set of
black-box evaluations can locate thresholds exactly), exact-chunkability results that
make the method streamable on GPUs, and an explicit conditional cost model that says
when NB-EGM beats brute force — and when it does not. Validation combines a dense
brute-force oracle on toy models, solved under the same interpolation and envelope
convention, with full-model agreement gates; a structural retirement model of the
Affordable Care Act with 18 regimes serves as the running application.

---

## 1 Introduction

Dynamic structural models of household behavior increasingly embed detailed
institutional rules. A Medicaid asset test switches out-of-pocket medical costs
discontinuously at an asset threshold; ACA premium subsidies follow bracketed
schedules of income; benefit programs impose notches; consumption floors truncate the
budget from below. Each such rule inserts a kink or a cliff into the household's
cash-on-hand as a function of its liquid assets, and — through backward induction — a
kink or a jump into the value function itself, recurring period after period.

These features are hostile to the endogenous-grid method (Carroll 2006). EGM's
appeal is amortization: invert the Euler equation once per post-decision (savings)
node and read the resulting policy correspondence at every current state. That logic
requires (i) a smooth, invertible marginal utility, (ii) a budget that maps the
recovered consumption–savings pair back to a unique current state, and (iii) a
continuation whose marginal is well defined where the inversion samples it. Kinked
budgets violate (ii) piecewise; jump discontinuities violate (iii) outright; and if
the institutional rule makes the Euler equation's right-hand side depend on the
*current* state even after conditioning on savings, the amortization is lost entirely
— the exact fallback is one EGM problem per current-state node ("asset-row" mode),
which forfeits EGM's advantage. Discrete–continuous extensions — DC-EGM
(Iskhakov, Jørgensen, Rust, and Schjerning 2017) — handle the non-concavities induced
by discrete choices via upper envelopes, and upper-envelope refinements
(Dobrescu and Shanker 2022; Druedahl and Jørgensen 2017; Druedahl 2021) extend the
reach of EGM to non-convex problems, but none of these treats *institutional*
breakpoints — thresholds known to the modeler in closed form — as first-class
algorithmic objects.

Meanwhile, the practical benchmark has quietly changed. On GPUs, brute-force grid
search over the continuous action is a dense map-reduce with static shapes, perfect
chunkability (max is associative), and no sequential scans. An algorithm with lower
arithmetic complexity loses to it whenever the algorithm materializes large
transients, carries loop-borne dependencies, or compiles many shape variants. Any
honest proposal of an EGM-family method today must therefore be *conditional*: state
the cost model under which it wins, and concede the regimes where brute force remains
the right default.

This paper contributes NB-EGM ("non-smooth-budget EGM"; code identifier `NBEGM`), a
solver for one-dimensional consumption–saving regimes inside finite-horizon
discrete–continuous lifecycle models, built on three ideas:

1. **Declared breakpoints.** Institutional boundaries are not discovered; they are
   declared. The model author decorates the relevant model functions with metadata:
   a case boundary (a Boolean predicate with an equality-owner and a discontinuity
   kind) or a piecewise-affine schedule on a monotone derived variable. We prove that
   this is not a convenience but a necessity: no solver can recover threshold
   locations exactly from finitely many black-box evaluations (Theorem 3).
2. **Per-interval EGM with an explicit candidate system.** The declared thresholds
   are mapped to their preimages on the liquid-asset axis, producing a sorted
   interval partition on which cash-on-hand is affine. Each interval is solved by
   ordinary EGM; non-concavity is handled not by a topology-discovering scan but by
   an explicit, provenance-tagged candidate set — Euler paths split at folds,
   borrowing and grid-saturation corners, candidates that target the eligible side of
   each cliff, and a dense savings-node floor — merged by an exact query-side upper
   envelope that never bridges unrelated branches.
3. **A conditional cost theory.** We give exact-chunkability results (the envelope
   and the brute-force reduction stream identically), a tautological-but-disciplining
   hardware inequality, and the explicit inequality under which NB-EGM dominates
   asset-row EGM and brute force. The message is deliberately non-triumphalist: with
   few declared breakpoints and bounded segment counts, NB-EGM restores EGM's
   amortization through recurring cliffs; when case structure explodes or the
   continuation's state dependence survives conditioning, brute force stays the
   honest choice.

The method is implemented in the open-source lifecycle-model framework `pylcm` on top
of JAX; every algorithmic object in this paper corresponds to a static-shape,
JIT-compiled kernel. Section 2 defines the problem class, Section 3 the algorithm,
Section 4 the correctness results, Section 5 the cost model, Section 6 the validation
methodology, Section 7 the performance levers, and Section 8 concludes. An appendix
lists where this paper deviates from the project's earlier design document in favor of
the code as built.

---

## 2 Problem class and notation

### 2.1 The dynamic program

Time is finite, $t = 0, \dots, T$. At period $t$ an agent occupies a regime $r_t$ (a
dynamic-programming mode with its own value function, transition targets, and
solver), a finite state $z_t$, and a continuous state $x_t \in X_t \subset
\mathbb{R}^{d_x}$. It chooses a finite action $d_t \in \mathcal{D}_t$ and a continuous
action $y_t \in Y_t(x_t, z_t, d_t)$. The Bellman equation is

$$
V_t(r, z, x)
= \max_{d \in \mathcal{D}_t(r,z,x)} \;
  \sup_{y \in Y_t(r,z,x,d)}
  \Big\{ u_t(r,z,x,d,y) + \beta_t\,
  \mathcal{C}_t\big[ V_{t+1}(r',z',x') \,\big|\, r,z,x,d,y \big] \Big\},
$$

with $(r', z', x') = F_t(r, z, x, d, y, \epsilon)$ and $\epsilon$ approximated by
finite quadrature or a finite Markov transition. $\mathcal{C}_t$ is the continuation
operator; in the expected-utility case treated here it is a linear expectation over
the joint next-period lottery (regime transition, stochastic states, shocks).

NB-EGM addresses the sub-class in which one continuous state — the *liquid* (Euler)
state $a$, with post-decision savings $s = \mathrm{coh}(a, \cdot) - c \ge 0$ — carries
the Euler equation, one continuous action $c$ (consumption) solves it, at most one
finite action $d$ enters the period problem, and every other state (a continuous
co-state such as pension claims, discrete health or preference types, stochastic
processes) *rides along*: it enters the budget, utility, and transitions but carries
no first-order condition of its own. Each combination of ride-along values is a
*cell*; the solver batches the one-dimensional liquid solve over cells.

### 2.2 Declared breakpoints

The institutional structure is exposed through model metadata of two kinds.

- **Case boundary and pieces.** A Boolean predicate $p$ on model variables (e.g.
  `assets < medicaid_asset_limit`) is decorated as a case boundary. Each boundary
  surface carries (i) the variable and threshold, (ii) an *equality owner*
  $e \in \{\texttt{when}, \texttt{otherwise}\}$ — which predicate side owns the exact
  boundary point — and (iii) a *kind*
  $k \in \{\texttt{continuous\_kink}, \texttt{jump}, \texttt{hard\_constraint}\}$.
  A split output (e.g. out-of-pocket medical costs) is covered by exactly one smooth
  formula per predicate side. The v1 scope is one binary predicate per regime.
- **Piecewise-affine schedule.** A budget component declared piecewise-affine in a
  monotone derived variable (taxable income, countable resources), with a list of
  breakpoints, each carrying a threshold (possibly a state-indexed table), a kind,
  and an equality owner. Multiple schedules on different variables may coexist; their
  breakpoints merge into one partition of the liquid axis (Section 3.2).

The equality owner is part of the feasible-set definition, not a tie-break
convenience: if the left limit of the value is higher at the threshold but the model
excludes that side at equality, the solver must not select the left-limit policy at
the exact boundary point.

**Smoothness contract.** Within each case (each side of every declared boundary),
every user-authored economic function reachable from resources, utility, feasibility,
transitions, or the Euler residual must be smooth on the interior of its domain: the
marginal utility $u_c'(c)$ strictly decreasing and the Euler residual continuously
differentiable in $c$. Solver infrastructure (grid location, interpolation, envelope
evaluation) is trusted and exempt. The contract is enforced mechanically at model
build (Section 3.7).

### 2.3 The discretized numerical target

All solvers compared here solve a *discretized* problem. Notation:

| Symbol | Meaning |
|---|---|
| $N_X$ | current liquid-state grid nodes |
| $N_S$ | post-decision savings-grid nodes |
| $N_A$ | brute-force continuous-action candidates |
| $N_Q$ | query nodes at which the solution is published (here $N_Q = N_X$, plus jump augmentations) |
| $N_B$ | declared breakpoints; $N_B + 1$ intervals |
| $N_{\mathrm{seg}}$ | candidate segments after masks, folds, corners, and point candidates |
| $N_D$ | discrete-action branches |
| $N_{\mathrm{cell}}$ | ride-along cells |
| $N_Z$ | stochastic quadrature nodes per cell |
| $N_R$ | regimes read by the continuation |

A solver is **correct for the discretized target** if it returns the value and policy
that exhaustive evaluation of the *same* finite candidate records under the *same*
declared interpolation, endpoint-ownership, and envelope convention would return.
Correctness claims in Section 4 are of this form; convergence of the discretized
target to the continuous problem under mesh refinement is a separate question that we
do not address formally.

---

## 3 The NB-EGM algorithm

### 3.1 Overview

One period of one regime, per ride-along cell and discrete branch, proceeds as:

```text
NB-EGM period step (one regime, period t)
Inputs: child carries {C_r'} on their liquid grids, params θ, grids.

1. Breakpoint geometry (per cell, per branch if the action enters a schedule):
   map every declared threshold to its liquid-axis preimage; clamp degenerate
   preimages outside the grid; sort into a partition of N_B + 1 intervals
   (lower-closed / upper-open); recover the active affine budget
   coh = α_k · a + γ_k on each interval k by differentiating the composed
   budget DAG at the interval midpoint.
2. Continuation read (transition-aware, per cell [and interval, and branch]):
   expected continuation value W(s) and expected marginal W'(s) on the savings
   grid, integrating the regime transition, stochastic nodes, and co-state
   laws; jump-aware near published cliffs (Section 3.6).
3. Per-interval EGM: invert u'_c(c_j) = β W'(s_j) at every savings node
   (analytic or safeguarded-numeric inverse); recover coh_j = c_j + s_j and the
   endogenous liquid a_j by inverting the interval's affine budget (with linear
   extension past the grid range); marginal value of liquid by the envelope
   theorem: μ_j = α_k(a_j) · u'_c(c_j).
4. Candidate generation and masking (Section 3.3): Euler interior paths,
   corners, boundary-targeting and save-to-cliff candidates, savings-node
   floor; NaN-dead masking of every candidate inconsistent with its interval /
   predicate side; segment ids per maximal ascending, hole-free run.
5. Query-side upper envelope (Section 3.4) at the liquid query grid
   (jump-augmented under the one-sided cliff mode): exact max over bracketing
   segments; policy and marginal from the winning segment.
6. Discrete branch envelope (Section 3.5): steps 1–5 run once per discrete-
   action value on a shared parent query grid, streamed through a compiled
   batch axis; hard max (or EV1 logsum) over branches.
7. Publish: value array on the state grid; continuation carry — plain rows
   (bridged) or jump-augmented rows with duplicated abscissae carrying
   one-sided limits plus the jump locations (one-sided).
```

### 3.2 Breakpoint geometry

A boundary declared on a derived variable $z$ with threshold $\bar z$ becomes a
breakpoint on the liquid axis at the preimage $a^\ast = (\bar z - \gamma)/\alpha$,
where the slope $\alpha$ and intercept $\gamma$ of the affine map $z(a)$ are read by
automatic differentiation of the composed model DAG at $a = 0$, with the cell's other
states and the parameters bound. Monotonicity of $z$ in $a$ (nonzero, single-signed
slope) makes the preimage unique. A boundary whose variable has (near-)zero slope in
the cell — the threshold is never crossed within reach — has a non-finite preimage; it
is clamped to a margin just outside the grid span, collapsing to an empty edge
interval rather than poisoning a live interval's affine segment.

The preimages of all declared sources merge into one sorted partition. Intervals are
**lower-closed / upper-open**: a liquid value exactly on a breakpoint belongs to the
interval above it, which implements equality ownership by the upper side; the binary
case-piece path additionally supports the `when`-owner convention through a
strict/non-strict mask split (Section 3.3). Jump breakpoints are located within the
sorted order — statically when one variable fixes the order, or per cell from the
sorting permutation when breakpoints declared on several variables interleave
differently across cells. Per-interval budget coefficients $(\alpha_k, \gamma_k)$ are
recovered by evaluating the budget DAG's value and derivative at each interval's
midpoint, a point strictly inside the interval where the active affine segment is
unambiguous.

### 3.3 Candidate generation and NaN-dead masking

Within an interval the budget is affine and the problem smooth, so EGM applies; but
the merged problem is non-concave (kinked or jumped continuations, interval seams), so
the Euler path alone is *not* a complete candidate system. NB-EGM generates, per case
or interval:

- **Euler interior path.** One candidate per savings node from the Euler inversion.
  A kinked continuation can fold the endogenous grid back on itself (the secondary
  kinks of Iskhakov et al. 2017); the path is split into maximal strictly-ascending,
  hole-free runs, each carrying its own segment id, so the envelope never links
  across a fold. Nodes where the expected marginal continuation falls below a small
  tolerance are dropped as degenerate (the inversion sends consumption to infinity);
  corner and floor candidates cover them.
- **Borrowing corner ($s = 0$).** Consume all cash-on-hand at each liquid grid
  point, earning the continuation at zero savings. Because a jumped continuation is
  non-concave, this corner can dominate the Euler path even where an Euler segment
  brackets the query, so it competes over the *whole* grid — not only in the
  constrained tail a concave-EGM shortcut would assume.
- **Saturation corner ($s = s_{\max}$).** With a finite savings grid, the
  finite-domain optimum at high cash-on-hand can be to save the grid maximum, above
  where the recovered endogenous grid reaches. Feasible where residual consumption
  is positive.
- **Savings-node floor.** In the savings-space kernels, a dense family: at every
  liquid grid point and every savings node, the zero-width candidate "consume
  $\mathrm{coh}(a) - s_i$, earn $W_k(s_i)$". This is a Bellman floor on the savings
  grid at every query point: an optimum at a continuation kink *between* Euler roots
  (where the inversion has no root) is still represented. Each pair is a zero-width
  segment the envelope brackets exactly at its own abscissa. The floor bounds the
  merged solution below by the dense-grid solution on the savings grid, so envelope
  or masking imperfections can cost at most interpolation-level accuracy relative to
  a savings-grid search.
- **Boundary-targeting candidates.** Per jump cliff in the *continuation*: save
  exactly enough to land next-period liquid one floating-point step inside the
  cliff's eligible side, paired with that side's one-sided continuation limit. The
  target is the open one-sided limit (one ulp inside), so the reported policy and
  the value it is paired with are mutually consistent — a maximum at a
  representable point, not a supremum dressed as a maximum. One-sided limits are
  extrapolated from the two nearest same-side nodes, with the stencil bounded to the
  segment between adjacent cliffs so close cliffs never bridge.
- **Save-to-cliff candidates** (one-sided cliff mode, self-referencing regimes).
  When the regime reads its own carry and publishes jump topology, the cell inverts
  its affine savings-to-next-liquid law at each published jump preimage and adds two
  savings targets one relative margin inside each side of every cliff, evaluated
  through the full blended continuation (all targets, all shocks). Targets outside
  the savings span or under a non-increasing liquid law are NaN-dead. These
  candidates capture the generic optimum "save to just inside the cliff", which
  falls strictly between savings nodes.
- **Hard-constraint floors.** A slope-zero interval pins cash-on-hand at the floor;
  the value there is a single-point Bellman maximum over consumption, evaluated by a
  dense consumption search (robust to a recurring flat continuation where the Euler
  inversion is degenerate). The floor's interior candidates are pulled onto its
  crossing breakpoint and a constant-value corner segment spans the interval.
  Flatness is classified by the *relative* span of cash-on-hand across the grid.

**NaN-dead masking.** A candidate is valid only if its recovered state is consistent
with its case: inside the interval (lower-closed/upper-open), on the correct predicate
side under the declared equality owner (strict/non-strict comparison split in the
binary path), non-degenerate, and feasible ($c > 0$, $s \ge 0$). Invalid candidates
are set to NaN in *every* channel — abscissa, value, policy, marginal. The convention
matters: a finite abscissa with $-\infty$ value is not absent for a link-based
envelope — it still forms a live link and can emit NaN through $0 \cdot (-\infty)$
in linear interpolation. $-\infty$ with zero marginal is reserved for *published*
infeasible-choice rows after the envelope, where downstream probability-weighted
aggregation requires a finite-probability convention.

**Segment ids.** After masking, ids label maximal strictly-ascending, hole-free runs;
dead candidates carry NaN ids. Candidate families and cases are offset by static
strides so ids never collide across the merge.

### 3.4 The query-side upper envelope

All candidates from all cases merge in a single exact envelope evaluated directly at
the query abscissae, without materializing a refined row. A *link* is a
consecutive-candidate pair sharing a finite segment id with both endpoints live. For
query $q$:

$$
V(q) \;=\; \max_{\ell \,:\, q \in [\underline{a}_\ell,\, \overline{a}_\ell]}
\; I_\ell[V](q),
$$

the maximum of the linear interpolants of all bracketing live links; the policy and
marginal are the winning link's. A query no live link brackets yields NaN in all
channels (surfaced by runtime diagnostics rather than silently patched). Ties within
an absolute tolerance are broken *right-continuously*: among near-maximal bracketing
links, the one with the largest value slope — higher just to the right of $q$ —
carries the policy and marginal.

The evaluation is a fixed-shape bracket-and-reduce over $(N_Q, N_{\mathrm{seg}})$: no
sequential scan, branch-parallel, reduction-heavy — the shape an accelerator runs
fastest. When the dense matrix is itself the memory wall, a two-pass blocked scan
(running max, then max-slope-among-near-max against the fixed envelope value) peaks at
$(N_Q, \text{block})$ and returns the identical result (Theorem 4).

### 3.5 The discrete branch envelope

A single discrete action $d$ with $N_D$ values is handled by an outer envelope over
per-branch continuous solves. Five channels may depend on the branch, and all are
routed per branch: the co-state laws of motion, the off-budget liquid law, period
utility, the regime-transition probabilities, and the schedule variable (each branch
then owns its own breakpoint partition and interval midpoints). The branch axis is
executed through `jax.lax.map` with a batch-size knob: the branch body compiles
*once* and is streamed in blocks (the whole axis in one vectorized pass by default),
so per-branch EGM intermediates never all sit in flight and the axis is never
Python-unrolled.

The discrete choice is the hard maximum over branch value rows,

$$
V(q) = \max_{d} V_d(q), \qquad
\mu(q) = \mu_{d^\ast(q)}(q), \quad d^\ast(q) = \arg\max_d V_d(q),
$$

with the winning branch's marginal by Danskin's theorem; the `argmax` convention
selects the *lowest-index* tied branch, a well-defined subgradient (not the
set-valued derivative) at a tie. An EV1 taste-shock scale replaces the hard max by
the scaled logsum with the probability-weighted branch marginal.

**Shared-parent-grid invariant.** The pointwise max over branches is valid only
because every branch's row is evaluated on the same parent query grid.
Branch-specific inputs — continuation slices, per-branch breakpoints, cliff
candidates — may change branch values and candidate sets, never the parent abscissae.
The one violation (an action moving a *published* jump preimage per branch under the
one-sided cliff mode) is refused at model build with an actionable error; so are
multiple discrete actions and an action-dependent discount factor.

### 3.6 Cross-regime continuation and the two cliff-read modes

The continuation is read through a transition-aware reader: per savings node it
integrates the regime-transition probabilities, the stochastic next-states (over
their joint node mesh, optionally in blocks), the co-state transitions, and each
target's carry interpolation, returning the expected value $W(s)$ and expected
marginal $W'(s)$ in savings space (the marginal already carries the gross-return
factor $\partial R/\partial s$, so the Euler inversion reads it directly).
Unreachable targets are zeroed on results, never by multiplying into a possibly
non-finite value. The carry's value read is upgraded from linear to monotone cubic
Hermite using the marginal row as exact node slopes (Fritsch–Carlson limited);
brackets with non-finite data fall back to the linear rule.

A child regime's value cliffs pose the representation problem: a plain aggregate
carry (values and marginals on a coarse grid) lets the parent's interpolation bridge
across a cliff, misstating the continuation in the cliff cell (Theorem 6 pins the
sign for linear reads; under the implemented Hermite read the sign is empirical).
NB-EGM offers two modes:

- **`one_sided` (exact cliff reads).** Each carry row holds every jump preimage as a
  *duplicated abscissa* carrying the exact one-sided value and marginal limits: the
  period solve is evaluated at a point one ulp inside each side of every jump, and
  the two results are published at the same abscissa (the jump location). The
  interpolation contract at a duplicated abscissa is one-sided by construction —
  queries strictly below interpolate toward the left value, queries at or above use
  the right value, and the zero-width bracket is never used as a divisor. The carry
  additionally publishes the jump locations, which feed the parent's save-to-cliff
  candidates. This mode gates off a stochastic-dimension pre-folding optimization on
  dimensions that move any jump preimage (their rows' duplicated abscissae differ),
  so it trades runtime for cliff fidelity.
- **`bridged` (fast approximate reads).** Plain liquid-grid rows with no
  breakpoints; the parent's interpolation may average across a cliff, exactly like
  any finite-grid solver reading the same rows. The stochastic fold stays available.
  The *within-period* case solve is jump-aware in both modes — masked cases,
  equality ownership, boundary targeting — the mode selects only what the carry
  publishes for parents. Bridged is the intended mode for solves whose consumer
  tolerates finite-grid cliff error (estimation inner loops), polished afterwards
  under `one_sided`.

**Per-interval continuation.** When the continuation reads the *current* liquid
state — a carry target's next-state law reads it (a transfer or pension adjustment
switched at a declared cliff) or the regime-transition probabilities read it
(survival switched at an asset test) — the continuation is no longer a single
function of savings. NB-EGM then binds the liquid state to each interval's
representative node and evaluates one continuation row per interval, solving each
interval as its own case against its own row. This is exact if and only if the
liquid-dependence is *piecewise-constant on the declared partition*; a build-time
probe differentiates every liquid-reading law at finitely many fixed liquid values
(absolute sample points, not per-interval representatives) — sweeping each
integer-coded argument over its grid's actual codes one at a time — and rejects any
detected smooth (affine or curved) dependence with an explanation.
The probe is a **finite diagnostic, not a certificate**: by the same argument as
Theorem 3, no finite set of black-box evaluations can prove constancy (or
affinity) of an arbitrary smooth law — a dependence whose derivative vanishes at
every probed point passes undetected. Exactness therefore rests on the hypothesis
being *true*: it holds by construction for structure declared through the
breakpoint metadata, is screened (not proven) by the probe for everything else,
and is otherwise discharged only by independent validation. The probe evaluates
the model's DAG on synthetic inputs — scalar fills first, retried with unit-length
array fills for DAGs holding plain array-valued schedule parameters. A DAG it
cannot evaluate either way (structured parameters, mixed scalar/array signatures
under strict runtime type contracts) leaves the precondition machine-unverified,
and the build then **refuses by default**. The model author may instead assert the
precondition explicitly (`probe_failure="assume_declared"`, emitting a warning).
Such runs are not probe-screened at all: every exactness claim is conditional on
the asserted precondition actually holding, and the assertion must be discharged
by an independent reference — the production application takes this path, because
its budget mixes scalar- and array-annotated tax and threshold parameters, and
discharges the burden with the full-model brute-agreement gates of Section 6. The
same fail-closed-or-assert semantics and the same finite-diagnostic status govern
the affine-budget probe. This trigger
is precisely the mechanism of Theorem 2: case conditioning removes the
current-state dependence that would otherwise force asset-row replication.

### 3.7 The smoothness gate

Declared breakpoints are only as good as the smoothness of what lies between them. At
model build, NB-EGM runs two complementary validators over the user's economic
functions: an AST gate rejecting Python branching and hidden comparisons in smooth
pieces (boundary predicates may compare — that is their job), and a JAXPR gate
tracing each smooth piece and rejecting piecewise primitives (`select_n`,
comparisons, …) hidden inside called helpers the AST cannot see. A reviewed numerical
helper (a `clip`/`max`/`abs` guard with a stated domain argument) is exempted by an
explicit attestation decorator. Solver infrastructure — continuation interpolation,
grid location, the envelope, the EGM kernel — is outside the gate's scope; a global
primitive ban would reject every realistic model, because continuation reads
necessarily interpolate.

---

## 4 Correctness

Throughout, "exact" means exact for the discretized target of Section 2.3: the same
finite candidate records under the same declared convention $\mathcal{K}_h$
(interpolation rule, endpoint ownership, tie-breaking). Comparisons to a brute-force
solver on a *different* candidate grid or interpolation rule are approximate by
construction and belong to validation (Section 6), not to these results.

**Theorem 1 (exactness of the case-enumeration envelope).** *Fix a discretized target
with convention $\mathcal{K}_h$, and let $\mathcal{S}_h$ be its finite set of
convention-level candidate records (links and one-sided boundary records with their
eligibility). Suppose (i) the case assignments cover the candidate set,
$\mathcal{S}_h = \bigcup_\sigma \mathcal{S}_{\sigma,h}$; (ii) for each $\sigma$ the
generator produces exactly the records $\mathcal{S}_{\sigma,h}$, with invalid
pre-envelope candidates NaN-dead in all channels and boundary ownership as declared;
and (iii) the envelope takes the maximum over all eligible generated records, with
policy and marginal selected from the winning record under $\mathcal{K}_h$'s tie
rule. Then the returned value and policy equal
$V_h(q) = \max_{s \in \mathcal{S}_h(q)} I_s[V](q)$ and the associated selection, for
every query $q$.*

*Proof.* By the cover, $\max_{s\in\mathcal S_h(q)} I_s[V](q) = \max_\sigma
\max_{s\in\mathcal S_{\sigma,h}(q)} I_s[V](q)$. By (ii) the generated records for
$\sigma$ are exactly $\mathcal S_{\sigma,h}$ under $\mathcal K_h$; NaN-dead masking
removes exactly the ineligible ones from the link structure. By (iii) the maximum over
all eligible generated records equals the right-hand side, and the tie rule pins a
unique selection. $\square$

*Remark (where the content lives).* Hypothesis (ii) — candidate *completeness* — is
the substantive burden, and it is discharged per kernel family, not generically: the
Euler path covers interior optima on each smooth piece; fold-splitting covers
secondary kinks; the corners cover binding constraints; the boundary-targeting and
save-to-cliff candidates cover the one-sided optima a jumped continuation creates;
and the savings-node floor bounds the result below by the dense savings-grid
solution at every query, so any optimum representable on the savings grid is never
lost. Theorem 1 converts "the candidate system is complete for the convention" into
"the answer is exact for the convention"; it does not prove the former.

**Theorem 2 (restored amortization; necessity of replication without it).**
*(a) Suppose that after fixing a case assignment $\sigma$ the Euler equation has the
form $u_c'(c) = \Phi_\sigma(s)$, independent of current resources $m$. Then a single
endogenous-grid correspondence per $\sigma$ — the Euler path plus the constrained and
boundary candidates of Section 3.3, split into eligible segments — serves all
current-resource queries under the declared convention. (b) Conversely, if the RHS is
$\Phi(s; m)$ with $\Phi(s;m_1) \neq \Phi(s;m_2)$ for two grid states at some savings
node, then no single-valued shared curve $c(s)$ can satisfy the Euler equation at
both: $u_c'(c(s)) = \Phi(s;m_1) = \Phi(s;m_2)$ is a contradiction. Exactness then
requires per-state replication (asset-row mode) or another representation that
reinstates case structure.*

*Proof.* (a) For fixed $\sigma$, $c_\sigma(s) = (u_c')^{-1}(\Phi_\sigma(s))$ does not
read $m$; the correspondence $s \mapsto (m_\sigma(s), c_\sigma(s))$, masked and
segmented, is queried at every $m$, and the non-Euler candidates are already indexed
by the query point. (b) Immediate: one number cannot equal two distinct numbers.
$\square$

*Remark.* Part (b) rules out only single-valued shared curves. NB-EGM's per-interval
continuation path (Section 3.6) is exactly the loophole: when $m$ enters $\Phi$ only
through a piecewise-constant term on the declared partition, one curve *per interval*
— $N_B + 1$ curves, not $N_X$ — restores exactness; a build-time probe screens for
(but, per Theorem 3, cannot certify the absence of) the smooth dependence that
would void it.

**Theorem 3 (declared breakpoints are necessary).** *No deterministic solver that
receives only finitely many (possibly adaptively chosen) black-box evaluations of a
piecewise function can identify all threshold locations exactly.*

*Proof.* Run the solver against $f_1(x) = \mathbf 1\{x \ge 1/2\}$ on $[0,1]$ and let
$S = \{x_1, \dots, x_n\}$ be the realized (possibly adaptive) query set. Since $S$
is finite, choose $\delta > 0$ with
$S \cap \bigl((1/2 - \delta, 1/2) \cup (1/2, 1/2 + \delta)\bigr) = \emptyset$. If
$1/2 \notin S$, set $f_2(x) = \mathbf 1\{x \ge 1/2 + \delta\}$; every $x \in S$ with
$x \ge 1/2$ satisfies $x \ge 1/2 + \delta$ and every $x \in S$ with $x < 1/2$
satisfies $x < 1/2 - \delta$, so $f_2 = f_1$ on $S$. If $1/2 \in S$, shift the other
way: $f_2(x) = \mathbf 1\{x \ge 1/2 - \delta\}$ agrees with $f_1$ at $1/2$ (both
$1$) and, by the same disjointness, on all of $S$. In either case the solver is
deterministic and its queries depend only on previous responses, so its interaction
with $f_2$ realizes the same query set and the same output, while the thresholds
differ. $\square$

*Corollary.* Exact treatment of institutional thresholds requires exposed boundary
metadata. NB-EGM's decorators are the mechanism; a numerical cliff detector can only
be a diagnostic.

**Theorem 4 (exact chunkability of the query-side envelope).** *Let the live links
be partitioned into blocks $\mathcal{S} = \cup_{b=1}^{B} \mathcal{S}_b$. For every
query $q$,*

$$
\max_{\ell \in \mathcal{S}(q)} I_\ell[V](q)
= \max_{b} \; \max_{\ell \in \mathcal{S}_b(q)} I_\ell[V](q),
$$

*so segment blocks can be folded with a running maximum without changing the value.
If the tie rule is block-order-invariant, the policy and marginal selections also
coincide with the dense evaluation.*

*Proof.* Max over a finite set is associative and commutative, and bracket
eligibility is evaluated per link before the reduction, so blocking does not alter
the eligible set. For the selection channels, a two-pass evaluation — fold the value
maximum first, then re-scan all blocks for the tie-rule winner against the now-fixed
envelope value — reproduces the dense selection whenever the tie rule (here:
max slope among near-maximal links, earliest on exact slope ties) does not depend on
block boundaries; the two-pass construction makes it not. $\square$

The same associativity underlies streaming over ride-along cells, discrete branches,
intervals, and stochastic nodes: every batch-size knob of Section 7 changes peak
memory, never the result (up to floating-point reassociation between compiled
reductions).

**Theorem 5 (continuous interpolants cannot represent a cliff).** *Let $V$ have a
jump discontinuity at $x_0$ and let $J$ be a closed interval containing $x_0$ in its
interior. No sequence of continuous functions on $J$ converges uniformly to $V$ on
$J$.*

*Proof.* A uniform limit of continuous functions on $J$ is continuous on $J$; $V$ is
not. $\square$

*Implication.* Node insertion alone cannot fix a notch: without side ownership, any
single-valued continuous row bridges it. Duplicated abscissae carrying one-sided
limits (the `one_sided` carry) represent the cliff exactly at the level of the read's
piecewise-linear-with-jumps interpolant.

**Theorem 6 (aggregate-max interpolation bridges upward under linear reads).** *Let $I$ be a positive
linear interpolation operator (nonnegative weights on shared nodes) and $V_b$
branch-specific values sampled at the same nodes. Then for every query $q$,*

$$
\max_b I[V_b](q) \;\le\; I\big[\max_b V_b\big](q).
$$

*Proof.* $V_b(x_i) \le \max_j V_j(x_i)$ at every node; positivity of $I$ preserves
the pointwise order, so $I[V_b](q) \le I[\max_j V_j](q)$ for each $b$; take the max
over $b$. $\square$

*Implications.* (i) Under a positive linear read, interpolating already-maximized
values can only *overstate* the envelope near switches — the bridged mode's error at
the cliff cell is upward. (ii) The inequality is proven **only** for positive linear
$I$. The implementation's value read is monotone-limited cubic Hermite, which is not
a linear operator of the node values, and the sign result does *not* transfer: with
the limiter active, a three-node, two-branch monotone configuration exists where the
Hermite read of the aggregate-max row *understates* the maximum of the per-branch
Hermite reads (the slope assignments of the maximized row differ from either
branch's, and the limited correction can pull the aggregate read below both). The
bridged mode's signed cliff-cell error under the Hermite read is therefore an
empirical quantity to be measured, not a guaranteed direction; only side-aware
(one-sided) representation removes the error by construction.

**Theorem 7 (conditional dominance over asset-row EGM).** *Let asset-row EGM and
NB-EGM have per-period, per-cell costs*

$$
T_{\mathrm{row}} = N_X\, C_E + C_{\mathrm{pub}}^{\mathrm{row}}
+ C_{\mathrm{compile}}^{\mathrm{row}} + C_{\mathrm{mem}}^{\mathrm{row}},
\qquad
T_{\mathrm{NB}} = (N_B{+}1)\, C_E + N_{\mathrm{bd}} C_{\mathrm{bd}}
+ N_Q N_{\mathrm{seg}} C_I + C_{\mathrm{pub}}^{\mathrm{topo}}
+ C_{\mathrm{compile}}^{\mathrm{NB}} + C_{\mathrm{mem}}^{\mathrm{NB}},
$$

*where $C_E$ is the cost of one EGM correspondence (scaling with $N_S$ and the
continuation read), $N_{\mathrm{bd}} C_{\mathrm{bd}}$ the boundary-candidate work,
$N_Q N_{\mathrm{seg}} C_I$ the envelope, and the publication, compile, and memory
terms are as labeled. NB-EGM is faster than asset-row EGM if and only if
$T_{\mathrm{NB}} < T_{\mathrm{row}}$. The heuristic $N_B + 1 < N_X$ is sufficient
only under side conditions: comparable $C_E$ across the two methods, $N_{\mathrm{seg}}$
bounded so the envelope term is lower order, and publication/compile/memory terms
that do not dominate.*

*Proof.* Direct comparison of the cost accountings. The heuristic fails without the
side conditions because $N_{\mathrm{seg}}$ scales with cases × folds × boundary
splits, and with $N_Q \sim N_X$ the envelope term is $O(N_X N_{\mathrm{seg}})$ —
potentially the same order as asset-row's $N_X C_E$. $\square$

*Remark (accounting versus roofline).* The additive decompositions are work
accountings; realized wall time on a device is roofline-shaped,
$T \ge \max\{W / (\eta_F F),\ R / (\eta_B B),\ D\lambda\}$ (Section 5.1). The additive
form upper-bounds relative comparisons when both methods sit on the same side of the
roofline; the decisive experiments of Section 6 measure wall time directly.

### 4.1 Remarks and unproven assertions

The design work behind NB-EGM also asserts a *multidimensional necessity* claim: when
the optimum is characterized by a genuinely coupled system of $d_y > 1$ first-order
conditions and no one-dimensional nesting exists, a one-dimensional endogenous-grid
correspondence cannot represent the solution, and one must solve a multidimensional
inverse system, search over $d_y - 1$ outer dimensions, or revert to direct
optimization. We record this as a structural remark rather than a theorem: making it
precise requires formalizing "represent", and we have not done so. Likewise, we make
no claim that the discretized NB-EGM solution converges to the continuous Bellman
solution under mesh refinement; the jump-augmented representation makes the natural
conjecture plausible away from bifurcation points of the argmax, but it is open here.

---

## 5 Cost model and solver choice

### 5.1 Hardware model

A device is described by peak throughput $F$, bandwidth $B$, memory $M$, launch/
dependency latency $\lambda$, and utilization factors $\eta_F, \eta_B \in (0,1]$.
An implementation with arithmetic work $W$, bytes moved $R$, resident memory $m$, and
dependency depth $D$ obeys

$$
T_{\mathrm{wall}} \;\ge\; \max\!\Big\{ \frac{W}{\eta_F F},\ \frac{R}{\eta_B B},\
D \lambda \Big\}, \qquad m \le M .
$$

An algorithm with lower nominal work loses whenever it materializes larger
transients, carries longer sequential dependencies, achieves lower utilization, or
compiles many static-shape variants. Brute force wins on GPUs precisely because it is
a dense map-reduce: for a finite action partition $A = \cup_k A_k$,
$\max_{a \in A} Q(x,a) = \max_k \max_{a \in A_k} Q(x,a)$, so it streams action chunks
with resident memory proportional to the largest chunk.

### 5.2 The three-way comparison

Per period, per regime, with $N_{\mathrm{cell}}$ ride-along cells and $N_D$ branches:

- **Brute force:** $W_B \approx N_{\mathrm{cell}} N_X N_D N_A\, C_Q$, where $C_Q$
  includes the continuation read per candidate. Dense, chunkable, no topology.
- **Asset-row EGM:** $W_{\mathrm{row}} \approx N_{\mathrm{cell}} N_X N_D\, C_E$ —
  one Euler correspondence per current node; exact under current-state dependence
  but with EGM's amortization lost.
- **NB-EGM:** $W_{\mathrm{NB}} \approx N_{\mathrm{cell}} N_D \big[ K\, C_E +
  N_{\mathrm{bd}} C_{\mathrm{bd}} + N_Q N_{\mathrm{seg}} C_I \big]$, with $K = 1$
  when the continuation is savings-only and $K = N_B + 1$ under the per-interval
  path; $C_E$ itself is dominated by $N_S$ continuation reads, each integrating
  $N_Z$ stochastic nodes over $N_R$ targets.

The qualitative rules that fall out:

1. **NB-EGM beats brute force** when $K (N_S/N_A)\, (C_E'/C_Q) +
   N_Q N_{\mathrm{seg}} C_I / (N_X N_A C_Q) < 1$ at matched accuracy — typically:
   few declared breakpoints, a savings grid much coarser than the action grid brute
   needs to resolve cliff-adjacent optima, and bounded segment counts. The decisive
   accuracy asymmetry is at cliffs: brute needs $N_A$ dense enough to place a
   candidate within tolerance of every save-to-cliff optimum in every cell, while
   NB-EGM places those candidates *exactly* at machine-precision one-sided targets
   regardless of $N_S$.
2. **Brute force beats NB-EGM** when case structure is rich ($N_{\mathrm{seg}}$
   large), when the one-sided mode's fold gating forces per-node continuation loops
   the fold would have removed, when the compile-shape multiplicity (per-period
   target splits, branch axes) dominates, or simply when $N_A$ needed for the
   application's accuracy is modest — the dense map-reduce's utilization advantage
   then buries the work advantage.
3. **Asset-row is the exact fallback** when current-state dependence survives all
   declared conditioning — i.e., when the build-time probe of Section 3.6 rejects
   the model. NB-EGM dominates it if and only if Theorem 7's inequality holds; the
   headline case $K = N_B + 1 \ll N_X$ is the design target.
4. **Asymptotics favor EGM in action resolution.** If matched accuracy
   $\varepsilon$ requires $N_A(\varepsilon) = \Theta(\varepsilon^{-1/p_A})$ brute
   candidates but only $N_S(\varepsilon) = \Theta(\varepsilon^{-1/p_S})$ savings
   nodes with $1/p_A > 1/p_S$, and the envelope and boundary terms do not add a term
   of equal or higher order, EGM-family methods eventually dominate as
   $\varepsilon \to 0$. On fixed production grids this asymptotic regime may or may
   not have set in; only the matched-accuracy benchmark decides.

The framing is deliberately conditional. The correct default in a general-purpose
framework remains brute force until a structure-specific solver demonstrates a
matched-accuracy advantage on the target hardware; NB-EGM's role is to make that
advantage achievable — and exactly quantifiable — for the institutional-breakpoint
class.

---

## 6 Validation methodology

Correctness for the discretized target (Section 4) is conditional on candidate
completeness; validation supplies the evidence per model class.

**Three evidence layers, in decreasing strength.** Validation separates what each
layer can and cannot certify:

1. **Convention-matched host oracles (exact, selection layer).** The envelope and
   selection logic is tested against an independent host-side NumPy oracle that
   evaluates the *exact* pointwise upper envelope of the candidate polylines under
   the declared topology contract — no concavity, monotonicity, or scan-window
   assumption, folds evaluated as true polylines, malformed inputs rejected. This
   layer certifies the selection rule (endpoint ownership, tie behavior, fold
   handling) at the candidate-record level, which a finite brute grid cannot: a
   dense brute oracle can miss off-grid save-to-cliff actions and can share the
   solver's own interpolation convention error in a narrow cliff band. Candidate
   *completeness* per kernel family (Euler records, corners, point candidates,
   save-to-cliff records) is covered per family by targeted record-level tests and
   remains, per Theorem 1's hypothesis, an evidence claim rather than a generic
   proof.

2. **Dense-brute agreement on toys (numerical regression evidence).** For each
   kernel family — the binary Medicaid asset-test toy, multi-kink schedules,
   recurring multi-cliff schedules, mixed kink-and-jump schedules, floors, the
   discrete-action envelope, ride-along co-states, liquid-reading transitions — a
   brute-force solver solves the same discretized problem over a *dense*
   savings/consumption grid. Agreement is asserted in concrete values with explicit
   tolerances, with adversarial queries at thresholds and one ulp on either side.
   These tests validate *numerical agreement on the toy*; they are regression and
   smoke evidence, not an exact certificate of envelope selection — that is layer
   1's job.

**Property tests of the streaming knobs.** Every batching knob (Section 7) is tested
for bit-level (or reassociation-level) equality against its dense counterpart:
blocked envelope versus dense envelope, cell blocks versus full vmap, branch blocks
versus a single pass, interval batches, stochastic-node batches.

**Full-model gates (split cliff/non-cliff scoring).** On the production
application — a structural retirement model of the Affordable Care Act with 18
regimes (working/retired × insurance and eligibility strata), Medicaid asset and
income tests, subsidy brackets, and survival/eligibility transitions switched at
declared thresholds — NB-EGM is gated against the framework's brute-force solver on
the production grids. A single aggregate tail quantile cannot distinguish harmless
finite-grid cliff convention error from wrong candidate selection, because the tail
concentrates exactly at cliff preimages (Theorem 5: no continuous interpolant
uniformly approximates a jump on a cell containing it). The gate therefore scores
two regions separately, with the cliff band defined as the grid cells adjacent to
each published jump abscissa:

- **outside the cliff band**, p99 *and* maximum relative disagreement must sit below
  a tight interpolation-error threshold (with an absolute floor for near-zero
  values), and no unexpected NaNs may appear;
- **inside the cliff band**, disagreement is scored under the declared read
  convention — signed tail quantiles, maxima, and explicit counts of cells breaching
  hard caps are reported, and under `one_sided` the disagreement must collapse to
  the finite-grid comparison error of the brute reference itself. Under `bridged`
  the signed error is *measured*, not assumed directional (Theorem 6's sign holds
  for linear reads; the implemented Hermite read voids the guarantee).

Simulation-moment deltas complement the state-space gates. Euler residuals are
reported but are not the acceptance criterion — they are blind to the corner and
boundary candidates that carry exactly the economics of interest.

**Diagnostics that must fail loudly.** A query no live segment brackets publishes
NaN and is surfaced by the runtime NaN gate rather than patched; the smoothness gate
(Section 3.7) and the piecewise-constancy probe (Section 3.6) reject at build time
the model classes they detect — a finite screen for, not a proof of, the kernels'
preconditions (Section 3.6).

---

## 7 Performance levers

The chunking levers below are exact — by the associativity results of Theorem 4
they change peak memory and schedule, never the solution (up to floating-point
reassociation). One lever is *not* purely a schedule change: stochastic-dimension
pre-folding under the Hermite value read commutes with the read only where the slope
limiter is inactive; where the limiter binds (near jumps), the folded expectation is
a different valid interpolant of the same data, deviating at interpolation-error
order. The fold is accordingly gated off on jump-moving dimensions under
`one_sided`, and its residual effect under `bridged` is part of that mode's measured
(not guaranteed) approximation error.

| Knob | Axis streamed | Peak-memory term bounded |
|---|---|---|
| `cell_block_size` | ride-along cells (both cores) | per-cell candidate and continuation buffers |
| `branch_batch_size` | discrete-action branches (`lax.map`, body compiled once) | per-branch EGM intermediates and continuation rows |
| `interval_batch_size` | per-interval continuation reads | per-interval continuation DAG buffers |
| `stochastic_node_batch_size` | child stochastic-node mesh | joint quadrature intermediates |
| `envelope_segment_block_size` | envelope segment blocks (two-pass scan) | the $(N_Q, N_{\mathrm{seg}})$ bracket matrix |

Two structural design choices matter as much as the knobs. First, the period step is
split into two independently compiled cores: a *continuation core* carrying the heavy
transition fan-out (regime blend, shocks, child interpolation) and an *envelope core*
carrying the EGM and envelope math; neither compiled program contains the other's
memory profile, and the cores are deduplicated across periods by their reachable
target split. Second, the branch axis is never Python-unrolled: `lax.map` with a
batch size compiles the branch body once, so adding discrete-action values does not
grow the program.

**The speed-versus-fidelity dial.** `jump_read="bridged"` is the warm-start and
screening mode: plain carry rows, stochastic-dimension pre-folding available (the
child's shock expectation folds into the carry once per cell instead of looping
nodes per savings query), cliff-cell error at finite-grid interpolation level with
sign properties as discussed under Theorem 6 (guaranteed upward only for linear
reads; empirical under the implemented Hermite read).
`jump_read="one_sided"` is the exact-convention mode: duplicated one-sided abscissae,
published jump locations, save-to-cliff candidates, fold gated off on jump-moving
dimensions. **Estimation protocol.** Because the bridged and one-sided solves define
*different objective surfaces* near institutional cliffs, an optimizer run under
`bridged` can converge to a point that is not the one-sided optimum, and evaluating
the one-sided objective once at the bridged optimum cannot detect this. `bridged` is
therefore a warm-start and screening mode only: final estimates require either
re-optimization under `one_sided` from the bridged optimum until the one-sided
first-order/trust-region criteria are met, or an empirical objective-surface
comparison over the relevant parameter region showing the two modes' minimizers
coincide within the reported precision. One caveat is inherited by *both* modes:
wherever the value read's monotone slope limiter binds (near jumps), the folded
expectation is a different valid interpolant of the same data than the node-looped
one, deviating at interpolation-error order — an engineering trade documented, not a
silent approximation.

---

## 8 Conclusion

NB-EGM turns institutional kinks and cliffs from a reason to abandon the endogenous-
grid method into declared structure the method exploits: thresholds become interval
partitions, one-sided limits become duplicated abscissae, and the non-concavities
they induce become an explicit, provenance-tagged candidate system merged by an exact
streamable envelope. The correctness results are honest about their shape — exactness
for the declared discretized target, conditional on candidate completeness that is
argued per kernel family and evidenced by same-convention oracles — and the
performance claim is deliberately conditional: NB-EGM dominates asset-row EGM when
declared cases are few and segment counts bounded, dominates brute force when cliff
accuracy would otherwise force dense action grids, and loses to brute force when case
structure explodes or when a modest action grid suffices. The impossibility result
draws the design line cleanly: threshold locations must be declared, because no
finite black-box probing can recover them exactly. Open problems include a refinement
theory for the jump-augmented discretization, multi-predicate case products beyond
the one-dimensional interval partition, hard-constraint floors under ride-along
co-states, and extending the exact one-sided machinery to taste-shock-smoothed
discrete envelopes.

---

## Appendix A: Deviations from the design document

This paper describes the implementation as built (`pylcm`, solver `NBEGM`); where the
project's design document ("A Conditional Theory of Solver Choice…", v3) and the code
disagree, the paper follows the code. The load-bearing deviations:

1. **Implementation status.** The design document states NB-EGM is not implemented as
   a named solver. It now is, with four kernel families: binary case-piece,
   piecewise-affine schedule, schedule + discrete action, and ride-along co-state
   (with per-interval continuation and one-sided cliff publication).
2. **No endpoint open/closed flags.** The document specifies segments carrying
   $(e_L, e_R) \in \{\text{open}, \text{closed}\}$ and an endpoint-aware envelope.
   The implementation's envelope brackets closed intervals only; equality ownership
   is enforced upstream — strict/non-strict validity masks in the binary path,
   one-ulp-shifted duplicated abscissae in every jump-aware interpolation and in the
   published carry, and a fixed lower-closed/upper-open interval partition. This is
   the document's own sanctioned alternative ("pre-process boundary queries"),
   realized via `nextafter` arithmetic.
3. **Equality ownership in the schedule path is one-sided by convention.** The
   interval partition gives an exact-breakpoint state to the interval above it
   regardless of the declared owner; only the binary case-piece step and the
   continuation-read interpolants honor both `when` and `otherwise` owners. Models
   needing lower-side ownership of a schedule breakpoint at the *current-state*
   level are outside the current scope.
4. **No $2^K$ case products.** The document enumerates case assignments
   $\sigma \in \{0,1\}^K$. The implementation solves either one binary predicate
   (v1 case-piece scope) or a one-dimensional sorted breakpoint partition with
   $N_B + 1$ interval cases; Cartesian case products never arise.
5. **Richer candidate system.** Beyond the document's Euler path and one-sided
   boundary candidates, the implementation adds: a whole-grid borrowing corner, a
   grid-saturation corner, a dense savings-node point-candidate floor (a Bellman
   floor on the savings grid at every query), save-to-cliff candidates through the
   blended continuation, and dense-search floor optima on hard-constraint intervals.
   Theorem 1's completeness hypothesis is discharged through this larger set.
6. **The carry is not a topology payload.** Instead of segment ids, ownership flags,
   and top-two records, the one-sided carry publishes jump-augmented rows (duplicated
   abscissae with exact one-sided limits) plus the jump locations — closer to the
   document's "switch-refined aggregate grid" option. The `bridged` mode deliberately
   ships the plain aggregate carry the document calls insufficient in general, as an
   explicitly approximate fast mode with a known error sign at cliffs (Theorem 6).
7. **Value reads are monotone cubic Hermite, not linear.** Theorem 6 is proven for
   positive linear interpolation; the implementation's Hermite read (marginal row as
   exact node slopes, Fritsch–Carlson limited) is outside its literal hypothesis, and
   the paper says so (Section 4, remark to Theorem 6).
8. **Tie conventions are layered.** The continuous envelope breaks near-ties (within
   an absolute tolerance) right-continuously by maximal value slope, identically in
   the dense and blocked paths; the *discrete branch* envelope uses lowest-index
   `argmax`, returning a well-defined subgradient at ties. The document left the
   convention open and did not discuss the discrete layer.
9. **Per-interval continuation trigger and probe.** The document's Theorem 2
   dichotomy (shared curve or asset-row) is refined in code by a middle path: when a
   carry target's next-state law *or* the regime-transition probabilities read the
   current liquid state, one continuation row per declared interval is used,
   screened by a build-time probe that rejects detected smooth
   (non-piecewise-constant) liquid dependence (a finite diagnostic — Section 3.6).
10. **Numerical guards not in the document.** A degenerate-marginal tolerance drops
    Euler candidates where the continuation is flat; a relative-span test classifies
    hard-constraint (flat) intervals; degenerate breakpoint preimages are clamped
    outside the grid; the stochastic-dimension fold commutes with the read only where
    the Hermite slope limiter is inactive and is gated off on jump-moving dimensions
    under `one_sided`.
11. **Notation repairs.** The document's Theorem 1 mixes $I_s[Q_h]$ and $I_s[V]$; its
    NEGM inequality reuses $N_C$ (case count) as a brute consumption grid; its
    Theorem 5 omits that the jump must lie in the interval's interior; its Theorem 2
    calls a single-valued curve a "correspondence"; its predicate definition
    $p(x) = \mathbf 1\{g_p(x) \le 0\}$ hard-codes equality into the `when` side while
    the ownership metadata treats it as free. All are repaired in the statements
    above.


## References

- Carroll, C. D. (2006). The method of endogenous gridpoints for solving dynamic
  stochastic optimization problems. *Economics Letters*, 91(3), 312–320.
- Dobrescu, L., and Shanker, A. (2022). Fast upper-envelope scan for
  discrete-continuous dynamic programming. SSRN working paper 4181302 / CEPAR working
  paper, UNSW Sydney.
- Druedahl, J. (2021). A guide on solving non-convex consumption-saving models.
  *Computational Economics*, 58(3), 747–775. doi:10.1007/s10614-020-10045-x
- Druedahl, J., and Jørgensen, T. H. (2017). A general endogenous grid method for
  multi-dimensional models with non-convexities and constraints. *Journal of Economic
  Dynamics and Control*, 74, 87–107.
- Iskhakov, F., Jørgensen, T. H., Rust, J., and Schjerning, B. (2017). The endogenous
  grid method for discrete-continuous dynamic choice models with (or without) taste
  shocks. *Quantitative Economics*, 8(2), 317–365. doi:10.3982/QE643
