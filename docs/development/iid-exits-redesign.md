# IID integration, exits and addressed simulation

Implementation contract for
[issue 607](https://github.com/OpenSourceEconomics/pylcm/issues/607). The base is
`671129d7ebdf1e1af005a3c658e9e73f6556dac5`, the head of PR 604 when this work started.
The issue defines the public semantics; this page resolves its remaining numerical
choices and records the ordered acceptance gates.

The owner authorized choosing these contracts in the implementation conversation on
2026-10-23. They are implementation decisions, not measured accuracy claims. Passing a
standalone reference establishes its arithmetic; each engine route must also be
connected to that reference through the public model API.

## Invariants

Current IID shocks have no axis in stored values, continuation carries or policies.
Resolve feasibility, participation, action selection, exit routing and EV1 readout per
node before integrating. Never average an action index or an endogenous-grid coordinate.
Persistent states and transition-local draws retain their timing.

Every policy query returns the values of one common selected action. `Raw`,
`Participating` and `Full` are distinct query identities, and only a regime's own
occupants and publication read `Full`. Exit targets read `Participating`. Same-period
query consumers belong to the query DAG and buffer lifetime graph.

Ordinary feasibility precedes participation. Dissolution participation is weak,
formation consent is strict, zero-weight members retain vetoes, and stable global action
identifiers decide exact objective ties. Utility adjustments change both the exit payoff
and participation benchmark, once each at their specified use.

Source-read shocks are observed; target-only shocks are hidden unless declared observed.
Taste draws stay hidden. Shared innovations retain identity in base space and use the
law at each evaluated state. A formation query cannot silently use an unobserved
participation shock.

Simulation uses immutable original subject rows and typed, versioned addresses. Each
period permits one same-period move, one final-regime action and one scheduled
transition, with no transition after the terminal period. Moves retain the row's shock
overrides. Regime order, chunking and group edits cannot move an address.

The RNG address version is `pylcm.random.v1`. Its domain, immutable scalar name and
nonnegative component number form compact UTF-8 JSON with the version string. SHA-256 of
those bytes is decoded into big-endian uint32 words and folded into the typed root,
followed by the original row's high and low uint32 words and the computational period.
Domains `shock`, `state`, `regime` and `taste` remain disjoint. Scalar shock identities
survive changes to their containing group; lowered joint-normal coordinates retain the
declared scalar shock names. The regime lottery uses the reserved name `regime`, and a
destination's taste stream uses its regime name. Consumer DAG bindings do not identify
the innovation.

A transition executed from period `t` uses address period `t + 1`, while its law still
receives period-`t` states and actions. Current-period draws and entry draws use their
current computational period. Original rows span uint64 independently of enabled
floating precision; computational periods fit uint32. Padded rows repeat the final real
row. Roots are immutable, and panels record their address version, PRNG implementations
and floating precision. Categorical draws use half-open inverse-CDF cells, preserve
literal zero masses, and close the CDF plateau from its final positive category without
renormalizing the supplied probabilities.

## EGM publication and refinement

The projection grid is the declared exogenous pre-shock liquid-state grid, resolved at
the exact age for each passive-state row. Its domain is the closed declared interval.
The projected carry rejects queries outside that domain; resources-row replay keeps its
existing boundary rule. A resources grid is not a substitute for a pre-shock grid.
Refuse publication without a suitable declared projection grid.

Case R requires a dependency proof, treating resources as an opaque leaf, over the
complete action problem, including constraints, bounds, laws, aggregation and
parameters. A failed proof selects Case U; it does not grant a resources-only
assumption. Case U builds candidate rows transiently and evaluates them at common
pre-shock coordinates before any average. Plain EGM continues to reject shocks.

Use a piecewise-affine value representation between smooth neighboring projected nodes.
Preserve each transformed kink and each side of a jump as explicit records. Their
ownership follows the source candidate's one-sided ownership and stable id. Do not
interpolate across an unrepresented discontinuity: refine or refuse that cell. Pull back
all boundaries with their owning sides; nonmonotone resource maps require all preimages.
Certified boundary inversion requires supported exact topology or a refusal; balanced
numerical inversion records bracket uncertainty. Preserve exact-node queries before
applying interpolation.

The economic marginal is a separate, explicitly labelled channel. On smooth cells it
includes derivatives of weights, physical-law transforms and resources. In particular,
`dW/db = sum(dw/db * V + w * dV/db)`. The derivative of the projected value interpolant
is a different statistic and cannot substitute in an Euler equation. At a kink use the
selected candidate's one-sided economic marginal; at a jump, label the owning-side
marginal and retain jump metadata. Reject an unresolved owning side. A zero primal
weight can have a nonzero derivative, so differentiate weights before dropping zero-mass
nodes. Nonlinear CE channels use their own stable typed sufficient statistic and full
chain rule.

Refinement bisects failed smooth cells and inserts known transformed breakpoints.
Compare the coarse representation with quarter, midpoint and three-quarter queries from
the solver's candidate construction, exact endpoints and adjacent representable points
at boundaries, then repeat with the refined representation. Stop only when every channel
meets its own budget and topology is represented; exhausted capacity, unresolved
ownership or an unbounded domain fails loudly. Refining a savings grid and refining this
projection are distinct operations.

Define normalized discrepancy as `abs(actual - reference) / max(scale, abs(reference))`.
Use unit scale for fixture value/marginal, action-domain span for policy, and a declared
positive natural unit for each moment. Fix scales before measurement. The acceptance
budgets are:

| Quantity                          | Certified                  | Balanced                   |
| --------------------------------- | -------------------------- | -------------------------- |
| Projected value and CE level      | `max(1e-8, 64 * eps)`      | `max(1e-6, 64 * eps)`      |
| Economic marginal on smooth cells | `max(1e-6, 8 * sqrt(eps))` | `max(1e-4, 8 * sqrt(eps))` |
| Continuous policy level           | `max(1e-6, 8 * sqrt(eps))` | `max(1e-4, 8 * sqrt(eps))` |
| Declared population moment        | `max(1e-6, 64 * eps)`      | `max(1e-4, 64 * eps)`      |

Here `eps` belongs to the active floating dtype. These are refinement and fixture
acceptance budgets, not a universal theorem about arbitrary economic models. Moments
must identify the population, common base draws and normalization. A discrete owner,
participation decision, kink side or exit id has no tolerance. Candidate omission for
`replay="grid"` must satisfy the policy and moment budgets separately from projection
error; simulation Q regret is not stored-value error. Sampled refinement validates its
declared query population. A uniform domain bound additionally requires an analytic
interval bound, a justified curvature or Lipschitz bound, or exhaustive piecewise-affine
verification. Diagnostics distinguish sampled validation from uniform bounds.

## What certified means

Certification concerns represented operands, finite candidate banks, declared quadrature
and piecewise representations. It does not certify the continuous economic optimum, the
adequacy of calibration, or an integration rule's error against the underlying
probability law.

Ownership and equality of represented candidates require exact comparison or an explicit
unresolved error. Existing exact-affine native requirements remain. The certified
profile refuses `replay="grid"`; realised-shock replay reconstructs the same candidate
bank and selection as solve. At a solve node the action id and candidate ownership must
match exactly; levels obey the dtype's rounding contract.

Measure quadrature, projection and candidate omission separately. The certified profile
does not combine them into a single tolerance or use a projection tolerance to excuse a
changed structural decision. Both profiles require public-API oracle comparisons on
eager/JIT, relevant vmap, dense/streamed/partitioned, supported precision, sharded and
AOT routes. Unsupported native hardware remains an open acceptance gate; source
inspection is not an execution receipt.

Signed quadrature is linear-only. Propagate its provenance into stored channels and
reject every nonlinear downstream reader, including CE, Euler inversion, constraints,
availability, exit rules and simulation sampling.

## Conditional query representation

The conditional operator is versioned as `pylcm.conditional.v1`. It conditions under the
declared scientific law and preserves the joint rows of the hidden numerical rule. A
finite node table is an integration rule; its accidental empirical dependence does not
define a copula. Conditional refactorization has its own quadrature error. There is no
general finite tower identity: a rule on the diagonal points `(-1, -1), (1, 1)` can
integrate `x * y` differently from nested conditioning under independent normal laws.
Retained and recomputed execution must still agree for the same declared conditional
operator.

A generated mathematical physical signal has an explicit internal representation: its
rounded physical value, the current source-law binding and, for an informative
normal-base marginal, its authorized correlated Gaussian coordinate. That coordinate
represents the invertible mathematical observation before rounding. It is a dynamic
operand of the query, and its representation and source binding enter query identity.
This is an augmented signal representation, not conditioning on the physical scalar's
bits alone: distinct signals can round to the same scalar and give different conditional
prices. Certification names this represented object. It does not claim a scalar inverse
can recover information lost to rounding.

Informative eligibility uses exact represented law-domain tests. Normal and lognormal
zero scales and uniform equal endpoints reveal no innovation. A categorical outcome
reveals its common-uniform interval. Neither permits a generated latent witness to
disclose additional information. An informative generated witness requires valid
source-law provenance and a finite, resolved physical transform. A physical override has
no generated witness; its inverse and support consistency need separate admission.
Hidden simulation draws remain in the revelation record until the destination acts.

For Gaussian conditioning, order informative observed coordinates first, followed by
hidden coordinates in original declared order. Exact host admission supplies the pivot
support of that permutation. Traced factor and triangular-solve arithmetic use safe
operands and mandatory unresolved status. Redundant observed pivots contribute no
independent innovation; engine-generated supported observations do not require an exact
rounded residual-equality test. Project the declared base rows onto the named hidden
innovation columns, retaining weights, order and duplicates, and apply the joint
conditional factor. Do not construct independent marginal rules. Correlation,
informative subset, factor convention, source/target bindings, clock, dtype and rule
bytes belong to the derived identity.

Without informative observations, retain the original forward operator exactly. An
informative view with a wholly deterministic conditional state evaluates that state
directly with unit mass. Residual uncertainty uses the projected rule's unchanged
weights. Record rule-mass discrepancy separately; these branches do not claim identical
finite rounding across different information views. Exact categorical event conditioning
uses common-CDF interval intersections and event-mass division where needed, subject to
the shared-discrete projection contract.

Actual zero-probability observations require an integrity refusal. Null solve nodes are
different: they retain safe finite conditional payloads where a zero mass has a nonzero
derivative. With unchanged shared categorical partitions the target category is the same
category, including a null solve category, without division by its mass.
Changed-partition zero-width limits require a qualified extension; unresolved consumed
derivatives fail separately from primal declaration validity. Never turn an unreachable
solve node into a global invalid-law failure or drop a finite `dmass * value` term.

State-dependent informative subsets are prepared within the admitted query chunk by
canonical programs. Host support classification and its transfers, metadata, numeric
variants and failed-reader owners share the query budget and lifetime contract. Avoid
population-sized preparation and a global subset bank. Keep target prescribed-policy
information separate from caller pricing information: additional caller knowledge does
not change the target's participation policy.

Acceptance separates observation representation/inversion, factor arithmetic,
conditional quadrature, nested-rule discrepancy, mapped-state representation and
execution equality. Qualify each with its own independent witnesses and scales;
arbitrary user rules and payoffs have no universal error bound. Structural comparisons
remain exact on the declared represented prices. These are implementation contracts;
native public solve/simulation and override qualification remain required gates.

## Runtime correlation admission

Each concrete solve, simulation and period replay validates the supplied correlation
matrix and its active-dtype representation on the host. The engine owns the exact
zero-pivot support mask and passes it beside dynamic correlation values to the native
program. The mask is not a user-supplied validation bypass. A later invocation validates
again; correlation values are not frozen by the first call.

An admitted zero pivot contributes an exact zero factor column. Select safe operands
before square root and division, rather than hiding an invalid inactive expression
afterward. An active pivot must be finite and strictly positive, and its factor entries
must remain finite. Unresolved active arithmetic returns a dedicated invalid-law status;
clipping an active pivot is not proof of factorization. Rounded residual equality is not
required at host-admitted zero pivots.

Derivatives hold admitted support fixed. Interior positive-definite derivatives use the
dynamic factor arithmetic. At singular support, derivatives have the interpretation of
tangents within the declared PSD stratum; crossing rank has no derivative guarantee.
Primal validity and finiteness of consumed derivatives are separate obligations.

## Legacy constructor rules

Retained IID constructor spellings return named shock groups at retirement. Their
compatibility rule preserves the finite operator of formerly admitted fixed folded
declarations. Standard new declarations retain their explicitly selected base rules.
Continuous law identity and finite quadrature identity remain separate facts.

Fixed normal-mixture physical-bin rules use frozen pullbacks under their bound law.
Fixed conditioned-normal rules use the union of category-specific pullbacks with the
original category weights. Preserve original physical-node bytes, category identity,
active support order and normalization semantics. A finite base-array round trip cannot
represent every such pullback exactly. Constructor-generated compatibility rules may
therefore retain typed frozen pullback anchors, authenticated against their bound law;
arbitrary user nodes cannot forge a physical-value override. Anchors are fixed numerical
coordinates, never persistent states or an alternative simulation law.

Simulation always applies the true continuous law to addressed base draws. Compatibility
anchors only preserve fixed-law solve operators. Unsupported cross-state anchor
evaluation and unrestricted runtime mixture-shape compatibility require a named refusal
or an explicit numerical migration; they cannot silently regenerate parameter-dependent
base nodes. Uniform singleton compatibility retains its declared left endpoint.

Preserving an atomic operator does not establish bitwise equality across changed
compiler topologies. Compare levels using a declared summand scale and compare
structural action ownership separately. A value tolerance cannot excuse a changed
structural decision; an unresolved parity decision fails acceptance. Keep legacy
consumers until their replacement's public-API parity and independent-oracle gates pass.

## EKL migration and resources

The exact reference household comes from lcm-zoo
`ad2658dae4df00aaf53934f9011adc4c60dcc959`,
`tests/test_ecksteinCareerFamilyDecisions2019_reference_household.py`. Its literal
Fraction loops are independent of production lowering. Its shared solve/simulation
decision helpers test routing agreement, not an independent derivation of household
economics.

For the full-model S5 precondition, make a per-shock table of law inputs, draw date and
every formation, divorce and counterparty projection. Action-dependent next-period
offer-set laws remain persistent/transition-local processes. They are not admitted as
current IID law callables. For every adopted shared discrete IID law, prove projection
invariance of its inputs or implement the common-uniform joint CDF refinement required
by the issue. Toy-law invariance alone is insufficient.

The initial source review uses lcm-zoo `ad2658dae4df00aaf53934f9011adc4c60dcc959`,
`src/lcm_zoo/ecksteinCareerFamilyDecisions2019/model.py`:

| Declaration                    | Actual source inputs and date                                                                                                           | Migration decision                                                                                                                                                                                                |
| ------------------------------ | --------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `_job_offer_probs`             | Current hours action, current education/health and experience advanced by hours; next-period offer set (`:451`).                        | Retain a persistent offer-set state with its edge law. Do not admit this callable as current IID.                                                                                                                 |
| `marriage_offer_young_f/m`     | Current school action and an arrival-age logit (`:2120`, `:2168`).                                                                      | Preserve existing transition-local timing. A new current-IID declaration needs a genuine period-current state law; it cannot relabel the school action as a parameter.                                            |
| `_spouse_offer_probs_from_f/m` | Own education and fixed parameters (`:1389`); one partner education draw also determines partner experience (`:2717`).                  | Retain one shared innovation for counterparty and couple reads. Same-period own education is preserved by the inspected single projections (`:2234`, `:2238`); changed-law cases require the joint CDF partition. |
| `_single_projection`           | Own economic states project identically; young projections also set `married_last_period` and phase-specific schooling years (`:2323`). | Check each adopted law against changed fields. Preserving education does not establish invariance for a callable reading marital history.                                                                         |

These are source-based migration constraints, not a revalidation of the full paper
model. S5 must connect adopted declarations and projection-invariance checks through the
public API before acceptance.

For S6, use symbolic full-model dimensions, dtype bytes, query channels and all consumer
counts with the resolved per-device chunk budget. Account for source, target and
transient workspace simultaneously. Never allocate full household arrays merely to size
them. If retaining queries exceeds the budget, recompute inside the same declared chunk.
Assert retained/recomputed equality and release after the last registered consumer. A
reduced-model run cannot certify full-model memory admission.

S10 also requires the specified production regime-period reconstruction-cost measurement
through the perf-loop workflow. Preserve existing production owners and paper-ledger
holds; a correctness fixture is not that performance receipt.

## Ordered delivery

### Gate migration map

The legacy gate date is the arrival period, after the scheduled transition has produced
the landing state and before the destination action. An arrival adapter retains
`(incoming source, proposed target, source role, arrival period)` provenance; a
destination-global exit cannot stand in for an edge-specific fallback. An adapter
retains its declared Boolean comparison, rather than substituting a utility maximum.

| Legacy behavior                                           | Replacement and preserved contract                                                                                                                         | Baseline witness                                                                             |
| --------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------- |
| Strict unanimous formation                                | Consent at the existing arrival date; native current-period Consent for the new household. Both values belong to the target's common participating policy. | `tests/regime_building/test_gated_edges_collective_solve.py`                                 |
| Collective participation and empty support                | Dissolution exit at arrival; realized participating-query support replaces interpolated dissolution flags. Keep zero-weight vetoes and one final action.   | `tests/simulation/test_realised_dissolution_route.py`                                        |
| Incoming-edge-specific fallback                           | Arrival adapter with incoming-leg provenance; initial and ungated occupants cannot borrow another leg's fallback.                                          | `tests/simulation/test_dissolution_route_provenance.py`                                      |
| Moving or retirement comparison                           | Arrival adapter retains the strict comparator, mapped landing and source-period utility/discounting.                                                       | `tests/solution/test_terminal_carry_kernel_edge_kwargs.py`                                   |
| Target value floor, window or mixed state/value predicate | Arrival adapter retains the entire threshold/window predicate, arrival-age grids and temporal parameter namespace.                                         | `tests/test_discrete_grid_values.py`, `tests/params/test_gate_reads_on_the_target_domain.py` |
| Reference-only or self-reference hurdle                   | Explicit typed projected reader at arrival; a value dependency is not another move.                                                                        | `tests/solution/test_gated_edge_self_referencing_ref_age_specialized_axes.py`                |
| Collective component comparison                           | Adapter over one selected joint-policy tuple; do not infer unanimous consent from a one-sided comparator.                                                  | `tests/solution/test_decoupled.py`                                                           |
| Value threshold on a self-loop                            | Adapter scoped to the edge and arrival date, carrying states once and preventing another same-period move.                                                 | `tests/regime_building/test_carried_state_through_gated_self_loop.py`                        |
| State-only or layout-only crossing                        | State-only Gate or predicate-free layout route, keeping role mapping, phase settlement and off-grid behavior.                                              | `tests/simulation/test_phased_edge_fallback.py`                                              |

Each migrated behavior retains an explicit replacement public-API test before
retirement. Apply projections to the realized landing before reading a value; apply a
predicate after reading its operands at the query point. Neither operation commutes with
interpolation. Preserve source, target, reference and edge parameter namespaces,
phase-specific settlements and exact arrival-period demand. The new realized policy
query intentionally replaces the interpolation-of-maxima residual; changed off-grid
outcomes need oracle evidence, not a claim of legacy bitwise parity.

### Stage gates

The child issues of 607 carry the native dependency graph. The implementation PR closes
the parent and children together. A child is accepted only with its own tests and all
inherited behavior, never by deleting a failing economic behavior.

S0 establishes standalone contracts and references. S1 and S2 follow S0; S3 follows S2;
S4 follows S3. S5 and S6 follow S4 and their issue-specific preconditions. S7 follows
S5/S6, S8 follows S7, and S9 follows S7/S8 with an explicit timing-preserving gate
migration map. S10 follows S2/S4 and the performance receipt. S11 follows S1/S5/S10. S12
retires legacy topology only after all prior stages are green.

Each receipt records source identity and dirty diff, exact command, environment,
precision/backend, owner, exit code and raw log/XML paths. New RNG fixtures are new
files beside frozen evidence. Completion requires the full suite, certificate and
resource gates, independent branch review with fixes, and a review-ready PR.
