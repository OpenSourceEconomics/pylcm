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

## Accepted scope

On 2026-10-24 the owner confirmed that faithful EKL remains a required acceptance case,
while the engine also supports generic positive-cost formation and independent hidden
shocks. EKL's formation adjustment uses the mapped lagged marital state, so newlyweds
have no lagged divorce cost; both spouses' offers are observed/shared at marital choice.
Action-dependent next-period offer laws retain their dating. The generic positive-cost
and hidden-offer witnesses test additional engine behavior.

Partial observation of correlated shocks is deferred. Refuse a query requiring that
conditional distribution explicitly. Wholly hidden groups retain their joint law.
Partial correlated physical overrides are also deferred: require a complete physical
frame for that group or use named base-innovation overrides and the forward transform.
Independent physical overrides remain supported. These decisions supersede the earlier
general Gaussian conditional-query and partial-physical-completion plan.

Legacy compatibility is not an acceptance requirement. Migrate callers directly to the
new API. Use focused local correctness checks; long certificate and broad platform
campaigns run in CI rather than blocking every implementation change locally.

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
Target taste draws stay hidden at the source choice. Shared innovations retain identity
in base space and use the law at each evaluated state. A formation query cannot silently
use an unobserved participation shock.

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

The discovery reviews in `sciebo:pro-audits/pylcm-pr621-grid_search/`,
`sciebo:pro-audits/pylcm-pr621-egm/` and `sciebo:pro-audits/pylcm-pr621-ce/` support a
sampled balanced acceptance contract. Keep the numerical budgets below and the economic
invariants above. Whole-domain topology proofs, exhaustive feature intersections,
arbitrary-finite exact accumulation, whole-Model reverse and higher-order AD are
separate capabilities, not routine balanced release prerequisites. Certification remains
optional and must fail closed whenever it is actually claimed. Source seals and broad
certificate populations run offline or in CI; they do not replace independent economic
checks.

A public balanced route must name its supported workload and serve every required
continuation, parent, replay and archive read within it. A refused required read is a
delivery gap. Private staged examples and exact samples do not establish public
admission or complete S10. Production cost requires a matched captured regime-period and
replay slice; fixture durations alone do not establish a speedup.

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
At served kinks and jumps, preserve the owning-side value, policy and economic marginal,
through side records or original candidate replay. Ownership follows the source
candidate's one-sided ownership and stable id. Do not interpolate across an
unrepresented discontinuity: refine or refuse that cell. Pull back all boundaries with
their owning sides; nonmonotone resource maps require all preimages. Certified boundary
inversion requires supported exact topology or a refusal; balanced numerical inversion
records bracket uncertainty. Preserve exact-node queries before applying interpolation.

Coverage may be local during private staged implementation. An explicit finite-demand
bank may retain exact declared and demanded coordinates while unproved open intervals
remain holes. Every required coordinate must be finite, inside the declared closed
domain and exactly retained after capacity admission. A nonexact query through a hole
fails before discounting or economic masking. Intact exact samples do not establish
whole-domain topology or smooth support. Keep complete declared-workload coverage
separate from sample integrity; an optional whole-domain certificate outside that
workload is not a balanced prerequisite. A private fixture bank does not install
arbitrary production parent-demand preparation or forward replay.

Exact-demand selection evidence retains the actual live-node owner, side and corner
tuple in original quadrature order, bound to its coordinate and original producing
row/date/law/context. Inactive quadrature rows are not corner selections. A common
partition index is only an address and cannot substitute for the economic roster. Keep
any fixture inspection evidence separate from durable carry axes; production evidence
retention must declare its own schema, native roles, reads and lifetime.

The economic marginal is a separate, explicitly labelled channel. On smooth cells it
includes derivatives of weights, physical-law transforms and resources. In particular,
`dW/db = sum(dw/db * V + w * dV/db)`. The derivative of the projected value interpolant
is a different statistic and cannot substitute in an Euler equation. At a kink use the
selected candidate's one-sided economic marginal; at a jump, label the owning-side
marginal and retain jump metadata. Reject an unresolved owning side. A zero primal
weight can have a nonzero derivative, so differentiate weights before dropping zero-mass
nodes. Nonlinear CE channels use their own stable typed sufficient statistic and full
chain rule.

The private unit-risk CE direction rule keeps the original power-mean primal and
ordinary coefficient path. When its nonnegative exponent coefficient is outside the
ordinary normal range, first-order transport carries binary scale separately and uses an
explicit transpose. Forward and transpose each meet the same independent numerical
reference; they need not be bit-identical across compiler paths. The transport fixtures
compare normal results after normalization, and subnormal products within one final
subnormal ULP of the represented binary-coefficient product. This bound allows
significand-product rounding followed by final scaling; it is not a bound on arbitrary
exponential/logarithm approximation or the economic model. True zero is allowed;
nonfinite base or direction operands, an unretained nonzero final underflow and
unresolved signed-scale cancellation refuse. Coefficient-residual differentiation beyond
the first-order rule remains unsupported. This kernel transpose does not add reverse
mode to the forward-only native exact-affine reader. The owned primitive's original
state, bindings and execution registrations are part of semantic capture authority.

The shared anchor product rule retains `exp(d) * Da + CE * Dd`, including at equal
payoffs. Its second term uses the actual represented CE coefficient for safe ordinary
products and sums, with the same owned first-order transpose. Tiny or unretained
products and sums retain exceptional scaled transport. The default logarithmic primitive
keeps its strict span refusal; ordinary anchor directions do not inherit that refusal
merely because one immaterial term is much smaller than the other.

The private balanced CE direction also checks conditioning before publishing validity.
Keep an uncancelled magnitude envelope `S` from the original child, weight and exponent
contributions, including absolute inner mass before opposing signed reductions. Refuse
when `64 * eps * S` exceeds `max(1e-4, 8 * sqrt(eps)) * max(1, abs(direction))` in the
profile's final units. Empty directions and ordinary finite cancellation remain
supported. This is a conditioning diagnostic backed by sampled independent economic
references, not a certified transcendental error bound or universal exact accumulator.
Existing probability, input, span and final-status checks still apply.

The law envelope also retains uncertainty from original payoffs whose separate logs can
lose their contrast. With `L = max_needed(abs(log(value)))` and
`r_i = exp(p * d_i - log_moment)`, it includes
`2 * CE / Z * sum(abs(Dw_i) * max(1, r_i)) * (1 + 2 * L)` before the same `64 * eps`
floor. Each differentiated factor retains its own binary scale. This component is empty
for zero law directions or bit-identical needed payoffs, including derivative-only null
events. Its positive sum is bounded by the largest active term times the active count;
the combined contribution and uncertainty envelopes are bounded by twice their maximum.
An empty uncertainty envelope preserves the original contribution envelope exactly.
These positive bounds do not enter the signed numerical reductions or their span gates.
High-scale unresolved law directions may refuse; ordinary finite and exact-equality
controls must remain usable. This is a conservative admission diagnostic, not a sharp
error bound for library logarithms.

Refinement bisects failed smooth cells and inserts known transformed breakpoints.
Compare the coarse representation with quarter, midpoint and three-quarter queries from
the solver's candidate construction, exact endpoints and adjacent representable points
at boundaries, then repeat with the refined representation. Report acceptance for that
declared sample when every channel meets its budget and known discontinuities are not
bridged. Required reads with exhausted capacity or unresolved ownership fail loudly.
Refining a savings grid and refining this projection are distinct operations.

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
participation decision, kink side or exit id has no tolerance. Assess candidate omission
for `replay="grid"` separately from projection error against the declared policy and
moment budgets. If that assessment is unavailable, report omission error as unknown; do
not claim an unrestricted optimum. Simulation Q regret is not stored-value error.
Sampled refinement validates its declared query population. A uniform domain bound
additionally requires an analytic interval bound, a justified curvature or Lipschitz
bound, or exhaustive piecewise-affine verification. Diagnostics distinguish sampled
validation from uniform bounds.

The private scalar DC Case-U route declares its immutable choice on the solver:
`DCEGM(..., replay="candidate", accuracy="balanced")` retains original candidate
reconstruction; `replay="grid"` explicitly selects the existing Cartesian forward
argmax. The latter is a distinct approximate replay identity, including in saved
results, and does not require the candidate-reconstruction future payload. A paired
assessment uses the same model equations, continuation, addressed frames and declared
population, reporting candidate omission separately from projection and moment errors.
It does not give every later population an accuracy guarantee. `accuracy="certified"`
refuses grid replay. The private publisher does not yet implement the complete certified
profile and refuses that current-IID declaration. Public EGM admission and the remaining
family/discrete-branch witnesses remain required.

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

For the added owned-resources-row versus borrowing-corner selector, compare the actual
canonical scores evaluated in the active dtype, backend and execution route. The row
wins equality under the canonical ordered comparison, without an ownership tolerance.
This compares represented scores, not ideal transcendental expressions, and preserves
the existing native exact-affine comparison requirements. Invalid candidate arithmetic,
unresolved source sides or a different original candidate roster fails the consumed
query. Same-point replay ownership needs the actual original roster or an applicable
arithmetic-transfer argument; a scalar receipt or a probe roster is insufficient.

Certified smooth links require proof for the actual represented arithmetic or an
applicable conservative enclosure. An ideal log bound alone does not enclose the target
JAX log program. Without that proof, keep unresolved intervals as holes and refuse their
consumption; exact finite-demand checks do not grant a uniform bound.

Measure quadrature, projection and candidate omission separately. The certified profile
does not combine them into a single tolerance or use a projection tolerance to excuse a
changed structural decision. Public-API oracle comparisons cover the relevant supported
execution routes of each declared workload, including its precision and device. A route
claimed as supported needs execution evidence; source inspection is not an execution
receipt. A full Cartesian product of optional solver, topology and transformation
features is not a balanced acceptance campaign.

Signed quadrature is linear-only. Propagate its provenance into stored channels and
reject every nonlinear downstream reader, including CE, Euler inversion, constraints,
availability, exit rules and simulation sampling.

## Information and hidden-query integration

For source EV1 Full choices, optimize continuous actions within each existing
discrete-action combination, then append one source taste component per exit in
declaration order. The source regime's existing taste address supplies that one roster;
continuous/global stay action IDs do not create additional independent tastes. Solve
takes the logsum over admitted stay-combination maxima and admitted exit payoffs.
Built-in dissolution (`unilateral` and `mutual`) admits its exit only when the kept stay
set is empty; an exit roster position is masked otherwise. Consent admits formation only
after its existing strict Raw comparisons pass. Simulation selects the route with the
same source draw and publishes the selected unshocked value. Target taste remains hidden
at the source, priced at its own logsum, then drawn at the destination address in its
acting pass. An arbitrary custom Full selector is not generally an argmax and therefore
has no canonical logsum integration; source EV1 with such a selector is unsupported and
explicitly refused. This does not prevent a custom source rule from reading a
canonically priced hidden target EV1 policy. Collective EV1 keeps its existing
capability refusal.

Keep target prescribed-policy information separate from caller pricing information.
Additional caller knowledge does not change the target's participation policy. Evaluate
that policy at each hidden node before pricing its selected action and stakeholder
tuple. Independent hidden shocks, including target taste shocks, are integrated before
the source decision and revealed only when the destination acts. A supplied hidden
override does not become an observed pricing operand.

For a wholly hidden group, use its original joint forward operator, preserving row
order, weights, duplicates and dependence. For independent observed/hidden coordinates,
fix the observed physical operands and project the same numerical rows onto the hidden
coordinates. Do not infer scientific dependence by filtering a finite node table or
replace a hidden correlated group by independent marginals. A finite rule has no general
tower identity; retained and recomputed execution must agree for the same chosen
operator.

If declared dependence connects observed and hidden coordinates, refuse the requested
partial observation instead of performing Gaussian conditional refactorization. Validate
runtime correlation at each invocation; a cached independent view cannot admit later
cross-correlation. Derivatives of an admitted independent view hold that independence
partition fixed. General correlated conditional moments and derivatives are deferred.

Ordinary equations consume the sealed rounded physical value. Retain original named base
innovations and producing-law context for shared-law replay and later revelation; they
are not extra observations available to utility, availability or participation.
Source/target bindings, observed names, target policy information, clock, dtype and rule
identity belong to the query identity. Keep this preparation within the admitted query
chunk and account for its outputs, transfers and failed-reader lifetimes.

Shared categorical observations retain the common-uniform interval semantics required by
the shared-discrete projection contract. Unchanged partitions preserve the category,
including a null solve category, without dividing by its mass. Changed partitions use
qualified common-CDF refinement; unresolved consumed derivatives fail separately from
primal validity. An actual zero-probability observation is an integrity failure. A null
solve node can still contribute a finite `dmass * value`; never drop that derivative.

Named base overrides use the admitted forward law. Independent physical overrides and
complete correlated physical frames require source-law inversion/support validation;
being complete is not a validation bypass. Refuse partial correlated physical frames and
mixed base/physical forms within a correlated component. Preserve producing context,
original row/address, path-versus-entry precedence and actual revelation timing. Full
frames, singular support and unresolved inverses retain their explicit integrity gates.

Acceptance covers independent hidden pricing, preserved wholly hidden joint dependence,
fully observed correlated inputs, explicit partial-observation/override refusals,
shared-U boundaries, original-row revelation and execution equality. Numerical levels
and structural action/exit decisions remain distinct acceptance channels.

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

Fixed declarations obey the same active-pivot validity requirement: exact host PSD
admission alone does not certify rounded factor arithmetic.

Derivatives hold admitted support fixed. Interior positive-definite derivatives use the
dynamic factor arithmetic. At singular support, derivatives have the interpretation of
tangents within the declared PSD stratum; crossing rank has no derivative guarantee.
Primal validity and finiteness of consumed derivatives are separate obligations.

## Native singleton GridSearch integration

Each real state cell evaluates complete joint node rows, applies its phase-specific law,
and invokes the canonical feasible action reducer before the weighted sum. Hard maximum
and EV1 keep that order. Node coordinates are temporary operands, with exact resolved
support and algorithm identity; stored values retain only real state axes. Ordinary
state tiling, hard-max action streaming and state/subject sharding use the existing
native program and admission paths. Every retained period's support bank and published
physical simulation bank contributes to memory admission.

Group law parameters use direct `params[group][slot]` leaves, including fixed slots. Law
helpers use their transitive dependency closure. Shared economic or report helpers
retain their ordinary regime parameters independently. A native cell requires scalar
resolved marginal parameters, or a categorical probability vector; a helper may consume
an array and select a scalar. Standalone group transforms retain their cell-batching
contract. Solve laws use carried-state imputations; simulate laws receive actual carried
rows. Law callables receive categorical state-grid values or explicit state-code
arguments at their usual boundary.

Discrete physical shocks expose their category code through either the bare name or the
explicit `<shock>_code` alias. Continuous physical draws have no node-code alias.
Categorical carried states use their declared domain for both solve imputations and
actual simulation rows. Invalid imputed codes feed mandatory law integrity even when a
code-reading callable would otherwise return finite parameters.

Shared discrete projection admission consumes the demanded query's actual physical read
closure. An unused group declaration does not create an observed category. Admitted
reuse requires the same semantic law/helper bindings, parameter operand paths, ordered
categorical domains and effective law-input projections. Continuous identity requires an
effect-free direct-return JAXpr without equations or closed constants; finite discrete
identity may instead enumerate the complete code domain with matching codecs. A private
cardinality cap bounds exhaustive build-time work; direct-return proofs apply to large
domains without enumeration. A failed proof names the shock, query and projection input.
If an adopted EKL current law changes a relevant demanded input, the required joint CDF
refinement remains separate from this private admission. The verified transition-local
next-offer law is not itself a new current law.

Simulation draws fresh addressed base innovations before the decision. Ordinary utility,
reporting and transitions consume the same rounded physical realization, which survives
result save/load. Compiler barriers preserve that physical boundary; an ordinary
consumer cannot recover an unrepresented base coordinate by reassociation. Mandatory law
and factor integrity is enforced even with logging and optional checks disabled.

Weight factors retain support before their rounded product underflows, including factors
inside a joint categorical group. Finite rare-weight times large-value terms and finite
zero-primal weight derivatives use the shared scaled arithmetic. Active-dtype/backend
value cotangents can still underflow before a large upstream payoff derivative; this
extreme sensitivity is not an accuracy-qualified marginal channel. Later economic
marginals must weight their explicitly represented derivative values.

Uncomposed current-IID intersections fail explicitly: non-GridSearch decisions,
Decoupled objectives, legacy folded axes, action partitions, co-mapped or invariant
state blocks, bound/query reconstruction, and gated or same-period value readers.
Refusals follow actual owners and incoming readers, preserving unrelated legacy
components. Policy lookup awaits explicit observation/conditional semantics. Signed rule
ancestry is refused at the native decision boundary. These scoped holds do not retire
legacy routes or complete the later exit, query and signed-reader stages.

## Legacy migration

Migrate legacy IID-state and folded declarations directly to named shock groups with
explicit base rules. Retaining constructor spellings, physical-bin nodes or the former
finite quadrature operator is not an acceptance requirement. Do not introduce frozen
pullback anchors or a compatibility bridge to preserve those numerical choices.

Migration preserves the economic law, information available at choice, draw date and
joint dependence. A current observed shock and a post-action transition-local lottery
are distinct: moving the latter before choice changes the model. Persistent processes
remain real states, with their node-mode and code support retained. State-conditioned
laws must keep their actual conditioning date rather than merely reuse a state name.
Simulation applies the declared physical law to addressed base draws.

Validate each replacement through the public API against its declared quadrature and
independent economic oracle. Continuous-law identity and finite quadrature identity
remain separate facts; neither implies bitwise equality across compiler topologies.
Compare structural action ownership separately from value tolerances. Retire legacy
consumers only after their replacement's required solver, query and replay gates pass.

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
files beside frozen evidence. Completion requires the applicable CI suite and resource
gates, independent branch review with fixes, and a review-ready PR. Broad certificate
campaigns remain in CI or offline; optional certification does not block balanced
semantic closure.

### Private balanced Case-R implementation frontier

The private connected BALANCED slice authenticates the full unbound scalar utility
JAXpr. Its initial syntactic class admits one unmodified `log(consumption)`, optional
single action-independent multiplication, and action-independent shifts and supported
shape-preserving wrappers. Coefficient readers evaluate the original independent
subgraphs; they do not infer log utility from an inverse, samples or AD. Runtime scale
must be finite and positive and shift finite. More general algebraically related graphs
and utility classes remain required extensions, with explicit refusal until their
coefficient provenance is implemented.

The canonical Hermite reader supplies its actual represented width and limited
correction operands. Exact-dyadic Bernstein control signs exclude ideal critical roots
on an authenticated unclamped local piece. They establish neither a target log error
bound nor monotone rounded ownership. Original rounded endpoint scores propose
resource-space row/corner regions. Mixed control signs, contacts, score crossings,
unsupported sides and applicability remain uncertainty holes, with bounded subdivision
and explicit reason/status records. The initial locator inserts at most
`min(capacity, 16)` additional resource coordinates; unresolved neighborhoods remain
holes rather than obtaining a smooth certificate from convergence or finite probes. A
sufficient initial legality class places both endpoint savings inside the literal saving
bounds and below the piece's lower resource coordinate. This convex-range condition
avoids replacing rounded width by exact endpoint subtraction. Interior legality cuts
remain required constructive work; actual rounded legality is checked at every consumed
ownership-sensitive query.

The structural catalog has a resource axis independent of quadrature rows. Native
piece/run, row/corner branch, side and legality records at deduplicated exact resource
coordinates grant only exact-coordinate authority. No sampled per-current-node policy or
statistic bank becomes durable projection authority. Required liquid coordinates and
their resource images, catalog cuts, actual `>=` and strict `>` rounded pullbacks,
adjacent represented records and refinement share the admitted capacities. Overflow
invalidates; allocation never grows or silently drops mandatory coordinates. Adjacent
represented queries are actual floats, not analytical one-sided limits.

The private projected subtype retains numeric integrated PWA levels, a separate economic
marginal, and a separately acquired numeric guard containing the original row, dated
context, physical/factor law operands and resource catalog. Static original callbacks
belong to the actual producer specification. A parent recomputes original rounded
native-piece/run/side/corner/legality tuples transiently against the catalog and merges
failure before discount or masking. It returns PWA levels and separately interpolated
economic marginals, not reference levels or the derivative of projected values. Guard
arrays participate in original native roles, actual continuation reads and rolling
lifetime/admission; stored leaf identities and tuple-backed consumer paths are distinct.

Refinement checks exact endpoints, adjacent floats and quarter/mid/three-quarter queries
using the adopted BALANCED value and marginal budgets. It refines only classified smooth
links, preserves holes and mandatory exact records, and refuses failed-channel capacity
exhaustion or nonrepresentable refinement midpoints. Source invalidity is global; a hole
is a mandatory local read failure. An entirely disconnected result is not admitted as
successful BALANCED PWA publication. These are sampled accuracy and consumed-query
ownership contracts, not certified or uniform full-domain coverage.

The explicit private installer derives the guard/schema from the actual native core and
actual materialization argument tree, then supplies the same immutable dated spec to the
genuine parent builder. The private dated path below installs these declarations
automatically for its admitted source classes. The shared-declaration identity check
remains mandatory and broad public admission stays closed. The BALANCED graph does not
publish the exact-demand replay selection bank as a forward artifact. Action-valued
forward reconstruction, owning-side outputs on served reads, passive/choice axes, CE,
law-weight derivatives, other EGM families and Case U remain required economic
capabilities. Complete declared-workload coverage remains required; certified uniform
qualification is separate.

### Private Case-U stable PowerMean economic frontier

The private scalar point path may consume the actual exact `PowerMean` declaration
through the same canonical anchored core as its public primal. Its savings candidate and
selected source envelope use the same original child-node records and separately
published economic child marginal. Null events needed by a weight derivative retain
their actual positive finite value, even outside the primal anchor range. Every event
and differentiated factor product retains its own coefficient and binary scale; signed
contributions stay scaled through final multiplication by the CE. The public
scaled-aggregate AD refusal remains unchanged.

At exact unit risk aversion the explicit risk-parameter contribution is
`-CE * Var(log(value)) * D(risk_aversion) / 2`. The variance uses centered nonnegative
terms with probability scales retained through final CE scaling. The finite-transform
path retains its original product and reduction arithmetic.

The private near-unit divided differences use the integral-derived power series for
`exprel(t)`, `psi(t)` and `phi(z)`. Their small domain is `abs(argument) <= 1/8`. The
exprel/psi degrees are 8 and 14 for float32 and float64; phi degrees are 10 and 20. The
omitted tails are bounded by `eps/16`, using respectively `exp(A)*A**(m+1)/(m+2)!`,
`exp(A)*A**(m+1)/((m+1)!*(m+3))` and `A**(m+1)/(1-A)`. Ordinary round-to-nearest
coefficient/Horner error plus those tails is bounded by `64*eps` in absolute value on
that domain. This is a local arithmetic contract, not a backend transcendental ULP
theorem or a topology certificate. The existing canonical near/far branch decides which
derivative expression is used; derivative coordinates do not reuse primal null-node
masks.

The first private arithmetic class refuses nonpositive needed payoffs, nonfinite needed
coordinate/economic/law operands, unresolved intermediate exponent geometry and
nonfinite final channels. It also refuses every reduction whose logarithmic spread
exceeds the dtype mantissa width, including same-sign inner reductions: a lost small
constituent could later be the residual after another channel opposes that aggregate.
Refusal occurs before aggregation, so nested reductions cannot hide the lost scale. A
complete multi-scale signed accumulator is deferred unless a required application needs
that refused class. Ordinary finite economic directions must remain accurate and usable;
high-condition cases may explicitly refuse. Finite-final derivative cases, rare-event
arithmetic and the zero-payoff family need their own declared domains and evidence.

The complete transient bank has an immutable layout bound by identity to the actual
point owner, original plan/read objects and exact CE declaration. Its joint event length
and factor arity come from the actual ordered roster, including derivative-only events.
It exposes actual bank shapes/templates; dispatch width is not total capacity. The
actual enclosing Case-U CoreProgram must additionally account for original
read/block/index/padding buffers, overlapping assembly copies, anchored/signed reduction
temporaries, outputs and resident child roots through its genuine compiler reservation
and existing solve budget. Point-only numerical qualification does not grant that
production admission. Automatic model scheduling, the real enclosing
materializer/admission, required streamed equivalent-family implementation and all
remaining economic, shape, served-side, parent/native and realized-forward cases stay
required for the declared workload. A uniform topology proof is separate. The broad
public current-IID non-GridSearch guard remains closed.

### Private genuinely owned Case-U enclosing program frontier

The private Case-U assembler acquires `GenerationSolverPreparation` from the real
call-owned Model generation and revalidates its original context, dated build, step,
current plan, declarations and installed-kernel identities. It does not manufacture an
authority token or substitute an ordinary grid program. The authentic constructor
currently supplies the resources-only subclass as an intermediate ownership witness;
general Case-U classification remains required.

The enclosing `CoreProgram` runs original `CaseUPointSolver.node/integrate` across the
actual liquid grid and realises the current law at each liquid coordinate. Its original
DC argument projection retains actual children, canonical fixed and runtime namespaces,
group/node operands, source clock and explicit savings/floor/ future-node grid operands.
Immutable code-grid owners remain in the genuine generation/program root census. Its
abstract output and roles come from tracing the real
`BoundedResult[CurrentResourcesEvaluation]` computation, including its actual
bound/query/count trees; no Case-R template, fabricated EGM carry or policy artifact
stands in for that output. Actual published child leaf schemas declare the native
argument locators.

The initial enclosing program is axis-free, without donation or host-buffer
substitution. The private profiling path lowers that exact whole function, holds the
actual generation/code/input roots through an owned workspace record and uses existing
concrete buffer accounting, resident inventory and compiler reservation against an
explicit existing budget. It conservatively excludes none of those held roots. This
admits only that private probe's represented lifetime; the genuine Model
scheduler/publisher/parent/forward installation remains required. An explicit
installation check rejects a privately assembled program unless the same program object
belongs to the real installed generation kernel graph.

Source fixtures use real private Model construction and real retained dated preparation
for the intermediate subclass, actual child schemas and native input projection. Missing
Model installation and resident-budget exhaustion before compilation remain explicit
refusal controls. General utility/constraint/law shock classification, producer
publication/shape preservation, full economic capabilities and realized forward
reconstruction remain required next work.

### Private immutable dated BALANCED preparation

The private dated preparation path splits normalized regime metadata from real
solver-program finalization. prepare_dated_parameters exposes the original parameter
namespaces and normalizes fixed operands through the same resolver, temporal conversion,
canonical casts and granular expansion used by ordinary construction. It does not derive
runtime shapes from annotations.

build_private_dated_balanced_regimes requires the exact active roster of real
CoreBuildContext operands from canonical parameter/support preparation. Its
producer-date DAG prepares next-date children first, including self transitions. The
final aggregate spec aliases those original date objects. Every incoming structural
Euler consumer is closed before final program assembly, and a second shape trace must
match the frozen producer template. Zero probability or discount cannot remove a
consumer.

The private Model.\_from_call_owned_params staging entry now consumes genuine
constructor declarations and actual parameters through the shared declaration
preparation and post-program installer. It normalizes parameters before dated
finalization, resolves the original support algorithm, and gives producer-date
declarations the actual runtime liquid nodes. It creates real programs before exposing
the Model. The supported public scalar route is described below.

One immutable generation keeps those real execution programs, their source seal,
declaration identity and lazily derived authority together. The cache walks actual
regime, edge and named-group namespaces before leaf schema hashing, and serializes cold
preparation. Lower, solve and forward select the same generation. Owned results retain
it after cache eviction, and entry residency observes initial, cached and
evicted-but-retained roots by actual buffer identity. The genuine dated contexts,
numerical steps and installed kernel identities are available to private enclosing-core
builders through generation.solver_preparation; no archive or arbitrary token supplies
those owners. Serialization preserves the actual initial program metadata and resets
derived caches and locks.

Initial private construction and cold specialization with a device-memory budget require
a genuine admission owner before dated bank/template materialization. Separate
simulation support, gated/arrival composition, other admitted solver families,
helper/query paths and unsupported consumed array/grouped proof classes remain required
extensions with explicit refusals. Private source support alone does not establish
native numerical or public Model-to-forward acceptance for those extensions.

### Private connected Case-U publication and replay

The genuine dated builder selects the private U branch from the complete original action
dependency census: current physical ancestry outside resources, pre-state context
outside the resource cut, or a genuine nonlinear CE declaration. It does not classify
arbitrary failed R admission as U. The original R admission remains binding; public
entry is limited to the scalar route described below.

The U branch shares the actual point assembler between original pre-install dated
context/build/step/plan/call objects and the final installed generation. Its native
output contains integrated source levels/economic marginals, economic emptiness,
original source validity/status/read counts and the actual same-invocation law witness.
Selected candidate/owner/side/corner checks remain transient inside the original
conditional solve. Durable projection does not retain a per-current-node policy bank.

The exact-point publisher closes its admitted incoming scalar deterministic Euler
parents using original next-state laws, parameters, fixed operands, source date, age and
savings/floor support. It unions represented demands with the declared liquid grid
inside the admitted projection capacity and checks that every mandatory coordinate
survived. Out-of-domain demands, capacity, law/source corruption and missing exact
support fail explicitly. Coordinate owner indices address stored exact records; they do
not certify source-candidate topology. Unsupported links and unproved sides fail through
the real dated reader before discount or masking. The separate saturated-span BALANCED
publisher retains its own source proof and refinement requirements.

Candidate replay retains the required future continuations and original dated solver
specification. At the realised physical frame it reconstructs the original alternatives
and selects their associated action, saving and native owner before advancing the
original transition. It does not interpolate an average policy. Default retention and
fresh-model save/load preserve that route. Explicit grid replay remains a separate
approximation with the declared omission and moment requirements.

The connected private families include scalar DC/NB, NB passive cells with discrete
branches, NB ride with the original PowerMean/CES recursion, and finite NEGM with its
original DC keeper and adjuster constructors. The recursive NB slice retains the
original joint ordinary-terminal target roster, stable CE channels and actual
transform/aggregator identities. The DCEGM nonlinear-CE refusal remains in place. NEGM
retains the actual outer target and complete inner candidate tuple; its branch authority
cannot authorize ordinary child publication. Ordinary passive future rows keep their
actual carry schema, numeric generation roots and original row bindings. Mapped NB
branches return only consumed associated channels; unused query metadata stays inside
the original branch solve. Their numeric record annotations permit the actual leading
mapped axes.

The finite additive NNBEGM source constructs its original keeper and every declared
outer alternative over the full original NB geometry before current-shock integration. A
fixed-capacity query insertion retains the complete bank, budget-run topology and
separate native winner. The selected record supplies its associated value, policy,
saving and economic marginal. Value germs still order ties; the closed upper node keeps
its economic marginal even when its right-extension germ is dead. This slice uses
ordinary terminal children, a literal floor and a common positive affine resource
Jacobian.

The NB recursion now retains the missing exponent direction at unit risk aversion:
`dCE/dp = CE * Var(log(value)) / 2` for power exponent `p = 1 - risk_aversion`. The
continuation kernel reuses the scaled pair variance and supports first-order reverse
mode for this correction. Its original primal and non-unit-risk derivative remain
unchanged. A nonzero coefficient outside the retained normal finite range refuses this
scoped derivative class; rare-event cases with a finite final direction remain required.

The shared signed-log materializer uses IEEE downscaling for negative binary exponents
and checks overflow before packing positive exponents. Its floating decomposition and
zero/invalid conventions remain unchanged. Boundary-value encoding checks do not
establish full finite-final scaled arithmetic, cancellation or additional AD support.

Native first-order parameter directions follow the represented winning affine segment,
including its moving endpoints. The model controls retain the separate continuous
optimum reference and check the original represented derivative. The native kernel's
forward-only AD contract remains unchanged. Default-retention archive replay also
reacquires the original model authority before reading recursive child carries.

An ordinary scalar DC parent can now consume the admitted NB passive child's exact
liquid demands. The producer inserts each demand into every actual passive row; the
consumer uses the original passive landing law and stencil on the stored values and
economic marginals. Physical replay retains that same row schema and checks every source
row, including rows with zero stencil weight. This connected unbudgeted ND slice does
not admit arbitrary incoming families or replace the passive axes by a flat liquid row.

The scalar NB finite-budget path retains the original probe arithmetic while building
JAX derivative transforms inside the enclosing JIT. Its owned-result solve and physical
replay now connect through an exact census of built-in future-child and consumed-input
records. Shared buffers are charged by their actual identity; foreign record subclasses
remain refused. A detached foreign solution can require authority or payload copies
before replay, and a finite budget still refuses those unprofiled copies. The public
source variant remains subject to its original Model-to-forward check; this scoped
connection does not complete cold admission.

This family scope does not complete S10. Broader nested NB and nonlinear-CE
intersections, broader incoming ND families and passive/choice/phase/query/gated/sharded
intersections, remaining cold finite-budget admission and derivative/archive controls
stay required. Correct U source-context/selector/corner outputs, sampled refinement,
complete declared-workload support and scheduling/admission also remain required.
Production cost needs a representative capture before any performance claim; universal
full-domain topology and certified refinement remain separate capabilities. Exact-demand
integrity and scoped Model-to-forward qualification do not establish those gates or
extend public admission beyond the scalar route below. Cold allocation remains owned by
the existing generation materializer; this route cannot bypass its budget guards.

### Public scalar Case-R lifecycle

Public `Model` construction retains genuine parameter metadata for the supported
unbudgeted scalar DCEGM declarations. `get_params_template()` uses the existing
canonical namespaces. A call with actual parameters acquires a genuine installed
execution. Calls with the same parameter schema and grid support reuse it; changed
support acquires a separate generation. Retained results keep their producing generation
alive.

The admitted workload uses `ExactEnvelope`, balanced accuracy, candidate replay, one
continuous state/action per source and ordinary scalar action-free terminal regimes. A
parent reads its Case-R child's original guarded projected continuation through the
native leaf roster. Default retention and archives preserve that continuation and the
original resources row required by physical candidate replay. A fresh public Model
reconstructs its own execution before reading the archive. Producer/date/schema, law,
clock, domain, hole and source-validity checks remain mandatory. Unsupported projected
reads retain their explicit refusal.

This public slice does not complete S10. Other producer families, public finite-budget
declaration admission, remaining required workloads and the production reconstruction
cost receipt remain open. Prepared public Models explicitly refuse policy lookup,
state-grid inspection and initial-state bounds until those helpers have a supported
execution-selection contract. Whole-domain topology and broad certificate campaigns
remain separate from the sampled balanced acceptance gate.
