---
title: Public API index
---

# Public API index

This curated index covers the names exported from the top-level `lcm` and `lcm.solvers`
namespaces. It is a navigation-completeness check, not an exhaustive semantic
specification: submodule-only public objects, constructor fields, behavioral contracts,
and the correctness of destinations require page-specific documentation and tests.

## Model and economic declarations

| Public name                                                                                          | Canonical documentation                 |
| ---------------------------------------------------------------------------------------------------- | --------------------------------------- |
| [`lcm.Model`](model_and_regime.md#api-model)                                                         | Model assembly and execution            |
| [`lcm.Regime`](model_and_regime.md#api-regime)                                                       | General regime                          |
| [`lcm.ConsumptionSavingsRegime`](consumption_savings.md#api-consumption-savings-regime)              | One liquid margin                       |
| [`lcm.NestedConsumptionSavingsRegime`](consumption_savings.md#api-nested-consumption-savings-regime) | Liquid plus outer margin                |
| [`lcm.LiquidMargin`](consumption_savings.md#api-liquid-margin)                                       | Liquid roles                            |
| [`lcm.OuterContinuousMargin`](consumption_savings.md#api-outer-continuous-margin)                    | Outer roles                             |
| [`lcm.NetOfAdjustmentCost`](consumption_savings.md#api-liquid-margin)                                | Resources composition                   |
| [`lcm.post_decision_lower_bound`](consumption_savings.md#api-post-decision-lower-bounds)             | Checkable borrowing bound               |
| [`lcm.outer_unchanged`](consumption_savings.md#api-outer-continuous-margin)                          | Identity no-adjustment sentinel         |
| [`lcm.cash_on_hand_with_subsidy`](case_pieces.md)                                                    | Supported case-piece fixed form         |
| [`lcm.liquid_law_from_resources`](consumption_savings.md)                                            | Conventional liquid law                 |
| [`lcm.liquid_law_from_savings`](consumption_savings.md)                                              | Conventional savings-written liquid law |
| [`lcm.ExtremeValueTasteShocks`](model_and_regime.md#api-extreme-value-taste-shocks)                  | EV1 discrete-choice shocks              |

## Grids, categories, and processes

| Public name                                                                             | Canonical documentation              |
| --------------------------------------------------------------------------------------- | ------------------------------------ |
| [`lcm.AgeGrid`](grids_and_processes.md#api-lifecycle-and-categorical-grids)             | Lifecycle grid                       |
| [`lcm.categorical`](grids_and_processes.md#api-lifecycle-and-categorical-grids)         | Category declaration                 |
| [`lcm.DiscreteGrid`](grids_and_processes.md#api-lifecycle-and-categorical-grids)        | Discrete outcome grid                |
| [`lcm.LinSpacedGrid`](grids_and_processes.md#api-continuous-grids)                      | Linear grid                          |
| [`lcm.LogSpacedGrid`](grids_and_processes.md#api-continuous-grids)                      | Log grid                             |
| [`lcm.IrregSpacedGrid`](grids_and_processes.md#api-continuous-grids)                    | Explicit irregular grid              |
| [`lcm.GridBreakpoint`](grids_and_processes.md#api-continuous-grids)                     | Piecewise-grid boundary              |
| [`lcm.PiecewiseLinSpacedGrid`](grids_and_processes.md#api-continuous-grids)             | Piecewise linear grid                |
| [`lcm.PiecewiseLogSpacedGrid`](grids_and_processes.md#api-continuous-grids)             | Piecewise log grid                   |
| [`lcm.UniformIIDProcess`](grids_and_processes.md#api-stochastic-processes)              | IID uniform process                  |
| [`lcm.NormalIIDProcess`](grids_and_processes.md#api-stochastic-processes)               | IID normal process                   |
| [`lcm.LogNormalIIDProcess`](grids_and_processes.md#api-stochastic-processes)            | IID log-normal process               |
| [`lcm.NormalMixtureIIDProcess`](grids_and_processes.md#api-stochastic-processes)        | IID normal mixture                   |
| [`lcm.TauchenAR1Process`](grids_and_processes.md#api-stochastic-processes)              | Tauchen AR(1)                        |
| [`lcm.RouwenhorstAR1Process`](grids_and_processes.md#api-stochastic-processes)          | Rouwenhorst AR(1)                    |
| [`lcm.TauchenNormalMixtureAR1Process`](grids_and_processes.md#api-stochastic-processes) | Mixture AR(1)                        |
| [`lcm.StateConditioned`](grids_and_processes.md#api-stochastic-processes)               | State-conditioned process parameters |

## Transitions, phases, and structured declarations

| Public name                                                           | Canonical documentation            |
| --------------------------------------------------------------------- | ---------------------------------- |
| [`lcm.MarkovTransition`](transitions.md#api-state-transitions)        | Stochastic transition wrapper      |
| [`lcm.JointTransition`](transitions.md#api-joint-transitions)         | Shared-draw joint law              |
| [`lcm.fixed_transition`](transitions.md#api-state-transitions)        | Identity law                       |
| [`lcm.AgeSpecializedFunction`](transitions.md#api-age-specialization) | Age-varying function               |
| [`lcm.AgeSpecializedGrid`](transitions.md#api-age-specialization)     | Age-varying grid                   |
| [`lcm.Phased`](transitions.md#api-solve-and-simulation-phases)        | Solve/simulate variants            |
| [`lcm.Condition`](conditions.md)                                      | Structured Boolean expression      |
| [`lcm.ref`](conditions.md#api-condition-syntax)                       | Named reference                    |
| [`lcm.implies`](conditions.md#api-condition-syntax)                   | Conditional requirement            |
| [`lcm.case_boundary`](case_pieces.md#api-boundary-predicate)          | Structured case boundary           |
| [`lcm.piece`](case_pieces.md#api-piece-formulas)                      | Piece decorator                    |
| [`lcm.smooth_helper`](case_pieces.md#api-piece-formulas)              | Reviewed smooth-helper attestation |
| [`lcm.affine_breakpoint`](piecewise_affine.md#api-affine-breakpoint)  | Schedule breakpoint                |
| [`lcm.piecewise_affine`](piecewise_affine.md)                         | Piecewise-affine schedule          |

## Preferences, results, and persistence

| Public name                                                                 | Canonical documentation        |
| --------------------------------------------------------------------------- | ------------------------------ |
| [`lcm.LinearAggregator`](../methods/preferences.md)                         | Linear Koopmans form           |
| [`lcm.CESAggregator`](../methods/preferences.md)                            | CES Koopmans form              |
| [`lcm.CertaintyEquivalent`](../methods/preferences.md)                      | Certainty-equivalent contract  |
| [`lcm.LinearExpectation`](../methods/preferences.md)                        | Linear lottery reduction       |
| [`lcm.PowerMean`](../methods/preferences.md)                                | Power-mean reduction           |
| [`lcm.QuasiArithmeticMean`](../methods/preferences.md)                      | Quasi-arithmetic reduction     |
| [`lcm.SimulationResult`](runtime_and_results.md#api-simulation-result)      | Deferred simulation result     |
| [`lcm.save_solution`](runtime_and_results.md#api-standalone-persistence)    | Save value functions           |
| [`lcm.load_solution`](runtime_and_results.md#api-standalone-persistence)    | Load value functions           |
| [`lcm.SolveSnapshot`](runtime_and_results.md#api-standalone-persistence)    | Solve diagnostic snapshot      |
| [`lcm.SimulateSnapshot`](runtime_and_results.md#api-standalone-persistence) | Simulation diagnostic snapshot |
| [`lcm.load_snapshot`](runtime_and_results.md#api-standalone-persistence)    | Load diagnostic snapshot       |
| [`lcm.__version__`](public_api.md)                                          | Installed pylcm version        |

## Explicit public submodule surfaces

Most user-facing names are re-exported from `lcm`. These deliberately public submodule
surfaces remain outside that top-level namespace:

| Public name                                                                       | Canonical documentation             |
| --------------------------------------------------------------------------------- | ----------------------------------- |
| [`lcm.params.MappingLeaf`](../user_guide/parameters.md)                           | Mapping parameter leaf              |
| [`lcm.params.SequenceLeaf`](../user_guide/parameters.md)                          | Sequence parameter leaf             |
| [`lcm.params.UserMappingLeaf`](../user_guide/parameters.md)                       | User mapping parameter leaf         |
| [`lcm.params.UserSequenceLeaf`](../user_guide/parameters.md)                      | User sequence parameter leaf        |
| [`lcm.params.as_leaf`](../user_guide/parameters.md)                               | Explicit parameter-leaf wrapper     |
| [`lcm.koopmans_aggregation.KoopmansAggregator`](../methods/preferences.md)        | Koopmans-form base contract         |
| [`lcm.solver_api.ResultRetention`](runtime_and_results.md#api-solution-result)    | Solution result retention           |
| [`lcm.solver_api.ArtifactKey`](runtime_and_results.md#api-solution-result)        | Versioned artifact identity         |
| [`lcm.solver_api.ArtifactRef`](runtime_and_results.md#api-solution-result)        | Period/regime artifact address      |
| [`lcm.solver_api.ArtifactStore`](runtime_and_results.md#api-solution-result)      | Immutable artifact store            |
| [`lcm.solver_api.KernelOutput`](custom_solvers.md)                                | What a period kernel returns        |
| [`lcm.solver_api.OmissionReason`](runtime_and_results.md#api-solution-result)     | Recorded reason for absence         |
| [`lcm.solver_api.SolutionMetadata`](runtime_and_results.md#api-solution-result)   | Labelled-solution metadata          |
| [`lcm.solver_api.SolutionResult`](runtime_and_results.md#api-solution-result)     | Labelled solution result            |
| [`lcm.solver_api.ValueArraySchema`](runtime_and_results.md#api-solution-result)   | Named value-array shape/dtype       |
| [`lcm.solver_api.SIMULATION_POLICY`](runtime_and_results.md#api-solution-result)  | Replay-policy schema key            |
| [`lcm.solver_api.DISSOLUTION_FLAG`](runtime_and_results.md#api-solution-result)   | Dissolution-flag schema key         |
| [`lcm.solver_api.EGM_CONTINUATION`](runtime_and_results.md#api-solution-result)   | Continuation schema key             |
| [`lcm.solver_api.SOLVER_DIAGNOSTICS`](runtime_and_results.md#api-solution-result) | Diagnostics schema key              |
| [`lcm.solver_api.ContinuationArtifact`](custom_solvers.md)                        | Keyed rolling-continuation protocol |
| [`lcm.solver_api.ReplayMode`](custom_solvers.md)                                  | How a regime's decision is obtained |
| [`lcm.solver_api.ReplayRoute`](custom_solvers.md)                                 | A regime's declared replay route    |

## Collective regimes and value-dependent choice

| Public name                                                                            | Canonical documentation                  |
| -------------------------------------------------------------------------------------- | ---------------------------------------- |
| [`lcm.CollectiveUtility`](collective_regimes.md#api-collective-utility)                | One utility per stakeholder, one action  |
| [`lcm.ParetoObjective`](collective_regimes.md#api-pareto-objective)                    | Weighted household objective             |
| [`lcm.ValueDependentConstraint`](collective_regimes.md#api-value-dependent-constraint) | Constraint that reads stakeholder values |
| [`lcm.ValueDependentTransition`](collective_regimes.md#api-value-dependent-transition) | Gated transition to a target regime      |
| [`lcm.StakeholderRoute`](collective_regimes.md#api-stakeholder-route)                  | One source stakeholder's route across it |
| [`lcm.ProjectedRegimeValue`](collective_regimes.md#api-projected-regime-value)         | Another regime's value, projected        |

## Solvers and configurations

| Public name                                                                                | Canonical documentation         |
| ------------------------------------------------------------------------------------------ | ------------------------------- |
| [`lcm.solvers.GridSearch`](solvers.md#api-grid-search)                                     | General grid search             |
| [`lcm.solvers.EGM`](solvers.md#api-egm)                                                    | Plain endogenous-grid method    |
| [`lcm.solvers.DCEGM`](solvers.md#api-dcegm)                                                | General-resources EGM           |
| [`lcm.solvers.NEGM`](solvers.md#api-negm)                                                  | Nested DCEGM                    |
| [`lcm.solvers.NBEGM`](solvers.md#api-nbegm)                                                | Declared non-convex-budget EGM  |
| [`lcm.solvers.NNBEGM`](solvers.md#api-nnbegm)                                              | Nested NBEGM                    |
| [`lcm.solvers.EnvelopeConfig`](envelopes.md)                                               | DCEGM envelope union            |
| [`lcm.solvers.ExactEnvelope`](envelopes.md#api-exact-envelope-availability)                | Certified envelope              |
| [`lcm.solvers.FUESEnvelope`](envelopes.md#api-approximate-envelope-backends)               | FUES configuration              |
| [`lcm.solvers.RFCEnvelope`](envelopes.md#api-approximate-envelope-backends)                | RFC configuration               |
| [`lcm.solvers.LTMEnvelope`](envelopes.md#api-approximate-envelope-backends)                | LTM configuration               |
| [`lcm.solvers.MSSEnvelope`](envelopes.md#api-approximate-envelope-backends)                | MSS configuration               |
| [`lcm.solvers.OuterSearch`](outer_search.md#api-outer-search)                              | Outer-search marker             |
| [`lcm.solvers.FiniteOuterGrid`](outer_search.md#api-finite-outer-grid)                     | Finite search                   |
| [`lcm.solvers.AdaptiveOuterMesh`](outer_search.md#api-adaptive-outer-mesh)                 | Adaptive continuous search      |
| [`lcm.solvers.OuterBranchAggregator`](outer_search.md#api-branch-aggregation)              | Branch-aggregation marker       |
| [`lcm.solvers.DeterministicOuterMaximum`](outer_search.md#api-deterministic-outer-maximum) | Hard maximum                    |
| [`lcm.solvers.UniformObservedFixedCost`](outer_search.md#api-uniform-observed-fixed-cost)  | Analytic fixed-cost integration |
| [`lcm.solvers.BranchAggregateResult`](outer_search.md#api-uniform-observed-fixed-cost)     | Aggregation output              |

The remaining exported solver infrastructure is the out-of-tree solver surface: a solver
can be written against these names without importing anything private. It stays
experimental until persistence, durable identity, a conformance suite, and a versioning
policy exist — see [Custom solvers](custom_solvers.md).

| Public name                                                      | Canonical documentation              |
| ---------------------------------------------------------------- | ------------------------------------ |
| [`lcm.solvers.Solver`](custom_solvers.md)                        | Solver base contract                 |
| [`lcm.solvers.OneMarginSolver`](custom_solvers.md)               | One-liquid-margin marker             |
| [`lcm.solvers.TwoMarginSolver`](custom_solvers.md)               | Nested-margin marker                 |
| [`lcm.solvers.SolverBuildContext`](custom_solvers.md)            | Per-regime build context             |
| [`lcm.solvers.SolutionKernels`](custom_solvers.md)               | Per-period kernel bundle             |
| [`lcm.solvers.PeriodKernel`](custom_solvers.md)                  | One period's execution protocol      |
| [`lcm.solvers.CoreProgram`](custom_solvers.md)                   | One declared compiled program        |
| [`lcm.solvers.CoreBuildContext`](custom_solvers.md)              | Argument-builder input               |
| [`lcm.solvers.CoreExecutionRequirements`](custom_solvers.md)     | Declared execution requirements      |
| [`lcm.solvers.CoreExecutionDisposition`](custom_solvers.md)      | Planned or deliberately dense        |
| [`lcm.solvers.ProgramScope`](custom_solvers.md)                  | Retention scope of a program         |
| [`lcm.solvers.StreamableProductAxis`](custom_solvers.md)         | Declared streamable action axis      |
| [`lcm.solvers.ReductionSemantics`](custom_solvers.md)            | Reduction a streamed axis performs   |
| [`lcm.solvers.OutputRole`](custom_solvers.md)                    | Value and dissolution-flag roles     |
| [`lcm.solvers.StateAxesLeading`](custom_solvers.md)              | Parametrized output-placement role   |
| [`lcm.solvers.StateActionSpace`](custom_solvers.md)              | Per-period state and action arrays   |
| [`lcm.solvers.KernelOutput`](custom_solvers.md)                  | What a period kernel returns         |
| [`lcm.solvers.ArtifactKey`](custom_solvers.md)                   | Versioned artifact identity          |
| [`lcm.solvers.EGM_CONTINUATION`](custom_solvers.md)              | Continuation schema key              |
| [`lcm.solvers.SIMULATION_POLICY`](custom_solvers.md)             | Replay-policy schema key             |
| [`lcm.solvers.DISSOLUTION_FLAG`](custom_solvers.md)              | Dissolution-flag schema key          |
| [`lcm.solvers.SOLVER_DIAGNOSTICS`](custom_solvers.md)            | Diagnostics schema key               |
| [`lcm.solvers.ContinuationArtifact`](custom_solvers.md)          | Keyed rolling-continuation protocol  |
| [`lcm.solvers.ContinuationSpec`](custom_solvers.md)              | Template and key of a continuation   |
| [`lcm.solvers.EGMContinuationSpec`](custom_solvers.md)           | EGM carry template and layout        |
| [`lcm.solvers.EGMContinuationLayout`](custom_solvers.md)         | How a reading parent interprets rows |
| [`lcm.solvers.ReplayMode`](custom_solvers.md)                    | How a regime's decision is obtained  |
| [`lcm.solvers.ReplayRoute`](custom_solvers.md)                   | A regime's declared replay route     |
| [`lcm.solvers.period_to_continuation_target`](custom_solvers.md) | Target a period carries into         |
| [`lcm.solvers.target_period_grid`](custom_solvers.md)            | A target's grid in one period        |
| [`lcm.solvers.union_free_params`](custom_solvers.md)             | Free params of a regime and targets  |
| [`lcm.solvers.union_fixed_params`](custom_solvers.md)            | Fixed params of a regime and targets |
