---
title: Public API index
---

# Public API index

This curated index covers the names exported from the top-level `lcm` and `lcm.solvers`
namespaces. It is a navigation-completeness check, not an exhaustive semantic
specification: submodule-only public objects, constructor fields, behavioral contracts,
and the correctness of destinations require page-specific documentation and tests.

## Model and economic declarations

| Public name                                                                                   | Canonical documentation                 |
| --------------------------------------------------------------------------------------------- | --------------------------------------- |
| [`lcm.Model`](model_and_regime.md#model)                                                      | Model assembly and execution            |
| [`lcm.Regime`](model_and_regime.md#regime)                                                    | General regime                          |
| [`lcm.ConsumptionSavingsRegime`](consumption_savings.md#consumptionsavingsregime)             | One liquid margin                       |
| [`lcm.NestedConsumptionSavingsRegime`](consumption_savings.md#nestedconsumptionsavingsregime) | Liquid plus outer margin                |
| [`lcm.LiquidMargin`](consumption_savings.md#liquidmargin)                                     | Liquid roles                            |
| [`lcm.OuterContinuousMargin`](consumption_savings.md#outercontinuousmargin)                   | Outer roles                             |
| [`lcm.NetOfAdjustmentCost`](consumption_savings.md#liquidmargin)                              | Resources composition                   |
| [`lcm.post_decision_lower_bound`](consumption_savings.md#post-decision-lower-bounds)          | Checkable borrowing bound               |
| [`lcm.outer_unchanged`](consumption_savings.md#outercontinuousmargin)                         | Identity no-adjustment sentinel         |
| [`lcm.cash_on_hand_with_subsidy`](case_pieces.md)                                             | Supported case-piece fixed form         |
| [`lcm.liquid_law_from_resources`](consumption_savings.md)                                     | Conventional liquid law                 |
| [`lcm.liquid_law_from_savings`](consumption_savings.md)                                       | Conventional savings-written liquid law |
| [`lcm.ExtremeValueTasteShocks`](model_and_regime.md#regime)                                   | EV1 discrete-choice shocks              |

## Grids, categories, and processes

| Public name                                                                         | Canonical documentation              |
| ----------------------------------------------------------------------------------- | ------------------------------------ |
| [`lcm.AgeGrid`](grids_and_processes.md#lifecycle-and-categorical-grids)             | Lifecycle grid                       |
| [`lcm.categorical`](grids_and_processes.md#lifecycle-and-categorical-grids)         | Category declaration                 |
| [`lcm.DiscreteGrid`](grids_and_processes.md#lifecycle-and-categorical-grids)        | Discrete outcome grid                |
| [`lcm.LinSpacedGrid`](grids_and_processes.md#continuous-grids)                      | Linear grid                          |
| [`lcm.LogSpacedGrid`](grids_and_processes.md#continuous-grids)                      | Log grid                             |
| [`lcm.IrregSpacedGrid`](grids_and_processes.md#continuous-grids)                    | Explicit irregular grid              |
| [`lcm.GridBreakpoint`](grids_and_processes.md#continuous-grids)                     | Piecewise-grid boundary              |
| [`lcm.PiecewiseLinSpacedGrid`](grids_and_processes.md#continuous-grids)             | Piecewise linear grid                |
| [`lcm.PiecewiseLogSpacedGrid`](grids_and_processes.md#continuous-grids)             | Piecewise log grid                   |
| [`lcm.UniformIIDProcess`](grids_and_processes.md#stochastic-processes)              | IID uniform process                  |
| [`lcm.NormalIIDProcess`](grids_and_processes.md#stochastic-processes)               | IID normal process                   |
| [`lcm.LogNormalIIDProcess`](grids_and_processes.md#stochastic-processes)            | IID log-normal process               |
| [`lcm.NormalMixtureIIDProcess`](grids_and_processes.md#stochastic-processes)        | IID normal mixture                   |
| [`lcm.TauchenAR1Process`](grids_and_processes.md#stochastic-processes)              | Tauchen AR(1)                        |
| [`lcm.RouwenhorstAR1Process`](grids_and_processes.md#stochastic-processes)          | Rouwenhorst AR(1)                    |
| [`lcm.TauchenNormalMixtureAR1Process`](grids_and_processes.md#stochastic-processes) | Mixture AR(1)                        |
| [`lcm.StateConditioned`](grids_and_processes.md#stochastic-processes)               | State-conditioned process parameters |

## Transitions, phases, and structured declarations

| Public name                                                       | Canonical documentation            |
| ----------------------------------------------------------------- | ---------------------------------- |
| [`lcm.MarkovTransition`](transitions.md#state-transitions)        | Stochastic transition wrapper      |
| [`lcm.JointTransition`](transitions.md#joint-transitions)         | Shared-draw joint law              |
| [`lcm.fixed_transition`](transitions.md#state-transitions)        | Identity law                       |
| [`lcm.AgeSpecializedFunction`](transitions.md#age-specialization) | Age-varying function               |
| [`lcm.AgeSpecializedGrid`](transitions.md#age-specialization)     | Age-varying grid                   |
| [`lcm.Phased`](transitions.md#solve-and-simulation-phases)        | Solve/simulate variants            |
| [`lcm.Condition`](conditions.md)                                  | Structured Boolean expression      |
| [`lcm.ref`](conditions.md#syntax)                                 | Named reference                    |
| [`lcm.implies`](conditions.md#syntax)                             | Conditional requirement            |
| [`lcm.case_boundary`](case_pieces.md#boundary-predicate)          | Structured case boundary           |
| [`lcm.piece`](case_pieces.md#piece-formulas)                      | Piece decorator                    |
| [`lcm.smooth_helper`](case_pieces.md#piece-formulas)              | Reviewed smooth-helper attestation |
| [`lcm.affine_breakpoint`](piecewise_affine.md#affine-breakpoint)  | Schedule breakpoint                |
| [`lcm.piecewise_affine`](piecewise_affine.md)                     | Piecewise-affine schedule          |

## Preferences, results, and persistence

| Public name                                                             | Canonical documentation        |
| ----------------------------------------------------------------------- | ------------------------------ |
| [`lcm.LinearAggregator`](../methods/preferences.md)                     | Linear Koopmans form           |
| [`lcm.CESAggregator`](../methods/preferences.md)                        | CES Koopmans form              |
| [`lcm.CertaintyEquivalent`](../methods/preferences.md)                  | Certainty-equivalent contract  |
| [`lcm.LinearExpectation`](../methods/preferences.md)                    | Linear lottery reduction       |
| [`lcm.PowerMean`](../methods/preferences.md)                            | Power-mean reduction           |
| [`lcm.QuasiArithmeticMean`](../methods/preferences.md)                  | Quasi-arithmetic reduction     |
| [`lcm.SimulationResult`](runtime_and_results.md#simulationresult)       | Deferred simulation result     |
| [`lcm.save_solution`](runtime_and_results.md#standalone-persistence)    | Save value functions           |
| [`lcm.load_solution`](runtime_and_results.md#standalone-persistence)    | Load value functions           |
| [`lcm.SolveSnapshot`](runtime_and_results.md#standalone-persistence)    | Solve diagnostic snapshot      |
| [`lcm.SimulateSnapshot`](runtime_and_results.md#standalone-persistence) | Simulation diagnostic snapshot |
| [`lcm.load_snapshot`](runtime_and_results.md#standalone-persistence)    | Load diagnostic snapshot       |
| [`lcm.__version__`](public_api.md)                                      | Installed pylcm version        |

## Explicit public submodule surfaces

Most user-facing names are re-exported from `lcm`. Two deliberately public submodule
surfaces remain outside that top-level namespace:

| Public name                                                                | Canonical documentation         |
| -------------------------------------------------------------------------- | ------------------------------- |
| [`lcm.params.MappingLeaf`](../user_guide/parameters.md)                    | Mapping parameter leaf          |
| [`lcm.params.SequenceLeaf`](../user_guide/parameters.md)                   | Sequence parameter leaf         |
| [`lcm.params.UserMappingLeaf`](../user_guide/parameters.md)                | User mapping parameter leaf     |
| [`lcm.params.UserSequenceLeaf`](../user_guide/parameters.md)               | User sequence parameter leaf    |
| [`lcm.params.as_leaf`](../user_guide/parameters.md)                        | Explicit parameter-leaf wrapper |
| [`lcm.koopmans_aggregation.KoopmansAggregator`](../methods/preferences.md) | Koopmans-form base contract     |

## Collective regimes

| Public name                                                                           | Canonical documentation              |
| ------------------------------------------------------------------------------------- | ------------------------------------ |
| [`lcm.SamePeriodRef`](collective_regimes.md#participation-constraints)                | Same-period outside-option value     |
| [`lcm.EdgeLeg`](collective_regimes.md#gated-transitions-across-stakeholder-layouts)   | One stakeholder's gated continuation |
| [`lcm.GatedEdge`](collective_regimes.md#gated-transitions-across-stakeholder-layouts) | Consent/dissolution edge             |

## Solvers and configurations

| Public name                                                                          | Canonical documentation         |
| ------------------------------------------------------------------------------------ | ------------------------------- |
| [`lcm.solvers.GridSearch`](solvers.md#gridsearch)                                    | General grid search             |
| [`lcm.solvers.EGM`](solvers.md#egm)                                                  | Plain endogenous-grid method    |
| [`lcm.solvers.DCEGM`](solvers.md#dcegm)                                              | General-resources EGM           |
| [`lcm.solvers.NEGM`](solvers.md#negm)                                                | Nested DCEGM                    |
| [`lcm.solvers.NBEGM`](solvers.md#nbegm)                                              | Declared non-convex-budget EGM  |
| [`lcm.solvers.NNBEGM`](solvers.md#nnbegm)                                            | Nested NBEGM                    |
| [`lcm.solvers.EnvelopeConfig`](envelopes.md)                                         | DCEGM envelope union            |
| [`lcm.solvers.ExactEnvelope`](envelopes.md#exact-envelope-availability)              | Certified envelope              |
| [`lcm.solvers.FUESEnvelope`](envelopes.md#approximate-backends)                      | FUES configuration              |
| [`lcm.solvers.RFCEnvelope`](envelopes.md#approximate-backends)                       | RFC configuration               |
| [`lcm.solvers.LTMEnvelope`](envelopes.md#approximate-backends)                       | LTM configuration               |
| [`lcm.solvers.MSSEnvelope`](envelopes.md#approximate-backends)                       | MSS configuration               |
| [`lcm.solvers.OuterSearch`](outer_search.md#outer-search)                            | Outer-search marker             |
| [`lcm.solvers.FiniteOuterGrid`](outer_search.md#finiteoutergrid)                     | Finite search                   |
| [`lcm.solvers.AdaptiveOuterMesh`](outer_search.md#adaptiveoutermesh)                 | Adaptive continuous search      |
| [`lcm.solvers.LegacyGoldenSection`](outer_search.md#legacygoldensection)             | Historical search               |
| [`lcm.solvers.OuterBranchAggregator`](outer_search.md#branch-aggregation)            | Branch-aggregation marker       |
| [`lcm.solvers.DeterministicOuterMaximum`](outer_search.md#deterministicoutermaximum) | Hard maximum                    |
| [`lcm.solvers.UniformObservedFixedCost`](outer_search.md#uniformobservedfixedcost)   | Analytic fixed-cost integration |
| [`lcm.solvers.BranchAggregateResult`](outer_search.md#uniformobservedfixedcost)      | Aggregation output              |

The remaining exported solver infrastructure is currently contributor-facing, not a
supported out-of-tree extension seam:

| Public name                                           | Status                           |
| ----------------------------------------------------- | -------------------------------- |
| [`lcm.solvers.Solver`](custom_solvers.md)             | Contributor-facing base contract |
| [`lcm.solvers.OneMarginSolver`](custom_solvers.md)    | Contributor-facing marker        |
| [`lcm.solvers.TwoMarginSolver`](custom_solvers.md)    | Contributor-facing marker        |
| [`lcm.solvers.SolverBuildContext`](custom_solvers.md) | Contributor-facing build context |
| [`lcm.solvers.SolutionKernels`](custom_solvers.md)    | Contributor-facing kernel bundle |
