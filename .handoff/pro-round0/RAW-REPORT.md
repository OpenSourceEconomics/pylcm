{
  "schema_version": "3.2",
  "workflow_class": "computational-method",
  "mode": "audit",
  "round": 0,
  "repair_attempt": 0,
  "reviewer_turn_status": "complete",
  "baseline_commit": "7e6fff631856572c63ac63cad43cfdf2e8c29c5c",
  "audit_scope": {
    "scope_mode": "delta",
    "target": "Design consultation with a bounded audit of the Stage 3 blocked route (ExecutionConfig(invariant_block_widths={\"pref_type\": 1}), commits 266927d9..ec882386 on 7e6fff63, PR #486): decide whether and how to remove the two residual cold-only helper compiles of the blocked solve -- one jit(_select_view_blocks) per distinct stored shape read through a selected view (value_transfer.py) and one jit(_write_block) per distinct blocked output shape (invariant_blocks.py), neither growing with the number of types -- choosing among Option 1 (write-back inside the bound core through a donated full-size accumulator plus a start-offset operand), Option 2 (selection inside the bound core, slicing the stored value by the code operand, which on several devices puts a resharding collective inside the core while keeping the no-full-type-gather rule, changes Stage 2's \"selection is a planned transfer stage\" contract and moves transfer_workspace_bytes into the compiler reservation), or neither; and judge whether structural_resolution (re-run for every program on every solve, about 2 ms host time per program, mostly code-dependent) should be cached across warm calls on a params-independent key for both the blocked and the unblocked route. Return a decision with reasons; the invariants the chosen design must preserve (bitwise parity blocked vs unblocked at fp32 and fp64, donation and liveness, admission and footprint, certified corridors, multi-device layouts); the minimal changeset outline; and the tests that would show the fold red first.",
    "manifest_digest": "aad5686b9c73b03da0fce9f99b396a6e1b2a43c6a98e3f613b26dd2e753f90d6",
    "profile_contract_digest": "486c5c43025d84d22cd553d638a1bb3956810f19357213f980c34322018e8fa1",
    "base_ref": "7e6fff631856572c63ac63cad43cfdf2e8c29c5c",
    "head_ref": "ec882386ca8fe0717ed878b8eb8e7a74173c3df4",
    "changed_paths": [
      "CHANGES.md",
      "docs/user_guide/tuning.md",
      "src/_lcm/execution/core_program.py",
      "src/_lcm/execution/execution_plan.py",
      "src/_lcm/execution/invariant_blocks.py",
      "src/_lcm/execution/value_transfer.py",
      "src/_lcm/regime_building/invariant_blocking.py",
      "src/_lcm/regime_building/processing.py",
      "src/_lcm/solution/backward_induction.py",
      "src/_lcm/solution/contract.py",
      "src/_lcm/solution/grid_search.py",
      "src/_lcm/solution/period_replay.py",
      "src/lcm/execution.py",
      "src/lcm/model.py",
      "tests/candidate_certificate/direct_flow.py",
      "tests/candidate_certificate/sources.json",
      "tests/ci/ci-workloads.json",
      "tests/solution/test_invariant_blocking.py",
      "tests/test_continuous_assets_sharding.py"
    ],
    "dependency_closure": [
      {
        "path": "tests/test_models/independent_types.py",
        "entered_from": [
          "tests/solution/test_invariant_blocking.py"
        ],
        "reason": "oracle_dependency",
        "finding_ids": [
          "F1"
        ]
      },
      {
        "path": "src/_lcm/execution/value_views.py",
        "entered_from": [
          "src/_lcm/execution/value_transfer.py"
        ],
        "reason": "claim_dependency",
        "finding_ids": [
          "F1"
        ]
      },
      {
        "path": "tests/execution/test_value_views.py",
        "entered_from": [
          "src/_lcm/execution/value_transfer.py"
        ],
        "reason": "oracle_dependency",
        "finding_ids": [
          "F1"
        ]
      },
      {
        "path": "tests/test_continuous_transfer_admission.py",
        "entered_from": [
          "src/_lcm/solution/backward_induction.py"
        ],
        "reason": "oracle_dependency",
        "finding_ids": [
          "F1"
        ]
      }
    ],
    "required_profiles": [
      "fast",
      "certified"
    ],
    "profile_applicability": "required",
    "declared_budgets": "Grid-conditional exact value and argmax agreement for fp32 fast and fp64 certified; warm compile growth 0; no cold compile growth per type; current per-shape select/write helpers explicitly accepted; peak device memory no increase over A; fail-closed admission; sealed corridors and no full-type gather. Repair review triggers 5% warm / 10% compile or memory, noise floor 3%. Candidate-set error beyond the fixed grid is not supplied and is not assessed.",
    "environment": "Reviewer Python 3.13.5; project requires Python >=3.14. Native project execution not_run as instructed; no interpreter search, syntax rewrite or shim. Complete static source/AST/token/JSON analysis, including direct delta/dependency review. Six delivered Python artifacts were syntax-parsed; only the independent standard-library reference was executed. Supplied forced-host-device and GPU evidence retains its actual source revision and is not reviewer execution."
  },
  "ingestion": {
    "manifest_digest": "aad5686b9c73b03da0fce9f99b396a6e1b2a43c6a98e3f613b26dd2e753f90d6",
    "manifest_files_declared": 53,
    "manifest_files_fully_read": 49,
    "manifest_paths_left_as_reference": [
      "project/.pre-commit-config.yaml",
      "project/AGENTS.md",
      "project/README.md",
      "project/pyproject.toml"
    ],
    "unreadable_or_truncated": [],
    "required_read_paths_unread": [],
    "coverage_complete_for_verdict": true,
    "reading_order": [
      "Control record and manifest; all ZIP and manifest integrity checks.",
      "Profile/return contracts and authoritative complete diff; full-file static AST/token/JSON traversal and base/head changed-symbol mapping.",
      "Changed source and runtime dependency seams: blocked declarations, views, transfers, ownership/admission, core resolution, GridSearch, result assembly, model lifecycle.",
      "Supplied reports, compile logs/drivers, full profile tables and complete benchmark-summary JSON; independent aggregation and version segregation.",
      "Regression fixtures, certificate/CI data, defect casebook; full generated-data comparisons and available-source digest checks.",
      "Candidate patches, references and tests; coverage/evidence records and reply validation."
    ]
  },
  "verdict": "serious_gap",
  "closure_status": "implementation_required",
  "verdict_explanation": "Neither helper fold is justified as the next change: the supplied contract accepts both residual per-shape cold helpers. Two supported target-owned source defects survive: aligned selected views are misclassified in ownership/admission, and singleton blocked cores can bypass selected-read planning. The package supplies minimal source patches and native red-first regressions, plus a safe immutable-blueprint cache contract for both routes. Native verification is not_run under the expressly stated Python mismatch, so the candidate repairs are not closed. Historical GPU evidence is retained with its real revision rather than promoted to current-head acceptance.",
  "audit_confidence": "moderate",
  "root_causes": [
    {
      "id": "R1",
      "defect_class_id": "selected-layout-is-not-owner-alias",
      "title": "Fresh aligned selections receive stored-owner credit or lose allocation charges",
      "finding_ids": [
        "F1"
      ],
      "repair_attempt": 0,
      "diagnosis": "Transport alignment after selecting an invariant coordinate is confused with receiving the original stored allocation. Four ownership/admission predicates still use ALIGNED_LOCAL instead of delivers_stored_buffer.",
      "counterexample_class": "A selected value view on the same placement, with its parent still resident; compiler-live and compiler-pruned reads, shared and unshared consumers, any selected axis position and K>1.",
      "repair_invariant": "Only delivers_stored_buffer may earn full stored-owner credit. All fresh selected destinations and declared stage workspace remain represented, including pruned operands and endpoint devices.",
      "repair_strategy": "local_patch",
      "closure_criterion": "RT1 is red at the stated ownership/admission assertions on head and green with P1 at both precisions; ordinary unsliced-aligned controls remain green; native owner/liveness, budget, multi-device and sealed-corridor gates pass.",
      "expected_resource_delta": "No changed Bellman arithmetic or helper-count policy. Corrected conservative admission may reduce selected widths and increase runtime; current physical resource delta is not measured. Do not waive omitted charges to keep prior timings.",
      "escalation_required": false
    },
    {
      "id": "R2",
      "defect_class_id": "bound-scalar-core-skips-value-view-planning",
      "title": "A bound no-axis GridSearch core omits its mandatory selected read",
      "finding_ids": [
        "F2"
      ],
      "repair_attempt": 0,
      "diagnosis": "GridSearch selects execution disposition from the existence of tiling/reduction axes alone, although invariant binding independently requires selected-view planning.",
      "counterexample_class": "A fixed discrete type is the only nontrivial state coordinate; after width-one binding both local state and action products have extent one, and the child value retains the type.",
      "repair_invariant": "Every bound continuation is supplied in its declared selected representation even when arithmetic is dense and no width axis exists.",
      "repair_strategy": "local_patch",
      "closure_criterion": "RT2 structural and numeric witnesses plus MT2 permutations/scales pass at fp32/fp64, against the literal one-step Bellman equation; nonblocked and ordinary dense controls stay unchanged; native admission/certification pass.",
      "expected_resource_delta": "Newly correct selection on this boundary may add the accepted per-shape helper, but no per-type specialization. Dense arithmetic and the ordinary unblocked route remain unchanged; measured resource delta not_run.",
      "escalation_required": false
    },
    {
      "id": "R3",
      "defect_class_id": "repeated-structural-reconstruction-design",
      "title": "Cache-boundary design for repeated structural resolution (nonblocking optimization work)",
      "finding_ids": [],
      "repair_attempt": 0,
      "diagnosis": "The repeated resolution function mixes reusable structural recipes with current numeric arguments, donation/liveness state and budget-dependent candidate processing. Wholesale memoization is unsafe.",
      "counterexample_class": "Repeated valid same-schema calls and equal-schema type families, including changed numerical parameter values; changed runtime support, layout or ownership must not reuse stale decisions.",
      "repair_invariant": "Cache immutable schema-certified blueprints only; always bind and validate current data and independently admit current memory.",
      "repair_strategy": "architecture_change",
      "closure_criterion": "RT3 and the cache safety/retention/schema/resource matrix in report.md pass for blocked and unblocked GridSearch; current-head paired resources meet the supplied contract or require an explicit workload-specific decision.",
      "expected_resource_delta": "Expected reduction in host structural work and repeated Python-object creation; bounded small metadata cache. No numerical speedup estimate and no elimination of select/write compiles promised.",
      "escalation_required": false
    }
  ],
  "findings": [
    {
      "id": "F1",
      "root_cause_id": "R1",
      "type": "code",
      "primary_type": "RESOURCE",
      "severity": "serious",
      "severity_scope": "Supported Stage 3 route, not repository-wide severity.",
      "status": "open",
      "blocks_clean": true,
      "title": "Selected aligned views are misclassified in four ownership/admission predicates",
      "location_text": "protocol/profile-contract.yaml: fail-closed admission and exact grid-conditional parity",
      "location_code": "src/_lcm/solution/backward_induction.py:3074,3274,3300,3314; src/_lcm/execution/value_transfer.py:delivers_stored_buffer,stages,cost",
      "claim": "A supported invariant-bound GridSearch call preserves its selected-value representation, correct ownership/admission, and the ordinary numerical result.",
      "issue": "ALIGNED_LOCAL can describe a fresh selected block; using it for full-owner credit and scratch/pruned-input exclusions contradicts delivers_stored_buffer and the selection stage allocation contract.",
      "classification": "code_text_mismatch",
      "hinge_step": true,
      "in_supported_envelope": true,
      "reachable": true,
      "trigger_path": "Model(invariant_block_widths={pref_type:1}, devices=(0,), finite device_memory_bytes).solve -> bound program -> selected ALIGNED_LOCAL transfer -> period/candidate residency accounting.",
      "witness": "Normal independent_types solve reaches selected compact compiler inputs while full continuation owners remain live. RT1 also isolates a valid pruned selected input: owner C plus destination B plus declared selection scratch B must be reserved; head skips the fresh-selection predicates.",
      "counterexample_class_id": "R1",
      "evidence": "Exact source/contract contradiction; candidate native regression delivered but not executed. The independent literal reference was executed without importing pylcm/JAX. See report.md section 2 and the artifact IDs.",
      "scope_ownership": {
        "category": "NEWLY_REACHABLE_BY_TARGET",
        "target_changed_failing_path": false,
        "target_newly_exposes_path": true,
        "target_uses_as_runtime_dependency": true,
        "target_uses_as_oracle_or_benchmark": false,
        "target_claim_invalidated": true,
        "rationale": "The baseline snapshot has no public Stage 3 blocking request. The target wires selected reads into this path. This is not a finding on unrelated private-object misuse; the public witness constructs a documented model."
      },
      "base_head_evidence": {
        "comparison_possible": false,
        "reason_if_impossible": "Native project execution was expressly excluded: Python 3.13.5 here versus required >=3.14; the shipped Git snapshot is also partial. No interpreter/syntax workaround was attempted. Original SHAs are mapped by synthetic snapshot subjects.",
        "environment": "Static original baseline/head snapshot comparison; native matched dtype/backend comparison not_run.",
        "command": "See evidence/commands.md for native witness and closure commands.",
        "reproducer_path": "tests/solution/test_audit486_selected_view_admission.py",
        "base_reproduces": null,
        "head_reproduces": null,
        "base_result": "not_run; source baseline lacks the blocking API and does not feed selected views through this Stage 3 path. The ordinary route is the intended control, not an observed pass.",
        "head_result": "not_run; source path contradicts the declared selected-view/ownership contract.",
        "behavior_delta": "newly reachable"
      },
      "level_effect": "The planner can underrepresent required resident storage and admit an unsafe width/budget. Actual OOM and production numerical differences are not claimed.",
      "decision_effect": "No measured policy or simulated-moment change is claimed; admission/completion or the value representation itself is affected.",
      "affects_reported_result": "unknown",
      "numerical_effect": {
        "max_value_regret": null,
        "q95_value_regret": null,
        "policy_disagreement_mass": null,
        "boundary_shift_grid_cells": null,
        "moment_or_objective_change": null,
        "parameter_or_welfare_change": null,
        "unresolved_or_certification_rate": null,
        "measured_under": "not measured natively; static proof and literal equations only",
        "candidate_set_error": {
          "profile": "fast, certified",
          "eps_candidate": "unknown",
          "assessed_by": "none",
          "command": null,
          "why_unknown": "No refinement/candidate-set budget supplied; this audit concerns the same finite candidate grid and representation/admission, not a full-action optimum.",
          "on_unavailable_applied": "report_unknown",
          "inherited_bound": {
            "value": null,
            "source": null,
            "source_digest": null,
            "assumptions": null,
            "applies_to": null
          }
        }
      },
      "resource_effect": {
        "measured": false,
        "benchmark_command": null,
        "base_warm_seconds": null,
        "head_warm_seconds": null,
        "base_compile_seconds": null,
        "head_compile_seconds": null,
        "base_peak_host_gib": null,
        "head_peak_host_gib": null,
        "base_peak_device_gib": null,
        "head_peak_device_gib": null,
        "cache_or_shape_growth": "No helper fold or new per-type numerical specialization in the candidate patch.",
        "ci_viable": null
      },
      "profile_blocking": {
        "fast": true,
        "balanced": false,
        "certified": true,
        "rationale": "Required fast/certified fail-closed memory and selected-input semantic contracts apply on this supported path. Balanced is not a required/supplied profile."
      },
      "target_blocking": {
        "blocks_target": true,
        "blocking_route": "NEWLY_REACHABLE",
        "required_merge_profiles": [
          "fast",
          "certified"
        ],
        "rationale": "The target newly introduces this public blocked route; exact static contradiction meets the ownership gate despite explicitly not_run native execution."
      },
      "repairability": "local",
      "suggestion": "Only delivers_stored_buffer may earn full stored-owner credit. All fresh selected destinations and declared stage workspace remain represented, including pruned operands and endpoint devices.",
      "smallest_repair": "Apply P1 source change, install and run the delivered regressions, regenerate touched seals/corridors using the native tools.",
      "proposed_disposition": "FIX_IN_TARGET",
      "issue_and_exit": {
        "eligible": false,
        "issue_filed": false,
        "issue_url": null,
        "issue_draft_path": null,
        "evidence_directory": null,
        "likely_provenance": null,
        "provenance_certainty": "proven",
        "bisection_command": null,
        "followup_repair": "",
        "followup_tests": [],
        "non_blocking_rationale": "",
        "would_become_blocking_if": []
      },
      "maintainer_decision": {
        "status": "PENDING",
        "decided_by": null,
        "decided_at": null,
        "rationale": "No waiver or acceptance is asserted by the reviewer."
      },
      "patch_ids": [
        "P1"
      ],
      "oracle_ids": [
        "OR1"
      ],
      "test_ids": [
        "RT1"
      ],
      "mutation_ids": [],
      "blocking_artifacts": [],
      "verification": {
        "status": "not_run",
        "basis": "static_proof",
        "commands": [],
        "result": "Native red/green not_run by the requested environment escape. Source changes apply cleanly; syntax parsing and independent stdlib equations do not establish native closure."
      },
      "confidence": "high"
    },
    {
      "id": "F2",
      "root_cause_id": "R2",
      "type": "code",
      "primary_type": "SEMANTIC",
      "severity": "serious",
      "severity_scope": "Supported Stage 3 route, not repository-wide severity.",
      "status": "open",
      "blocks_clean": true,
      "title": "One-cell/one-action blocked cores bypass selected continuation planning",
      "location_text": "protocol/profile-contract.yaml: fail-closed admission and exact grid-conditional parity",
      "location_code": "src/_lcm/solution/grid_search.py:280-303,400-423; src/_lcm/regime_building/processing.py:3417-3423; src/_lcm/solution/backward_induction.py:6097-6139",
      "claim": "A supported invariant-bound GridSearch call preserves its selected-value representation, correct ownership/admission, and the ordinary numerical result.",
      "issue": "No width axes makes the bound core DENSE, so prepare does not plan a selected read although Q continuation construction already removed the invariant coordinate.",
      "classification": "code_text_mismatch",
      "hinge_step": true,
      "in_supported_envelope": true,
      "reachable": true,
      "trigger_path": "Public two-period model with only fixed pref_type, one action and typed terminal -> width-one binding -> requirements.axes=() -> DENSE -> whole typed continuation supplied to a bound reader.",
      "witness": "For flow=(1,2,3), terminal=(4,8,12), beta=1/2 the literal values are (3,6,9). RT2 first asserts the absent PLANNED disposition, then checks literal output; the exact erroneous native outcome is not asserted without execution.",
      "counterexample_class_id": "R2",
      "evidence": "Exact source/contract contradiction; candidate native regression delivered but not executed. The independent literal reference was executed without importing pylcm/JAX. See report.md section 2 and the artifact IDs.",
      "scope_ownership": {
        "category": "NEWLY_REACHABLE_BY_TARGET",
        "target_changed_failing_path": true,
        "target_newly_exposes_path": true,
        "target_uses_as_runtime_dependency": true,
        "target_uses_as_oracle_or_benchmark": false,
        "target_claim_invalidated": true,
        "rationale": "The baseline snapshot has no public Stage 3 blocking request. The target wires selected reads into this path. This is not a finding on unrelated private-object misuse; the public witness constructs a documented model."
      },
      "base_head_evidence": {
        "comparison_possible": false,
        "reason_if_impossible": "Native project execution was expressly excluded: Python 3.13.5 here versus required >=3.14; the shipped Git snapshot is also partial. No interpreter/syntax workaround was attempted. Original SHAs are mapped by synthetic snapshot subjects.",
        "environment": "Static original baseline/head snapshot comparison; native matched dtype/backend comparison not_run.",
        "command": "See evidence/commands.md for native witness and closure commands.",
        "reproducer_path": "tests/solution/test_audit486_scalar_block.py",
        "base_reproduces": null,
        "head_reproduces": null,
        "base_result": "not_run; source baseline lacks the blocking API and does not feed selected views through this Stage 3 path. The ordinary route is the intended control, not an observed pass.",
        "head_result": "not_run; source path contradicts the declared selected-view/ownership contract.",
        "behavior_delta": "newly reachable"
      },
      "level_effect": "The advertised supported scalar boundary can receive a wrong-shaped/wrong-type continuation, jeopardizing values or completion. The single-action witness makes no policy-change claim.",
      "decision_effect": "No measured policy or simulated-moment change is claimed; admission/completion or the value representation itself is affected.",
      "affects_reported_result": "unknown",
      "numerical_effect": {
        "max_value_regret": null,
        "q95_value_regret": null,
        "policy_disagreement_mass": null,
        "boundary_shift_grid_cells": null,
        "moment_or_objective_change": null,
        "parameter_or_welfare_change": null,
        "unresolved_or_certification_rate": null,
        "measured_under": "not measured natively; static proof and literal equations only",
        "candidate_set_error": {
          "profile": "fast, certified",
          "eps_candidate": "unknown",
          "assessed_by": "none",
          "command": null,
          "why_unknown": "No refinement/candidate-set budget supplied; this audit concerns the same finite candidate grid and representation/admission, not a full-action optimum.",
          "on_unavailable_applied": "report_unknown",
          "inherited_bound": {
            "value": null,
            "source": null,
            "source_digest": null,
            "assumptions": null,
            "applies_to": null
          }
        }
      },
      "resource_effect": {
        "measured": false,
        "benchmark_command": null,
        "base_warm_seconds": null,
        "head_warm_seconds": null,
        "base_compile_seconds": null,
        "head_compile_seconds": null,
        "base_peak_host_gib": null,
        "head_peak_host_gib": null,
        "base_peak_device_gib": null,
        "head_peak_device_gib": null,
        "cache_or_shape_growth": "No helper fold or new per-type numerical specialization in the candidate patch.",
        "ci_viable": null
      },
      "profile_blocking": {
        "fast": true,
        "balanced": false,
        "certified": true,
        "rationale": "Required fast/certified fail-closed memory and selected-input semantic contracts apply on this supported path. Balanced is not a required/supplied profile."
      },
      "target_blocking": {
        "blocks_target": true,
        "blocking_route": "NEWLY_REACHABLE",
        "required_merge_profiles": [
          "fast",
          "certified"
        ],
        "rationale": "The target newly introduces this public blocked route; exact static contradiction meets the ownership gate despite explicitly not_run native execution."
      },
      "repairability": "local",
      "suggestion": "Every bound continuation is supplied in its declared selected representation even when arithmetic is dense and no width axis exists.",
      "smallest_repair": "Apply P2 source change, install and run the delivered regressions, regenerate touched seals/corridors using the native tools.",
      "proposed_disposition": "FIX_IN_TARGET",
      "issue_and_exit": {
        "eligible": false,
        "issue_filed": false,
        "issue_url": null,
        "issue_draft_path": null,
        "evidence_directory": null,
        "likely_provenance": null,
        "provenance_certainty": "proven",
        "bisection_command": null,
        "followup_repair": "",
        "followup_tests": [],
        "non_blocking_rationale": "",
        "would_become_blocking_if": []
      },
      "maintainer_decision": {
        "status": "PENDING",
        "decided_by": null,
        "decided_at": null,
        "rationale": "No waiver or acceptance is asserted by the reviewer."
      },
      "patch_ids": [
        "P2"
      ],
      "oracle_ids": [
        "OR1"
      ],
      "test_ids": [
        "RT2"
      ],
      "mutation_ids": [
        "MT2"
      ],
      "blocking_artifacts": [],
      "verification": {
        "status": "not_run",
        "basis": "static_proof",
        "commands": [],
        "result": "Native red/green not_run by the requested environment escape. Source changes apply cleanly; syntax parsing and independent stdlib equations do not establish native closure."
      },
      "confidence": "high"
    }
  ],
  "hardening_notes": [
    {
      "id": "H1",
      "title": "Do not promote historical resource receipts to current-head acceptance",
      "scope": "Evidence limitation, not an additional native-reproduced current-head blocker.",
      "in_supported_envelope": true,
      "reachable": true,
      "affects_reported_result": "Resource claims only; no reviewer reproduction",
      "blocks_clean": false,
      "suggestion": "Run same-head A/B after the fixes and cache work; retain historical actual SHAs and distinguish cold public wall from compilation time."
    },
    {
      "id": "H2",
      "title": "Bitwise acceptance needs signed-zero-sensitive comparators",
      "scope": "Test-strength recommendation, not an independently demonstrated numerical defect.",
      "in_supported_envelope": true,
      "reachable": true,
      "affects_reported_result": "No numerical discrepancy established",
      "blocks_clean": false,
      "suggestion": "Use byte comparisons with explicit NaN-mask policy and one-ULP/signed-zero negative controls; do not infer signed-zero identity from array_equal alone."
    }
  ],
  "resource_report": [
    {
      "id": "B1",
      "profile": "fast",
      "workload": "reduced2 fp32; historical single-A100 supplied evidence",
      "command": "Implementer stage3_arms.py arms as described in context/evidence/03-REPORT.md; exact original launcher command is not shipped. See evidence/resource-summary.json for complete numeric inputs.",
      "comparison": "A blocking off -> B blocking on at 9f03c34e0844d2e6da5fcf8bf3d79bfb8b1854a7; NOT original baseline -> audited head",
      "warm_seconds_base": 7.580958443228155,
      "warm_seconds_head": 9.035050147678703,
      "compile_seconds_base": null,
      "compile_seconds_head": null,
      "peak_host_gib_base": null,
      "peak_host_gib_head": null,
      "peak_device_gib_base": 0.708176851272583,
      "peak_device_gib_head": 0.6876528263092041,
      "value_regret": 0.0,
      "policy_disagreement_mass": null,
      "moment_or_objective_change": null,
      "within_declared_budget": null,
      "source": "supplied_evidence",
      "measurement_notes": "Four steady observations across two isolated arms per mode; historical revision, not audit head. Warm increase exceeds the review trigger at that historical revision; there is no measured current-head verdict. Value equality is reported by supplied summary; raw arrays not rerun."
    },
    {
      "id": "B2",
      "profile": "fast",
      "workload": "reduced3 fp32; historical single-A100 supplied evidence",
      "command": "Implementer stage3_arms.py arms as described in context/evidence/03-REPORT.md; exact original launcher command is not shipped. See evidence/resource-summary.json for complete numeric inputs.",
      "comparison": "A blocking off -> B blocking on at 9f03c34e0844d2e6da5fcf8bf3d79bfb8b1854a7; NOT original baseline -> audited head",
      "warm_seconds_base": 8.316951557528228,
      "warm_seconds_head": 10.793254820629954,
      "compile_seconds_base": null,
      "compile_seconds_head": null,
      "peak_host_gib_base": null,
      "peak_host_gib_head": null,
      "peak_device_gib_base": 0.7453267574310303,
      "peak_device_gib_head": 0.6981780529022217,
      "value_regret": 0.0,
      "policy_disagreement_mass": null,
      "moment_or_objective_change": null,
      "within_declared_budget": null,
      "source": "supplied_evidence",
      "measurement_notes": "Four steady observations across two isolated arms per mode; historical revision, not audit head. Warm increase exceeds the review trigger at that historical revision; there is no measured current-head verdict. Value equality is reported by supplied summary; raw arrays not rerun."
    },
    {
      "id": "B3",
      "profile": "certified",
      "workload": "reduced3 fp64; historical single-A100 supplied evidence",
      "command": "Implementer stage3_arms.py arms as described in context/evidence/03-REPORT.md; exact original launcher command is not shipped. See evidence/resource-summary.json for complete numeric inputs.",
      "comparison": "A blocking off -> B blocking on at 9f03c34e0844d2e6da5fcf8bf3d79bfb8b1854a7; NOT original baseline -> audited head",
      "warm_seconds_base": 9.259227780625224,
      "warm_seconds_head": 11.835180788766593,
      "compile_seconds_base": null,
      "compile_seconds_head": null,
      "peak_host_gib_base": null,
      "peak_host_gib_head": null,
      "peak_device_gib_base": 0.5556533336639404,
      "peak_device_gib_head": 0.34554624557495117,
      "value_regret": 0.0,
      "policy_disagreement_mass": null,
      "moment_or_objective_change": null,
      "within_declared_budget": null,
      "source": "supplied_evidence",
      "measurement_notes": "Two steady observations in one arm per mode; below the requested three repetitions, historical revision. Warm increase exceeds the review trigger at that historical revision; there is no measured current-head verdict. Value equality is reported by supplied summary; raw arrays not rerun."
    },
    {
      "id": "B4",
      "profile": "fast",
      "workload": "current-overhead-patch CPU tiny compile-count fixtures",
      "command": "Native count driver is context/evidence/08-count_compiles.py (hard-coded implementer checkout path); supplied outputs 05/06. Not executed by reviewer.",
      "comparison": "A vs B for overhead patch replayed as ec882386 according to report 02",
      "warm_seconds_base": null,
      "warm_seconds_head": null,
      "compile_seconds_base": null,
      "compile_seconds_head": null,
      "peak_host_gib_base": null,
      "peak_host_gib_head": null,
      "peak_device_gib_base": null,
      "peak_device_gib_head": null,
      "value_regret": null,
      "policy_disagreement_mass": null,
      "moment_or_objective_change": null,
      "within_declared_budget": true,
      "source": "supplied_evidence",
      "measurement_notes": "Limited to explicitly accepted cold-helper counts: 12->14,13->16,13->15. This is not a complete resource-acceptance statement. Warm-helper execution and GPU costs remain. The standalone count driver does not itself set precision; its counts are not a replacement for separate fp32/fp64 native acceptance."
    }
  ],
  "patches": [
    {
      "id": "P1",
      "related_findings": [
        "F1"
      ],
      "target_paths": [
        "src/_lcm/solution/backward_induction.py"
      ],
      "format": "unified_diff",
      "artifact_ref": "P1",
      "filename_hint": "P1-selected-owner-accounting.patch",
      "package_path": "patches/P1-selected-owner-accounting.patch",
      "rationale": "Use actual stored-buffer delivery, not post-selection transport alignment, for all four ownership/cost decisions.",
      "expected_resource_delta": "Unmeasured. Arithmetic unchanged; corrected admission may narrow widths. Native sealed-source regeneration is required."
    },
    {
      "id": "P2",
      "related_findings": [
        "F2"
      ],
      "target_paths": [
        "src/_lcm/solution/grid_search.py"
      ],
      "format": "unified_diff",
      "artifact_ref": "P2",
      "filename_hint": "P2-plan-bound-scalar-reads.patch",
      "package_path": "patches/P2-plan-bound-scalar-reads.patch",
      "rationale": "Require planning for invariant-bound cores even when the arithmetic declares no execution-width axes.",
      "expected_resource_delta": "Unmeasured. Correct selected transfer on scalar boundary; keeps dense arithmetic, no new per-code specialization. Native sealed-source regeneration is required."
    }
  ],
  "implementation_plan": [
    {
      "order": 1,
      "root_cause_id": "R1",
      "finding_ids": [
        "F1"
      ],
      "action": "Run RT1 red at the specified assertions, apply P1, and run green at fp32/fp64. Check compiler-live and pruned selected inputs, shared copies and endpoint scratch; unsliced owner credits remain valid. Do not globally rewrite transport-dispatch kinds.",
      "repair_strategy": "local_patch",
      "repair_cost_tier": "semantic_repair",
      "patch_ids": [
        "P1"
      ],
      "oracle_ids": [
        "OR1"
      ],
      "test_ids": [
        "RT1"
      ],
      "mutation_ids": [],
      "done_when": "RT1 is red at the stated ownership/admission assertions on head and green with P1 at both precisions; ordinary unsliced-aligned controls remain green; native owner/liveness, budget, multi-device and sealed-corridor gates pass."
    },
    {
      "order": 2,
      "root_cause_id": "R2",
      "finding_ids": [
        "F2"
      ],
      "action": "Run the public scalar RT2 graph/numerical witness red, apply P2, and run RT2/MT2 green at both precisions. PLANNED disposition enables selected reads without adding artificial numerical tiles.",
      "repair_strategy": "local_patch",
      "repair_cost_tier": "semantic_repair",
      "patch_ids": [
        "P2"
      ],
      "oracle_ids": [
        "OR1"
      ],
      "test_ids": [
        "RT2"
      ],
      "mutation_ids": [
        "MT2"
      ],
      "done_when": "RT2 structural and numeric witnesses plus MT2 permutations/scales pass at fp32/fp64, against the literal one-step Bellman equation; nonblocked and ordinary dense controls stay unchanged; native admission/certification pass."
    },
    {
      "order": 3,
      "root_cause_id": "R3",
      "finding_ids": [],
      "action": "Retain both per-shape select/write helpers. Split structural blueprint construction from current invocation binding in core_program/backward_induction. Cache only adapter-certified built-in GridSearch structure keyed by sealed program identity and current validated abstract schema, layout/view structure, compiler options and recipe policy. Start with warm reuse in both routes, then equal-schema family reuse. Exact cache content and mandatory fresh fields are in report.md section 3.",
      "repair_strategy": "architecture_change",
      "repair_cost_tier": "redesign",
      "patch_ids": [],
      "oracle_ids": [],
      "test_ids": [
        "RT3"
      ],
      "mutation_ids": [],
      "done_when": "No concrete arguments, mutable liveness/donation/frontier objects, old transfer generations or numerical results live in the cache; warm same-schema changed-parameter outputs equal fresh models; schema changes miss/refuse correctly."
    },
    {
      "order": 4,
      "root_cause_id": "R3",
      "finding_ids": [],
      "action": "Add a bounded model-local runtime cache with lifecycle/serialization exclusions; current params, runtime grids, exact code/offset/read addresses, actual owners, transfer generations, liveness, donation, budgets and admission are rebound each solve. Leave uncertified third-party builders uncached. Test retention, layout/precision/weak-type changes, stale seals and lowered available budgets.",
      "repair_strategy": "architecture_change",
      "repair_cost_tier": "redesign",
      "patch_ids": [],
      "oracle_ids": [],
      "test_ids": [
        "RT3"
      ],
      "mutation_ids": [],
      "done_when": "RT3 and the cache safety/retention/schema/resource matrix in report.md pass for blocked and unblocked GridSearch; current-head paired resources meet the supplied contract or require an explicit workload-specific decision."
    },
    {
      "order": 5,
      "root_cause_id": "R1",
      "finding_ids": [
        "F1",
        "F2"
      ],
      "action": "Regenerate changed certificate/corridor anchors and CI registrations using native project tooling; run targeted ownership/donation/liveness, scalar and ordinary blocked/unblocked 1/2/8-device gates and mutation self-tests. Run one representative smoke per repair and one matched full resource/profile suite at closure. Preserve helper-count acceptance and exact numerical/RNG controls.",
      "repair_strategy": "local_patch",
      "repair_cost_tier": "invariant_or_diagnostic",
      "patch_ids": [
        "P1",
        "P2"
      ],
      "oracle_ids": [
        "OR1"
      ],
      "test_ids": [
        "RT1",
        "RT2",
        "RT3"
      ],
      "mutation_ids": [
        "MT2"
      ],
      "done_when": "Current exact source has native red/green, regenerated seals/corridor mutation receipts, no full-type gather, safe actual admission and matched current-head resource evidence. Historical receipts remain separately labelled."
    }
  ],
  "todo_responses": [
    {
      "id": "D1",
      "marker_location": "audit target: Option 1 / Option 2 / neither",
      "instruction": "Choose whether/how to remove residual helpers.",
      "response": "Neither now. The supplied contract accepts the residual per-shape helpers. Option 1 is a possible later measured write/launch optimization; Option 2 is a larger transfer/accounting redesign. Removing just one does not remove the other. Conditional preconditions and red-first fold gates are specified in report.md section 4; no fold is required for F1/F2. RT4 delivers an optional executable probe for the two fold-specific red assertions and the accepted-current-design control."
    },
    {
      "id": "D2",
      "marker_location": "audit target: structural_resolution",
      "instruction": "Judge params-independent warm caching for blocked and unblocked routes.",
      "response": "Yes for an immutable schema-certified blueprint, no for wholesale materialized/resolved programs. Numerical-payload-independent does not mean parameter-dependent schemas can be ignored. The minimal split, exact key categories, per-call rebinding, fresh admission and lifecycle tests are in plan steps 3-4 and report.md section 3."
    }
  ],
  "failed_attacks": [
    {
      "attack": "Residual helper compile count grows once per preference code",
      "where_it_would_enter": "invariant_blocks and selected view runtime operands",
      "how_tested": "Read raw count outputs, helper implementations and shape/runtime bindings.",
      "why_no_objection": "Current supplied counts identify select/write families per shape, not a per-code compile multiplier; the profile explicitly accepts them.",
      "residual_uncertainty": "Current real-GPU helper count and runtime cost not measured by reviewer."
    },
    {
      "attack": "Nonzero code is silently reindexed to zero or result cache aliases two types",
      "where_it_would_enter": "block_state_action_space, selected_block_view, transfer_result_key",
      "how_tested": "Read explicit start/code bindings, view structure/identity split and fresh per-solve generation wiring plus supplied tests.",
      "why_no_objection": "The design keeps original codes as runtime data and result identity includes the selected coordinates. No supported collision found in this bounded audit.",
      "residual_uncertainty": "Proposed structural cache is not implemented and needs its own stale-reuse tests."
    },
    {
      "attack": "A selected block currently communicates the whole type axis",
      "where_it_would_enter": "apply_value_transfer selection before device_put",
      "how_tested": "Read ordered transfer stages, selection shapes, and supplied standalone HLO report.",
      "why_no_objection": "The shipped explicit selected route slices before transport; neither proposed fold is already present.",
      "residual_uncertainty": "Supplied forced-host evidence is not real multi-GPU production acceptance."
    },
    {
      "attack": "Unsupported cross-type resets/gated/folded/reference routes are silently optimized",
      "where_it_would_enter": "invariant_blocking and construction/processing gates",
      "how_tested": "Read request validation, unsafe-analysis call and supported-route exclusions.",
      "why_no_objection": "The target explicitly refuses these extensions; no objection imported from an unsupported path.",
      "residual_uncertainty": "This does not certify all invariant-analysis internals outside the supplied delta."
    },
    {
      "attack": "Old GPU warm regression proves the final overhead patch has the same measured regression",
      "where_it_would_enter": "evidence attribution and final resource verdict",
      "how_tested": "Read every arm revision in the supplied benchmark summary and report 02 rebase history.",
      "why_no_objection": "The GPU arrays/timings belong to 9f03c34e, not ec882386. The known regression motivates caching but cannot be silently promoted to a current-head measurement.",
      "residual_uncertainty": "Current-head matched GPU/resource suite remains required."
    }
  ],
  "attack_ledger_updates": [
    {
      "dimension": "selected-view ownership vs transport layout",
      "previous_status": "unknown",
      "new_status": "finding",
      "note": "F1: four remaining ALIGNED_LOCAL ownership/admission predicates."
    },
    {
      "dimension": "blocked degenerate products",
      "previous_status": "unknown",
      "new_status": "finding",
      "note": "F2: absent width axes must not suppress selected-read planning."
    },
    {
      "dimension": "cold helper growth vs type count",
      "previous_status": "unknown",
      "new_status": "attacked_clean",
      "note": "Supplied evidence and current profile accept per-shape select/write helpers; do not mistake them for a K-fold executable explosion."
    }
  ],
  "defect_class_sweep": [
    {
      "class_id": "DC-1",
      "result": "NOT_APPLICABLE",
      "finding_ids": [],
      "note": "Casebook cancellation-before-reduction class: no new cancellation/reduction arithmetic in the audited changes; no supported target witness established. This is not a repository-wide absence claim."
    },
    {
      "class_id": "DC-2",
      "result": "NOT_APPLICABLE",
      "finding_ids": [],
      "note": "Casebook single-rounded evaluation trusted for exact ordering: no new exact-order predicate or arithmetic-certificate decision in this delta. The representation/admission findings do not assert an arithmetic cancellation or near-tie defect."
    },
    {
      "class_id": "DC-3",
      "result": "NOT_APPLICABLE",
      "finding_ids": [],
      "note": "Casebook stabilized-primitive bypass: no new exact-affine/stable-primitive dispatch to audit; existing sealed corridors still need regeneration for source changes."
    }
  ],
  "reference_implementations": [
    {
      "id": "OR1",
      "language": "python",
      "oracle_type": "structural",
      "purpose": "Independent owner/block reservation equations and literal beta=1/2 scalar Bellman values, without pylcm/JAX.",
      "related_finding": "F1",
      "filename_hint": "selected_view_reference.py",
      "artifact_ref": "OR1",
      "package_path": "audit-oracles/selected_view_reference.py",
      "independence_basis": "Counts distinct logical allocations arithmetically and evaluates one-action values with fractions.Fraction. It does not duplicate transfer classification or solver control flow. The C+2B pruned reservation is explicitly the existing conservative policy, not a measured allocator peak.",
      "contamination_checked": true,
      "run_command": "python audit-oracles/selected_view_reference.py",
      "expected_result": "24 byte-accounting checks and literal values (3,6,9) pass; no production correctness assertion.",
      "executed": true,
      "observed_result": "PASS: 24 byte-accounting cases; exact one-action values = (Fraction(3, 1), Fraction(6, 1), Fraction(9, 1))\nThis checks independent equations only. No pylcm/JAX execution is claimed."
    }
  ],
  "reusable_tests": [
    {
      "id": "RT1",
      "test_kind": "profile_budget",
      "language": "python",
      "purpose": "Public selected-owner-credit observation plus exact declared threshold, pruned/shared variants and unsliced control.",
      "related_finding": "F1",
      "profile": "any",
      "filename_hint": "test_audit486_selected_view_admission.py",
      "artifact_ref": "RT1",
      "package_path": "tests/solution/test_audit486_selected_view_admission.py",
      "run_command": "pixi run --as-is -e tests-cpu pytest tests/solution/test_audit486_selected_view_admission.py --precision=32 -n 0 -v --junitxml=/absolute/evidence/rt1-p32.xml",
      "expected_result": "Specified assertions red on head and green after P1; repeat precision=64. Import/fixture errors do not count as reproducing the finding.",
      "executed": false,
      "observed_result": "not_run: mandated Python >=3.14 native environment unavailable. AST syntax and supplied-interface checks only."
    },
    {
      "id": "RT2",
      "test_kind": "witness",
      "language": "python",
      "purpose": "No-axis bound graph still plans selected continuation and matches literal finite Bellman values.",
      "related_finding": "F2",
      "profile": "any",
      "filename_hint": "test_audit486_scalar_block.py",
      "artifact_ref": "RT2",
      "package_path": "tests/solution/test_audit486_scalar_block.py",
      "run_command": "pixi run --as-is -e tests-cpu pytest tests/solution/test_audit486_scalar_block.py --precision=32 -n 0 -v --junitxml=/absolute/evidence/rt2-p32.xml",
      "expected_result": "PLANNED assertion red on head; literal outputs green with P2 at both precisions. Actual prepatch native error/value must be recorded rather than guessed.",
      "executed": false,
      "observed_result": "not_run: required native environment absent. Syntax parsed only."
    },
    {
      "id": "RT3",
      "test_kind": "benchmark",
      "language": "python",
      "purpose": "Future immutable-blueprint cache skips expensive rebuilds on same-schema same/changed params in blocked and unblocked built-in GridSearch, while fresh-model byte equality holds.",
      "related_finding": null,
      "profile": "any",
      "filename_hint": "test_audit486_structural_cache.py",
      "artifact_ref": "RT3",
      "package_path": "tests/solution/test_audit486_structural_cache.py",
      "run_command": "pixi run --as-is -e tests-cpu pytest tests/solution/test_audit486_structural_cache.py --precision=32 -n 0 -v --junitxml=/absolute/evidence/rt3-p32.xml",
      "expected_result": "Intentionally red on ec882386 and P1/P2 alone, green after the recommended cache split. If old names become cheap hit dispatchers, move spies to actual expensive builders rather than removing work-elimination assertions.",
      "executed": false,
      "observed_result": "not_run. This is a future optimization acceptance test, not a third current correctness finding."
    },
    {
      "id": "RT4",
      "test_kind": "benchmark",
      "language": "python",
      "purpose": "Optional helper-specific red-first probe: accepted current-design control and separate select/write removal hypotheses. Not a new required fold or merge gate.",
      "related_finding": null,
      "profile": "any",
      "filename_hint": "helper_fold_probe.py",
      "artifact_ref": "RT4",
      "package_path": "audit-tests/helper_fold_probe.py",
      "run_command": "pixi run --as-is -e tests-cpu python /absolute/reply/audit-tests/helper_fold_probe.py --repo-root /absolute/full-checkout --expect-removed write",
      "expected_result": "Current source is expected to fail the named write-removal assertion; select-removal is separately red; none observes the accepted two-helper design. A future authorized fold must pass its matching probe without relaxing numerical, ownership or layout gates.",
      "executed": false,
      "observed_result": "not_run. Native-only probe syntax parsed; helper-removal expectations are proposed, not measured."
    }
  ],
  "mutation_suites": [
    {
      "id": "MT2",
      "language": "python",
      "purpose": "Tiny type-label permutation and exact binary-scale family for the scalar bound-read defect.",
      "related_root_cause": "R2",
      "dimensions": [
        "type permutation",
        "binary scale",
        "mixed-sign flow",
        "nonzero type code"
      ],
      "filename_hint": "test_audit486_scalar_block_family.py",
      "artifact_ref": "MT2",
      "package_path": "tests/solution/test_audit486_scalar_block_family.py",
      "run_command": "pixi run --as-is -e tests-cpu pytest tests/solution/test_audit486_scalar_block_family.py --precision=64 -n 0 -v --junitxml=/absolute/evidence/mt2-p64.xml",
      "expected_result": "Six bounded cases agree byte-for-byte with independent dyadic equations; repeat fp32.",
      "executed": false,
      "observed_result": "not_run; AST syntax checked, no native execution."
    }
  ],
  "executions": [
    {
      "id": "E1",
      "command": "Safe ZIP extraction, zipfile.testzip, manifest byte/SHA256/UTF8 and canonical manifest digest checks (see evidence/coverage.json and static-checks.json).",
      "status": "passed",
      "source": "reviewer_execution",
      "exit_code": 0,
      "establishes": "Input integrity and full static read surface, not numerical execution.",
      "output_excerpt": "53/53 manifest entries match; 45 required paths present; manifest digest matches."
    },
    {
      "id": "E2",
      "command": "git apply --check P1-selected-owner-accounting.patch P2-plan-bound-scalar-reads.patch on the extracted head",
      "status": "passed",
      "source": "reviewer_execution",
      "exit_code": 0,
      "establishes": "Both source patches apply to the shipped snapshot.",
      "output_excerpt": "Exit 0, no stderr. No project source was executed or certificate regenerated."
    },
    {
      "id": "E3",
      "command": "ast.parse on every delivered .py artifact",
      "status": "passed",
      "source": "reviewer_execution",
      "exit_code": 0,
      "establishes": "Syntax only, not imports, native JAX compatibility or tests.",
      "output_excerpt": "6 Python artifacts parsed successfully."
    },
    {
      "id": "E4",
      "command": "python audit-oracles/selected_view_reference.py",
      "status": "passed",
      "source": "reviewer_execution",
      "exit_code": 0,
      "establishes": "Independent small allocation equations and one-step finite values.",
      "output_excerpt": "PASS: 24 byte-accounting cases; exact one-action values = (Fraction(3, 1), Fraction(6, 1), Fraction(9, 1))\nThis checks independent equations only. No pylcm/JAX execution is claimed."
    },
    {
      "id": "E5",
      "command": "Native RT1/RT2/MT2 red-green and RT3 cache acceptance; native certificate/mutation and matched resource closure commands in evidence/commands.md",
      "status": "not_run",
      "source": "reviewer_execution",
      "exit_code": null,
      "establishes": "Required future native closure, not a reviewer pass.",
      "output_excerpt": "Project requires >=3.14; reviewer Python 3.13.5; no shim, syntax rewrite or interpreter search."
    },
    {
      "id": "E6",
      "command": "Supplied Stage 3 CPU reports and historical Marvin arm summary (source paths/digests in resource-summary.json).",
      "status": "passed",
      "source": "supplied_evidence",
      "exit_code": 0,
      "establishes": "Only the reported fixture/run results at their stated revisions.",
      "output_excerpt": "Reports state successful parity/regression runs; GPU metrics are at 9f03c34e, not the final audited head. No new-witness pass is inferred."
    }
  ],
  "unresolved": [
    {
      "id": "U1",
      "finding_ids": [
        "F1",
        "F2"
      ],
      "exact_question": "Do the delivered public/native regressions fail for their specified source defects at ec882386 and pass after P1/P2 at fp32/fp64?",
      "required_artifact": "Native red/green JUnit XML and full logs for RT1, RT2 and MT2 with exact source/env identities.",
      "why_blocking": "Required native validation of candidate repairs is not_run, not a missing input that requires repackaging."
    },
    {
      "id": "U2",
      "finding_ids": [
        "F1",
        "F2"
      ],
      "exact_question": "Do corrected ownership, transfers, output layouts and dense scalar planning preserve sealed corridors and actual 1/2/8-device admission?",
      "required_artifact": "Native regenerated seal/corridor and mutation-self-test receipts plus ownership/liveness/transfer tests and real-GPU selected-layout evidence.",
      "why_blocking": "Patched source is not covered by old certificate receipts. No full-type gather or budget fit may be inferred from source patch application."
    },
    {
      "id": "U3",
      "finding_ids": [],
      "exact_question": "After repairs and optional blueprint caching, what is the same-head blocked/unblocked warm, cold, host/device peak and compile-family trade-off on required ACA lanes?",
      "required_artifact": "Current-head matched A1 B1 B2 A2 resource ledger with three steady observations, exact per-arm source/environment, compile logs and memory peaks.",
      "why_blocking": "Blocks performance/default-promotion closure, not the design decision or a source-audit finish. Historical 9f03c34e measurements cannot certify ec882386 or a future cached build."
    }
  ],
  "next_action": "implement_plan",
  "summary": "Keep select and write helpers for now; cache only immutable, schema-certified structural blueprints with current-data rebinding and fresh admission. Repair the two target-owned selected-read defects using P1/P2 and the delivered tests, then regenerate certificates and run matched native closure. No GPU speedup or native test pass is claimed by the reviewer.",
  "evidence_files": [
    {
      "id": "EV-COVERAGE",
      "kind": "FILE",
      "path": "evidence/coverage.json",
      "package_path": "evidence/coverage.json"
    },
    {
      "id": "EV-STATIC",
      "kind": "PROVENANCE",
      "path": "evidence/static-checks.json",
      "package_path": "evidence/static-checks.json"
    },
    {
      "id": "EV-RESOURCE",
      "kind": "FILE",
      "path": "evidence/resource-summary.json",
      "package_path": "evidence/resource-summary.json"
    },
    {
      "id": "EV-REFERENCE",
      "kind": "LOG",
      "path": "evidence/reference-check.log",
      "package_path": "evidence/reference-check.log"
    },
    {
      "id": "EV-COMMANDS",
      "kind": "FILE",
      "path": "evidence/commands.md",
      "package_path": "evidence/commands.md"
    }
  ],
  "package_roles": {
    "audit-oracles/selected_view_reference.py": "artifact_payload",
    "audit-result.json": "artifact_payload",
    "audit-tests/helper_fold_probe.py": "artifact_payload",
    "evidence/commands.md": "human_action_required",
    "evidence/coverage.json": "supporting_evidence",
    "evidence/reference-check.log": "supporting_evidence",
    "evidence/resource-summary.json": "supporting_evidence",
    "evidence/static-checks.json": "supporting_evidence",
    "patches/P1-selected-owner-accounting.patch": "artifact_payload",
    "patches/P2-plan-bound-scalar-reads.patch": "artifact_payload",
    "report.md": "human_action_required",
    "tests/solution/test_audit486_scalar_block.py": "artifact_payload",
    "tests/solution/test_audit486_scalar_block_family.py": "artifact_payload",
    "tests/solution/test_audit486_selected_view_admission.py": "artifact_payload",
    "tests/solution/test_audit486_structural_cache.py": "artifact_payload"
  }
}
