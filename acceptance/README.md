# ACA continuous-sharding acceptance artifacts

These artifacts describe a bounded extension of continuous GridSearch sharding. They do
not certify production capacity or performance.

- `inventory_aca.py` constructs the frozen ACA factory from the original sealed
  eleven-file input packet. Run in a fresh fp32 process with eight visible devices,
  candidate engine/caller imports and the frozen ACA model. It does not solve.
- `aca-phase-inventory.json` is the native factory receipt from job 27535505.
- `aca-inventory-provenance.json` identifies executed source bytes and later
  source/schema changes; the native receipt is not silently relabelled as current.
- `aca-capacity.json` derives necessary lower bounds from that inventory. Compiler,
  simultaneous ownership, replay, physical GPU peaks and admission remain unknown.
- `test_aca_phase_boundaries.py` runs structural checks using the frozen ACA model and
  original inputs. It is explicitly selected by the acceptance harness, outside the
  engine's regular test dependencies.

The synthetic numerical and refusal tests live in
`tests/test_continuous_assets_aca_{vocabulary,numerics,certificate}.py`. They cover
represented finite-grid semantics; their small fixture is not production equality. The
caller's explicit opt-in retains the economic grids and legacy chunk policy.

The implementation work order requires separate physical GPU semantic evidence,
production-shaped resource admission, and reviewed production release. Original
continuous-sharding evidence remains scoped to its original source and workload.
