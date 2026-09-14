# Supported discrete baseline/head resource control

This benchmark compares the same discrete GridSearch witness at the two source revisions
in SOURCE-MANIFEST.json. It requests four A40 GPUs, eight CPUs, 64 GiB and two hours
through pytask-slurm. It runs both precisions and retains separate correctness,
compilation, warm runtime and live-memory observations.

HMG authorized this control after reviewing its preparation packet, with separate Marvin
directories and no ACA use. The controller uses an already installed non-ACA
pytask-slurm environment read-only. Baseline, head and benchmark tooling have
independent Git checkouts; commit, push and pull the tooling before running it. Preserve
any other queued measurement checkout unchanged.

See WORKORDER.md for source pins, exact commands, numerical contracts, installation
gates and stopping rules. B2-DECISION.md describes the separate unresolved main/head
comparison. Neither a successful job nor this control's completion automatically accepts
the full resource contract.
