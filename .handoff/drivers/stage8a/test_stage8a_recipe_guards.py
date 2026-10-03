"""Owner recipes reject relative output roots before checkout or cluster commands."""

import hashlib
import os
import shlex
import subprocess
from pathlib import Path

import pytest


@pytest.mark.parametrize("phase", ["production", "reference"])
def test_recipe_requests_user_selected_partition(*, phase: str) -> None:
    """Both full-production allocations request the user's mlgpu partition."""
    recipe = Path(__file__).with_name(f"stage8a_{phase}.sbatch")
    partitions = [
        line
        for line in recipe.read_text().splitlines()
        if line.startswith("#SBATCH --partition=")
    ]

    assert partitions == ["#SBATCH --partition=mlgpu"]


@pytest.mark.parametrize("phase", ["production", "reference"])
def test_recipe_refuses_relative_output_root_before_changing_directory(
    *,
    tmp_path: Path,
    phase: str,
) -> None:
    """Full output paths cannot change meaning when the recipe enters the checkout."""
    recipe = Path(__file__).with_name(f"stage8a_{phase}.sbatch")
    caller = tmp_path / "caller"
    caller.mkdir()
    (caller / "relative-output").mkdir()
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    commands = tmp_path / "commands"
    commands.mkdir()
    marker = tmp_path / "unexpected-command.txt"
    for command in ("git", "srun", "scontrol", "pixi", "findmnt"):
        script = commands / command
        script.write_text(
            "#!/bin/sh\n"
            f"printf '%s\\n' '{command}' >> {shlex.quote(str(marker))}\n"
            + ("printf '%s\\n' lustre\n" if command == "findmnt" else "exit 97\n")
        )
        script.chmod(0o755)
    environment = {
        **os.environ,
        "PATH": f"{commands}:{os.environ['PATH']}",
        "PYLCM_DIR": str(checkout),
        "PYLCM_COMMIT": "1" * 40,
        "ACA_MODEL_DIR": str(checkout),
        "ACA_MODEL_COMMIT": "ad38653696ec366e318ac61b9a81b597a4ecb700",
        "ACA_SLURM_DIR": str(checkout),
        "ACA_SLURM_COMMIT": "2" * 40,
        "PIXI_LOCK_SHA256": "a" * 64,
        "DRIVER_SHA256": "b" * 64,
        "BENCHMARK_HELPER_SHA256": "c" * 64,
        "OUT_ROOT": "relative-output",
        "REFERENCE_DIR": str(tmp_path / "reference"),
        "REFERENCE_RECEIPT_SHA256": "d" * 64,
    }

    result = subprocess.run(  # noqa: S603
        ["/bin/bash", str(recipe)],
        cwd=caller,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )

    assert (result.returncode, result.stderr.strip(), marker.exists()) == (
        98,
        "OUT_ROOT must be an absolute path to an existing owner Lustre workspace",
        False,
    )


@pytest.mark.parametrize("phase", ["production", "reference"])
@pytest.mark.parametrize("root_name", ["PYLCM_DIR", "ACA_MODEL_DIR", "ACA_SLURM_DIR"])
def test_recipe_refuses_relative_owner_source_paths_before_external_commands(
    *,
    tmp_path: Path,
    phase: str,
    root_name: str,
) -> None:
    """Owner source roots retain their meaning after entering the checkout."""
    recipe = Path(__file__).with_name(f"stage8a_{phase}.sbatch")
    output = tmp_path / "output"
    output.mkdir()
    commands = tmp_path / "commands"
    commands.mkdir()
    marker = tmp_path / "unexpected-command"
    findmnt = commands / "findmnt"
    findmnt.write_text(
        f"#!/bin/sh\nprintf '%s' invoked > {shlex.quote(str(marker))}\nexit 97\n"
    )
    findmnt.chmod(0o755)
    environment = {
        **os.environ,
        "PATH": f"{commands}:{os.environ['PATH']}",
        "HOME": str(tmp_path / "home"),
        "OUT_ROOT": str(output),
        "PYLCM_DIR": str(tmp_path / "checkout"),
        "PYLCM_COMMIT": "1" * 40,
        "ACA_MODEL_DIR": str(tmp_path / "aca-model"),
        "ACA_MODEL_COMMIT": "ad38653696ec366e318ac61b9a81b597a4ecb700",
        "ACA_SLURM_DIR": str(tmp_path / "aca-slurm"),
        "ACA_SLURM_COMMIT": "2" * 40,
        "PIXI_LOCK_SHA256": "a" * 64,
        "DRIVER_SHA256": "b" * 64,
        "BENCHMARK_HELPER_SHA256": "c" * 64,
        "REFERENCE_DIR": str(tmp_path / "reference"),
        "REFERENCE_RECEIPT_SHA256": "d" * 64,
        root_name: "relative-owner-source",
    }

    result = subprocess.run(  # noqa: S603
        ["/bin/bash", str(recipe)],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )

    assert (result.returncode, result.stderr.strip(), marker.exists()) == (
        98,
        "PYLCM_DIR, ACA_MODEL_DIR and ACA_SLURM_DIR must be absolute paths",
        False,
    )


@pytest.mark.parametrize("phase", ["production", "reference"])
def test_recipe_forwards_immutable_receipts_without_installing_worker_environment(
    *,
    tmp_path: Path,
    phase: str,
) -> None:
    """Actual recipes bind receipts and use a preinstalled prefix in every step.

    External allocation/source commands are recording stand-ins. This checks
    shell routing and cannot certify a GPU allocation, checkout or environment.
    """
    recipe = Path(__file__).with_name(f"stage8a_{phase}.sbatch")
    checkout = tmp_path / "checkout"
    driver = checkout / ".handoff/drivers/stage8a/stage8a_production.py"
    helper = checkout / ".handoff/drivers/stage5b/stage3_arms.py"
    driver.parent.mkdir(parents=True)
    helper.parent.mkdir(parents=True)
    (tmp_path / "aca-model").mkdir()
    (tmp_path / "aca-slurm").mkdir()
    driver.write_bytes(b"source guard fixture: driver")
    helper.write_bytes(b"source guard fixture: inherited builder")
    lock = checkout / "pixi.lock"
    lock.write_bytes(b"source guard fixture: frozen lock")
    output = tmp_path / "lustre-routing-fixture"
    output.mkdir()
    reference = tmp_path / "immutable-reference"
    reference.mkdir()
    reference_receipt = reference / "receipt.json"
    reference_receipt.write_bytes(b'{"scope":"reference-binding-fixture"}')
    commands = tmp_path / "commands"
    commands.mkdir()
    traces = tmp_path / "traces"
    traces.mkdir()
    pylcm_commit = "1" * 40
    slurm_commit = "2" * 40
    model_commit = "ad38653696ec366e318ac61b9a81b597a4ecb700"
    scripts = {
        "findmnt": "#!/bin/sh\nprintf '%s\\n' lustre\n",
        "git": (
            "#!/bin/sh\n"
            'if [ "$3" = rev-parse ]; then\n'
            f'  case "$2" in */aca-model) printf "%s\\n" {model_commit};;\n'
            f'    */aca-slurm) printf "%s\\n" {slurm_commit};;\n'
            f'    *) printf "%s\\n" {pylcm_commit};; esac\nfi\n'
        ),
        "scontrol": (
            "#!/bin/sh\n"
            'if [ "$2" = hostnames ]; then printf "%s\\n" node0 node1 node2;\n'
            'else printf "%s\\n" allocation-routing-fixture; fi\n'
        ),
        "srun": (
            "#!/bin/bash\nset -eu\n"
            "count=0\n"
            '[[ ! -f "$TRACE_ROOT/count" ]] || read -r count < "$TRACE_ROOT/count"\n'
            'count=$((count + 1)); printf "%s\\n" "$count" > "$TRACE_ROOT/count"\n'
            'printf "%s\\0" "$@" > "$TRACE_ROOT/$count.args"\n'
            'publish_receipt=false; output=""; next_output=false\n'
            'for arg in "$@"; do\n'
            '  if "$next_output"; then output="$arg"; next_output=false; fi\n'
            '  [[ "$arg" != --out ]] || next_output=true\n'
            '  [[ "$arg" != plan && "$arg" != reference ]] || publish_receipt=true\n'
            "done\n"
            'if "$publish_receipt"; then mkdir -p "$output";\n'
            '  printf "%s" \'{"scope":"plan-binding-fixture"}\' '
            '> "$output/receipt.json"\nfi\n'
        ),
    }
    for command, source in scripts.items():
        executable = commands / command
        executable.write_text(source)
        executable.chmod(0o755)
    environment = {
        **os.environ,
        "PATH": f"{commands}:{os.environ['PATH']}",
        "HOME": str(tmp_path / "home"),
        "TRACE_ROOT": str(traces),
        "PYLCM_DIR": str(checkout),
        "PYLCM_COMMIT": pylcm_commit,
        "ACA_MODEL_DIR": str(tmp_path / "aca-model"),
        "ACA_MODEL_COMMIT": model_commit,
        "ACA_SLURM_DIR": str(tmp_path / "aca-slurm"),
        "ACA_SLURM_COMMIT": slurm_commit,
        "PIXI_LOCK_SHA256": hashlib.sha256(lock.read_bytes()).hexdigest(),
        "DRIVER_SHA256": hashlib.sha256(driver.read_bytes()).hexdigest(),
        "BENCHMARK_HELPER_SHA256": hashlib.sha256(helper.read_bytes()).hexdigest(),
        "OUT_ROOT": str(output),
        "REFERENCE_DIR": str(reference),
        "REFERENCE_RECEIPT_SHA256": hashlib.sha256(
            reference_receipt.read_bytes(),
        ).hexdigest(),
        "SLURM_JOB_ID": "4242",
        "SLURM_JOB_NODELIST": "recorded-nodes",
    }

    result = subprocess.run(  # noqa: S603
        ["/bin/bash", str(recipe)],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    steps = [
        path.read_bytes().decode().rstrip("\0").split("\0")
        for path in sorted(traces.glob("*.args"))
    ]
    installed_runtime_only = all(
        "--frozen" not in " ".join(step) and "--as-is" in " ".join(step)
        for step in steps
    )
    bindings = True
    if phase == "production" and len(steps) == 3:
        plan_receipt = (
            output / "stage8a-production-fp32-11111111-4242/planning/receipt.json"
        )
        plan_sha = hashlib.sha256(plan_receipt.read_bytes()).hexdigest()
        for step in steps[1:]:
            bindings = bindings and all(
                name in step and step[step.index(name) + 1] == value
                for name, value in (
                    ("--planning-receipt", str(plan_receipt)),
                    ("--planning-receipt-sha256", plan_sha),
                )
            )
        bindings = bindings and (
            "--worker-receipts" in steps[2]
            and steps[2][steps[2].index("--worker-receipts") + 1]
            == str(plan_receipt.parent.parent / "workers")
        )
        bindings = bindings and (
            "--reference-receipt-sha256" in steps[2]
            and steps[2][steps[2].index("--reference-receipt-sha256") + 1]
            == environment["REFERENCE_RECEIPT_SHA256"]
        )

    assert (result.returncode, len(steps), installed_runtime_only, bindings) == (
        0,
        3 if phase == "production" else 1,
        True,
        True,
    )


def test_production_recipe_refuses_relative_reference_before_external_commands(
    *,
    tmp_path: Path,
) -> None:
    """The separately admitted reference retains its location after checkout changes."""
    recipe = Path(__file__).with_name("stage8a_production.sbatch")
    output = tmp_path / "output"
    output.mkdir()
    commands = tmp_path / "commands"
    commands.mkdir()
    marker = tmp_path / "external-command"
    command = commands / "findmnt"
    command.write_text(
        f"#!/bin/sh\nprintf invoked > {shlex.quote(str(marker))}\nexit 97\n"
    )
    command.chmod(0o755)
    environment = {
        **os.environ,
        "PATH": f"{commands}:{os.environ['PATH']}",
        "PYLCM_DIR": str(tmp_path / "pylcm"),
        "PYLCM_COMMIT": "1" * 40,
        "ACA_MODEL_DIR": str(tmp_path / "model"),
        "ACA_MODEL_COMMIT": "2" * 40,
        "ACA_SLURM_DIR": str(tmp_path / "slurm"),
        "ACA_SLURM_COMMIT": "3" * 40,
        "PIXI_LOCK_SHA256": "a" * 64,
        "DRIVER_SHA256": "b" * 64,
        "BENCHMARK_HELPER_SHA256": "c" * 64,
        "OUT_ROOT": str(output),
        "REFERENCE_DIR": "relative-reference",
        "REFERENCE_RECEIPT_SHA256": "d" * 64,
    }
    result = subprocess.run(  # noqa: S603 - execute the authenticated recipe with fixed arguments
        ["/bin/bash", str(recipe)],
        env=environment,
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )

    assert (result.returncode, result.stderr.strip(), marker.exists()) == (
        98,
        "REFERENCE_DIR must be an absolute path",
        False,
    )
