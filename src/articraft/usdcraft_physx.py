"""File boundary between WSL modeling and the separate Windows Isaac runtime.

This module never simulates physics. It verifies native reports against the
requested asset and frozen profile before attaching feedback to a modeling run.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

from articraft.usdcraft_benchmark import save_json


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare_job(
    asset: Path,
    profile: Path,
    native_asset: str,
    native_profile: str,
    native_output: str,
    case: str,
) -> dict:
    """Paths prefixed 'native' are explicit Windows paths, never guessed from cwd."""
    return {
        "profile": native_profile,
        "expected_profile_sha256": sha256(profile),
        "output": native_output,
        "assets": [{"case": case, "asset": native_asset, "expected_sha256": sha256(asset)}],
    }


def native_path(path: Path) -> str:
    """Convert shared-drive paths only; native USD never opens Linux-only paths."""
    value = path.resolve().as_posix()
    if sys.platform == "win32":
        return value
    if value.startswith("/mnt/") and len(value) > 7 and value[5].isalpha() and value[6] == "/":
        return value[5].upper() + ":/" + value[7:]
    raise ValueError("Native jobs require artifacts on a shared /mnt/<drive>/ path")


def native_identity(config_file: Path) -> dict:
    config = json.loads(config_file.read_text())
    scripts = Path(config["script_dir"])
    return {
        "config": sha256(config_file),
        "profile": sha256(Path(config["profile"])),
        "adapter": sha256(scripts / "apply_compatibility.py"),
        "partition": sha256(scripts / "orthogonal_compound.py"),
        "runner": sha256(scripts / "validate_asset.py"),
        "launcher": sha256(scripts / "launch_native.py"),
    }


def invoke_native(config: dict, script: Path, job: Path, folder: Path) -> dict:
    """The Windows helper owns the real PID, timeout and log across WSL interruptions."""
    request = folder / (job.stem + "-launch.json")
    state = folder / (job.stem + "-process.json")
    log = folder / (job.stem + "-stdout.log")
    save_json(
        request,
        {
            "command": [
                config["isaacsim"],
                config["experience"],
                "--no-window",
                "--portable-root",
                config["portable_root"],
                "--exec",
                native_path(script),
                "--/usdcraft/job=" + native_path(job),
                "--/log/file=" + native_path(folder / (job.stem + "-kit.log")),
            ],
            "state": native_path(state),
            "log": native_path(log),
            "environment": config.get("environment", {}),
            "timeout_seconds": config["timeout_seconds"],
        },
    )
    command = [
        config["windows_python"],
        native_path(Path(config["script_dir"]) / "launch_native.py"),
        native_path(request),
    ]
    completed = subprocess.run(command, text=True, capture_output=True, check=False)
    if state.exists():
        log = folder / json.loads(state.read_text()).get("log_name", log.name)
    if completed.returncode:
        return {
            "status": "execution_error",
            "returncode": completed.returncode,
            "launcher_error": completed.stderr[-3000:],
            "log": str(log),
            "error": log.read_text(errors="replace")[-6000:]
            if log.exists()
            else "Native launch failed",
        }
    return {"status": "completed", "log": str(log)}


def validate_native(asset: Path, config_file: Path, output: Path, case: str) -> dict:
    """Adapt and actually simulate a compiled candidate, returning model-readable feedback."""
    config = json.loads(config_file.read_text())
    scripts = Path(config["script_dir"])
    profile = Path(config["profile"])
    identity = {"source": sha256(asset), **native_identity(config_file)}
    folder = output / hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:16]
    folder.mkdir(parents=True, exist_ok=True)
    receipt = folder / "native-feedback.json"
    if receipt.exists():
        return json.loads(receipt.read_text()) | {"reused_completed_result": True}
    started = time.monotonic()
    candidate = folder / "input.usdz"
    if not candidate.exists():
        candidate.write_bytes(asset.read_bytes())
    adapted = folder / "adapted.usdz"
    manifest = adapted.with_suffix(".patchmanifest.json")
    if not manifest.exists():
        adapter_job = folder / "adapt-job.json"
        save_json(
            adapter_job,
            {
                "asset": native_path(candidate),
                "output": native_path(adapted),
                "collision_mode": "standard_joints_orthogonal_compound",
            },
        )
        execution = invoke_native(config, scripts / "apply_compatibility.py", adapter_job, folder)
        if execution["status"] != "completed":
            return {
                **execution,
                "passed": False,
                "stage": "native_adaptation",
                "directory": str(folder),
            }
    lineage = json.loads(manifest.read_text())
    if lineage["source_sha256"] != identity["source"] or lineage["output_sha256"] != sha256(
        adapted
    ):
        raise ValueError("Native adaptation does not match the compiled candidate")
    simulation_job = folder / "simulate-job.json"
    job = prepare_job(
        adapted,
        profile,
        native_path(adapted),
        native_path(profile),
        native_path(folder / "simulation"),
        case,
    )
    job["expected_runner_sha256"] = identity["runner"]
    save_json(simulation_job, job)
    execution = invoke_native(config, scripts / "validate_asset.py", simulation_job, folder)
    if execution["status"] != "completed":
        return {
            **execution,
            "passed": False,
            "stage": "native_simulation",
            "directory": str(folder),
        }
    summary = summarize_native(folder / "simulation/index.json", adapted, profile, case)
    result = {
        **summary,
        "status": "completed",
        "stage": "native_physics",
        "source_asset_sha256": identity["source"],
        "adapted_asset": str(adapted),
        "adaptation_manifest": str(manifest),
        "directory": str(folder),
        "elapsed_seconds": time.monotonic() - started,
        "reused_completed_result": False,
    }
    save_json(receipt, result)
    return result


def summarize_native(index: Path, asset: Path, profile: Path, case: str) -> dict:
    """Reject stale, incomplete, duplicated, or differently configured evidence."""
    expected = json.loads(profile.read_text())
    digest, profile_digest = sha256(asset), sha256(profile)
    rows = [r for r in json.loads(index.read_text()) if r["case"] == case]
    required = {(s, r) for s in expected["scenarios"] for r in range(1, expected["repeats"] + 1)}
    actual = [(r["scenario"], r["repeat"]) for r in rows]
    if len(actual) != len(required) or set(actual) != required:
        raise ValueError("Native scenario/repeat matrix is incomplete or duplicated")
    reports = []
    for row in rows:
        path = (index.parent / row["result"]).resolve()
        path.relative_to(index.parent.resolve())
        result = json.loads(path.read_text())
        if result["asset_sha256"] != digest:
            raise ValueError("Native result belongs to another asset version")
        if result["profile_sha256"] != profile_digest or result["profile"] != expected:
            raise ValueError("Native result used another solver profile")
        if result["engine"] != "omni.physx native CPU solver" or result["steps"] <= 0:
            raise ValueError("Not a native PhysX simulation report")
        if result["passed"] != row["passed"]:
            raise ValueError("Native index disagrees with its detailed report")
        if result["full_native_gate_passed"] and (
            not result["passed"]
            or result["errors"]
            or result["adjacent_collision_policy"] != "standard-joints-enabled"
        ):
            raise ValueError("Full native acceptance contradicts dynamics or collision policy")
        reports.append(result)
    failures = [
        {"scenario": r["scenario"], "errors": r["errors"]} for r in reports if not r["passed"]
    ]
    return {
        "engine": "omni.physx native CPU solver",
        "asset_sha256": digest,
        "profile_sha256": profile_digest,
        "runs": len(reports),
        "dynamics_passed": all(r["passed"] for r in reports),
        "passed": all(r["full_native_gate_passed"] for r in reports),
        "collision_policies": sorted({r["adjacent_collision_policy"] for r in reports}),
        "peak_floor_penetration_m": max(r["peak_floor_penetration_m"] for r in reports),
        "resting_floor_penetration_m": max(r["resting_floor_penetration_m"] for r in reports),
        "max_joint_anchor_error_m": max(r["max_joint_anchor_error_m"] for r in reports),
        "failures": failures,
        "index": str(index.resolve()),
        "scope": "native engineering acceptance; no paper accuracy equivalence",
    }


def attach_native_feedback(
    run: Path,
    attempt: int,
    index: Path,
    asset: Path,
    profile: Path,
    case: str,
    manifest: Path | None = None,
) -> dict:
    """Attach native evidence without rewriting frozen compile/geometry feedback."""
    folder = run / "result" / f"attempt-{attempt:02d}"
    compiled = json.loads((folder / "compile.json").read_text())
    original = Path(compiled["usdz"])
    if not original.is_absolute():
        original = run / original
    source_digest, digest = sha256(original), sha256(asset)
    if source_digest != digest:
        lineage = json.loads(manifest.read_text()) if manifest else []
        if not any(
            r["source_sha256"] == source_digest and r["output_sha256"] == digest for r in lineage
        ):
            raise ValueError("Adapted asset has no matching source/output lineage manifest")
    native = summarize_native(index, asset, profile, case)
    geometry = json.loads((folder / "feedback.json").read_text())
    result = {
        "attempt": attempt,
        "source_candidate_sha256": source_digest,
        "native_asset_sha256": digest,
        "adaptation_manifest": str(manifest) if manifest else None,
        "combined_geometry_gate_passed": geometry["combined_gate_passed"],
        "native_physics": native,
        "full_gate_passed": bool(geometry["combined_gate_passed"] and native["passed"]),
    }
    save_json(folder / f"native-feedback-{digest[:12]}.json", result)
    save_json(run / "native-feedback-latest.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    new = sub.add_parser("prepare")
    for name in ["asset", "profile", "output"]:
        new.add_argument("--" + name, type=Path, required=True)
    for name in ["native-asset", "native-profile", "native-output", "case"]:
        new.add_argument("--" + name, required=True)
    read = sub.add_parser("summarize")
    for name in ["asset", "profile", "index", "output"]:
        read.add_argument("--" + name, type=Path, required=True)
    read.add_argument("--case", required=True)
    attach = sub.add_parser("attach")
    for name in ["run", "asset", "profile", "index"]:
        attach.add_argument("--" + name, type=Path, required=True)
    attach.add_argument("--attempt", type=int, required=True)
    attach.add_argument("--case", required=True)
    attach.add_argument("--manifest", type=Path)
    args = parser.parse_args()
    if args.command == "prepare":
        result = prepare_job(
            args.asset,
            args.profile,
            args.native_asset,
            args.native_profile,
            args.native_output,
            args.case,
        )
    elif args.command == "summarize":
        result = summarize_native(args.index, args.asset, args.profile, args.case)
    else:
        result = attach_native_feedback(
            args.run, args.attempt, args.index, args.asset, args.profile, args.case, args.manifest
        )
    if args.command != "attach":
        save_json(args.output, result)
    print(json.dumps(result))


if __name__ == "__main__":
    main()
