"""File-based modeling tools for an explicitly authorized Codex session backend.

This module prepares evidence and compiles submitted source. It never fabricates
model replies or substitutes known CAD programs for the modeling agent.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import trimesh

from articraft.agent.workspace import LocalWorkspace
from articraft.source_geometry import (
    clip_surface,
    measure_face,
    measure_mesh,
    measure_plane,
    planar_observations,
)
from articraft.usdcraft_benchmark import render_meshes, save_json
from articraft.usdcraft_evaluation import load_parts
from articraft.usdcraft_motion import pose_parts
from articraft.usdcraft_program import API_CONTRACT, audit_program, program_source
from articraft.usdcraft_workflow import ModelingTools

VIEWS = {"oblique": (-1.0, 1.5, -0.8), "minus_y": (0.0, 1.0, -0.15), "plus_y": (0.0, -1.0, -0.15)}
PALETTE = [(83, 135, 170), (218, 148, 86), (92, 170, 144), (164, 132, 187)]


def compact_observations(analysis: dict) -> dict:
    patches = []
    for i, patch in enumerate(analysis["planar_observations"]["patches"]):
        patches.append(
            {
                "id": f"P{i + 1:02d}",
                "axis": patch["axis"],
                "sign": patch["normal_sign"],
                "coordinate": round(patch["coordinate"], 6),
                "area": round(patch["area"], 8),
                "bounds": np.round(patch["bounds"], 6).tolist(),
            }
        )
    return {
        "bounds": analysis["measurement"]["bounds"],
        "units": "metres",
        "observed_planes": patches,
        "oriented_planes": analysis.get("oriented_planar_observations", {}),
        "surface_grid": analysis["grid"],
        "axial_surface_encoding": analysis["encoding"],
        "unknown_space": "no observed surface does not imply empty or solid",
    }


def evidence_views(
    source: trimesh.Trimesh,
    directory: Path,
    parts: dict | None = None,
    assembly: dict | None = None,
) -> dict:
    states = {"closed": parts or {"source": source}}
    if parts is not None:
        joints = {j["child"]: j for j in assembly["joints"]}
        coordinates = {j["name"]: max(j["limits"], key=abs) for j in joints.values()}
        states["open"] = pose_parts(parts, joints, coordinates)
    bounds = np.stack(
        [m.bounds for group in states.values() for m in group.values()] + [source.bounds]
    )
    frame = np.stack([bounds[:, 0].min(0), bounds[:, 1].max(0)])
    files = []
    for name, view in VIEWS.items():
        path = directory / f"source-{name}.png"
        render_meshes([source], path, 480, view=view, frame_bounds=frame)
        files.append(path)
        if parts is not None:
            for state, meshes in states.items():
                path = directory / f"candidate-{state}-{name}.png"
                render_meshes(
                    list(meshes.values()),
                    path,
                    480,
                    view=view,
                    frame_bounds=frame,
                    part_colors=[PALETTE[i % len(PALETTE)] for i in range(len(meshes))],
                )
                files.append(path)
    record = {
        "shared_frame_bounds": frame.tolist(),
        "views": VIEWS,
        "units": "metres",
        "part_color_order": list(parts or {}),
        "files": [
            {"path": str(p), "sha256": hashlib.sha256(p.read_bytes()).hexdigest()} for p in files
        ],
        "scope": "CPU inspection images; no PhysX or photorealistic rendering claim",
    }
    save_json(directory / "views.json", record)
    return record


def concise_feedback(feedback: dict) -> dict:
    result = {
        k: feedback.get(k)
        for k in [
            "compiler_status",
            "compiler_gate_passed",
            "geometry_gate_passed",
            "combined_gate_passed",
            "error",
            "compiler_diagnostics",
            "source_bounds",
            "candidate_bounds",
            "part_bounds",
            "source_surface_within_one_cell",
            "relative_extent_errors",
            "geometry_motion_gate_passed",
            "local_geometry_required",
            "local_geometry_gate_passed",
            "local_geometry",
            "native_physics_required",
            "native_physics",
            "full_gate_passed",
            "acceptance_scope",
            "status",
        ]
    }
    if "geometry" in feedback:
        g = feedback["geometry"]
        result["geometry"] = {
            k: g[k]
            for k in ["coverage", "missing_observed", "added_in_unknown", "first_surface_maps"]
        }
    if "motion" in feedback:
        result["motion"] = feedback["motion"]
    return result


def plane_feedback(analysis: dict, candidate: trimesh.Trimesh) -> dict:
    """Locate missing observed planes; the candidate never receives part/joint GT."""
    observed = compact_observations(analysis)["observed_planes"]
    rebuilt = planar_observations(candidate)["patches"]
    tolerance = analysis["grid"]["pitch"] * 1.5
    missing = []
    for patch in observed:
        axis = "XYZ".index(patch["axis"])
        other = [i for i in range(3) if i != axis]
        lo, hi = np.asarray(patch["bounds"])[:, other]
        matches = []
        for new in rebuilt:
            if new["axis"] != patch["axis"] or new["normal_sign"] != patch["sign"]:
                continue
            b = np.asarray(new["bounds"])[:, other]
            overlap = np.maximum(0, np.minimum(hi, b[1]) - np.maximum(lo, b[0])).prod()
            if overlap > 0:
                matches.append(new["coordinate"] - patch["coordinate"])
        closest = min(matches, key=abs) if matches else None
        if closest is None or abs(closest) > tolerance:
            missing.append({**patch, "nearest_parallel_plane_offset": closest})
    return {
        "missing_or_displaced_observed_planes": sorted(
            missing, key=lambda p: p["area"], reverse=True
        )[:10],
        "total_missing_plane_groups": len(missing),
        "coordinate_tolerance": tolerance,
        "scope": "diagnostic axis-plane and projected-bounds comparison; not an acceptance replacement",
    }


def host_for_run(run: Path) -> ModelingTools:
    config = json.loads((run / "session.json").read_text())
    host = ModelingTools(
        Path(config["public_input"]),
        LocalWorkspace(output_dir=run.parent, physics_enabled=True, timeout_seconds=180),
        run,
        Path(config["native_config"]) if config.get("native_config") else None,
        Path(config["geometry_profile"]) if config.get("geometry_profile") else None,
    )
    host.attempt = max(
        (int(p.name.split("-")[1]) for p in (run / "result").glob("attempt-*")), default=0
    )
    return host


def prepare(
    public: Path,
    output: Path,
    backend: str,
    native_config: Path | None = None,
    geometry_profile: Path | None = None,
) -> Path:
    workspace = LocalWorkspace(output_dir=output, physics_enabled=True, timeout_seconds=180)
    run = workspace.create_run(f"session-{public.name}-{datetime.now():%Y%m%d-%H%M%S}")
    frozen = {}
    for name, path in [("native_config", native_config), ("geometry_profile", geometry_profile)]:
        if path:
            target = run / (name + ".json")
            target.write_bytes(path.read_bytes())
            frozen[name] = str(target.resolve())
    host = ModelingTools(
        public,
        workspace,
        run,
        Path(frozen["native_config"]) if native_config else None,
        Path(frozen["geometry_profile"]) if geometry_profile else None,
    )
    save_json(
        run / "session.json",
        {
            "backend": backend,
            "public_input": str(public.resolve()),
            "case": public.name,
            "private_truth_supplied": False,
            **frozen,
        },
    )
    request = {
        "task": host.input,
        "source_observations": compact_observations(host.analysis),
        "source_mesh_sha256": hashlib.sha256(
            (public / host.input["source"]).read_bytes()
        ).hexdigest(),
        "required_process": "Observe source images and measurements; write a measured plan; author full Python; compile; inspect actual feedback and images; revise natural errors; never access GT or category target code",
    }
    save_json(run / "model-request.json", request)
    (run / "SDK_CONTRACT.md").write_text(API_CONTRACT, encoding="utf-8")
    evidence_views(host.source, run / "source-views")
    return run


def submit(run: Path, source_file: Path, notes_file: Path, backend: str) -> dict:
    host = host_for_run(run)
    source = program_source(source_file.read_text(encoding="utf-8"))
    notes = json.loads(notes_file.read_text())
    previous = run / "result" / f"attempt-{host.attempt:02d}"
    same_source = (previous / "main.py").exists() and (previous / "main.py").read_text() == source
    if same_source and (previous / "feedback.json").exists():
        saved = json.loads((previous / "feedback.json").read_text())
        if (
            saved.get("status", "completed") == "completed"
            and host.validation_matches(saved)
            and saved.get("native_physics", {}).get("status") != "execution_error"
        ):
            return {
                "attempt": host.attempt,
                "result_directory": str(previous),
                "source_unchanged": True,
                "compiled_this_call": False,
                **concise_feedback(saved),
            }
    if same_source:
        feedback = host.evaluate(json.loads((previous / "compile.json").read_text()), previous)
    else:
        feedback = host.submit(source)
    out = run / "result" / f"attempt-{host.attempt:02d}"
    receipt = {
        "backend": backend,
        "source_sha256": hashlib.sha256(source.encode()).hexdigest(),
        "submitted_file": str(source_file),
        "notes": notes,
        "static_audit": audit_program(source),
        "code_transport": "outer whitespace/newlines normalized; one optional fence removed; program body unchanged",
    }
    if not (out / "agent-receipt.json").exists():
        save_json(out / "agent-receipt.json", receipt)
    if feedback["compiler_status"] == "success":
        assembly = json.loads((out / "parts/assembly.json").read_text())
        parts = load_parts(out / "parts", assembly)
        feedback["plane_feedback"] = plane_feedback(
            host.analysis, trimesh.util.concatenate(list(parts.values()))
        )
        save_json(out / "plane-feedback.json", feedback["plane_feedback"])
        evidence_views(host.source, out / "views", parts, assembly)
        for region in feedback["local_geometry"]["regions"][:3]:
            if region["critical"]:
                bounds = np.asarray(region["bounds"])
                margin = host.grid.pitch * 2
                bounds = bounds + np.array([-margin, margin])[:, None]
                patch = clip_surface(host.source, bounds)
                candidate = clip_surface(trimesh.util.concatenate(list(parts.values())), bounds)
                if len(patch.faces) and len(candidate.faces):
                    directory = out / "local-views" / region["id"]
                    render_meshes([patch], directory / "source.png", 480, frame_bounds=bounds)
                    render_meshes(
                        [candidate], directory / "candidate.png", 480, frame_bounds=bounds
                    )
    compact = concise_feedback(feedback)
    if "plane_feedback" in feedback:
        compact["plane_feedback"] = feedback["plane_feedback"]
    save_json(out / "agent-feedback.json", compact)
    return {
        "attempt": host.attempt,
        "result_directory": str(out),
        "compiled_this_call": not same_source,
        "resumed_existing_submission": same_source,
        **compact,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    new = sub.add_parser("prepare")
    new.add_argument("--public", type=Path, required=True)
    new.add_argument("--output", type=Path, required=True)
    new.add_argument("--backend", required=True)
    new.add_argument("--native-config", type=Path)
    new.add_argument("--geometry-profile", type=Path)
    build = sub.add_parser("submit")
    build.add_argument("--run", type=Path, required=True)
    build.add_argument("--source", type=Path, required=True)
    build.add_argument("--notes", type=Path, required=True)
    build.add_argument("--backend", required=True)
    query = sub.add_parser("measure")
    query.add_argument("--run", type=Path, required=True)
    query.add_argument("--bounds", type=str)
    query.add_argument("--axis", type=int, choices=[0, 1, 2])
    query.add_argument("--normal", type=str, help="Observed world XYZ normal as JSON array")
    query.add_argument("--side", type=int, choices=[-1, 1], default=1)
    args = parser.parse_args()
    if args.command == "prepare":
        result = {
            "run": str(
                prepare(
                    args.public,
                    args.output,
                    args.backend,
                    args.native_config,
                    args.geometry_profile,
                )
            )
        }
    elif args.command == "submit":
        result = submit(args.run, args.source, args.notes, args.backend)
    else:
        host = host_for_run(args.run)
        bounds = np.asarray(json.loads(args.bounds)) if args.bounds else None
        result = (
            measure_plane(host.source, np.asarray(json.loads(args.normal)), bounds)
            if args.normal
            else (
                measure_mesh(host.source, bounds)
                if args.axis is None
                else measure_face(host.source, args.axis, args.side, bounds)
            )
        )
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
