"""Local development evidence: prepare sources, probe diagnostics, evaluate USDZ.

Run from the repository root with python -m articraft.usdcraft_stage --help.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import trimesh

from articraft.geometry_recheck import feedback_summary, recheck_geometry
from articraft.source_geometry import analyze_source
from articraft.usdcraft_benchmark import build_benchmark, render_meshes, save_json
from articraft.usdcraft_evaluation import evaluate_assembly, load_parts
from articraft.usdcraft_io import read_usd_asset


def prepare(root: Path) -> None:
    benchmark = build_benchmark(root)
    results = []
    for record in benchmark["cases"]:
        public = Path(record["public"])
        source = trimesh.load_mesh(public / "source.ply", process=False)
        grid, cells, analysis = analyze_source(source)
        out = root / "observations" / record["case"]
        save_json(out / "source-analysis.json", analysis)
        np.savez_compressed(
            out / "surface-support.npz",
            cells=cells,
            origin=grid.origin,
            pitch=grid.pitch,
            shape=grid.shape,
        )
        (out / "axial.txt").write_text(analysis["encoding"], encoding="utf-8")
        # These are instrument checks, never model reconstruction results.
        probes = {
            "identity": source.copy(),
            "shift_10mm_x": source.copy(),
            "width_plus_10_percent": source.copy(),
            "missing_upper_surface": source.copy(),
        }
        probes["shift_10mm_x"].apply_translation([0.01, 0, 0])
        probes["width_plus_10_percent"].vertices[:, 0] *= 1.1
        cut = probes["missing_upper_surface"]
        cut.update_faces(cut.triangles_center[:, 2] < source.bounds[1, 2] * 0.8)
        cut.remove_unreferenced_vertices()
        for name, candidate in probes.items():
            report = recheck_geometry(source, candidate, grid, cells)
            save_json(out / f"probe-{name}.json", report)
            results.append(
                {
                    "case": record["case"],
                    "probe": name,
                    "scope": "injected diagnostic, not generation",
                    **feedback_summary(report),
                }
            )
        print(
            json.dumps(
                {
                    "prepared": record["case"],
                    "surface_cells": len(cells),
                    "pitch_mm": grid.pitch * 1000,
                }
            ),
            flush=True,
        )
    save_json(root / "diagnostic-probes.json", results)


def evaluate(root: Path, case: str, asset: Path, out: Path) -> dict:
    public, truth = root / "inputs" / case, root / "private_truth" / case
    reference = json.loads((truth / "assembly.json").read_text())
    reference_parts = load_parts(truth, reference)
    prediction, parts = read_usd_asset(asset, out / "parts")
    scores = evaluate_assembly(reference, prediction, reference_parts, parts)
    scores["reference_frame_metrics"] = evaluate_assembly(
        reference, prediction, reference_parts, parts, reference_frame=True
    )
    source = trimesh.load_mesh(public / "source.ply", process=False)
    recheck = recheck_geometry(source, trimesh.util.concatenate(list(parts.values())))
    save_json(out / "recheck.json", recheck)
    save_json(out / "metrics.json", scores)
    render_meshes(list(parts.values()), out / "candidate.png")
    render_meshes(list(reference_parts.values()), out / "reference.png")
    # Render the inferred joint endpoint as a separate inspection artifact.
    from articraft.usdcraft_evaluation import _endpoint

    endpoint = _endpoint(parts, {j["child"]: j for j in prediction["joints"]})
    render_meshes(list(endpoint.values()), out / "endpoint.png")
    print(json.dumps({"case": case, "asset": str(asset), **scores}), flush=True)
    return scores


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("prepare")
    check = sub.add_parser("evaluate")
    check.add_argument("--case", required=True)
    check.add_argument("--asset", type=Path, required=True)
    check.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args.root)
    else:
        evaluate(args.root, args.case, args.asset, args.out)


if __name__ == "__main__":
    main()
