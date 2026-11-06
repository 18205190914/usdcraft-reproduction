"""Compare candidate observed surfaces in the SOURCE grid; unknown is not empty."""

from __future__ import annotations

import igl
import numpy as np
import trimesh
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

from articraft.source_geometry import SurfaceGrid, surface_support


def sample_surface(mesh: trimesh.Trimesh, count: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    triangles = mesh.triangles[rng.choice(len(mesh.faces), count, p=mesh.area_faces / mesh.area)]
    uv = rng.random((count, 2))
    uv[uv.sum(1) > 1] = 1 - uv[uv.sum(1) > 1]
    return (
        triangles[:, 0]
        + uv[:, :1] * (triangles[:, 1] - triangles[:, 0])
        + uv[:, 1:] * (triangles[:, 2] - triangles[:, 0])
    )


def chamfer_squared(first: np.ndarray, second: np.ndarray) -> float:
    return float(
        (cKDTree(second).query(first)[0] ** 2).mean()
        + (cKDTree(first).query(second)[0] ** 2).mean()
    )


def _first_surfaces(cells: np.ndarray, axis: int, side: int) -> dict:
    other = [a for a in range(3) if a != axis]
    values: dict[tuple[int, int], int] = {}
    for cell in cells:
        key = tuple(cell[other])
        depth = int(cell[axis])
        if key not in values or side * depth > side * values[key]:
            values[key] = depth
    return values


def recheck_geometry(
    source: trimesh.Trimesh,
    candidate: trimesh.Trimesh,
    grid: SurfaceGrid | None = None,
    source_cells: np.ndarray | None = None,
) -> dict:
    grid = grid or SurfaceGrid.from_mesh(source)
    source_cells = surface_support(source, grid) if source_cells is None else source_cells
    candidate_cells = surface_support(candidate, grid)
    observed = set(map(tuple, source_cells))
    rebuilt = set(map(tuple, candidate_cells))
    missed = sorted(observed - rebuilt)
    added = sorted(rebuilt - observed)
    faces = {}
    for axis in range(3):
        for side in (-1, 1):
            first = _first_surfaces(source_cells, axis, side)
            second = _first_surfaces(candidate_cells, axis, side)
            common = first.keys() & second.keys()
            shifts = np.asarray([second[k] - first[k] for k in common]) * grid.pitch
            faces[f"{'XYZ'[axis]}{'+' if side > 0 else '-'}"] = {
                "common_columns": len(common),
                "source_only_columns": len(first.keys() - second.keys()),
                "candidate_only_columns": len(second.keys() - first.keys()),
                "median_axis_offset": float(np.median(shifts)) if len(shifts) else None,
                "p90_absolute_offset": float(np.quantile(abs(shifts), 0.9))
                if len(shifts)
                else None,
            }

    def region(cells):
        if not cells:
            return None
        values = np.asarray(cells)
        return [
            list(grid.origin + values.min(0) * grid.pitch),
            list(grid.origin + (values.max(0) + 1) * grid.pitch),
        ]

    return {
        "coverage": len(observed & rebuilt) / len(observed),
        "grid": grid.to_dict(),
        "source_surface_cells": len(observed),
        "candidate_surface_cells": len(rebuilt),
        "missing_observed": {
            "count": len(missed),
            "region": region(missed),
            "cells": np.asarray(missed).tolist(),
        },
        "added_in_unknown": {
            "count": len(added),
            "region": region(added),
            "cells": np.asarray(added).tolist(),
            "interpretation": "requires inspection; may be justified completion",
        },
        "candidate_cells_outside_source_bounds": int(
            ((candidate_cells < 0) | (candidate_cells >= grid.shape)).any(1).sum()
        ),
        "first_surface_maps": faces,
        "first_surface_depth_uncertainty": grid.pitch,
        "units": "input metric frame; no alignment or rescaling performed",
    }


def compare_part(source: trimesh.Trimesh, candidate: trimesh.Trimesh, count: int = 20000) -> dict:
    a, b = sample_surface(source, count), sample_surface(candidate, count, 1)
    shift = np.median(a, axis=0) - np.median(b, axis=0)
    distances = cKDTree(a).query(b)[0]
    return {
        "raw_chamfer_squared": chamfer_squared(a, b),
        "candidate_to_source_p90": float(np.quantile(distances, 0.9)),
        "diagnostic_translation": shift.tolist(),
        "translation_only_diagnostic_chamfer_squared": chamfer_squared(a, b + shift),
        "diagnostic_alignment_used_for_acceptance": False,
        "note": "translation estimate is diagnostic, not a registration-based evaluation score",
    }


def triangle_distances(
    points: np.ndarray, mesh: trimesh.Trimesh
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    squared, faces, closest = igl.point_mesh_squared_distance(
        np.asarray(points, dtype=np.float64),
        np.asarray(mesh.vertices, dtype=np.float64),
        np.asarray(mesh.faces, dtype=np.int64),
    )
    return np.sqrt(np.maximum(squared, 0)), faces, closest


def diagnose_geometry(
    source: trimesh.Trimesh,
    parts: dict[str, trimesh.Trimesh],
    grid: SurfaceGrid,
    samples: int = 20000,
    distance_tolerance_pitch: float = 0.5,
    normal_tolerance_degrees: float = 25.0,
    critical_region_fraction: float = 0.005,
) -> dict:
    """Locate observed surface disagreements with exact triangle distances.

    Regions are diagnostic source observations; added unknown-space surfaces are
    never automatically classified as solid/empty or penalized here.
    """
    candidate = trimesh.util.concatenate(list(parts.values()))
    face_parts = np.concatenate([np.repeat(name, len(mesh.faces)) for name, mesh in parts.items()])
    points = sample_surface(source, samples)
    # Repeat the deterministic sampling choice to recover source face normals.
    rng = np.random.default_rng(0)
    source_faces = rng.choice(len(source.faces), samples, p=source.area_faces / source.area)
    distances, faces, closest = triangle_distances(points, candidate)
    cosine = np.abs((source.face_normals[source_faces] * candidate.face_normals[faces]).sum(1))
    angles = np.degrees(np.arccos(np.clip(cosine, 0, 1)))
    tolerance = grid.pitch * distance_tolerance_pitch
    bad_distance = distances > tolerance
    bad_normal = angles > normal_tolerance_degrees
    bad = bad_distance | bad_normal
    # Sampling is sparser than the axial surface grid: group spatially connected
    # evidence at its sampling density, while measuring errors at the original tolerance.
    grouping_pitch = max(grid.pitch, 2 * np.sqrt(source.area / samples))
    grouping_cells = np.floor((points[bad] - grid.origin) / grouping_pitch).astype(np.int64)
    cells, inverse = np.unique(grouping_cells, axis=0, return_inverse=True)
    regions = []
    if len(cells):
        pairs = cKDTree(cells).query_pairs(1.01, p=np.inf, output_type="ndarray")
        graph = coo_matrix(
            (np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(len(cells), len(cells))
        )
        count, labels = connected_components(graph, directed=False)
        for label in range(count):
            selected = np.flatnonzero(bad)[labels[inverse] == label]
            fraction = len(selected) / samples
            values = points[selected]
            names, counts = np.unique(face_parts[faces[selected]], return_counts=True)
            regions.append(
                {
                    "fraction_of_source_samples": fraction,
                    "critical": fraction >= critical_region_fraction,
                    "bounds": [values.min(0).tolist(), values.max(0).tolist()],
                    "p95_surface_distance_m": float(np.quantile(distances[selected], 0.95)),
                    "maximum_surface_distance_m": float(distances[selected].max()),
                    "p95_normal_angle_degrees": float(np.quantile(angles[selected], 0.95)),
                    "distance_failure_samples": int(bad_distance[selected].sum()),
                    "normal_failure_samples": int(bad_normal[selected].sum()),
                    "candidate_to_source_median_delta_m": np.median(
                        points[selected] - closest[selected], axis=0
                    ).tolist(),
                    "nearest_candidate_parts": dict(
                        zip(names.tolist(), counts.tolist(), strict=True)
                    ),
                    "interpretation": "observed local surface discrepancy; infer CAD repair from measurements, not GT",
                }
            )
    regions.sort(key=lambda r: r["fraction_of_source_samples"], reverse=True)
    for i, region in enumerate(regions):
        region["id"] = f"R{i + 1:02d}"
    return {
        "samples": samples,
        "region_grouping_pitch_m": float(grouping_pitch),
        "distance_tolerance_m": tolerance,
        "normal_tolerance_degrees": normal_tolerance_degrees,
        "critical_region_fraction": critical_region_fraction,
        "source_support_within_tolerance": float((~bad_distance).mean()),
        "p95_surface_distance_m": float(np.quantile(distances, 0.95)),
        "critical_regions": sum(r["critical"] for r in regions),
        "passed": not any(r["critical"] for r in regions),
        "regions": regions[:12],
        "total_regions": len(regions),
        "alignment_applied": False,
        "scope": "observed source surfaces only; source-to-candidate exact triangle distance; unknown additions require review",
    }


def feedback_summary(report: dict) -> dict:
    """Keep cells in the evidence file, but return bounded summaries to the agent."""
    return {
        k: ({x: y for x, y in v.items() if x != "cells"} if isinstance(v, dict) else v)
        for k, v in report.items()
    }
