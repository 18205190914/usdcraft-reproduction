from __future__ import annotations

import numpy as np
import pytest
import trimesh

from articraft.geometry_recheck import compare_part, recheck_geometry
from articraft.source_geometry import SurfaceGrid, analyze_source, measure_mesh, surface_support


def cube():
    return trimesh.creation.box(extents=[0.24, 0.16, 0.12])


def test_surface_support_does_not_fill_unknown_interior():
    mesh = cube()
    grid = SurfaceGrid.from_mesh(mesh, 32)
    cells = surface_support(mesh, grid)
    center = tuple(grid.indices(np.zeros((1, 3)))[0])
    assert center not in set(map(tuple, cells))
    assert max(grid.shape) == 32
    assert len(cells) > 100


def test_identical_surface_and_shared_frame():
    mesh = cube()
    report = recheck_geometry(mesh, mesh.copy(), SurfaceGrid.from_mesh(mesh, 32))
    assert report["coverage"] == 1
    assert report["missing_observed"]["count"] == 0
    assert report["added_in_unknown"]["count"] == 0
    assert all(f["p90_absolute_offset"] == 0 for f in report["first_surface_maps"].values())


def test_translation_is_detected_without_alignment():
    mesh = cube()
    moved = mesh.copy()
    moved.apply_translation([0.03, 0, 0])
    report = recheck_geometry(mesh, moved, SurfaceGrid.from_mesh(mesh, 48))
    assert report["coverage"] < 0.8
    assert report["candidate_cells_outside_source_bounds"] > 0
    assert (
        abs(report["first_surface_maps"]["X+"]["median_axis_offset"] - 0.03)
        < report["grid"]["pitch"]
    )
    assert report["added_in_unknown"]["interpretation"].startswith("requires inspection")


def test_local_query_clips_triangles_crossing_region():
    mesh = cube()
    bounds = np.array([[-0.02, -0.02, 0.05], [0.02, 0.02, 0.07]])
    report = measure_mesh(mesh, bounds)
    assert report["observed"]
    assert np.allclose(report["bounds"], [[-0.02, -0.02, 0.06], [0.02, 0.02, 0.06]])
    assert not measure_mesh(mesh, np.array([[-0.01] * 3, [0.01] * 3]))["observed"]
    assert measure_mesh(mesh, np.array([[-1.0] * 3, [1.0] * 3]))["watertight"]


def test_slice_text_is_deterministic_and_budgeted():
    _, _, first = analyze_source(cube(), 48, 16000)
    _, _, second = analyze_source(cube(), 48, 16000)
    assert first["encoding"] == second["encoding"]
    assert "?=unknown, never empty/solid" in first["encoding"]
    assert len(first["encoding"]) <= 16000


def test_local_comparison_does_not_use_alignment_for_score():
    mesh = cube()
    moved = mesh.copy().apply_translation([0.02, 0, 0])
    report = compare_part(mesh, moved, 5000)
    assert report["raw_chamfer_squared"] > report["translation_only_diagnostic_chamfer_squared"]
    assert not report["diagnostic_alignment_used_for_acceptance"]


@pytest.mark.parametrize("resolution", [0, 1])
def test_invalid_grid_resolution(resolution):
    with pytest.raises(ValueError):
        SurfaceGrid.from_mesh(cube(), resolution)
