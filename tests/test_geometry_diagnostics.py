"""Synthetic diagnostic controls, never counted as natural model repair evidence."""

from __future__ import annotations

import numpy as np
import pytest
import trimesh

from articraft.geometry_recheck import diagnose_geometry
from articraft.source_geometry import SurfaceGrid, measure_plane, oriented_planar_observations


def box(extents: list[float], center: list[float]) -> trimesh.Trimesh:
    mesh = trimesh.creation.box(extents)
    mesh.apply_translation(center)
    return mesh


def test_rotated_plane_is_measured_in_unchanged_world_frame() -> None:
    mesh = trimesh.creation.box([0.2, 0.1, 0.05])
    transform = trimesh.transformations.rotation_matrix(0.37, [0, 0, 1])
    mesh.apply_transform(transform)
    vertices = mesh.vertices.copy()
    normal = transform[:3, :3] @ [1, 0, 0]
    result = measure_plane(mesh, normal, angle_degrees=1)
    assert result["observed"]
    assert result["plane_offset_range_m"] == pytest.approx([0.1, 0.1])
    patches = oriented_planar_observations(mesh)["patches"]
    assert any(np.dot(p["normal"], normal) > 0.999 for p in patches)
    np.testing.assert_array_equal(vertices, mesh.vertices)


def test_local_detail_missing_without_a_bbox_change_is_localized() -> None:
    plate = box([0.30, 0.30, 0.01], [0, 0, 0.005])
    left = box([0.04, 0.04, 0.02], [-0.1, 0, 0.02])
    right = box([0.04, 0.04, 0.02], [0.1, 0, 0.02])
    source = trimesh.boolean.union([plate, left, right], engine="manifold")
    candidate = trimesh.boolean.union([plate, right], engine="manifold")
    np.testing.assert_allclose(source.bounds, candidate.bounds)
    grid = SurfaceGrid.from_mesh(source)
    report = diagnose_geometry(source, {"moving_part": candidate}, grid)
    assert not report["passed"]
    region = next(r for r in report["regions"] if r["critical"])
    assert region["bounds"][1][0] < -0.07
    assert "moving_part" in region["nearest_candidate_parts"]
    assert region["maximum_surface_distance_m"] > 0.015
    fixed = diagnose_geometry(source, {"moving_part": source.copy()}, grid)
    assert fixed["passed"] and fixed["p95_surface_distance_m"] < 1e-7


def test_added_geometry_in_unknown_space_is_not_automatically_an_error() -> None:
    source = trimesh.creation.box([0.2, 0.1, 0.05])
    addition = box([0.02, 0.02, 0.02], [0, 0, 0.1])
    report = diagnose_geometry(
        source, {"observed": source.copy(), "completion": addition}, SurfaceGrid.from_mesh(source)
    )
    assert report["passed"]
    assert report["source_support_within_tolerance"] == 1
    assert not report["alignment_applied"]


def test_translation_is_reported_without_registration() -> None:
    source = trimesh.creation.box([0.2, 0.1, 0.05])
    candidate = source.copy()
    candidate.apply_translation([0, 0, 0.01])
    result = diagnose_geometry(source, {"shifted": candidate}, SurfaceGrid.from_mesh(source))
    assert not result["passed"]
    assert result["p95_surface_distance_m"] > 0.009
    assert not result["alignment_applied"]
