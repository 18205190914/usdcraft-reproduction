"""Geometry tests for exact collision proxies; native contacts are tested separately."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest
import trimesh
from pxr import Usd, UsdGeom

spec = importlib.util.spec_from_file_location(
    "orthogonal_compound", Path(__file__).parents[1] / "scripts/isaac/orthogonal_compound.py"
)
compound = importlib.util.module_from_spec(spec)
spec.loader.exec_module(compound)


def usd_mesh(mesh):
    stage = Usd.Stage.CreateInMemory()
    shape = UsdGeom.Mesh.Define(stage, "/Shape")
    shape.CreatePointsAttr(mesh.vertices.tolist())
    shape.CreateFaceVertexCountsAttr([3] * len(mesh.faces))
    shape.CreateFaceVertexIndicesAttr(mesh.faces.ravel().tolist())
    return stage, shape


def l_shape():
    a = trimesh.creation.box([0.3, 0.1, 0.1])
    a.apply_translation([0.15, 0.05, 0.05])
    b = trimesh.creation.box([0.1, 0.3, 0.1])
    b.apply_translation([0.05, 0.15, 0.05])
    return trimesh.boolean.union([a, b], engine="manifold")


def test_concave_corner_stays_empty_and_volume_is_preserved():
    stage, mesh = usd_mesh(l_shape())
    boxes, metrics = compound.partition(mesh)
    assert metrics["compound_volume"] == pytest.approx(0.005, abs=1e-9)
    assert metrics["absolute_volume_error"] < 1e-12
    for box in boxes:
        center, size = np.asarray(box["center"]), np.asarray(box["dimensions"])
        assert not np.all(np.abs(np.array([0.2, 0.2, 0.05]) - center) < size / 2)
    root, authored = compound.author(stage, mesh, boxes)
    assert authored and stage.GetPrimAtPath(root).IsValid()


def test_baked_rotation_is_explicitly_unsupported():
    mesh = l_shape()
    mesh.apply_transform(trimesh.transformations.rotation_matrix(0.37, [0, 0, 1]))
    _stage, source = usd_mesh(mesh)
    with pytest.raises(compound.UnsupportedMesh, match="Non-axis-aligned"):
        compound.partition(source)


def test_open_mesh_cannot_be_filled_as_a_solid():
    mesh = l_shape()
    mesh.update_faces(np.arange(len(mesh.faces)) != 0)
    _stage, source = usd_mesh(mesh)
    with pytest.raises(compound.UnsupportedMesh, match="closed two-manifold"):
        compound.partition(source)


@pytest.mark.parametrize(
    "value,expected", [(-1, 0), (0, 0), (0.125, 0), (0.26, 1), (0.375, 1), (0.4, 2), (1.1, 3)]
)
def test_binary_nearest_plane_keeps_boundary_and_lower_tie_semantics(value, expected):
    assert compound.nearest_plane([0, 0.25, 0.5, 1], value) == expected
