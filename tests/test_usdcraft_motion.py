from __future__ import annotations

import numpy as np
import pytest
import trimesh

from articraft.usdcraft_benchmark import reference_cases
from articraft.usdcraft_motion import check_motion, pose_parts, solid_overlaps


def test_containment_is_detected_without_surface_intersection():
    outer = trimesh.creation.box(extents=[1, 1, 1])
    inner = trimesh.creation.box(extents=[0.2, 0.2, 0.2])
    overlaps = solid_overlaps({"outer": outer, "inner": inner})
    assert len(overlaps) == 1
    assert np.isclose(overlaps[0]["fraction_of_smaller_part"], 1)


def test_part_in_real_cavity_is_not_reported_as_solid_overlap():
    outer = trimesh.creation.box(extents=[1, 1, 1])
    cavity = trimesh.creation.box(extents=[0.8, 0.8, 0.8])
    shell = trimesh.boolean.difference([outer, cavity], engine="manifold")
    inner = trimesh.creation.box(extents=[0.2, 0.2, 0.2])
    assert not solid_overlaps({"shell": shell, "inner": inner})


def test_wrong_opening_direction_is_rejected_by_motion_sweep():
    case = reference_cases()["cabinet"]
    parts = dict(case["parts"])
    assembly = {"joints": case["joints"]}
    correct = check_motion(assembly, parts, 9)
    assert correct["assessed"] and correct["passed"]
    wrong = {"joints": [{**case["joints"][0], "axis": [0, 0, -1]}]}
    report = check_motion(wrong, parts, 9)
    assert report["assessed"] and not report["passed"]
    assert report["failed_states"] > 0


def test_nested_pose_does_not_depend_on_joint_dictionary_order():
    parts = {
        "base": trimesh.creation.box(extents=[0.01] * 3),
        "arm": trimesh.creation.box(extents=[0.01] * 3).apply_translation([0.1, 0, 0]),
        "tip": trimesh.creation.box(extents=[0.01] * 3).apply_translation([0.2, 0, 0]),
    }
    joints = {
        "tip": {
            "name": "wrist",
            "parent": "arm",
            "axis": [0, 0, 1],
            "origin": [0.1, 0, 0],
            "type": "revolute",
        },
        "arm": {
            "name": "shoulder",
            "parent": "base",
            "axis": [0, 0, 1],
            "origin": [0, 0, 0],
            "type": "revolute",
        },
    }
    result = pose_parts(parts, joints, {"wrist": np.pi / 2, "shoulder": np.pi / 2})
    assert np.allclose(result["tip"].centroid, [-0.1, 0.1, 0])


def test_open_mesh_is_unassessed_instead_of_passing_volume_check():
    part = trimesh.creation.box()
    part.update_faces(np.arange(len(part.faces)) != 0)
    report = check_motion({"joints": []}, {"open": part})
    assert not report["assessed"] and not report["passed"]


def test_missing_floor_contacts_cannot_pass_a_falling_body(monkeypatch, tmp_path):
    mujoco = pytest.importorskip("mujoco")
    from articraft import usdcraft_dynamics

    spec = mujoco.MjSpec()
    spec.option.gravity = [0, 0, -9.81]
    spec.worldbody.add_geom(
        name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE, size=[1, 1, 0.1], contype=0, conaffinity=0
    )
    body = spec.worldbody.add_body(name="falling", pos=[0, 0, 0.2])
    body.add_freejoint()
    body.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.03] * 3, mass=1, contype=0, conaffinity=0)
    monkeypatch.setattr(usdcraft_dynamics, "_build_spec", lambda asset: spec)
    result = usdcraft_dynamics.inspect_dynamics(tmp_path / "unused.usdz", "drop", True, seconds=0.5)
    assert result["finite"] and result["peak_floor_penetration_mm"] == 0
    assert result["fallen_through_floor"] and not result["strict_passed"]


def test_compiler_diagnostics_are_not_treated_as_asset_acceptance(tmp_path):
    import json

    from articraft.agent.workspace import LocalWorkspace
    from articraft.usdcraft_workflow import ModelingTools

    public = tmp_path / "public"
    public.mkdir()
    trimesh.util.concatenate(
        [trimesh.creation.box(extents=[0.1] * 3), trimesh.creation.box(extents=[0.02] * 3)]
    ).export(public / "source.ply")
    (public / "input.json").write_text(
        json.dumps({"source": "source.ply", "units": "metres", "request": "test geometry only"})
    )
    source = """from build123d import Box, Axis
from articraft.sdk import RigidBodyAssembly, Material, JointDOF, JointAxis, TestContext
m = RigidBodyAssembly("nested")
a = m.rigid_body("outer")
a.add(Box(.1,.1,.1),name="outer",material=Material.ABS_PLASTIC)
b = m.rigid_body("inner")
b.add(Box(.02,.02,.02),name="inner",material=Material.ABS_PLASTIC)
axis = Axis((0,0,0),(0,0,1))
m.joint("joint",a.at(axis),b.at(axis),dofs=(JointDOF(JointAxis.ROT_Z,limits=(-.5,.5)),))
m.articulation("main",root=a,joints=["joint"])
object_model = m
def run_tests():
    return TestContext(object_model).report()
"""
    workspace = LocalWorkspace(output_dir=tmp_path / "runs", physics_enabled=True)
    host = ModelingTools(public, workspace, workspace.create_run("diagnostic-gate"))
    feedback = host.submit(source)
    assert feedback["compiler_status"] == "success"
    assert feedback["geometry_gate_passed"]
    assert any(d["kind"] == "overlap" for d in feedback["compiler_diagnostics"])
    assert not feedback["compiler_gate_passed"]
    assert not feedback["combined_gate_passed"]
