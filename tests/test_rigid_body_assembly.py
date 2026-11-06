from __future__ import annotations

import math

import numpy as np
import pytest
from build123d import Box

from articraft.sdk.assembly import (
    WORLD,
    Articulation,
    JointAxis,
    JointDOF,
    JointFrame,
    Mat4,
    PhysicsState,
    ResolvedRigidBodyAssembly,
    RigidBodyAssembly,
)
from articraft.sdk.bodies import RigidBody
from articraft.sdk.errors import LoopClosureError, ValidationError


def add_body(assembly: RigidBodyAssembly, name: str) -> RigidBody:
    body = assembly.rigid_body(name)
    body.add(Box(0.1, 0.1, 0.1), name="shape")
    return body


def four_bar(*, second_closure: bool = False) -> RigidBodyAssembly:
    assembly = RigidBodyAssembly("four_bar")
    ground = add_body(assembly, "ground")
    crank = add_body(assembly, "crank")
    coupler = add_body(assembly, "coupler")
    rocker = add_body(assembly, "rocker")
    revolute = (JointDOF(JointAxis.ROT_Z),)
    ground_crank = assembly.joint("ground_crank", ground.at(), crank.at(), dofs=revolute)
    crank_coupler = assembly.joint(
        "crank_coupler",
        crank.at(JointFrame(xyz=(1.0, 0.0, 0.0))),
        coupler.at(),
        dofs=revolute,
    )
    coupler_rocker = assembly.joint(
        "coupler_rocker",
        coupler.at(JointFrame(xyz=(1.0, 0.0, 0.0))),
        rocker.at(),
        dofs=revolute,
    )
    assembly.joint(
        "rocker_ground",
        rocker.at(),
        ground.at(JointFrame(xyz=(2.0, 0.0, 0.0))),
        dofs=revolute,
    )
    if second_closure:
        assembly.joint(
            "crank_rocker",
            crank.at(JointFrame(xyz=(2.0, 0.0, 0.0))),
            rocker.at(),
            dofs=revolute,
        )
    assembly.articulation(
        "main",
        root=ground,
        joints=(ground_crank, crank_coupler, coupler_rocker),
    )
    return assembly


def test_joint_dofs_are_usd_d6_axes_and_unlisted_axes_are_locked() -> None:
    assembly = RigidBodyAssembly("slider_hinge")
    base = add_body(assembly, "base")
    moving = add_body(assembly, "moving")
    joint = assembly.joint(
        "motion",
        base.at(),
        moving.at(),
        dofs=(
            JointDOF("rotZ", limits=(-1.0, 1.0)),
            JointDOF(JointAxis.TRANS_X, limits=(-0.2, 0.3)),
        ),
    )
    assembly.articulation("main", root=base)

    resolved = assembly.resolve()

    assert [dof.axis for dof in joint.dofs] == [JointAxis.TRANS_X, JointAxis.ROT_Z]
    assert not resolved.has_closed_loops
    posed = resolved.forward_kinematics({"motion.transX": 0.1, "motion.rotZ": 0.5})
    assert posed.dof_positions["motion.transX"] == pytest.approx(0.1)
    assert posed.dof_positions["motion.rotZ"] == pytest.approx(0.5)


def test_joint_limits_must_include_the_frame_defined_zero_configuration() -> None:
    with pytest.raises(ValidationError, match="contain the zero configuration"):
        JointDOF(JointAxis.ROT_X, limits=(0.1, 1.0))
    with pytest.raises(ValidationError, match="duplicate DOF axes"):
        assembly = RigidBodyAssembly("duplicate")
        a = add_body(assembly, "a")
        b = add_body(assembly, "b")
        assembly.joint(
            "joint",
            a.at(),
            b.at(),
            dofs=(JointDOF("rotX"), JointDOF("rotX")),
        )


def test_ordinary_tree_articulation_is_inferred_and_supports_forward_kinematics() -> None:
    assembly = RigidBodyAssembly("arm")
    base = add_body(assembly, "base")
    link = add_body(assembly, "link")
    tip = add_body(assembly, "tip")
    hinge = assembly.joint(
        "hinge",
        base.at(JointFrame(xyz=(1.0, 0.0, 0.0))),
        link.at(),
        dofs=(JointDOF(JointAxis.ROT_Z, limits=(-math.pi, math.pi)),),
    )
    assembly.joint("tip_mount", link.at(JointFrame(xyz=(1.0, 0.0, 0.0))), tip.at())
    assembly.articulation("arm", root=base)

    resolved = assembly.resolve()
    posed = resolved.forward_kinematics({hinge.dof_id(hinge.dofs[0]): math.pi / 2})

    assert [item.joint.name for item in resolved.joints if item.articulation] == [
        "hinge",
        "tip_mount",
    ]
    assert np.allclose(posed.matrix(link)[:3, 3], (1.0, 0.0, 0.0))
    assert np.allclose(posed.matrix(tip)[:3, 3], (1.0, 1.0, 0.0), atol=1e-9)


@pytest.mark.parametrize("second_closure", [False, True])
def test_closed_loop_joints_are_derived_as_regular_excluded_constraints(
    second_closure: bool,
) -> None:
    resolved = four_bar(second_closure=second_closure).resolve()

    excluded = [item.joint.name for item in resolved.joints if item.exclude_from_articulation]
    assert excluded == (["rocker_ground", "crank_rocker"] if second_closure else ["rocker_ground"])
    assert resolved.has_closed_loops
    assert np.allclose(resolved.reference_state.matrix("rocker")[:3, 3], (2.0, 0.0, 0.0))

    with pytest.raises(LoopClosureError, match="cannot reach this pose"):
        resolved.forward_kinematics({"ground_crank.rotZ": 0.2})
    with pytest.raises(ValidationError, match="closure joint positions"):
        resolved.forward_kinematics({"rocker_ground.rotZ": 0.2})


def test_closed_loop_requires_an_explicit_articulation_tree() -> None:
    assembly = four_bar()
    assembly.articulations.clear()
    assembly.articulation("main", root=assembly.get_rigid_body("ground"))

    with pytest.raises(ValidationError, match="requires explicit joints"):
        assembly.resolve()


def test_reference_state_rejects_a_geometrically_open_closure() -> None:
    assembly = four_bar()
    closure = assembly.get_joint("rocker_ground")
    assembly.joints[assembly.joints.index(closure)] = type(closure)(
        name=closure.name,
        body0=closure.body0,
        frame0=closure.frame0,
        body1=closure.body1,
        frame1=JointFrame(xyz=(3.0, 0.0, 0.0)),
        dofs=closure.dofs,
    )

    with pytest.raises(ValidationError, match="violates locked axis"):
        assembly.resolve()


def test_state_body_poses_are_authoritative_and_checked_against_every_joint() -> None:
    resolved = four_bar().resolve()
    valid = resolved.validate_state(PhysicsState(resolved.reference_state.body_poses))
    shifted = np.asarray(valid.body_poses["rocker"], dtype=float)
    shifted[1, 3] = 0.1
    invalid: dict[str, Mat4] = {
        name: np.asarray(matrix, dtype=float) for name, matrix in valid.body_poses.items()
    }
    invalid["rocker"] = shifted

    assert set(valid.body_poses) == {"ground", "crank", "coupler", "rocker"}
    with pytest.raises(ValidationError, match="violates locked axis"):
        resolved.validate_state(PhysicsState(invalid))


def test_world_joint_is_the_fixed_articulation_root() -> None:
    assembly = RigidBodyAssembly("mounted")
    base = add_body(assembly, "base")
    link = add_body(assembly, "link")
    mount = assembly.joint("mount", WORLD.at(JointFrame(xyz=(1.0, 2.0, 3.0))), base.at())
    hinge = assembly.joint("hinge", base.at(), link.at(), dofs=(JointDOF(JointAxis.ROT_Y),))
    assembly.articulation("fixed", root=mount, joints=(mount, hinge))

    resolved = assembly.resolve()

    assert np.allclose(resolved.reference_state.matrix(base)[:3, 3], (1.0, 2.0, 3.0))
    assert resolved.articulations[0].articulation.root is mount


def test_multiple_articulations_do_not_share_bodies_and_cross_joints_are_excluded() -> None:
    assembly = RigidBodyAssembly("two_arms")
    a = add_body(assembly, "a")
    b = add_body(assembly, "b")
    c = add_body(assembly, "c")
    d = add_body(assembly, "d")
    ab = assembly.joint("ab", a.at(), b.at())
    cd = assembly.joint("cd", c.at(), d.at())
    assembly.joint("bridge", b.at(), c.at())
    assembly.articulation("left", root=a, joints=(ab,))
    assembly.articulation("right", root=c, joints=(cd,))

    resolved = assembly.resolve()

    assert resolved.get_joint("bridge").exclude_from_articulation
    with pytest.raises(ValidationError, match="only one articulation"):
        assembly.articulation("overlap", root=b, joints=(ab,))
        assembly.resolve()


def test_maximal_coordinate_assembly_needs_no_articulation() -> None:
    assembly = RigidBodyAssembly("maximal")
    a = add_body(assembly, "a")
    b = add_body(assembly, "b")
    assembly.joint("constraint", a.at(), b.at(), dofs=(JointDOF(JointAxis.ROT_X),))

    resolved = assembly.resolve()

    assert not resolved.articulations
    assert not resolved.joints[0].exclude_from_articulation


def test_invalid_graph_references_and_disconnected_bodies_fail() -> None:
    disconnected = RigidBodyAssembly("disconnected")
    add_body(disconnected, "left")
    add_body(disconnected, "right")
    with pytest.raises(ValidationError, match="connected joint graph"):
        disconnected.resolve()

    assembly = RigidBodyAssembly("foreign")
    local = add_body(assembly, "local")
    foreign = RigidBody("foreign")
    foreign.add(Box(0.1, 0.1, 0.1), name="shape")
    with pytest.raises(ValidationError, match="unknown endpoint1"):
        assembly.joint("bad", local.at(), foreign.at())


def test_articulation_selection_must_be_a_rooted_tree() -> None:
    assembly = four_bar()
    ground = assembly.get_rigid_body("ground")
    assembly.articulations.clear()
    with pytest.raises(ValidationError, match="must form a tree"):
        assembly.articulation("bad", root=ground, joints=assembly.joints)
        assembly.resolve()

    world_assembly = RigidBodyAssembly("wrong_root")
    body = add_body(world_assembly, "body")
    mount = world_assembly.joint("mount", WORLD.at(), body.at())
    world_assembly.articulation("bad", root=body, joints=(mount,))
    with pytest.raises(ValidationError, match="use that joint as the articulation root"):
        world_assembly.resolve()


def test_articulation_root_must_belong_to_the_assembly() -> None:
    assembly = RigidBodyAssembly("local")
    add_body(assembly, "body")
    foreign = RigidBody("foreign")
    foreign.add(Box(0.1, 0.1, 0.1), name="shape")
    assembly.articulations.append(Articulation("bad", root=foreign, joints=()))

    with pytest.raises(ValidationError, match="outside this assembly"):
        assembly.resolve()


def _ball(*axes: JointAxis) -> ResolvedRigidBodyAssembly:
    assembly = RigidBodyAssembly("ball")
    base = add_body(assembly, "base")
    head = add_body(assembly, "head")
    assembly.joint(
        "socket",
        base.at(JointFrame()),
        head.at(JointFrame()),
        dofs=tuple(JointDOF(axis, limits=(-math.pi, math.pi)) for axis in axes),
    )
    assembly.articulation("main", root=base, joints=["socket"])
    return assembly.resolve()


def test_multi_axis_joints_pose_through_gimbal_lock() -> None:
    """Euler angles are ambiguous at pitch +-90; the pose is not."""

    resolved = _ball(JointAxis.ROT_X, JointAxis.ROT_Y, JointAxis.ROT_Z)
    for pitch in (math.pi / 2.0, -math.pi / 2.0):
        state = resolved.forward_kinematics(
            {"socket.rotX": 0.3, "socket.rotY": pitch, "socket.rotZ": 0.4}
        )
        resolved.world_transforms(state)


def test_gimbal_lock_does_not_charge_a_locked_axis() -> None:
    resolved = _ball(JointAxis.ROT_Y, JointAxis.ROT_Z)
    state = resolved.forward_kinematics({"socket.rotY": math.pi / 2.0, "socket.rotZ": 0.4})

    resolved.world_transforms(state)


def test_structural_names_must_be_identifiers() -> None:
    """Assembly, body, joint, and articulation names flow verbatim into USD
    prim paths, DOF ids, MJCF names, and viewer keys; anything looser than an
    identifier silently diverges somewhere downstream."""

    with pytest.raises(ValidationError, match="identifier"):
        RigidBodyAssembly("my assembly")

    model = RigidBodyAssembly("named")
    with pytest.raises(ValidationError, match="identifier"):
        model.rigid_body("my part")
    body = add_body(model, "part")
    other = add_body(model, "other")
    with pytest.raises(ValidationError, match="identifier"):
        model.joint("hinge.rotZ", body.at(), other.at())
    hinge = model.joint("hinge", body.at(), other.at())
    with pytest.raises(ValidationError, match="identifier"):
        model.articulation("main tree", root=body, joints=[hinge])

    # Shape names stay display-only and permissive.
    body.add(Box(0.05, 0.05, 0.05), name="display-shape")
