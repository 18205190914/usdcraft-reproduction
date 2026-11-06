from __future__ import annotations

import numpy as np
import pytest
from build123d import Axis, Box, Location, Plane, Pos

from articraft.sdk import (
    WORLD,
    BodyFrame,
    BoxGeometry,
    JointAxis,
    JointDOF,
    JointFrame,
    MeshGeometry,
    RigidBody,
    RigidBodyAssembly,
    TestContext,
)
from articraft.sdk.assembly import _frame_matrix
from articraft.sdk.errors import ValidationError


def test_body_at_accepts_points_and_build123d_features() -> None:
    model = RigidBodyAssembly("anchors")
    body = model.rigid_body("body")
    shape = Pos(1.0, 2.0, 3.0) * Box(2.0, 4.0, 6.0)
    body.add(shape, name="box")
    top = shape.faces().sort_by(Axis.Z)[-1]

    assert body.at().frame == JointFrame()
    assert body.at((1.0, 2.0, 3.0)).xyz == (1.0, 2.0, 3.0)
    assert body.at(top.center()).xyz == pytest.approx((1.0, 2.0, 6.0))
    assert body.at(body.shape("box")).xyz == pytest.approx((1.0, 2.0, 3.0))
    assert body.at(top).xyz == pytest.approx((1.0, 2.0, 6.0))
    assert np.allclose(_frame_matrix(body.at(top).frame)[:3, 2], (0.0, 0.0, 1.0))

    edge = top.edges().filter_by(Axis.X)[0]
    assert np.allclose(
        abs(_frame_matrix(body.at(edge).frame)[:3, 2]),
        (1.0, 0.0, 0.0),
    )


def test_body_at_accepts_native_locations_axes_planes_and_meshes() -> None:
    model = RigidBodyAssembly("native_frames")
    body = model.rigid_body("body")
    body.add(Box(1.0, 1.0, 1.0), name="box")

    location = Location((1.0, 2.0, 3.0), (10.0, 20.0, 30.0))
    located = body.at(location)
    assert located.xyz == (1.0, 2.0, 3.0)
    transform = location.wrapped.Transformation()
    expected = np.identity(4)
    for row in range(3):
        for column in range(4):
            expected[row, column] = transform.Value(row + 1, column + 1)
    assert np.allclose(_frame_matrix(located.frame), expected)
    assert np.allclose(
        _frame_matrix(body.at(Axis((1.0, 2.0, 3.0), (1.0, 0.0, 0.0))).frame)[:3, 2],
        (1.0, 0.0, 0.0),
    )
    assert body.at(Plane(origin=(4.0, 5.0, 6.0))).xyz == (4.0, 5.0, 6.0)

    mesh = BoxGeometry((2.0, 4.0, 6.0)).translate(3.0, 4.0, 5.0)
    assert body.at(mesh).xyz == pytest.approx((3.0, 4.0, 5.0))
    with pytest.raises(ValidationError, match="must contain vertices"):
        body.at(MeshGeometry())


def test_frame_in_derives_an_exact_closure_frame_without_coordinate_arithmetic() -> None:
    model = RigidBodyAssembly("derived_closure")
    base = model.rigid_body("base")
    base.add(Box(1.0, 1.0, 1.0), name="shape")
    link = model.rigid_body("link")
    link.add(Box(1.0, 1.0, 1.0), name="shape")
    hinge = model.joint(
        "hinge",
        base.at((2.0, 0.0, 0.0)),
        link.at(),
        dofs=(JointDOF(JointAxis.ROT_Z),),
    )
    model.articulation("main", root=base, joints=(hinge,))

    link_tip = link.at((1.0, 0.0, 0.0))
    same_frame_on_base = model.frame_in(link_tip, base)
    same_frame_in_world = model.frame_in(link_tip)

    assert isinstance(same_frame_on_base, BodyFrame)
    assert same_frame_on_base.body is base
    assert same_frame_on_base.xyz == pytest.approx((3.0, 0.0, 0.0))
    assert same_frame_in_world.body is WORLD
    assert same_frame_in_world.xyz == pytest.approx((3.0, 0.0, 0.0))

    model.joint(
        "closure",
        same_frame_on_base,
        link_tip,
        dofs=(JointDOF(JointAxis.ROT_Z),),
    )
    resolved = model.resolve()
    assert resolved.has_closed_loops
    assert resolved.get_joint("closure").exclude_from_articulation

    foreign_base = RigidBody("base")
    foreign_base.add(Box(1.0, 1.0, 1.0), name="shape")
    with pytest.raises(ValidationError, match="source frame belongs"):
        model.frame_in(foreign_base.at())
    with pytest.raises(ValidationError, match="target body belongs"):
        model.frame_in(link_tip, foreign_base)


def test_joint_endpoints_are_body_bound_frames() -> None:
    model = RigidBodyAssembly("bound")
    first = model.rigid_body("first")
    first.add(Box(1.0, 1.0, 1.0), name="shape")
    second = model.rigid_body("second")
    second.add(Box(1.0, 1.0, 1.0), name="shape")

    joint = model.joint("fixed", first.at(), second.at())
    assert joint.body0 is first
    assert joint.body1 is second

    with pytest.raises(ValidationError, match=r"body\.at"):
        model.joint(
            "unbound",
            JointFrame(),  # pyright: ignore[reportArgumentType]
            second.at(),
        )

    with pytest.raises(ValidationError, match="RigidBody or WORLD"):
        BodyFrame("first")  # pyright: ignore[reportArgumentType]

    same_named_foreign = RigidBody("first")
    same_named_foreign.add(Box(1.0, 1.0, 1.0), name="shape")
    with pytest.raises(ValidationError, match="belongs to another assembly"):
        model.joint("foreign", same_named_foreign.at(), second.at())


def test_context_checks_frame_coincidence_and_axis_alignment() -> None:
    model = RigidBodyAssembly("frame_checks")
    base = model.rigid_body("base")
    base.add(Box(1.0, 1.0, 1.0), name="shape")
    link = model.rigid_body("link")
    link.add(Box(1.0, 1.0, 1.0), name="shape")
    model.joint("mount", base.at((1.0, 0.0, 0.0)), link.at())
    context = TestContext(model)

    assert context.expect_coincident(base.at((1.0, 0.0, 0.0)), link.at())
    assert context.expect_coaxial(
        base.at(Axis((1.0, 0.0, 0.0), (0.0, 0.0, 1.0))),
        link.at(Axis((0.0, 0.0, 0.0), (0.0, 0.0, -1.0))),
    )
    assert not context.expect_coincident(base.at(), link.at())
    assert not context.expect_coaxial(base.at(), link.at(), axis="z")

    failures = context.report().failures
    assert "position_error=1" in failures[0].details
    assert "radial_error=1" in failures[1].details


def test_round_features_anchor_their_axis_of_symmetry() -> None:
    """A bore, pin, or rim means the axis it spins about, not a surface point."""

    import math

    from build123d import Cylinder, Rot

    model = RigidBodyAssembly("round_anchors")
    body = model.rigid_body("body")
    pin = Pos(1.0, 2.0, 3.0) * Rot(0.0, 30.0, 0.0) * Cylinder(0.01, 0.05)
    body.add(pin, name="pin")
    direction = np.array([math.sin(math.radians(30.0)), 0.0, math.cos(math.radians(30.0))])

    lateral = next(f for f in pin.faces() if "CYLINDER" in str(f.geom_type).upper())
    frame = _frame_matrix(body.at(lateral).frame)
    assert abs(abs(float(frame[:3, 2] @ direction)) - 1.0) < 1e-9
    radial = (frame[:3, 3] - (1.0, 2.0, 3.0)) - (
        (frame[:3, 3] - (1.0, 2.0, 3.0)) @ direction
    ) * direction
    assert np.linalg.norm(radial) < 1e-9

    top = pin.faces().sort_by(Axis((1.0, 2.0, 3.0), tuple(direction)))[-1]
    rim = top.edges()[0]
    rim_frame = _frame_matrix(body.at(rim).frame)
    assert abs(abs(float(rim_frame[:3, 2] @ direction)) - 1.0) < 1e-9
    assert np.allclose(rim_frame[:3, 3], np.array([1.0, 2.0, 3.0]) + 0.025 * direction)


def test_frame_in_before_the_graph_connects_names_the_precondition() -> None:
    model = RigidBodyAssembly("early")
    first = model.rigid_body("first")
    first.add(Box(1.0, 1.0, 1.0), name="shape")
    second = model.rigid_body("second")
    second.add(Box(1.0, 1.0, 1.0), name="shape")

    with pytest.raises(ValidationError, match="authored first"):
        model.frame_in(first.at(), second)
