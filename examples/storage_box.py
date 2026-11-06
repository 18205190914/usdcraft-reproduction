from __future__ import annotations

import math

from build123d import Align, Axis, Box, Pos

from articraft.sdk import JointAxis, JointDOF, Material, RigidBodyAssembly, TestContext, TestReport

WIDTH = 0.30
DEPTH = 0.20
HEIGHT = 0.15
WALL = 0.002
LID_HEIGHT = 0.006
OPENING_DEGREES = 100.0
BASE_HEIGHT = HEIGHT - LID_HEIGHT


def hinge_frames(model):
    base = model.get_rigid_body("base")
    lid = model.get_rigid_body("lid")
    base_edge = (
        base.shape("shell").edges().filter_by(Axis.X).group_by(Axis.Y)[0].sort_by(Axis.Z)[-1]
    )
    lid_edge = lid.shape("plate").edges().filter_by(Axis.X).group_by(Axis.Y)[0].sort_by(Axis.Z)[0]
    return base.at(Axis(base_edge.center(), Axis.X.direction)), lid.at(
        Axis(lid_edge.center(), Axis.X.direction)
    )


def build_object_model() -> RigidBodyAssembly:
    model = RigidBodyAssembly("storage_box")
    base = model.rigid_body("base")
    outer = Box(WIDTH, DEPTH, BASE_HEIGHT, align=(Align.CENTER, Align.CENTER, Align.MIN))
    cutter = Pos(0, 0, WALL) * Box(
        WIDTH - 2 * WALL,
        DEPTH - 2 * WALL,
        BASE_HEIGHT,
        align=(Align.CENTER, Align.CENTER, Align.MIN),
    )
    base.add(outer - cutter, name="shell", material=Material.ABS_PLASTIC, color=(0.16, 0.35, 0.52))
    lid = model.rigid_body("lid")
    lid.add(
        Box(WIDTH, DEPTH, LID_HEIGHT, align=(Align.CENTER, Align.MIN, Align.MIN)),
        name="plate",
        material=Material.ABS_PLASTIC,
        color=(0.70, 0.78, 0.86),
    )
    base_hinge, lid_hinge = hinge_frames(model)
    model.joint(
        "rear_hinge",
        base_hinge,
        lid_hinge,
        dofs=(JointDOF(JointAxis.ROT_Z, limits=(0.0, math.radians(OPENING_DEGREES))),),
    )
    model.articulation("main", root=base, joints=["rear_hinge"])
    return model


object_model = build_object_model()


def run_tests() -> TestReport:
    ctx = TestContext(object_model)
    ctx.expect_contact("base", "lid")
    ctx.expect_positive_volume("base", shape="shell")
    ctx.expect_positive_volume("lid", shape="plate")
    ctx.expect_bounds(
        "base",
        minimum=(-WIDTH / 2, -DEPTH / 2, 0),
        maximum=(WIDTH / 2, DEPTH / 2, BASE_HEIGHT),
        tolerance=1e-5,
    )
    ctx.expect_bounds(
        "lid",
        minimum=(-WIDTH / 2, -DEPTH / 2, BASE_HEIGHT),
        maximum=(WIDTH / 2, DEPTH / 2, HEIGHT),
        tolerance=1e-5,
    )
    first, second = hinge_frames(object_model)
    ctx.expect_coaxial(first, second, name="closed_rear_hinge_coaxial")
    poses = ctx.sample_joint("rear_hinge", samples=9)
    # A real hinge keeps its surfaces in contact. Check solid interpenetration
    # at every sampled pose, with micrometer tolerance, instead of forbidding contact.
    for index, pose in enumerate(poses):
        with ctx.pose(pose.as_dict()):
            ctx.fail_if_parts_overlap_in_current_pose(
                overlap_tol=1e-6,
                overlap_volume_tol=1e-12,
                name=f"no_solid_interpenetration_pose_{index}",
            )
    for degrees in (OPENING_DEGREES * f for f in (0.25, 0.5, 0.75, 1.0)):
        with ctx.pose(rear_hinge=math.radians(degrees)):
            ctx.expect_coaxial(first, second, name=f"hinge_coaxial_at_{degrees}_degrees")
    return ctx.report()
