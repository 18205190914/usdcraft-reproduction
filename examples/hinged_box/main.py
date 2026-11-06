"""Canonical articulated example: a box whose lid opens on a real hinge.

Two parts, one REVOLUTE articulation, and an authored contact check. The
hinge axis sits exactly on the edge where the lid meets the base, so the
parts stay in contact through the whole motion range -- the property the
compiler's articulation-separation check verifies.

Compile it:  python -m articraft.compiler.worker <run_dir>
"""

from __future__ import annotations

from build123d import Align, Axis, Box

from articraft.sdk import (
    JointAxis,
    JointDOF,
    Material,
    RigidBodyAssembly,
    TestContext,
    TestReport,
)


def build_object_model() -> RigidBodyAssembly:
    model = RigidBodyAssembly("hinged_box")

    base = model.rigid_body("base")
    # Saying what a shape is made of settles its mass, how it behaves on contact,
    # and how it looks. Aluminum comes out light and metallic without asking.
    base_shape = Box(0.10, 0.08, 0.04)
    base.add(base_shape, name="body", material=Material.ALUMINUM)

    lid = model.rigid_body("lid")
    lid_shape = Box(
        0.10,
        0.08,
        0.015,
        align=(Align.CENTER, Align.MIN, Align.MIN),
    )
    lid.add(
        # The rear-bottom edge is the body's local origin, so it can be selected
        # directly as the matching hinge feature below.
        lid_shape,
        name="body",
        # A recolor keeps the material -- still plastic, still 1050 kg/m^3 --
        # and only changes what it looks like: an amber lid on a metal base.
        material=Material.ABS_PLASTIC,
        color=(0.62, 0.45, 0.16),
    )

    base_edge = base_shape.edges().filter_by(Axis.X).group_by(Axis.Y)[0].sort_by(Axis.Z)[-1]
    lid_edge = lid_shape.edges().filter_by(Axis.X).group_by(Axis.Y)[0].sort_by(Axis.Z)[0]
    base_hinge = base.at(Axis(base_edge.center(), Axis.X.direction))
    lid_hinge = lid.at(Axis(lid_edge.center(), Axis.X.direction))

    # The hinge frames are anchored to the actual contact edges. Axis directions
    # become local Z, so ROT_Z opens the lid around the selected edge.
    model.joint(
        "lid_hinge",
        base_hinge,
        lid_hinge,
        dofs=(JointDOF(JointAxis.ROT_Z, limits=(0.0, 1.5708)),),
    )
    model.articulation("main", root=base, joints=["lid_hinge"])
    return model


object_model = build_object_model()


def run_tests() -> TestReport:
    ctx = TestContext(object_model)
    ctx.expect_contact("base", "lid")
    base = object_model.get_rigid_body("base")
    lid = object_model.get_rigid_body("lid")
    base_shape = base.shape("body")
    lid_shape = lid.shape("body")
    assert isinstance(base_shape, Box)
    assert isinstance(lid_shape, Box)
    base_edge = base_shape.edges().filter_by(Axis.X).group_by(Axis.Y)[0].sort_by(Axis.Z)[-1]
    lid_edge = lid_shape.edges().filter_by(Axis.X).group_by(Axis.Y)[0].sort_by(Axis.Z)[0]
    ctx.expect_coaxial(
        base.at(Axis(base_edge.center(), Axis.X.direction)),
        lid.at(Axis(lid_edge.center(), Axis.X.direction)),
    )
    with ctx.pose(lid_hinge=0.8):
        ctx.expect_coaxial(
            base.at(Axis(base_edge.center(), Axis.X.direction)),
            lid.at(Axis(lid_edge.center(), Axis.X.direction)),
            name="hinge_edges_stay_coaxial_while_opening",
        )
    return ctx.report()
