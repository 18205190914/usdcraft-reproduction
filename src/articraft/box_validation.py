"""Independent exported-geometry acceptance checks for the rectangular box profile."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import trimesh
from pxr import Gf, Usd, UsdGeom, UsdPhysics
from scipy.spatial.transform import Rotation


@dataclass(frozen=True)
class BoxSpec:
    width: float
    depth: float
    height: float
    wall: float
    lid_height: float
    opening_degrees: float

    def __post_init__(self) -> None:
        if not (
            0 < self.wall < min(self.width, self.depth) / 2
            and self.wall < self.height - self.lid_height
            and 0 < self.lid_height < self.height
            and 0 < self.opening_degrees <= 180
        ):
            raise ValueError("Invalid rectangular box dimensions or opening angle")

    def prompt(self) -> str:
        return (
            f"Create a rectangular storage box, {self.width} m wide along X, "
            f"{self.depth} m deep along Y, and {self.height} m tall when closed. "
            f"Use an open-top hollow base with {self.wall} m uniform walls and bottom, "
            f"and a solid {self.lid_height} m thick separate lid. The base rests on Z=0. "
            f"One rear hinge at Y=-depth/2 opens upwards from 0 to {self.opening_degrees} "
            "degrees around world +X. Use ABS plastic with density 1050 kg/m^3 and "
            "derived explicit positive masses. Exactly two bodies named base and lid, "
            "one joint named rear_hinge. No latch or decorations."
        )


def _triangles(body: Usd.Prim, scale: float) -> np.ndarray:
    triangles = []
    transforms = UsdGeom.XformCache()
    for prim in Usd.PrimRange(body):
        if not prim.IsA(UsdGeom.Mesh):
            continue
        if UsdGeom.Imageable(prim).ComputeVisibility() == "invisible":
            continue
        mesh = UsdGeom.Mesh(prim)
        points = np.asarray(mesh.GetPointsAttr().Get(), dtype=float)
        world = np.column_stack((points, np.ones(len(points)))) @ np.asarray(
            transforms.GetLocalToWorldTransform(prim), dtype=float
        )
        indices = list(mesh.GetFaceVertexIndicesAttr().Get())
        offset = 0
        for count in mesh.GetFaceVertexCountsAttr().Get():
            face = indices[offset : offset + count]
            triangles.extend(
                world[[face[0], face[index], face[index + 1]], :3] * scale
                for index in range(1, count - 1)
            )
            offset += count
    return np.asarray(triangles)


def _ray_hits(triangles: np.ndarray, origin: np.ndarray, direction: np.ndarray) -> list[float]:
    edge1 = triangles[:, 1] - triangles[:, 0]
    edge2 = triangles[:, 2] - triangles[:, 0]
    perpendicular = np.cross(np.broadcast_to(direction, edge2.shape), edge2)
    determinant = np.einsum("ij,ij->i", edge1, perpendicular)
    valid = abs(determinant) > 1e-12
    inverse = np.zeros_like(determinant)
    inverse[valid] = 1 / determinant[valid]
    offset = origin - triangles[:, 0]
    u = np.einsum("ij,ij->i", offset, perpendicular) * inverse
    cross = np.cross(offset, edge1)
    v = np.einsum("j,ij->i", direction, cross) * inverse
    distance = np.einsum("ij,ij->i", edge2, cross) * inverse
    valid &= (u >= -1e-9) & (v >= -1e-9) & (u + v <= 1 + 1e-9) & (distance > 1e-8)
    hits = np.sort(distance[valid])
    return [float(t) for i, t in enumerate(hits) if i == 0 or t - hits[i - 1] > 1e-7]


def _solid(triangles: np.ndarray) -> trimesh.Trimesh:
    mesh = trimesh.Trimesh(
        vertices=triangles.reshape(-1, 3),
        faces=np.arange(triangles.size // 3).reshape(-1, 3),
        process=True,
    )
    mesh.fix_normals()
    if not mesh.is_volume:
        raise ValueError("Exported body is not a closed positive-volume triangle mesh")
    return mesh


def validate_box(asset: Path, spec: BoxSpec) -> dict[str, Any]:
    """Read the USDZ itself; never trust constants or tests in generated Python."""
    report: dict[str, Any] = {"asset": str(asset), "requested": asdict(spec), "checks": {}}
    checks = report["checks"]
    tolerance = 5e-5  # 0.05 mm: accommodates USD float mesh coordinates.
    try:
        stage = Usd.Stage.Open(str(asset))
        if not stage or not stage.GetDefaultPrim():
            raise ValueError("USDZ cannot be reopened")
        scale = UsdGeom.GetStageMetersPerUnit(stage)
        checks["z_up"] = str(UsdGeom.GetStageUpAxis(stage)) == "Z"
        bodies = {p.GetName(): p for p in stage.Traverse() if p.HasAPI(UsdPhysics.RigidBodyAPI)}
        joints = [p for p in stage.Traverse() if p.IsA(UsdPhysics.Joint)]
        checks["two_named_bodies_one_revolute_joint"] = (
            sum(p.HasAPI(UsdPhysics.RigidBodyAPI) for p in stage.Traverse()) == 2
            and set(bodies) == {"base", "lid"}
            and len(joints) == 1
            and joints[0].IsA(UsdPhysics.RevoluteJoint)
        )
        if not checks["two_named_bodies_one_revolute_joint"]:
            raise ValueError("Expected base, lid, and one revolute joint")
        triangles = {name: _triangles(body, scale) for name, body in bodies.items()}
        meshes = {name: _solid(value) for name, value in triangles.items()}
        checks["watertight_positive_volume_bodies"] = True
        points = np.concatenate(list(triangles.values())).reshape(-1, 3)
        lower, upper = points.min(axis=0), points.max(axis=0)
        report["closed_size_m"] = (upper - lower).tolist()
        checks["closed_dimensions"] = bool(
            np.allclose(
                upper - lower, [spec.width, spec.depth, spec.height], atol=tolerance, rtol=0
            )
        )
        checks["ground_and_center"] = bool(
            np.allclose(lower, [-spec.width / 2, -spec.depth / 2, 0], atol=tolerance, rtol=0)
        )
        base_height = spec.height - spec.lid_height
        expected_bounds = {
            "base": [
                [-spec.width / 2, -spec.depth / 2, 0],
                [spec.width / 2, spec.depth / 2, base_height],
            ],
            "lid": [
                [-spec.width / 2, -spec.depth / 2, base_height],
                [spec.width / 2, spec.depth / 2, spec.height],
            ],
        }
        checks["separate_base_and_lid_bounds"] = all(
            np.allclose(mesh.bounds, expected_bounds[name], atol=tolerance, rtol=0)
            for name, mesh in meshes.items()
        )

        walls, bottoms, open_rays = [], [], []
        base = triangles["base"]
        # Probe all four walls at two heights, and the floor/open top at nine XY points.
        for height_fraction in (0.4, 0.75):
            z = spec.wall + height_fraction * (base_height - spec.wall)
            for axis, dimension in ((0, spec.width), (1, spec.depth)):
                origin = np.array([0.0, 0.0, z])
                origin[axis] = -dimension / 2 - 0.01
                direction = np.eye(3)[axis]
                hits = _ray_hits(base, origin, direction)
                walls.extend([hits[1] - hits[0], hits[3] - hits[2]] if len(hits) == 4 else [None])
        for x in (-0.25 * spec.width, 0.0, 0.25 * spec.width):
            for y in (-0.25 * spec.depth, 0.0, 0.25 * spec.depth):
                hits = _ray_hits(base, np.array([x, y, -0.01]), np.array([0.0, 0.0, 1.0]))
                bottoms.append(hits[1] - hits[0] if len(hits) == 2 else None)
                hits = _ray_hits(
                    base, np.array([x, y, base_height * 0.8]), np.array([0.0, 0.0, 1.0])
                )
                open_rays.append(len(hits) == 0)
        report["wall_thickness_samples_m"] = walls
        report["bottom_thickness_samples_m"] = bottoms
        checks["four_wall_thicknesses"] = all(
            t is not None and abs(t - spec.wall) < tolerance for t in walls
        )
        checks["bottom_thickness"] = all(
            t is not None and abs(t - spec.wall) < tolerance for t in bottoms
        )
        checks["open_top"] = all(open_rays)
        expected_volumes = {
            "base": spec.width * spec.depth * base_height
            - (spec.width - 2 * spec.wall)
            * (spec.depth - 2 * spec.wall)
            * (base_height - spec.wall),
            "lid": spec.width * spec.depth * spec.lid_height,
        }
        checks["solid_volumes"] = all(
            np.isclose(mesh.volume, expected_volumes[name], rtol=0.001, atol=1e-10)
            for name, mesh in meshes.items()
        )
        masses = {
            name: UsdPhysics.MassAPI(body).GetMassAttr().Get() for name, body in bodies.items()
        }
        report["masses_kg"] = masses
        checks["explicit_abs_masses"] = all(
            mass is not None
            and mass > 0
            and np.isclose(mass, expected_volumes[name] * 1050, rtol=0.01)
            for name, mass in masses.items()
        )

        joint = UsdPhysics.RevoluteJoint(joints[0])
        checks["joint_name"] = joints[0].GetName() == "rear_hinge"
        checks["joint_body_targets"] = list(joint.GetBody0Rel().GetTargets()) == [
            bodies["base"].GetPath()
        ] and list(joint.GetBody1Rel().GetTargets()) == [bodies["lid"].GetPath()]
        lower_limit, upper_limit = joint.GetLowerLimitAttr().Get(), joint.GetUpperLimitAttr().Get()
        report["joint_limits_degrees"] = [lower_limit, upper_limit]
        checks["joint_limits"] = bool(
            np.allclose([lower_limit, upper_limit], [0, spec.opening_degrees], atol=1e-3, rtol=0)
        )
        transforms = UsdGeom.XformCache()
        body0 = stage.GetPrimAtPath(joint.GetBody0Rel().GetTargets()[0])
        body1 = stage.GetPrimAtPath(joint.GetBody1Rel().GetTargets()[0])
        transform0, transform1 = (transforms.GetLocalToWorldTransform(b) for b in (body0, body1))
        axis = Gf.Vec3d(*("XYZ".index(str(joint.GetAxisAttr().Get())) == i for i in range(3)))
        axis0 = Gf.Rotation(Gf.Quatd(joint.GetLocalRot0Attr().Get())).TransformDir(axis)
        axis1 = Gf.Rotation(Gf.Quatd(joint.GetLocalRot1Attr().Get())).TransformDir(axis)
        world_axis = np.asarray(transform0.TransformDir(axis0))
        child_axis = np.asarray(transform1.TransformDir(axis1))
        anchor = np.asarray(transform0.Transform(joint.GetLocalPos0Attr().Get())) * scale
        child_anchor = np.asarray(transform1.Transform(joint.GetLocalPos1Attr().Get())) * scale
        report["hinge_anchor_m"] = anchor.tolist()
        report["hinge_axis_world"] = world_axis.tolist()
        checks["rear_hinge_frame"] = bool(
            np.allclose(world_axis, [1, 0, 0], atol=1e-6)
            and np.allclose(child_axis, world_axis, atol=1e-6)
            and np.allclose(anchor, [0, -spec.depth / 2, base_height], atol=tolerance)
            and np.allclose(child_anchor, anchor, atol=tolerance)
        )
        overlaps = []
        # Rotate exported child triangles about the exported joint frame, independently
        # of the generated code and its own self-tests. Surface contact is permitted.
        for angle in np.linspace(0, spec.opening_degrees, 9):
            rotation = Rotation.from_rotvec(np.radians(angle) * world_axis).as_matrix()
            moved = (triangles["lid"] - anchor) @ rotation.T + anchor
            overlap = trimesh.boolean.intersection(
                [meshes["base"], _solid(moved)], engine="manifold"
            )
            # Signed tetrahedral volumes avoid computing the undefined center of
            # mass of a zero-volume surface-contact intersection.
            faces = overlap.triangles
            volume = (
                np.einsum("ij,ij->i", faces[:, 0], np.cross(faces[:, 1], faces[:, 2])).sum() / 6
            )
            overlaps.append(abs(float(volume)))
        report["sweep_intersection_volumes_m3"] = overlaps
        checks["nine_pose_no_solid_interpenetration"] = all(v < 1e-10 for v in overlaps)
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
    report["failed_checks"] = [name for name, passed in checks.items() if not passed]
    report["passed"] = bool(checks) and not report.get("error") and not report["failed_checks"]
    return report
