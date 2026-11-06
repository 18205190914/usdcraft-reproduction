"""Read compiled USD geometry and joint frames independently of its Python author."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import trimesh
from pxr import Gf, Usd, UsdGeom, UsdPhysics

from articraft.usdcraft_benchmark import save_json


def read_usd_asset(path: Path, directory: Path) -> tuple[dict, dict]:
    stage = Usd.Stage.Open(str(path))
    scale = UsdGeom.GetStageMetersPerUnit(stage)
    transforms = UsdGeom.XformCache()
    bodies = {p.GetName(): p for p in stage.Traverse() if p.HasAPI(UsdPhysics.RigidBodyAPI)}
    parts = {}
    directory.mkdir(parents=True, exist_ok=True)
    for name, body in bodies.items():
        meshes = []
        for prim in Usd.PrimRange(body):
            if (
                not prim.IsA(UsdGeom.Mesh)
                or UsdGeom.Imageable(prim).ComputeVisibility() == "invisible"
            ):
                continue
            mesh = UsdGeom.Mesh(prim)
            vertices = np.asarray(mesh.GetPointsAttr().Get(), dtype=float)
            world = np.column_stack((vertices, np.ones(len(vertices)))) @ np.asarray(
                transforms.GetLocalToWorldTransform(prim)
            )
            indices = mesh.GetFaceVertexIndicesAttr().Get()
            faces = []
            offset = 0
            for count in mesh.GetFaceVertexCountsAttr().Get():
                face = indices[offset : offset + count]
                faces.extend([face[0], face[i], face[i + 1]] for i in range(1, count - 1))
                offset += count
            meshes.append(trimesh.Trimesh(world[:, :3] * scale, faces, process=False))
        parts[name] = trimesh.util.concatenate(meshes)
        parts[name].export(directory / f"{name}.ply")
    joints = []
    for prim in stage.Traverse():
        if not (prim.IsA(UsdPhysics.RevoluteJoint) or prim.IsA(UsdPhysics.PrismaticJoint)):
            continue
        joint = UsdPhysics.Joint(prim)
        parent = stage.GetPrimAtPath(joint.GetBody0Rel().GetTargets()[0])
        child = stage.GetPrimAtPath(joint.GetBody1Rel().GetTargets()[0])
        frame = transforms.GetLocalToWorldTransform(parent)
        axis = Gf.Vec3d(
            *("XYZ".index(str(prim.GetAttribute("physics:axis").Get())) == i for i in range(3))
        )
        axis = Gf.Rotation(Gf.Quatd(joint.GetLocalRot0Attr().Get())).TransformDir(axis)
        origin = frame.Transform(joint.GetLocalPos0Attr().Get())
        revolute = prim.IsA(UsdPhysics.RevoluteJoint)
        limits = [prim.GetAttribute(f"physics:{name}Limit").Get() for name in ("lower", "upper")]
        joints.append(
            {
                "name": prim.GetName(),
                "parent": parent.GetName(),
                "child": child.GetName(),
                "type": "revolute" if revolute else "prismatic",
                "axis": list(frame.TransformDir(axis)),
                "origin": (np.asarray(origin) * scale).tolist(),
                "limits": (np.radians(limits) if revolute else np.asarray(limits) * scale).tolist(),
            }
        )
    assembly = {
        "parts": [{"name": n, "mesh": f"{n}.ply"} for n in parts],
        "joints": joints,
        "units": "metres",
    }
    save_json(directory / "assembly.json", assembly)
    return assembly, parts
