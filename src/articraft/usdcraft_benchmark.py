"""Five local synthetic cases. Public inputs and evaluator-only truth are separate."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import trimesh
from PIL import Image

from articraft.source_geometry import projected_samples


def save_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")


def box(extent: list[float], center: list[float]) -> trimesh.Trimesh:
    mesh = trimesh.creation.box(extents=extent)
    mesh.apply_translation(center)
    return mesh


def union(meshes: list[trimesh.Trimesh]) -> trimesh.Trimesh:
    return trimesh.boolean.union(meshes, engine="manifold")


def shell(w: float, d: float, h: float, t: float, front: bool) -> trimesh.Trimesh:
    outer = box([w, d, h], [0, 0, h / 2])
    if front:
        inner = box([w - 2 * t, d, h - 2 * t], [0, -t, h / 2])
    else:
        inner = box([w - 2 * t, d - 2 * t, h], [0, 0, h / 2 + t])
    return trimesh.boolean.difference([outer, inner], engine="manifold")


def door(
    left: float, right: float, y: float, h: float, t: float, handle_x: float
) -> trimesh.Trimesh:
    return union(
        [
            box([right - left, t, h - 0.02], [(left + right) / 2, y, h / 2]),
            box([0.012, 0.018, 0.09], [handle_x, y - t / 2 - 0.018, h / 2]),
            *[
                box([0.012, 0.025, 0.01], [handle_x, y - t / 2 - 0.012, z])
                for z in [h / 2 - 0.04, h / 2 + 0.04]
            ],
        ]
    )


def reference_cases() -> dict:
    cases = {}
    w, d, h, t, lid = 0.24, 0.16, 0.12, 0.002, 0.006
    cases["hinged_box"] = {
        "request": "Rebuild the storage box with its hollow base and a rear-hinged lid. Preserve measured dimensions.",
        "parts": [
            ("base", shell(w, d, h - lid, t, False)),
            ("lid", box([w, d, lid], [0, 0, h - lid / 2])),
        ],
        "joints": [
            {
                "name": "hinge",
                "parent": "base",
                "child": "lid",
                "type": "revolute",
                "axis": [1, 0, 0],
                "origin": [0, -d / 2, h - lid],
                "limits": [0, float(np.radians(100))],
            }
        ],
    }
    w, d, h, t = 0.36, 0.25, 0.20, 0.012
    tray = shell(w - 2 * t - 0.006, d - t - 0.012, h - 0.04, 0.008, False)
    tray.apply_translation([0, -0.006, 0.02])
    drawer = union(
        [
            tray,
            box([w - 0.008, 0.014, h - 0.014], [0, -d / 2 - 0.007, h / 2]),
            box([0.11, 0.018, 0.012], [0, -d / 2 - 0.025, h / 2]),
        ]
    )
    cases["drawer"] = {
        "request": "Rebuild the drawer housing and pull-out drawer with its front and handle. Preserve observed interior surfaces.",
        "parts": [("housing", shell(w, d, h, t, True)), ("drawer", drawer)],
        "joints": [
            {
                "name": "slide",
                "parent": "housing",
                "child": "drawer",
                "type": "prismatic",
                "axis": [0, -1, 0],
                "origin": [0, 0, 0],
                "limits": [0, 0.16],
            }
        ],
    }
    for name, w, d, h, double in [
        ("cabinet", 0.42, 0.27, 0.50, False),
        ("double_cabinet", 0.60, 0.30, 0.56, True),
        ("damaged_cabinet", 0.38, 0.25, 0.46, False),
    ]:
        t = 0.014
        y = -d / 2 - 0.007
        parts = [("frame", shell(w, d, h, t, True))]
        joints = []
        if double:
            for label, left, right, ax, sign in [
                ("left_door", -w / 2 + 0.003, -0.003, -w / 2 + 0.003, -1),
                ("right_door", 0.003, w / 2 - 0.003, w / 2 - 0.003, 1),
            ]:
                handle = -0.028 if sign < 0 else 0.028
                parts.append((label, door(left, right, y, h, 0.014, handle)))
                joints.append(
                    {
                        "name": label + "_hinge",
                        "parent": "frame",
                        "child": label,
                        "type": "revolute",
                        "axis": [0, 0, 1],
                        "origin": [ax, y, h / 2],
                        "limits": [
                            float(np.radians(-100)) if sign < 0 else 0,
                            0 if sign < 0 else float(np.radians(100)),
                        ],
                    }
                )
        else:
            parts.append(("door", door(-w / 2 + 0.003, w / 2 - 0.003, y, h, 0.014, w / 2 - 0.035)))
            joints.append(
                {
                    "name": "door_hinge",
                    "parent": "frame",
                    "child": "door",
                    "type": "revolute",
                    "axis": [0, 0, 1],
                    "origin": [-w / 2 + 0.003, y, h / 2],
                    "limits": [float(np.radians(-100)), 0],
                }
            )
        cases[name] = {
            "request": "Rebuild this cabinet with its visible door assembly, handles and usable interior; infer functional articulation from the object.",
            "parts": parts,
            "joints": joints,
        }
    return cases


def render_meshes(
    meshes: list[trimesh.Trimesh],
    path: Path,
    size: int = 640,
    *,
    view: tuple[float, float, float] = (-1.0, 1.5, -0.8),
    frame_bounds: np.ndarray | None = None,
    part_colors: list[tuple[int, int, int]] | None = None,
) -> None:
    """Deterministic CPU z-buffer inspection image; not a photorealistic renderer."""
    mesh = trimesh.util.concatenate(meshes)
    center = (mesh.bounds if frame_bounds is None else frame_bounds).mean(0)
    forward = np.array(view, dtype=float)
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, [0, 0, 1])
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)
    coords = (mesh.vertices - center) @ np.stack([right, -up, forward], axis=1)
    framing = mesh.vertices if frame_bounds is None else trimesh.bounds.corners(frame_bounds)
    projected_frame = (framing - center) @ np.stack([right, -up, forward], axis=1)
    scale = (size * 0.78) / np.ptp(projected_frame[:, :2], axis=0).max()
    pixels = coords[:, :2] * scale + size / 2
    depth = np.full((size, size), np.inf)
    image = np.full((size, size, 3), 241, dtype=np.uint8)
    light = np.array([0.4, -0.6, 0.7])
    light /= np.linalg.norm(light)
    colors = np.repeat(
        np.asarray(part_colors if part_colors is not None else [(100, 142, 169)] * len(meshes)),
        [len(m.faces) for m in meshes],
        axis=0,
    )
    for face, normal, color in zip(mesh.faces, mesh.face_normals, colors, strict=True):
        xy, bary = projected_samples(pixels[face])
        valid = ((xy >= 0) & (xy < size)).all(1)
        xy, bary = xy[valid], bary[valid]
        z = bary @ coords[face, 2]
        visible = z < depth[xy[:, 1], xy[:, 0]]
        xy, z = xy[visible], z[visible]
        depth[xy[:, 1], xy[:, 0]] = z
        shade = 0.45 + 0.55 * abs(float(normal @ light))
        image[xy[:, 1], xy[:, 0]] = color * shade
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(image).save(path)


def build_benchmark(root: Path) -> dict:
    records = []
    for name, case in reference_cases().items():
        public = root / "inputs" / name
        truth = root / "private_truth" / name
        public.mkdir(parents=True, exist_ok=True)
        truth.mkdir(parents=True, exist_ok=True)
        parts = []
        for label, mesh in case["parts"]:
            mesh.export(truth / f"{label}.ply")
            parts.append({"name": label, "mesh": f"{label}.ply"})
        gt = {"parts": parts, "joints": case["joints"], "units": "metres"}
        save_json(truth / "assembly.json", gt)
        merged = union([m for _, m in case["parts"]])
        if name == "damaged_cabinet":
            c = merged.triangles_center
            remove = (c[:, 0] < -0.05) & (c[:, 1] > 0.04) & (c[:, 2] > 0.25)
            merged.update_faces(~remove)
            merged.remove_unreferenced_vertices()
        # Export just geometry: no part names, joints, original program or labels.
        merged.visual = trimesh.visual.ColorVisuals(mesh=merged, face_colors=[170, 170, 170, 255])
        merged.export(public / "source.ply")
        render_meshes([m for _, m in case["parts"]], public / "reference.png")
        render_meshes([merged], public / "source.png")
        save_json(
            public / "input.json",
            {
                "source": "source.ply",
                "image": "reference.png",
                "units": "metres",
                "up_axis": "Z",
                "request": case["request"],
            },
        )
        records.append(
            {
                "case": name,
                "public": str(public),
                "truth": str(truth),
                "source_triangles": len(merged.faces),
                "degraded": name == "damaged_cabinet",
            }
        )
    result = {
        "scope": "local synthetic development benchmark; not USDCraft-bench or Lightwheel",
        "cases": records,
        "truth_access": "only evaluator; public modeling tools receive inputs directory only",
    }
    save_json(root / "benchmark.json", result)
    return result
