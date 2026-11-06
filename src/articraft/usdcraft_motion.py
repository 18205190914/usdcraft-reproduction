"""Independent solid-volume and sampled-motion checks for bounded joint trees."""

from __future__ import annotations

from itertools import combinations

import numpy as np
import trimesh
from scipy.spatial.transform import Rotation


def pose_parts(
    parts: dict[str, trimesh.Trimesh], joints: dict, coordinates: dict[str, float]
) -> dict[str, trimesh.Trimesh]:
    """Transform canonical world-space part meshes through the joint tree."""
    poses = {}

    def pose(name: str) -> np.ndarray:
        if name not in poses:
            local = np.eye(4)
            if name in joints:
                joint = joints[name]
                axis = np.asarray(joint["axis"], dtype=float)
                axis = axis / np.linalg.norm(axis)
                q = coordinates.get(joint["name"], 0.0)
                if joint["type"] == "revolute":
                    rotation = Rotation.from_rotvec(axis * q).as_matrix()
                    origin = np.asarray(joint["origin"])
                    local[:3, :3] = rotation
                    local[:3, 3] = origin - rotation @ origin
                else:
                    local[:3, 3] = axis * q
                local = pose(joint["parent"]) @ local
            poses[name] = local
        return poses[name]

    return {name: mesh.copy().apply_transform(pose(name)) for name, mesh in parts.items()}


def solid_overlaps(parts: dict[str, trimesh.Trimesh]) -> list[dict]:
    """Use actual CSG intersection volume, including complete containment."""
    overlaps = []
    for (name_a, first), (name_b, second) in combinations(parts.items(), 2):
        if np.any(
            np.maximum(first.bounds[0], second.bounds[0])
            >= np.minimum(first.bounds[1], second.bounds[1])
        ):
            continue
        intersection = trimesh.boolean.intersection([first, second], engine="manifold")
        if not len(intersection.faces):
            continue
        # Signed tetrahedra give volume without asking for an undefined mass
        # centroid when the Boolean result is a zero-volume contact sheet.
        triangles = intersection.triangles - intersection.vertices.mean(axis=0)
        volume = max(
            0.0,
            float(np.einsum("ij,ij->", triangles[:, 0], np.cross(triangles[:, 1], triangles[:, 2])))
            / 6,
        )
        ratio = volume / min(float(first.volume), float(second.volume))
        if ratio > 1e-4:
            overlaps.append(
                {
                    "parts": [name_a, name_b],
                    "intersection_volume": volume,
                    "fraction_of_smaller_part": ratio,
                }
            )
    return overlaps


def check_motion(assembly: dict, parts: dict[str, trimesh.Trimesh], samples: int = 13) -> dict:
    """Check rest, each joint, and synchronized extreme motion. Not continuous CCD."""
    clean = {}
    for name, mesh in parts.items():
        clean[name] = mesh.copy()
        clean[name].merge_vertices()
    invalid = [name for name, mesh in clean.items() if not mesh.is_volume]
    if invalid:
        return {
            "assessed": False,
            "passed": False,
            "reason": "non-volume candidate parts",
            "parts": invalid,
        }
    joints = {j["child"]: j for j in assembly["joints"]}
    if len(joints) != len(assembly["joints"]) or any(
        not np.isfinite(j["limits"]).all() for j in joints.values()
    ):
        return {
            "assessed": False,
            "passed": False,
            "reason": "requires bounded single-parent joint tree",
        }
    # Authoring compiles the tree; the inspector reports unsupported loops explicitly.
    for name in parts:
        ancestors = set()
        current = name
        while current in joints:
            if current in ancestors:
                return {"assessed": False, "passed": False, "reason": "closed joint graph"}
            ancestors.add(current)
            current = joints[current]["parent"]
    states = [("rest", {})]
    for joint in joints.values():
        states.extend(
            (joint["name"], {joint["name"]: float(q)})
            for q in np.linspace(*joint["limits"], samples)
            if q != 0
        )
    if len(joints) > 1:
        states.extend(
            (
                "synchronized",
                {j["name"]: float(f * max(j["limits"], key=abs)) for j in joints.values()},
            )
            for f in np.linspace(0, 1, samples)[1:]
        )
    failures = []
    for label, coordinates in states:
        overlaps = solid_overlaps(pose_parts(clean, joints, coordinates))
        if overlaps:
            failures.append({"sweep": label, "coordinates": coordinates, "overlaps": overlaps})
    maximum = max(
        (v["fraction_of_smaller_part"] for f in failures for v in f["overlaps"]), default=0.0
    )
    return {
        "assessed": True,
        "passed": not failures,
        "states_checked": len(states),
        "failed_states": len(failures),
        "max_overlap_fraction": maximum,
        "failures": failures,
        "overlap_fraction_threshold": 1e-4,
        "units": "metres and cubic metres; relative threshold is dimensionless",
        "scope": "solid intersection at sampled kinematic poses; no continuous or dynamics guarantee",
    }


def motion_summary(report: dict) -> dict:
    """Bound feedback size while preserving all failing states in the evidence file."""
    return {
        **report,
        "failures": sorted(
            report.get("failures", []),
            key=lambda f: max(o["fraction_of_smaller_part"] for o in f["overlaps"]),
            reverse=True,
        )[:3],
    }
