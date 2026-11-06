"""Paper-aligned local metrics. Missing-part Chamfer penalty is explicit/configurable.

This is not the unavailable official evaluator. F1 uses movable parts. Its
aggregation and missing-distance conventions are recorded, never silently claimed
to match every unpublished detail of the paper implementation.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import trimesh
from scipy.optimize import linear_sum_assignment

from articraft.geometry_recheck import chamfer_squared, sample_surface
from articraft.usdcraft_motion import pose_parts


def load_parts(directory: Path, assembly: dict) -> dict[str, trimesh.Trimesh]:
    return {
        p["name"]: trimesh.load_mesh(directory / p["mesh"], process=False)
        for p in assembly["parts"]
    }


def box_iou(first: np.ndarray, second: np.ndarray) -> tuple[float, float]:
    intersection = np.maximum(
        0, np.minimum(first[1], second[1]) - np.maximum(first[0], second[0])
    ).prod()
    union = np.prod(first[1] - first[0]) + np.prod(second[1] - second[0]) - intersection
    enclosing = np.prod(np.maximum(first[1], second[1]) - np.minimum(first[0], second[0]))
    iou = intersection / union if union else 0.0
    return float(iou), float(iou - (enclosing - union) / enclosing if enclosing else 0)


def axis_error(a: np.ndarray, b: np.ndarray) -> float:
    return float(
        np.degrees(
            np.arccos(np.clip(abs(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))), 0, 1))
        )
    )


def axis_line_distance(o1: np.ndarray, a1: np.ndarray, o2: np.ndarray, a2: np.ndarray) -> float:
    a1 = a1 / np.linalg.norm(a1)
    a2 = a2 / np.linalg.norm(a2)
    normal = np.cross(a1, a2)
    if np.linalg.norm(normal) < 1e-8:
        return float(np.linalg.norm(np.cross(o2 - o1, a1)))
    return float(abs(np.dot(o2 - o1, normal)) / np.linalg.norm(normal))


def _normalize(parts: dict, assembly: dict, frame: tuple | None = None) -> tuple[dict, dict, tuple]:
    bounds = trimesh.util.concatenate(list(parts.values())).bounds
    center, size = (
        frame if frame is not None else (bounds.mean(0), float(np.max(bounds[1] - bounds[0])))
    )
    normalized = {}
    for name, mesh in parts.items():
        transformed = mesh.copy()
        transformed.vertices = (transformed.vertices - center) / size
        normalized[name] = transformed
    joints = {
        j["child"]: {
            **j,
            "origin": ((np.asarray(j["origin"]) - center) / size).tolist(),
            "limits": (
                np.asarray(j["limits"]) / (size if j["type"] == "prismatic" else 1)
            ).tolist(),
        }
        for j in assembly["joints"]
    }
    return normalized, joints, (center, size)


def _endpoint(parts: dict, joints: dict) -> dict:
    coordinates = {j["name"]: max(j["limits"], key=abs) for j in joints.values()}
    return pose_parts(parts, joints, coordinates)


def evaluate_assembly(
    reference: dict,
    prediction: dict,
    reference_parts: dict,
    prediction_parts: dict,
    samples: int = 100000,
    reference_frame: bool = False,
    missing_part_pc: float = 2.0,
) -> dict:
    if not prediction_parts:
        joints = reference["joints"]
        return {
            "F1_percent": 0.0,
            "static_gIoU": -1.0,
            "static_PC": missing_part_pc,
            "static_mIoU": 0.0,
            "articulated_gIoU": -1.0,
            "articulated_PC": missing_part_pc,
            "articulated_OC": missing_part_pc,
            "AE_degrees": 90.0 if joints else None,
            "LE": 0.5 if any(j["type"] == "revolute" for j in joints) else None,
            "failed_output": True,
            "missing_part_PC_penalty": missing_part_pc,
        }
    a, ja, frame = _normalize(reference_parts, reference)
    b, jb, _ = _normalize(prediction_parts, prediction, frame if reference_frame else None)
    names_a, names_b = list(a), list(b)
    cost = np.array(
        [
            [np.linalg.norm(a[x].bounds.mean(0) - b[y].bounds.mean(0)) for y in names_b]
            for x in names_a
        ]
    )
    ii, jj = linear_sum_assignment(cost)
    pairs = {names_a[i]: names_b[j] for i, j in zip(ii, jj, strict=True)}
    end_a, end_b = _endpoint(a, ja), _endpoint(b, jb)
    static_iou = []
    static_giou = []
    static_pc = []
    end_giou = []
    end_pc = []
    angles = []
    locations = []
    matched_movable = 0
    details = []
    # One whole-object surface sample is partitioned by part area for PC.
    total_area = sum(m.area for m in a.values())
    for name, mesh in a.items():
        other = pairs.get(name)
        expected = ja.get(name)
        actual = jb.get(other)
        if other is None:
            static_iou.append(0)
            static_giou.append(-1)
            end_giou.append(-1)
            static_pc.append(missing_part_pc)
            end_pc.append(missing_part_pc)
            if expected:
                angles.append(90)
                locations.extend([0.5] if expected["type"] == "revolute" else [])
            details.append({"reference": name, "prediction": None})
            continue
        candidate = b[other]
        iou, giou = box_iou(mesh.bounds, candidate.bounds)
        count = max(32, round(samples * mesh.area / total_area))
        static_iou.append(iou)
        static_giou.append(giou)
        static_pc.append(
            chamfer_squared(sample_surface(mesh, count), sample_surface(candidate, count))
        )
        end_giou.append(box_iou(end_a[name].bounds, end_b[other].bounds)[1])
        end_pc.append(
            chamfer_squared(sample_surface(end_a[name], count), sample_surface(end_b[other], count))
        )
        ae = 90.0
        le = 0.5
        valid = False
        if expected and actual and expected["type"] == actual["type"]:
            ae = axis_error(np.asarray(expected["axis"]), np.asarray(actual["axis"]))
            le = (
                axis_line_distance(
                    np.asarray(expected["origin"]),
                    np.asarray(expected["axis"]),
                    np.asarray(actual["origin"]),
                    np.asarray(actual["axis"]),
                )
                if expected["type"] == "revolute"
                else None
            )
            distance = np.linalg.norm(mesh.bounds.mean(0) - candidate.bounds.mean(0))
            sizes = [mesh.extents.max(), candidate.extents.max()]
            spatial = giou >= 0.25 or (
                distance <= 0.6 * min(sizes) and max(sizes) / min(sizes) <= 2
            )
            valid = bool(spatial and ae <= 15 and (le is None or le <= 0.1))
            matched_movable += int(valid)
        if expected:
            angles.append(ae)
            if expected["type"] == "revolute":
                locations.append(le if le is not None else 0.5)
        details.append(
            {
                "reference": name,
                "prediction": other,
                "movable_match": valid,
                "axis_error_degrees": ae if expected else None,
            }
        )
    denominator = len(ja) + len(jb)
    return {
        "F1_percent": 200 * matched_movable / denominator if denominator else None,
        "static_gIoU": float(np.mean(static_giou)),
        "static_PC": float(np.mean(static_pc)),
        "static_mIoU": float(np.mean(static_iou)),
        "articulated_gIoU": float(np.mean(end_giou)),
        "articulated_PC": float(np.mean(end_pc)),
        "articulated_OC": chamfer_squared(
            sample_surface(trimesh.util.concatenate(list(end_a.values())), samples),
            sample_surface(trimesh.util.concatenate(list(end_b.values())), samples),
        ),
        "AE_degrees": float(np.mean(angles)) if angles else None,
        "LE": float(np.mean(locations)) if locations else None,
        "matched_movable": matched_movable,
        "reference_movable": len(ja),
        "prediction_movable": len(jb),
        "part_correspondence": details,
        "normalization": "reference frame"
        if reference_frame
        else "independent centers and unit longest side; no rotation fitting",
        "samples_per_asset": samples,
        "missing_part_PC_penalty": missing_part_pc,
        "protocol_status": "local paper-aligned implementation; movable-only F1 and missing-PC penalty need official verification",
    }
