"""Metric surface support, axial text encoding, and local mesh measurements.

Cells record observed SURFACE, never solid occupancy. All other space is unknown.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import trimesh


@dataclass(frozen=True)
class SurfaceGrid:
    origin: np.ndarray
    pitch: float
    shape: tuple[int, int, int]

    @classmethod
    def from_mesh(cls, mesh: trimesh.Trimesh, resolution: int = 192) -> SurfaceGrid:
        if resolution < 2 or not len(mesh.faces) or mesh.extents.max() <= 0:
            raise ValueError("A nonempty surface and resolution >= 2 are required")
        pitch = float(mesh.extents.max() / (resolution - 1))
        shape = tuple((np.ceil(mesh.extents / pitch - 1e-9).astype(int) + 1).tolist())
        return cls(mesh.bounds[0] - pitch / 2, pitch, shape)

    def indices(self, points: np.ndarray) -> np.ndarray:
        return np.floor((points - self.origin) / self.pitch + 1e-9).astype(np.int64)

    def to_dict(self) -> dict:
        return {"origin": self.origin.tolist(), "pitch": self.pitch, "shape": list(self.shape)}


def projected_samples(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Grid-center rays through a projected triangle, with barycentric weights."""
    a, b, c = points
    u, v = b - a, c - a
    determinant = u[0] * v[1] - u[1] * v[0]
    if abs(determinant) < 1e-12:
        return np.empty((0, 2), dtype=int), np.empty((0, 3))
    lo = np.ceil(points.min(0) - 0.5).astype(int)
    hi = np.floor(points.max(0) - 0.5).astype(int)
    xx, yy = np.meshgrid(np.arange(lo[0], hi[0] + 1), np.arange(lo[1], hi[1] + 1))
    pixels = np.column_stack((xx.ravel(), yy.ravel()))
    delta = pixels + 0.5 - a
    beta = (delta[:, 0] * v[1] - delta[:, 1] * v[0]) / determinant
    gamma = (u[0] * delta[:, 1] - u[1] * delta[:, 0]) / determinant
    bary = np.column_stack((1 - beta - gamma, beta, gamma))
    valid = (bary >= -1e-9).all(1)
    return pixels[valid], bary[valid]


def surface_support(mesh: trimesh.Trimesh, grid: SurfaceGrid) -> np.ndarray:
    """All-hit orthographic ray sampling along three axes, plus vertices/centroids.

    Triangle rasterization visits only projected footprints, avoiding a dense
    ray-by-face tensor. Out-of-source-grid candidate cells are retained.
    """
    triangles = (mesh.triangles - grid.origin) / grid.pitch
    support = [grid.indices(mesh.vertices), grid.indices(mesh.triangles_center)]
    for triangle in triangles:
        for axis in range(3):
            other = [i for i in range(3) if i != axis]
            pixels, bary = projected_samples(triangle[:, other])
            hit = np.empty((len(pixels), 3), dtype=np.int64)
            hit[:, other] = pixels
            hit[:, axis] = np.floor(bary @ triangle[:, axis] + 1e-9).astype(np.int64)
            support.append(hit)
    return np.unique(np.concatenate(support), axis=0)


def axial_text(grid: SurfaceGrid, cells: np.ndarray, max_chars: int = 16000) -> str:
    """Choose 32/24/16 longest-axis cells; repeat identical XY slices once."""
    for longest in (32, 24, 16):
        factor = max(1, int(np.ceil(max(grid.shape) / longest)))
        shape = np.ceil(np.asarray(grid.shape) / factor).astype(int)
        coarse = np.unique(np.floor_divide(cells, factor), axis=0)
        slices: dict[str, list[int]] = {}
        for z in range(shape[2]):
            layer = np.full((shape[1], shape[0]), "?", dtype="U1")
            xy = coarse[coarse[:, 2] == z, :2]
            xy = xy[((xy >= 0) & (xy < shape[:2])).all(1)]
            layer[xy[:, 1], xy[:, 0]] = "#"
            key = "/".join("".join(row) for row in layer)
            slices.setdefault(key, []).append(z)
        head = (
            f"SURFACE SUPPORT, not occupancy. #=observed surface; ?=unknown, never empty/solid.\n"
            f"origin_xyz={grid.origin.tolist()} coarse_pitch={grid.pitch * factor:.8g} "
            f"shape_xyz={shape.tolist()}; rows=+Y, columns=+X; slices=+Z.\n"
        )
        text = head + "\n".join(f"z={zs}: {layer}" for layer, zs in slices.items())
        if len(text) <= max_chars:
            return text
    raise ValueError("Text budget is too small for the 16-cell surface encoding")


def clip_surface(mesh: trimesh.Trimesh, bounds: np.ndarray) -> trimesh.Trimesh:
    """Clip triangles against six planes, without closing unobserved surfaces."""
    triangles = mesh.triangles
    for axis in range(3):
        for side, coordinate in ((1, bounds[0, axis]), (-1, bounds[1, axis])):
            clipped = []
            for triangle in triangles:
                polygon = []
                for start, end in zip(triangle, np.roll(triangle, -1, axis=0), strict=True):
                    ds = side * (start[axis] - coordinate)
                    de = side * (end[axis] - coordinate)
                    if ds >= 0:
                        polygon.append(start)
                    if (ds >= 0) != (de >= 0):
                        polygon.append(start + ds / (ds - de) * (end - start))
                clipped.extend(
                    [polygon[0], polygon[i], polygon[i + 1]] for i in range(1, len(polygon) - 1)
                )
            triangles = np.asarray(clipped).reshape(-1, 3, 3)
    result = trimesh.Trimesh(
        vertices=triangles.reshape(-1, 3),
        faces=np.arange(triangles.size // 3).reshape(-1, 3),
        process=False,
    )
    result.update_faces(result.nondegenerate_faces())
    result.merge_vertices()
    result.remove_unreferenced_vertices()
    return result


def measure_mesh(mesh: trimesh.Trimesh, bounds: np.ndarray | None = None) -> dict:
    """Clip to a metric AABB before measuring; do not infer hidden structure."""
    patch = mesh if bounds is None else clip_surface(mesh, bounds)
    if not len(patch.faces):
        return {"observed": False, "interpretation": "unknown; no observed surface in region"}
    normals = patch.face_normals
    areas = patch.area_faces
    return {
        "observed": True,
        "bounds": patch.bounds.tolist(),
        "extents": patch.extents.tolist(),
        "surface_area": float(patch.area),
        "triangles": len(patch.faces),
        "area_weighted_normal": np.average(normals, axis=0, weights=areas).tolist(),
        "absolute_normal_axis_areas": (np.abs(normals) * areas[:, None]).sum(0).tolist(),
        "watertight": bool(patch.is_watertight),
        "region_is_clipped": bounds is not None,
        "unobserved_space": "unknown",
    }


def measure_face(
    mesh: trimesh.Trimesh, axis: int, side: int, bounds: np.ndarray | None = None
) -> dict:
    selected = mesh.face_normals[:, axis] * side > 0.9
    patch = mesh.submesh([np.flatnonzero(selected)], append=True)
    result = measure_mesh(patch, bounds)
    result.update(axis=axis, side=side, normal_cosine_threshold=0.9)
    return result


def planar_observations(mesh: trimesh.Trimesh, max_patches: int = 40) -> dict:
    """Group observed axis-aligned faces into metric planes; never infer hidden walls."""
    tolerance = float(mesh.extents.max() * 1e-6)
    patches = []
    for axis in range(3):
        for side in (-1, 1):
            selected = np.flatnonzero(mesh.face_normals[:, axis] * side > 0.999999)
            levels = np.round(mesh.triangles_center[selected, axis] / tolerance).astype(int)
            for level in np.unique(levels):
                faces = selected[levels == level]
                area = mesh.area_faces[faces]
                vertices = mesh.triangles[faces].reshape(-1, 3)
                patches.append(
                    {
                        "axis": "XYZ"[axis],
                        "normal_sign": side,
                        "coordinate": round(
                            float(np.average(mesh.triangles_center[faces, axis], weights=area)), 9
                        ),
                        "area": float(area.sum()),
                        "bounds": np.round(
                            np.stack([vertices.min(0), vertices.max(0)]), 9
                        ).tolist(),
                    }
                )
    ranked = sorted(patches, key=lambda p: p["area"], reverse=True)
    return {
        "patches": sorted(
            ranked[:max_patches], key=lambda p: (p["axis"], p["coordinate"], p["normal_sign"])
        ),
        "total_plane_groups": len(patches),
        "coordinate_tolerance": tolerance,
        "scope": "observed axis-aligned planar faces only; no hidden surfaces or joint labels inferred",
    }


def measure_plane(
    mesh: trimesh.Trimesh,
    normal: np.ndarray,
    bounds: np.ndarray | None = None,
    angle_degrees: float = 10.0,
) -> dict:
    """Measure observed faces in any world direction; never rotate the asset."""
    normal = np.asarray(normal, dtype=float)
    length = np.linalg.norm(normal)
    if normal.shape != (3,) or not np.isfinite(length) or length == 0:
        raise ValueError("normal must be a finite nonzero XYZ vector")
    normal = normal / length
    selected = mesh.face_normals @ normal >= np.cos(np.radians(angle_degrees))
    patch = mesh.submesh([np.flatnonzero(selected)], append=True)
    result = measure_mesh(patch, bounds)
    result.update(normal=normal.tolist(), angle_tolerance_degrees=angle_degrees)
    if result["observed"]:
        patch = patch if bounds is None else clip_surface(patch, bounds)
        offsets = patch.vertices @ normal
        result["plane_offset_range_m"] = [float(offsets.min()), float(offsets.max())]
    return result


def oriented_planar_observations(mesh: trimesh.Trimesh, max_patches: int = 24) -> dict:
    """Deterministic face-plane consensus, including tilted/rotated source planes.

    This is an observation summary, not shape completion or a general scan denoiser.
    Distance and angular support are reported explicitly.
    """
    tolerance = SurfaceGrid.from_mesh(mesh).pitch * 0.2
    remaining = np.arange(len(mesh.faces))
    rng = np.random.default_rng(0)
    patches = []
    for _ in range(max_patches):
        if not len(remaining):
            break
        areas = mesh.area_faces[remaining]
        centers = mesh.triangles_center[remaining]
        normals = mesh.face_normals[remaining]
        seeds = rng.choice(
            len(remaining), min(48, len(remaining)), replace=False, p=areas / areas.sum()
        )
        best, score = np.array([], dtype=int), 0.0
        for seed in seeds:
            normal = normals[seed]
            offsets = (centers - centers[seed]) @ normal
            valid = (abs(offsets) <= tolerance) & (normals @ normal >= np.cos(np.radians(5)))
            area = float(areas[valid].sum())
            if area > score:
                best, score = np.flatnonzero(valid), area
        if score < mesh.area * 0.001:
            break
        faces = remaining[best]
        weights = mesh.area_faces[faces]
        normal = np.average(mesh.face_normals[faces], axis=0, weights=weights)
        normal /= np.linalg.norm(normal)
        center = np.average(mesh.triangles_center[faces], axis=0, weights=weights)
        vertices = mesh.triangles[faces].reshape(-1, 3)
        residual = abs((vertices - center) @ normal)
        patches.append(
            {
                "id": f"O{len(patches) + 1:02d}",
                "normal": normal.tolist(),
                "center": center.tolist(),
                "plane_offset_m": float(center @ normal),
                "area_m2": score,
                "bounds": [vertices.min(0).tolist(), vertices.max(0).tolist()],
                "p95_plane_residual_m": float(np.quantile(residual, 0.95)),
                "triangles": len(faces),
            }
        )
        remaining = np.delete(remaining, best)
    return {
        "patches": patches,
        "distance_tolerance_m": tolerance,
        "normal_tolerance_degrees": 5.0,
        "unassigned_area_fraction": float(mesh.area_faces[remaining].sum() / mesh.area),
        "scope": "source world-frame observed face-plane consensus; hidden surfaces remain unknown",
    }


def analyze_source(
    mesh: trimesh.Trimesh, resolution: int = 192, max_chars: int = 16000
) -> tuple[SurfaceGrid, np.ndarray, dict]:
    grid = SurfaceGrid.from_mesh(mesh, resolution)
    cells = surface_support(mesh, grid)
    report = {
        "grid": grid.to_dict(),
        "measurement": measure_mesh(mesh),
        "surface_cells": len(cells),
        "encoding": axial_text(grid, cells, max_chars),
        "planar_observations": planar_observations(mesh),
        "oriented_planar_observations": oriented_planar_observations(mesh),
        "units": "metres, supplied by input manifest; not inferred from geometry",
    }
    return grid, cells, report
