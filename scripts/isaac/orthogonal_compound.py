"""Exact box partition for closed orthogonal USD meshes; no case-specific data.

Unsupported/non-manifold/non-orthogonal/oversize meshes raise UnsupportedMesh.
Only source vertex planes and parity of source triangles determine occupancy.
Adjacent solid cells are greedily merged without changing their union.
"""

from bisect import bisect_left
from collections import Counter
from itertools import product

from pxr import Gf, Usd, UsdGeom, UsdPhysics


class UnsupportedMesh(ValueError):
    pass


def sub(a: tuple, b: tuple) -> tuple:
    return tuple(x - y for x, y in zip(a, b, strict=True))


def cross(a: tuple, b: tuple) -> tuple:
    return (a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0])


def dot(a: tuple, b: tuple) -> float:
    return sum(x * y for x, y in zip(a, b, strict=True))


def nearest_plane(planes: list[float], value: float) -> int:
    upper = bisect_left(planes, value)
    if upper == 0 or upper == len(planes):
        return max(0, upper - 1)
    return upper - 1 if value - planes[upper - 1] <= planes[upper] - value else upper


def partition(
    mesh: UsdGeom.Mesh, max_cells: int = 200000, max_boxes: int = 2048, tolerance: float = 1e-8
) -> tuple[list[dict], dict]:
    points = [tuple(float(v) for v in p) for p in mesh.GetPointsAttr().Get()]
    if mesh.GetHoleIndicesAttr().Get() or mesh.GetPointsAttr().GetNumTimeSamples():
        raise UnsupportedMesh("Hole faces and animated points are not supported")
    planes = []
    max_merge = 0.0
    for axis in range(3):
        clusters = []
        for x in sorted({p[axis] for p in points}):
            if clusters and x - clusters[-1][0] <= tolerance:
                clusters[-1].append(x)
            else:
                clusters.append([x])
        merged = [sum(c) / len(c) for c in clusters]
        max_merge = max(
            max_merge, *(abs(x - m) for c, m in zip(clusters, merged, strict=True) for x in c)
        )
        planes.append(merged)
    vertex_ids = [tuple(nearest_plane(planes[a], p[a]) for a in range(3)) for p in points]
    counts = list(mesh.GetFaceVertexCountsAttr().Get())
    indices = list(mesh.GetFaceVertexIndicesAttr().Get())
    edges = Counter()
    tris = []
    volume = 0.0
    start = 0
    for n in counts:
        face = indices[start : start + n]
        start += n
        if n < 3:
            raise UnsupportedMesh("Degenerate polygon")
        for i in range(n):
            u, v = vertex_ids[face[i]], vertex_ids[face[(i + 1) % n]]
            if u != v:
                edges[tuple(sorted((u, v)))] += 1
        for i in range(1, n - 1):
            a, b, c = [points[k] for k in (face[0], face[i], face[i + 1])]
            normal = cross(sub(b, a), sub(c, a))
            magnitude = sum(v * v for v in normal) ** 0.5
            if magnitude < 1e-15:
                continue
            axis = max(range(3), key=lambda j: abs(normal[j]))
            if any(abs(normal[j]) / magnitude > 1e-6 for j in range(3) if j != axis):
                raise UnsupportedMesh("Non-axis-aligned face normal")
            if max(abs(p[axis] - a[axis]) for p in (b, c)) > tolerance:
                raise UnsupportedMesh("Face is not on a single source coordinate plane")
            volume += dot(a, cross(b, c)) / 6
            if axis == 0:
                tris.append((a, b, c))
    if not edges or any(n != 2 for n in edges.values()):
        raise UnsupportedMesh("Mesh is not a closed two-manifold after vertex welding")
    dims = [len(p) - 1 for p in planes]
    cell_count = dims[0] * dims[1] * dims[2]
    if cell_count == 0 or cell_count > max_cells:
        raise UnsupportedMesh(f"Orthogonal grid cell count {cell_count} exceeds limit {max_cells}")
    # At each (y,z) cell center, find unique source boundary planes along x.
    occupied = set()
    for j, k in product(range(dims[1]), range(dims[2])):
        y = (planes[1][j] + planes[1][j + 1]) / 2
        z = (planes[2][k] + planes[2][k + 1]) / 2
        crossings = []
        for a, b, c in tris:
            # Barycentric point-in-triangle in the yz plane, including diagonals.
            v0 = (b[1] - a[1], b[2] - a[2])
            v1 = (c[1] - a[1], c[2] - a[2])
            v2 = (y - a[1], z - a[2])
            det = v0[0] * v1[1] - v1[0] * v0[1]
            if abs(det) < 1e-18:
                continue
            u = (v2[0] * v1[1] - v1[0] * v2[1]) / det
            v = (v0[0] * v2[1] - v2[0] * v0[1]) / det
            if (
                u >= -1e-9
                and v >= -1e-9
                and u + v <= 1 + 1e-9
                and not any(abs(a[0] - x) <= tolerance for x in crossings)
            ):
                crossings.append(a[0])
        crossings.sort()
        if len(crossings) % 2:
            raise UnsupportedMesh("Odd boundary crossing count; source does not bound a solid")
        for i in range(dims[0]):
            x = (planes[0][i] + planes[0][i + 1]) / 2
            if sum(t > x for t in crossings) % 2:
                occupied.add((i, j, k))
    solid_count = len(occupied)
    union_volume = sum(
        (planes[0][i + 1] - planes[0][i])
        * (planes[1][j + 1] - planes[1][j])
        * (planes[2][k + 1] - planes[2][k])
        for i, j, k in occupied
    )
    source_volume = abs(volume)
    if source_volume <= 0 or abs(union_volume - source_volume) > max(1e-12, source_volume * 1e-5):
        raise UnsupportedMesh("Cell union and oriented source volume disagree")
    boxes = []
    remaining = set(occupied)
    while remaining:
        i, j, k = min(remaining)
        i1 = i + 1
        while i1 < dims[0] and (i1, j, k) in remaining:
            i1 += 1
        j1 = j + 1
        while j1 < dims[1] and all((x, j1, k) in remaining for x in range(i, i1)):
            j1 += 1
        k1 = k + 1
        while k1 < dims[2] and all(
            (x, y, k1) in remaining for x, y in product(range(i, i1), range(j, j1))
        ):
            k1 += 1
        remaining.difference_update(product(range(i, i1), range(j, j1), range(k, k1)))
        lo = [planes[a][idx] for a, idx in enumerate((i, j, k))]
        hi = [planes[a][idx] for a, idx in enumerate((i1, j1, k1))]
        boxes.append(
            {
                "center": [(a + b) / 2 for a, b in zip(lo, hi, strict=True)],
                "dimensions": [b - a for a, b in zip(lo, hi, strict=True)],
            }
        )
        if len(boxes) > max_boxes:
            raise UnsupportedMesh(f"Merged box count exceeds {max_boxes}")
    return boxes, {
        "coordinate_plane_counts": [len(p) for p in planes],
        "grid_cell_count": cell_count,
        "solid_cell_count": solid_count,
        "box_count": len(boxes),
        "source_mesh_signed_volume": volume,
        "source_mesh_volume": source_volume,
        "compound_volume": union_volume,
        "absolute_volume_error": abs(union_volume - source_volume),
        "plane_merge_tolerance": tolerance,
        "maximum_plane_merge_displacement": max_merge,
        "closed_two_manifold": True,
        "all_face_normals_axis_aligned": True,
    }


def author(stage: Usd.Stage, mesh: UsdGeom.Mesh, boxes: list[dict]) -> tuple[str, list[dict]]:
    parent_path = str(mesh.GetPath()) + "/OrthogonalCollision"
    if stage.GetPrimAtPath(parent_path):
        raise UnsupportedMesh("Proxy path already exists")
    parent = UsdGeom.Xform.Define(stage, parent_path)
    parent.CreateVisibilityAttr("invisible")
    parent.CreatePurposeAttr("guide")
    authored = []
    for i, box in enumerate(boxes):
        path = f"{parent_path}/box_{i:04d}"
        cube = UsdGeom.Cube.Define(stage, path)
        cube.CreateSizeAttr(1.0)
        cube.AddTranslateOp().Set(Gf.Vec3d(*box["center"]))
        cube.AddScaleOp().Set(Gf.Vec3f(*box["dimensions"]))
        UsdPhysics.CollisionAPI.Apply(cube.GetPrim()).CreateCollisionEnabledAttr(True)
        authored.append(
            {
                "path": path,
                "type": "Cube",
                "api_schema": "PhysicsCollisionAPI",
                "size": 1.0,
                "translate": box["center"],
                "scale": box["dimensions"],
                "physics:collisionEnabled": True,
            }
        )
    return parent_path, authored
