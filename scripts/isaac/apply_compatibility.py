"""Minimal Isaac Kit USD/PhysX adapter. Run with --/usdcraft/job=<job.json>.

Job: {"asset": "original.usdz", "output": "adapted.usdz",
      "collision_mode": "articulation" | "standard_joints" |
                        "standard_joints_orthogonal_compound",
      "decomposition": {"errorPercentage": 0.1, "shrinkWrap": true,
                        "voxelResolution": 8000000, "maxConvexHulls": 128}}
An optional {"jobs": [...]} envelope runs several independent exports.
Only existing convexDecomposition collision meshes receive these parameters.
Orthogonal-compound mode replaces closed orthogonal mesh colliders with an
exact hidden compound of boxes; unsupported meshes retain native approximation.
No points, mass, inertia, joint frames/limits, drives, or materials are changed.
"""

import hashlib
import json
import sys
import traceback
from pathlib import Path

import carb
import omni.kit.app
from pxr import PhysxSchema, Sdf, Usd, UsdGeom, UsdPhysics, UsdUtils

sys.path.insert(0, str(Path(__file__).resolve().parent))
import orthogonal_compound

DEFAULTS = {
    "errorPercentage": 0.1,
    "shrinkWrap": True,
    "voxelResolution": 8000000,
    "maxConvexHulls": 128,
    "hullVertexLimit": 64,
    "minThickness": 0.001,
}
NATIVE_DEFAULTS = {
    "errorPercentage": 10.0,
    "shrinkWrap": False,
    "voxelResolution": 500000,
    "maxConvexHulls": 32,
    "hullVertexLimit": 64,
    "minThickness": 0.001,
}


def digest_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def value(v: object) -> object:
    if v is None or isinstance(v, (str, bool, int, float)):
        return v
    if isinstance(v, Sdf.AssetPath):
        return v.path
    if isinstance(v, dict):
        return {str(k): value(x) for k, x in v.items()}
    try:
        return [value(x) for x in v]
    except TypeError:
        return str(v)


def fingerprint(data: dict) -> str:
    return hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()


def invariants(stage: Usd.Stage, excluded_paths: list[str] | tuple[str, ...] = ()) -> dict:
    """Compare every authored value except the precisely allowed edits."""
    all_values, meshes, mass, joints = {}, {}, {}, {}
    ignored = {"physics:excludeFromArticulation", "physics:collisionEnabled"}
    for prim in stage.Traverse():
        if any(prim.GetPath().HasPrefix(Sdf.Path(p)) for p in excluded_paths):
            continue
        attrs = {}
        for a in prim.GetAuthoredAttributes():
            name = a.GetName()
            if name in ignored or name.startswith("physxConvexDecompositionCollision:"):
                continue
            attrs[name] = {
                "default": value(a.Get()),
                "samples": {str(t): value(a.Get(t)) for t in a.GetTimeSamples()},
            }
        rels = {
            r.GetName(): [str(x) for x in r.GetTargets()] for r in prim.GetAuthoredRelationships()
        }
        record = {"type": prim.GetTypeName(), "attributes": attrs, "relationships": rels}
        path = str(prim.GetPath())
        all_values[path] = record
        if prim.IsA(UsdGeom.Mesh):
            meshes[path] = record
        if prim.HasAPI(UsdPhysics.MassAPI):
            mass[path] = {
                k: v
                for k, v in attrs.items()
                if k
                in {
                    "physics:mass",
                    "physics:density",
                    "physics:centerOfMass",
                    "physics:diagonalInertia",
                    "physics:principalAxes",
                }
            }
        if prim.IsA(UsdPhysics.Joint):
            joints[path] = record
    return {
        "all_preserved_values_sha256": fingerprint(all_values),
        "mesh_values_sha256": fingerprint(meshes),
        "mesh_count": len(meshes),
        "mass_values_sha256": fingerprint(mass),
        "mass_body_count": len(mass),
        "joint_values_sha256": fingerprint(joints),
        "joint_count": len(joints),
        "prim_count": len(all_values),
    }


def edit_attr(
    changes: list[dict],
    prim: Usd.Prim,
    attr: Usd.Attribute,
    new: object,
    before_fallback: object = None,
) -> None:
    before = {"authored": attr.HasAuthoredValueOpinion(), "value": value(attr.Get())}
    if before["value"] is None and before_fallback is not None:
        before["native_default_without_API"] = before_fallback
    attr.Set(new)
    changes.append(
        {
            "path": str(prim.GetPath()),
            "attribute": attr.GetName(),
            "before": before,
            "after": {"authored": True, "value": value(attr.Get())},
        }
    )


def adapt(job: dict) -> dict:
    source = Path(job["asset"]).resolve()
    output = Path(job["output"]).resolve()
    if source == output or output.exists():
        raise ValueError("Output must be a new path, distinct from the source")
    if output.suffix.lower() not in (".usd", ".usda", ".usdc", ".usdz"):
        raise ValueError("Output must be USD, USDA, USDC, or USDZ")
    mode = job.get("collision_mode", "articulation")
    if mode not in ("articulation", "standard_joints", "standard_joints_orthogonal_compound"):
        raise ValueError("Unknown collision_mode: " + mode)
    params = dict(DEFAULTS)
    unknown = set(job.get("decomposition", {})) - set(params)
    if unknown:
        raise ValueError("Unknown decomposition parameters: " + str(unknown))
    params.update(job.get("decomposition", {}))
    if (
        not 0 < params["errorPercentage"] <= 100
        or not 1 <= params["maxConvexHulls"] <= 1024
        or not 4 <= params["hullVertexLimit"] <= 255
        or params["voxelResolution"] < 10000
        or params["minThickness"] < 0
        or not isinstance(params["shrinkWrap"], bool)
    ):
        raise ValueError("Invalid convex decomposition parameters")
    source_hash = digest_file(source)
    # Flatten composed opinions into an anonymous layer; never edit the source.
    original = Usd.Stage.Open(str(source))
    before = invariants(original)
    stage = Usd.Stage.Open(original.Flatten())
    changes = []
    compounds, proxy_roots = [], []
    for prim in list(stage.Traverse()):
        if prim.IsA(UsdGeom.Mesh) and prim.HasAPI(UsdPhysics.CollisionAPI):
            replaced = False
            if mode == "standard_joints_orthogonal_compound":
                info = {"source_mesh": str(prim.GetPath())}
                try:
                    boxes, metrics = orthogonal_compound.partition(UsdGeom.Mesh(prim))
                    proxy_root, authored = orthogonal_compound.author(
                        stage, UsdGeom.Mesh(prim), boxes
                    )
                    proxy_roots.append(proxy_root)
                    edit_attr(
                        changes,
                        prim,
                        UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr(),
                        False,
                    )
                    info.update(metrics)
                    info.update(
                        {
                            "status": "orthogonal_compound",
                            "proxy_root": proxy_root,
                            "authored_collision_prims": authored,
                        }
                    )
                    replaced = True
                except orthogonal_compound.UnsupportedMesh as error:
                    info.update({"status": "native_approximation_fallback", "reason": str(error)})
                compounds.append(info)
            mesh_api = UsdPhysics.MeshCollisionAPI(prim)
            if (
                not replaced
                and mesh_api
                and mesh_api.GetApproximationAttr().Get() == "convexDecomposition"
            ):
                had_api = prim.HasAPI(PhysxSchema.PhysxConvexDecompositionCollisionAPI)
                api = PhysxSchema.PhysxConvexDecompositionCollisionAPI.Apply(prim)
                if not had_api:
                    changes.append(
                        {
                            "path": str(prim.GetPath()),
                            "api_schema": "PhysxConvexDecompositionCollisionAPI",
                            "before": False,
                            "after": True,
                        }
                    )
                for name, new in params.items():
                    attr = getattr(api, "Get" + name[0].upper() + name[1:] + "Attr")()
                    edit_attr(changes, prim, attr, new, NATIVE_DEFAULTS[name])
        if mode.startswith("standard_joints"):
            if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
                prim.RemoveAPI(UsdPhysics.ArticulationRootAPI)
                changes.append(
                    {
                        "path": str(prim.GetPath()),
                        "api_schema": "PhysicsArticulationRootAPI",
                        "before": True,
                        "after": False,
                    }
                )
            if prim.IsA(UsdPhysics.Joint):
                joint = UsdPhysics.Joint(prim)
                edit_attr(changes, prim, joint.GetExcludeFromArticulationAttr(), True)
                edit_attr(changes, prim, joint.GetCollisionEnabledAttr(), True)
    output.parent.mkdir(parents=True, exist_ok=True)
    intermediate = output.with_suffix(".usda") if output.suffix.lower() == ".usdz" else output
    if intermediate.exists():
        raise ValueError("Intermediate path already exists: " + str(intermediate))
    stage.Export(str(intermediate))
    if output.suffix.lower() == ".usdz" and not UsdUtils.CreateNewUsdzPackage(
        Sdf.AssetPath(str(intermediate)), str(output)
    ):
        raise RuntimeError("USDZ packaging failed")
    reopened = Usd.Stage.Open(str(output))
    after = invariants(reopened, proxy_roots)
    preserved = before == after
    record = {
        "source_asset": str(source),
        "source_sha256": source_hash,
        "output_asset": str(output),
        "output_sha256": digest_file(output),
        "collision_mode": mode,
        "decomposition": params,
        "changes": changes,
        "orthogonal_compounds": compounds,
        "preservation_excluded_generated_proxy_roots": proxy_roots,
        "invariants_before": before,
        "invariants_after": after,
        "geometry_mass_joint_values_preserved": preserved,
        "source_file_unchanged": digest_file(source) == source_hash,
        "articulation_roots_after": [
            str(p.GetPath())
            for p in reopened.Traverse()
            if p.HasAPI(UsdPhysics.ArticulationRootAPI)
        ],
        "joint_collision_contract": (
            "Standard maximal-coordinate rigid bodies and original USD joints; "
            "excludeFromArticulation=true and collisionEnabled=true on each joint. "
            "This is not the original reduced-coordinate articulation."
            if mode.startswith("standard_joints")
            else "Original articulation retained. Per-joint collisionEnabled is ignored "
            "by native articulation parsing; adjacent-pair collision effectiveness "
            "is not claimed in this mode."
        ),
    }
    output.with_suffix(".patchmanifest.json").write_text(json.dumps(record, indent=2))
    if not preserved or not record["source_file_unchanged"]:
        raise RuntimeError("Preservation check failed; see patchmanifest")
    return record


def main() -> None:
    path = carb.settings.get_settings().get("/usdcraft/job")
    job = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    records = [adapt(j) for j in job.get("jobs", [job])]
    if "manifest" in job:
        Path(job["manifest"]).write_text(json.dumps(records, indent=2))
    print(
        json.dumps({"adapted": len(records), "outputs": [r["output_asset"] for r in records]}),
        flush=True,
    )


try:
    main()
except Exception:
    traceback.print_exc()
    omni.kit.app.get_app().post_quit(1)
else:
    omni.kit.app.get_app().post_quit(0)
