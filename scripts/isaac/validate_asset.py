"""Run authored USD articulation assets in Isaac Sim's native PhysX extension.

Execute with Kit --exec. Arguments are provided by /usdcraft/job in Kit settings.
The asset layer is read only; all test-scene edits use an anonymous session layer.
"""

from __future__ import annotations

import hashlib
import json
import math
import traceback
from pathlib import Path

import carb
import carb.settings
import numpy as np
import omni.kit.app
import omni.physx
from pxr import Gf, PhysicsSchemaTools, PhysxSchema, Usd, UsdGeom, UsdPhysics, UsdUtils


def matrix(prim: Usd.Prim) -> np.ndarray:
    return np.asarray(
        UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    ).T


def frame(joint: UsdPhysics.Joint, side: int) -> np.ndarray:
    pos = getattr(joint, f"GetLocalPos{side}Attr")().Get()
    quat = getattr(joint, f"GetLocalRot{side}Attr")().Get()
    transform = Gf.Matrix4d().SetRotate(Gf.Quatd(quat))
    transform.SetTranslateOnly(Gf.Vec3d(pos))
    return np.asarray(transform).T


def set_world_matrix(prim: Usd.Prim, value: np.ndarray) -> None:
    parent = prim.GetParent()
    local = np.linalg.inv(matrix(parent)) @ value if parent.IsA(UsdGeom.Xformable) else value
    xform = UsdGeom.Xformable(prim)
    xform.ClearXformOpOrder()
    xform.AddTransformOp().Set(Gf.Matrix4d(local.T.tolist()))


def joint_state(stage: Usd.Stage, joint: UsdPhysics.Joint) -> tuple[float, float]:
    p0 = stage.GetPrimAtPath(joint.GetBody0Rel().GetTargets()[0])
    p1 = stage.GetPrimAtPath(joint.GetBody1Rel().GetTargets()[0])
    relative = np.linalg.inv(matrix(p0) @ frame(joint, 0)) @ matrix(p1) @ frame(joint, 1)
    axis = "XYZ".index(joint.GetAxisAttr().Get())
    if joint.GetPrim().IsA(UsdPhysics.RevoluteJoint):
        i, k = (axis + 1) % 3, (axis + 2) % 3
        q = math.degrees(math.atan2(relative[k, i], relative[i, i]))
        anchor_error = np.linalg.norm(relative[:3, 3])
    else:
        q = float(relative[axis, 3])
        anchor_error = np.linalg.norm(np.delete(relative[:3, 3], axis))
    return q, float(anchor_error)


def initialize_joint(stage: Usd.Stage, joint: UsdPhysics.Joint, q: float) -> None:
    p0 = stage.GetPrimAtPath(joint.GetBody0Rel().GetTargets()[0])
    p1 = stage.GetPrimAtPath(joint.GetBody1Rel().GetTargets()[0])
    axis = "XYZ".index(joint.GetAxisAttr().Get())
    relative = np.eye(4)
    if joint.GetPrim().IsA(UsdPhysics.RevoluteJoint):
        rotation = Gf.Rotation(Gf.Vec3d(*np.eye(3)[axis]), q)
        relative[:3, :3] = np.asarray(Gf.Matrix3d(rotation)).T
    else:
        relative[axis, 3] = q
    set_world_matrix(p1, matrix(p0) @ frame(joint, 0) @ relative @ np.linalg.inv(frame(joint, 1)))


def configure(stage: Usd.Stage, profile: dict, scenario: str) -> tuple[list, list, list]:
    assert UsdGeom.GetStageMetersPerUnit(stage) == 1.0
    assert UsdGeom.GetStageUpAxis(stage) == "Z"
    bodies = [p for p in stage.Traverse() if p.HasAPI(UsdPhysics.RigidBodyAPI)]
    joints = [
        UsdPhysics.RevoluteJoint(p)
        if p.IsA(UsdPhysics.RevoluteJoint)
        else UsdPhysics.PrismaticJoint(p)
        for p in stage.Traverse()
        if p.IsA(UsdPhysics.RevoluteJoint) or p.IsA(UsdPhysics.PrismaticJoint)
    ]
    assert bodies and joints, "A dynamic articulated USD asset is required"
    scene = UsdPhysics.Scene.Define(stage, "/NativeTest/PhysicsScene")
    scene.CreateGravityDirectionAttr(Gf.Vec3f(0, 0, -1))
    scene.CreateGravityMagnitudeAttr(9.81)
    native = PhysxSchema.PhysxSceneAPI.Apply(scene.GetPrim())
    native.CreateEnableGPUDynamicsAttr(False)
    native.CreateEnableCCDAttr(profile["ccd"])
    native.CreateSolverTypeAttr(profile["solver"])
    native.CreateTimeStepsPerSecondAttr(round(1 / profile["dt_s"]))
    floor = UsdGeom.Cube.Define(stage, "/NativeTest/Floor")
    floor.CreateSizeAttr(1.0)
    floor.AddTranslateOp().Set(Gf.Vec3d(0, 0, -0.05))
    floor.AddScaleOp().Set(Gf.Vec3f(4, 4, 0.1))
    UsdPhysics.CollisionAPI.Apply(floor.GetPrim())
    for body in bodies:
        rigid = PhysxSchema.PhysxRigidBodyAPI.Apply(body)
        rigid.CreateSolverPositionIterationCountAttr(profile["position_iterations"])
        rigid.CreateSolverVelocityIterationCountAttr(profile["velocity_iterations"])
        rigid.CreateEnableCCDAttr(profile["ccd"])
        PhysxSchema.PhysxContactReportAPI.Apply(body).CreateThresholdAttr(0.0)
        if body.HasAPI(UsdPhysics.ArticulationRootAPI):
            art = PhysxSchema.PhysxArticulationAPI.Apply(body)
            art.CreateEnabledSelfCollisionsAttr(True)
            art.CreateSolverPositionIterationCountAttr(profile["position_iterations"])
            art.CreateSolverVelocityIterationCountAttr(profile["velocity_iterations"])
    shapes = [p for p in stage.Traverse() if p.HasAPI(UsdPhysics.CollisionAPI)]
    for prim in shapes:
        shape = PhysxSchema.PhysxCollisionAPI.Apply(prim)
        shape.CreateContactOffsetAttr(profile["contact_offset_m"])
        shape.CreateRestOffsetAttr(profile["rest_offset_m"])
    for joint in joints:
        joint.CreateCollisionEnabledAttr(True)
    if scenario == "drop":
        for body in bodies:
            pose = matrix(body)
            pose[2, 3] += profile["initial_conditions"]["drop_height_m"]
            set_world_matrix(body, pose)
    else:
        children = {j.GetBody1Rel().GetTargets()[0] for j in joints}
        root = next(p for p in bodies if p.GetPath() not in children)
        fixed = UsdPhysics.FixedJoint.Define(stage, "/NativeTest/BaseFixture")
        fixed.CreateBody1Rel().SetTargets([root.GetPath()])
        pose = Gf.Matrix4d(matrix(root).T.tolist())
        fixed.CreateLocalPos0Attr(Gf.Vec3f(pose.ExtractTranslation()))
        fixed.CreateLocalRot0Attr(Gf.Quatf(pose.ExtractRotationQuat()))
    for joint in joints:
        lo = joint.GetLowerLimitAttr().Get()
        hi = joint.GetUpperLimitAttr().Get()
        if scenario == "half_open_release":
            fraction = profile["initial_conditions"]["half_open_fraction"]
            initialize_joint(stage, joint, lo + (hi - lo) * fraction)
        if scenario in {"drive_cycle", "limit_load"}:
            angular = joint.GetPrim().IsA(UsdPhysics.RevoluteJoint)
            drive = UsdPhysics.DriveAPI.Apply(joint.GetPrim(), "angular" if angular else "linear")
            name = "revolute" if angular else "prismatic"
            drive.CreateTypeAttr("force")
            drive.CreateStiffnessAttr(profile[f"drive_{name}_stiffness"])
            drive.CreateDampingAttr(profile[f"drive_{name}_damping"])
            drive.CreateMaxForceAttr(profile[f"drive_{name}_max_force"])
            drive.CreateTargetPositionAttr(0.0)
    return bodies, joints, shapes


def test_asset(asset: Path, out: Path, profile: dict, scenario: str) -> dict:
    stage = Usd.Stage.Open(str(asset))
    stage.SetEditTarget(stage.GetSessionLayer())
    bodies, joints, shapes = configure(stage, profile, scenario)
    stage.Flatten().Export(str(out / "initial.usda"))
    points = [
        (p, np.asarray(UsdGeom.Mesh(p).GetPointsAttr().Get(), dtype=float))
        for p in stage.Traverse()
        if p.IsA(UsdGeom.Mesh)
    ]
    physics = omni.physx.get_physx_interface()
    sim = omni.physx.get_physx_simulation_interface()
    cache = UsdUtils.StageCache.Get()
    cache_id = cache.Insert(stage)
    stage_id = cache_id.ToLongInt()
    sim.attach_stage(stage_id)
    trajectories, q_history, minimum_z, contact_pairs = [], [], [], set()
    max_anchor, contact_count = 0.0, 0
    min_contact_separation = 0.0
    tracking_errors = []
    dt, duration = profile["dt_s"], profile["duration_s"]
    steps = round(duration / dt)
    final_targets = []
    previous_q = np.asarray([joint_state(stage, j)[0] for j in joints])
    conditions = profile["initial_conditions"]
    for i in range(steps):
        time = i * dt
        targets = []
        for joint in joints:
            angular = joint.GetPrim().IsA(UsdPhysics.RevoluteJoint)
            lo, hi = joint.GetLowerLimitAttr().Get(), joint.GetUpperLimitAttr().Get()
            endpoint = max([lo, hi], key=abs)
            target = 0.0
            if scenario == "drive_cycle":
                target = endpoint * (
                    conditions["drive_fraction"] if time < conditions["drive_switch_s"] else 0.0
                )
            elif scenario == "limit_load":
                target = endpoint * conditions["limit_drive_fraction"]
            if scenario in {"drive_cycle", "limit_load"}:
                drive = UsdPhysics.DriveAPI(joint.GetPrim(), "angular" if angular else "linear")
                drive.GetTargetPositionAttr().Set(target)
            if (
                scenario == "limit_load"
                and conditions["perturbation_s"][0] <= time < conditions["perturbation_s"][1]
            ):
                child = stage.GetPrimAtPath(joint.GetBody1Rel().GetTargets()[0])
                pos = matrix(child)[:3, 3]
                mass = UsdPhysics.MassAPI(child).GetMassAttr().Get()
                sim.apply_force_at_pos(
                    stage_id,
                    PhysicsSchemaTools.sdfPathToInt(child.GetPath()),
                    carb.Float3(conditions["perturbation_force_fraction"] * mass * 9.81, 0, 0),
                    carb.Float3(*pos),
                    "Force",
                )
            targets.append(target)
        sim.simulate(dt, time)
        sim.fetch_results()
        physics.update_transformations(False, True, True, False)
        if i == 0:
            assert all(
                physics.get_rigidbody_transformation(str(p.GetPath()))["ret_val"] for p in bodies
            ), "PhysX did not create all rigid bodies"
        states = [joint_state(stage, j) for j in joints]
        qs = [q for q, _ in states]
        assert all(np.isfinite(matrix(p)).all() for p in bodies), "Nonfinite native body pose"
        qdot = (np.asarray(qs) - previous_q) / dt
        previous_q = np.asarray(qs)
        q_history.append(qs)
        max_anchor = max(max_anchor, *(error for _, error in states))
        min_z = min(
            float((p @ matrix(prim)[:3, :3].T + matrix(prim)[:3, 3])[:, 2].min())
            for prim, p in points
        )
        minimum_z.append(min_z)
        headers, contacts = sim.get_contact_report()
        contact_count += len(contacts)
        for header in headers:
            contact_pairs.add(
                tuple(
                    sorted(
                        [
                            str(PhysicsSchemaTools.intToSdfPath(header.collider0)),
                            str(PhysicsSchemaTools.intToSdfPath(header.collider1)),
                        ]
                    )
                )
            )
        for contact in contacts:
            min_contact_separation = min(min_contact_separation, float(contact.separation))
        switch, window = conditions["drive_switch_s"], conditions["settling_window_s"]
        if scenario == "drive_cycle" and (
            switch - window <= time < switch or time >= duration - window
        ):
            tracking_errors.append([abs(q - target) for q, target in zip(qs, targets, strict=True)])
        if i % 32 == 0 or i == steps - 1:
            trajectories.append(
                {
                    "time_s": time + dt,
                    "q": qs,
                    "target": targets,
                    "qdot": qdot.tolist(),
                    "minimum_surface_z_m": min_z,
                    "body_world_matrices": {str(p.GetPath()): matrix(p).tolist() for p in bodies},
                }
            )
        if i in {steps // 4, steps // 2, steps - 1}:
            stage.Flatten().Export(str(out / f"frame-{i:04d}.usda"))
        final_targets = targets
    q_array = np.asarray(q_history)
    limits, excursions, errors = [], [], []
    for n, joint in enumerate(joints):
        angular = joint.GetPrim().IsA(UsdPhysics.RevoluteJoint)
        lo, hi = joint.GetLowerLimitAttr().Get(), joint.GetUpperLimitAttr().Get()
        tol = profile["revolute_limit_tolerance_deg" if angular else "prismatic_limit_tolerance_m"]
        tracking_tol = profile[
            "tracking_revolute_tolerance_deg" if angular else "tracking_prismatic_tolerance_m"
        ]
        violation = max(0.0, lo - q_array[:, n].min(), q_array[:, n].max() - hi)
        excursion = float(np.ptp(q_array[:, n]))
        tracking = max((row[n] for row in tracking_errors), default=0.0)
        limits.append(
            {
                "joint": str(joint.GetPath()),
                "units": "deg" if angular else "m",
                "limits": [lo, hi],
                "minimum": float(q_array[:, n].min()),
                "maximum": float(q_array[:, n].max()),
                "violation": float(violation),
                "excursion": excursion,
                "max_settled_tracking_error": tracking,
            }
        )
        excursions.append(excursion)
        if violation > tol:
            errors.append(f"joint_limit:{joint.GetPath()}")
        if scenario == "drive_cycle" and (excursion < 0.4 * (hi - lo) or tracking > tracking_tol):
            errors.append(f"drive_tracking:{joint.GetPath()}")
        if scenario == "limit_load" and abs(q_array[-1, n] - max([lo, hi], key=abs)) > tracking_tol:
            errors.append(f"limit_not_reached:{joint.GetPath()}")
    peak = max(0.0, -min(minimum_z))
    resting = max(0.0, -min(minimum_z[-round(conditions["resting_window_s"] / dt) :]))
    if peak > profile["peak_floor_penetration_m"]:
        errors.append("peak_floor_penetration")
    if resting > profile["resting_floor_penetration_m"]:
        errors.append("resting_floor_penetration")
    if not np.isfinite(q_array).all() or not np.isfinite(minimum_z).all():
        errors.append("nonfinite_state")
    if max_anchor > conditions["max_anchor_error_m"]:
        errors.append("joint_anchor_separation")
    if scenario == "drop" and (not contact_count or np.ptp(minimum_z) < 0.05):
        errors.append("missing_freefall_or_contact_evidence")
    adjacent_collisions = not any(
        p.HasAPI(UsdPhysics.ArticulationRootAPI) for p in stage.Traverse()
    ) and all(j.GetCollisionEnabledAttr().Get() for j in joints)
    result = {
        "asset": str(asset),
        "asset_sha256": hashlib.sha256(asset.read_bytes()).hexdigest(),
        "engine": "omni.physx native CPU solver",
        "scenario": scenario,
        "profile": profile,
        "passed": not errors,
        "errors": errors,
        "adjacent_collision_policy": "standard-joints-enabled"
        if adjacent_collisions
        else "articulation-parent-child-collision-ignored",
        "full_native_gate_passed": not errors and adjacent_collisions,
        "runtime": {
            "usd": list(Usd.GetVersion()),
            "physx_extension": omni.kit.app.get_app()
            .get_extension_manager()
            .get_enabled_extension_id("omni.physx"),
        },
        "steps": steps,
        "peak_floor_penetration_m": peak,
        "resting_floor_penetration_m": resting,
        "max_joint_anchor_error_m": max_anchor,
        "joints": limits,
        "contact_point_samples": contact_count,
        "contact_pairs": sorted(contact_pairs),
        "minimum_contact_separation_m": min_contact_separation,
        "collision_shapes": [
            {
                "path": str(p.GetPath()),
                "enabled": bool(UsdPhysics.CollisionAPI(p).GetCollisionEnabledAttr().Get()),
                "approximation": UsdPhysics.MeshCollisionAPI(p).GetApproximationAttr().Get()
                if p.IsA(UsdGeom.Mesh)
                else "primitive",
            }
            for p in shapes
        ],
        "trajectory": trajectories,
        "joint_coordinates_source": "solver USD writeback, relative joint local frames; no clipping",
        "final_targets": final_targets,
    }
    sim.detach_stage()
    cache.Erase(cache_id)
    return result


def main() -> None:
    job_file = Path(carb.settings.get_settings().get("/usdcraft/job"))
    job = json.loads(job_file.read_text())
    if "expected_runner_sha256" in job:
        assert (
            hashlib.sha256(Path(__file__).read_bytes()).hexdigest() == job["expected_runner_sha256"]
        ), "Runner hash differs from job"
    if "expected_profile_sha256" in job:
        assert (
            hashlib.sha256(Path(job["profile"]).read_bytes()).hexdigest()
            == job["expected_profile_sha256"]
        ), "Profile hash differs from job"
    profile = json.loads(Path(job["profile"]).read_text())
    results = []
    profile_hash = hashlib.sha256(Path(job["profile"]).read_bytes()).hexdigest()
    runner_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    for asset in job["assets"]:
        for scenario in profile["scenarios"]:
            for repeat in range(profile["repeats"]):
                slot = Path(job["output"]) / asset["case"] / scenario / f"repeat-{repeat + 1}"
                slot.mkdir(parents=True, exist_ok=True)
                previous = sorted(slot.glob("execution-*/result.json"))
                completed = None
                for path in previous:
                    try:
                        record = json.loads(path.read_text())
                    except json.JSONDecodeError:
                        continue  # Preserve a partially written record; use a new execution folder.
                    if "native_execution_error" not in record["errors"]:
                        assert record["asset_sha256"] == asset["expected_sha256"]
                        assert (
                            record["profile_sha256"] == profile_hash
                            and record["runner_sha256"] == runner_hash
                        )
                        completed = {
                            "case": asset["case"],
                            "scenario": scenario,
                            "repeat": repeat + 1,
                            "passed": record["passed"],
                            "errors": record["errors"],
                            "result": path.relative_to(Path(job["output"])).as_posix(),
                            "reused": True,
                        }
                if completed:
                    results.append(completed)
                    (Path(job["output"]) / "index.json").write_text(json.dumps(results, indent=2))
                    continue
                number = len(list(slot.glob("execution-*"))) + 1
                out = slot / f"execution-{number:02d}"
                out.mkdir()
                try:
                    if "expected_sha256" in asset:
                        assert (
                            hashlib.sha256(Path(asset["asset"]).read_bytes()).hexdigest()
                            == asset["expected_sha256"]
                        ), "Asset hash differs from job"
                    result = test_asset(Path(asset["asset"]), out, profile, scenario)
                    result["profile_sha256"] = hashlib.sha256(
                        Path(job["profile"]).read_bytes()
                    ).hexdigest()
                    result["runner_sha256"] = runner_hash
                except Exception:
                    result = {
                        "passed": False,
                        "errors": ["native_execution_error"],
                        "exception": traceback.format_exc(),
                    }
                (out / "result.json").write_text(json.dumps(result, indent=2, allow_nan=False))
                row = {
                    "case": asset["case"],
                    "scenario": scenario,
                    "repeat": repeat + 1,
                    "passed": result["passed"],
                    "errors": result["errors"],
                    "result": (out / "result.json").relative_to(Path(job["output"])).as_posix(),
                }
                results.append(row)
                print(json.dumps(row), flush=True)
                (Path(job["output"]) / "index.json").write_text(json.dumps(results, indent=2))
                if "native_execution_error" in result["errors"]:
                    omni.kit.app.get_app().post_quit(1)
                    return
    omni.kit.app.get_app().post_quit(0)


main()
