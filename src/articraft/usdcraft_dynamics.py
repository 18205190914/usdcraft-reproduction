"""Full-timestep MuJoCo diagnostics; explicit stricter solver profile, no q clipping.

This profile changes the evaluator, not the exported USD or PhysX behavior.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import mujoco
import numpy as np

from articraft.simulate import _build_spec
from articraft.usdcraft_benchmark import save_json


def inspect_dynamics(
    asset: Path,
    scenario: str,
    strict_solver: bool,
    seconds: float = 3.0,
    *,
    dt: float = 0.0005,
    constraint_time_constant: float = 0.002,
) -> dict:
    model = _build_spec(asset).compile()
    if strict_solver:
        model.opt.timestep = dt
        model.opt.iterations = 100
        model.jnt_solref[:] = [constraint_time_constant, 1.0]
        model.jnt_solimp[:] = [0.99, 0.999, 0.0001, 0.5, 2.0]
        model.geom_solref[:] = [constraint_time_constant, 1.0]
        model.geom_solimp[:] = [0.99, 0.999, 0.0001, 0.5, 2.0]
    data = mujoco.MjData(model)
    movable = [i for i in range(model.njnt) if model.jnt_limited[i]]
    if scenario == "release":
        for i in movable:
            data.qpos[model.jnt_qposadr[i]] = model.jnt_range[i].mean()
    mujoco.mj_forward(model, data)
    floor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    violations = np.zeros(model.njnt)
    peak_floor, rest_floor = 0.0, 0.0
    finite = True
    minimum_com_height = float(data.xipos[1:, 2].min())
    steps = round(seconds / model.opt.timestep)
    for step in range(steps):
        mujoco.mj_step(model, data)
        mujoco.mj_forward(model, data)
        for i in movable:
            q = data.qpos[model.jnt_qposadr[i]]
            lower, upper = model.jnt_range[i]
            violations[i] = max(violations[i], lower - q, q - upper)
        depth = max(
            (max(0.0, -c.dist) for c in data.contact if floor_id in (c.geom1, c.geom2)), default=0.0
        )
        peak_floor = max(peak_floor, depth)
        if step >= 0.6 * steps:
            rest_floor = max(rest_floor, depth)
        minimum_com_height = min(minimum_com_height, float(data.xipos[1:, 2].min()))
        if not np.isfinite(data.qpos).all():
            finite = False
            break
    joints = []
    for i in movable:
        hinge = model.jnt_type[i] == mujoco.mjtJoint.mjJNT_HINGE
        value = float(np.degrees(violations[i]) if hinge else violations[i] * 1000)
        joints.append(
            {
                "joint": mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i),
                "maximum_limit_violation": value,
                "unit": "degrees" if hinge else "mm",
                "passed": value <= 0.5,
            }
        )
    return {
        "asset": str(asset),
        "scenario": scenario,
        "profile": "strict-mujoco" if strict_solver else "original-mujoco",
        "timestep": model.opt.timestep,
        "joint_time_constants": model.jnt_solref[movable, 0].tolist(),
        "contact_time_constants": np.unique(model.geom_solref[:, 0]).tolist(),
        "steps": step + 1,
        "peak_floor_penetration_mm": peak_floor * 1000,
        "resting_floor_penetration_mm": rest_floor * 1000,
        "joints": joints,
        "finite": finite,
        "minimum_body_com_height": minimum_com_height,
        "fallen_through_floor": minimum_com_height < -0.01,
        "strict_passed": finite
        and minimum_com_height >= -0.01
        and peak_floor <= 0.001
        and rest_floor <= 0.0005
        and all(j["passed"] for j in joints),
        "thresholds": "local engineering gate: peak floor <=1mm, rest <=0.5mm, limits <=0.5deg or0.5mm, COM not below floor by 10mm; not paper thresholds",
        "q_positions_clipped": False,
        "exported_usd_modified": False,
        "engine_scope": "MuJoCo solver-profile test only, does not establish native PhysX fidelity",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets", nargs="+", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--refine",
        action="store_true",
        help="Also test half timestep, with fixed and halved constraint time constants",
    )
    args = parser.parse_args()
    results = [
        inspect_dynamics(asset, scenario, strict)
        for asset in args.assets
        for scenario in ("drop", "release")
        for strict in (False, True)
    ]
    if args.refine:
        results.extend(
            inspect_dynamics(asset, scenario, True, dt=0.00025, constraint_time_constant=tau)
            for asset in args.assets
            for scenario in ("drop", "release")
            for tau in (0.002, 0.001)
        )
    save_json(args.out, results)
    for result in results:
        print(
            {
                k: v
                for k, v in result.items()
                if k
                in (
                    "asset",
                    "scenario",
                    "profile",
                    "peak_floor_penetration_mm",
                    "joints",
                    "strict_passed",
                )
            },
            flush=True,
        )


if __name__ == "__main__":
    main()
