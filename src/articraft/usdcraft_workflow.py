"""Source-grounded, whole-program authoring with bounded free-model calls.

Modeling reads public inputs only. Ground-truth comparison is a separate command.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import time
from datetime import datetime
from pathlib import Path

import httpx
import numpy as np
import trimesh
from scipy.spatial import cKDTree

from articraft import package_dir
from articraft.agent.provider.openrouter import OpenRouterModel
from articraft.agent.workspace import LocalWorkspace
from articraft.box_workflow import free_model_preflight
from articraft.errors import ModelError
from articraft.geometry_recheck import (
    compare_part,
    diagnose_geometry,
    feedback_summary,
    recheck_geometry,
)
from articraft.source_geometry import (
    analyze_source,
    clip_surface,
    measure_face,
    measure_mesh,
    measure_plane,
    surface_support,
)
from articraft.usdcraft_benchmark import render_meshes, save_json
from articraft.usdcraft_io import read_usd_asset
from articraft.usdcraft_motion import check_motion, motion_summary
from articraft.usdcraft_physx import native_identity, validate_native


def tool(name: str, description: str, properties: dict, required: list[str]) -> dict:
    return {
        "type": "function",
        "name": name,
        "description": description,
        "parameters": {
            "type": "object",
            "properties": properties,
            "required": required,
            "additionalProperties": False,
        },
    }


BOUNDS = {
    "type": "array",
    "items": {"type": "array", "items": {"type": "number"}, "minItems": 3, "maxItems": 3},
    "minItems": 2,
    "maxItems": 2,
    "description": "Metric [[xmin,ymin,zmin],[xmax,ymax,zmax]]; omit for whole source",
}
TOOLS = [
    tool(
        "measure_mesh",
        "Measure observed source surfaces, clipped to optional region; absent surface is unknown.",
        {"bounds": BOUNDS},
        [],
    ),
    tool(
        "measure_face",
        "Measure observed source faces with axis-aligned normals in a region.",
        {
            "axis": {"type": "integer", "enum": [0, 1, 2]},
            "side": {"type": "integer", "enum": [-1, 1]},
            "bounds": BOUNDS,
        },
        ["axis", "side"],
    ),
    tool(
        "measure_plane",
        "Measure observed faces around any world-space normal, without rotating or aligning the object.",
        {
            "normal": {"type": "array", "items": {"type": "number"}, "minItems": 3, "maxItems": 3},
            "bounds": BOUNDS,
        },
        ["normal"],
    ),
    tool(
        "check_axial",
        "Get shared-source-grid coverage, missing/unknown regions and six depth maps for last compiled candidate.",
        {},
        [],
    ),
    tool(
        "compare_part",
        "Compare a candidate part to a source-region patch. Alignment is diagnostic only.",
        {"part": {"type": "string"}, "bounds": BOUNDS},
        ["part", "bounds"],
    ),
    tool(
        "render_candidate",
        "Save a CPU inspection image. This text-only adapter does not send pixels to the model.",
        {},
        [],
    ),
    tool(
        "submit_program",
        "Compile complete Python, export USDZ, independently recheck observed geometry. Repair and resubmit.",
        {
            "source": {
                "type": "string",
                "description": "Complete main.py with object_model and run_tests(); no Markdown fences",
            }
        },
        ["source"],
    ),
]


class FreeProgramModel(OpenRouterModel):
    async def _send_with_retries(self, request: dict) -> httpx.Response:
        return await super()._send_with_retries(
            request
            | {
                "max_tokens": 4096,
                "parallel_tool_calls": False,
                "reasoning": {"enabled": False},
                "tool_choice": (
                    {"type": "function", "function": {"name": "submit_program"}}
                    if len(request.get("tools", [])) == 1
                    else "auto"
                ),
            }
        )


class ModelingTools:
    def __init__(
        self,
        public: Path,
        workspace: LocalWorkspace,
        run: Path,
        native_config: Path | None = None,
        geometry_profile: Path | None = None,
    ):
        self.input = json.loads((public / "input.json").read_text())
        self.source = trimesh.load_mesh(public / self.input["source"], process=False)
        self.grid, self.cells, self.analysis = analyze_source(self.source)
        self.workspace, self.run = workspace, run
        self.parts: dict[str, trimesh.Trimesh] = {}
        self.attempt = 0
        self.recheck: dict = {}
        self.native_config = native_config
        self.validation_identity = {
            "workflow": "usdcraft-round5-v1",
            "native_config": native_identity(native_config) if native_config else None,
            "geometry_profile": hashlib.sha256(geometry_profile.read_bytes()).hexdigest()
            if geometry_profile
            else None,
        }
        self.geometry_profile = (
            json.loads(geometry_profile.read_text()) if geometry_profile else None
        )
        save_json(run / "source-analysis.json", self.analysis)

    def validation_matches(self, feedback: dict) -> bool:
        identity = feedback.get("validation_identity")
        if identity is None:
            return self.native_config is None and self.geometry_profile is None
        return identity == self.validation_identity

    def execute(self, name: str, arguments: dict) -> dict:
        bounds = np.asarray(arguments["bounds"]) if "bounds" in arguments else None
        if name == "measure_mesh":
            return measure_mesh(self.source, bounds)
        if name == "measure_face":
            return measure_face(self.source, arguments["axis"], arguments["side"], bounds)
        if name == "measure_plane":
            return measure_plane(self.source, np.asarray(arguments["normal"]), bounds)
        if name == "submit_program":
            return self.submit(arguments["source"])
        if not self.parts:
            raise ValueError("Compile a candidate successfully before querying candidate geometry")
        if name == "check_axial":
            return feedback_summary(self.recheck)
        if name == "compare_part":
            patch = clip_surface(self.source, bounds)
            if not len(patch.faces):
                return {"observed": False, "interpretation": "unknown source region"}
            return compare_part(patch, self.parts[arguments["part"]])
        if name == "render_candidate":
            path = self.run / "result" / f"attempt-{self.attempt:02d}" / "candidate.png"
            render_meshes(list(self.parts.values()), path)
            return {"path": str(path), "pixels_sent_to_model": False}
        raise ValueError(f"Unknown tool: {name}")

    def submit(self, source: str) -> dict:
        self.attempt += 1
        out = self.run / "result" / f"attempt-{self.attempt:02d}"
        out.mkdir()
        (out / "main.py").write_text(source, encoding="utf-8")
        (self.run / "workspace/main.py").write_text(source, encoding="utf-8")
        compiled = self.workspace.compile_path(self.run)
        save_json(out / "compile.json", compiled)
        return self.evaluate(compiled, out)

    def evaluate(self, compiled: dict, out: Path) -> dict:
        """Resume checks on an immutable compiled candidate without recompiling it."""
        diagnostics = (compiled.get("test_report") or {}).get("diagnostics", [])
        feedback = {
            "attempt": self.attempt,
            "compiler_status": compiled["status"],
            "geometry_gate_passed": False,
            "error": compiled.get("error"),
            "traceback": compiled.get("traceback", "")[-5000:],
            "test_report": compiled.get("test_report"),
            "compiler_diagnostics": diagnostics,
            "compiler_gate_passed": compiled["status"] == "success" and not diagnostics,
            "native_physics_required": self.native_config is not None,
            "local_geometry_required": self.geometry_profile is not None,
            "validation_identity": self.validation_identity,
            "status": "compiled",
        }
        save_json(out / "feedback-compiled.json", feedback)
        if compiled["status"] == "success":
            asset = Path(compiled["usdz"])
            asset = asset if asset.is_absolute() else self.run / asset
            assembly, self.parts = read_usd_asset(asset, out / "parts")
            candidate = trimesh.util.concatenate(list(self.parts.values()))
            self.recheck = recheck_geometry(self.source, candidate, self.grid, self.cells)
            save_json(out / "recheck.json", self.recheck)
            candidate_cells = surface_support(candidate, self.grid)
            distances = cKDTree(candidate_cells).query(self.cells)[0]
            tolerant = float((distances <= np.sqrt(3) + 1e-9).mean())
            extent_error = np.abs(candidate.extents - self.source.extents) / self.source.extents
            # Local development gate, set before experiments, not paper acceptance.
            passed = tolerant >= 0.95 and max(extent_error) <= 0.02 and bool(assembly["joints"])
            feedback.update(
                asset=str(asset),
                source_bounds=self.source.bounds.tolist(),
                candidate_bounds=candidate.bounds.tolist(),
                part_bounds={name: part.bounds.tolist() for name, part in self.parts.items()},
                geometry=feedback_summary(self.recheck),
                source_surface_within_one_cell=tolerant,
                relative_extent_errors=extent_error.tolist(),
                movable_joints=len(assembly["joints"]),
                geometry_gate_passed=bool(passed),
                gate_scope="geometry only: 95% support within one cell in each axis, 2% extents, a movable joint; not semantics or physics acceptance",
            )
            motion = check_motion(assembly, self.parts)
            save_json(out / "motion.json", motion)
            feedback["motion"] = motion_summary(motion)
            feedback["geometry_motion_gate_passed"] = bool(
                passed and motion["passed"] and not diagnostics
            )
            local = diagnose_geometry(
                self.source, self.parts, self.grid, **(self.geometry_profile or {})
            )
            save_json(out / "local-geometry.json", local)
            feedback["local_geometry"] = local
            feedback["local_geometry_gate_passed"] = local["passed"]
            feedback["combined_gate_passed"] = feedback["geometry_motion_gate_passed"] and (
                self.geometry_profile is None or local["passed"]
            )
            feedback["status"] = "geometry_checked"
            save_json(out / "feedback.json", feedback)
            save_json(out / "feedback-geometry.json", feedback)
            if self.native_config and feedback["combined_gate_passed"]:
                native = validate_native(
                    asset, self.native_config, out / "native", self.input.get("case", self.run.name)
                )
                feedback["native_physics"] = native
                feedback["combined_gate_passed"] = bool(native["passed"])
                feedback["full_gate_passed"] = bool(native["passed"])
            elif self.native_config:
                feedback["native_physics"] = {
                    "status": "not_run",
                    "reason": "compiler/geometry/motion gate failed",
                }
                feedback["full_gate_passed"] = False
            feedback["acceptance_scope"] = (
                "geometry_motion"
                + ("_local" if self.geometry_profile else "")
                + ("_native" if self.native_config else "; native not requested")
            )
            render_meshes(list(self.parts.values()), out / "candidate.png")
        feedback["status"] = "completed"
        save_json(out / "feedback.json", feedback)
        return feedback


async def run_case(
    model: OpenRouterModel,
    workspace: LocalWorkspace,
    public: Path,
    max_calls: int,
    native_config: Path | None = None,
    geometry_profile: Path | None = None,
) -> dict:
    run = workspace.create_run(f"source-{public.name}-{datetime.now():%Y%m%d-%H%M%S}")
    host = ModelingTools(public, workspace, run, native_config, geometry_profile)
    docs = "\n".join(
        (package_dir / "sdk/docs/common" / name).read_text()
        for name in ["00_quickstart.md", "35_joints.md", "37_materials.md"]
    )
    messages = [
        {
            "role": "system",
            "content": "Rebuild an articulated asset by authoring a COMPLETE Articraft/build123d Python program. "
            "Use tools to observe source geometry and correct your program after compilation and geometry feedback. "
            "No source code template or part/joint truth is supplied. Infer movable parts and joints. "
            "All measurements use supplied meters and Z up. Source cells mark SURFACE, other cells are UNKNOWN. "
            "Do not fill an observed cavity. Added geometry in unknown space may be justified completion. "
            "Use the build123d primitives, profiles, transforms, booleans, sweeps or lofts required by observations. "
            "Import only symbols you use, subtract observed cavities and keep handles within the moving body. "
            "Use local surface/normal feedback to repair the complete program. Subtraction uses the '-' operator. "
            "You may author all body geometry in the common source frame, then use identical global anchors "
            "at both endpoints so zero state matches the source. Orient using Axis(point,direction) with ROT_Z/TRANS_Z. "
            "Set material=Material.ABS_PLASTIC. Define object_model and run_tests() returning TestContext(object_model).report(). "
            "Do not use files, imports outside SDK/build123d/math/numpy, network, subprocesses, or introspection. "
            "You cannot access ground truth. You must call submit_program; plain explanations do not build an asset. "
            "The geometry gate checks only observations, not physics or semantics.\nSDK DOCUMENTATION:\n"
            + docs,
        },
        {
            "role": "user",
            "content": host.input["request"] + "\nOBSERVATIONS:\n" + json.dumps(host.analysis),
        },
    ]
    events, feedback = [], {}
    started = time.monotonic()
    error, cost, calls = "", 0.0, 0
    try:
        for calls in range(1, max_calls + 1):
            print(json.dumps({"case": public.name, "call": calls, "phase": "query"}), flush=True)
            # Reserve the remaining budget for construction and repair.
            # Measurement results already persist in the conversation.
            if calls <= 2:
                available_tools = TOOLS[:3]
            elif calls % 2 or calls == max_calls:
                available_tools = [TOOLS[-1]]
            else:
                available_tools = TOOLS
            if calls == 3:
                messages.append(
                    {
                        "role": "user",
                        "content": "The observation stage is complete. Now submit the complete Python program. "
                        "Use remaining calls to repair compiler and geometry failures.",
                    }
                )
            response = await model.query(messages, tools=available_tools)
            cost += float(response.get("cost") or 0)
            assistant = {
                "role": "assistant",
                "content": response.get("text", ""),
                "tool_calls": response.get("tool_calls", []),
            }
            if assistant["content"] or assistant["tool_calls"]:
                messages.append(
                    assistant | {"provider_content": response.get("provider_content", [])}
                )
            events.append(assistant)
            events.append(
                {
                    "type": "usage",
                    "tokens": response.get("token_usage"),
                    "finish_reason": response["response"]["choices"][0].get("finish_reason"),
                    "cost_usd": response.get("cost"),
                }
            )
            submitted = False
            for call in assistant["tool_calls"]:
                try:
                    if call["name"] == "submit_program" and submitted:
                        tool_feedback = {
                            "error": "Submit one complete program per response; other submissions were not executed"
                        }
                    else:
                        feedback = await asyncio.to_thread(
                            host.execute, call["name"], json.loads(call["arguments"])
                        )
                        tool_feedback = feedback
                        submitted |= call["name"] == "submit_program"
                except (ValueError, KeyError, TypeError) as exc:
                    feedback = {
                        "error": f"{type(exc).__name__}: {exc}",
                        "geometry_gate_passed": False,
                    }
                    tool_feedback = feedback
                output = {
                    "type": "function_call_output",
                    "call_id": call["id"],
                    "output": json.dumps(tool_feedback),
                }
                messages.append(output)
                events.append(output)
                print(
                    json.dumps(
                        {
                            "case": public.name,
                            "call": calls,
                            "tool": call["name"],
                            "compiler": feedback.get("compiler_status"),
                            "geometry_gate_passed": feedback.get("geometry_gate_passed"),
                        }
                    ),
                    flush=True,
                )
            save_json(run / "tool-events.json", events)
            if feedback.get("combined_gate_passed"):
                break
            if feedback.get("native_physics", {}).get("status") == "execution_error":
                error = "Native execution failed; repair infrastructure before changing CAD"
                break
            if not assistant["tool_calls"]:
                messages.append(
                    {"role": "user", "content": "Call submit_program with the complete Python now."}
                )
    except (ModelError, httpx.HTTPError) as exc:
        error = f"{type(exc).__name__}: {exc}"
    summary = {
        "case": public.name,
        "run": str(run),
        "model": model.config.openrouter_model,
        "model_calls": calls,
        "submissions": host.attempt,
        "cost_usd": cost,
        "elapsed_seconds": time.monotonic() - started,
        "error": error,
        "last_feedback": feedback,
        "private_truth_sent_to_model": False,
        "source_code_template_supplied": False,
        "pixels_sent_to_model": False,
    }
    save_json(run / "summary.json", summary)
    return summary


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--cases", nargs="+", default=["hinged_box", "drawer", "cabinet"])
    parser.add_argument("--max-calls", type=int, choices=range(1, 9), default=8)
    parser.add_argument("--native-config", type=Path)
    parser.add_argument("--geometry-profile", type=Path)
    parser.add_argument(
        "--model", help="Explicit zero-price :free model; default is existing settings"
    )
    args = parser.parse_args()
    from articraft.settings import Settings

    settings = Settings().model_copy(
        update={"openrouter_max_attempts": 1, "openrouter_request_timeout_seconds": 240}
    )
    if args.model:
        settings = settings.model_copy(update={"openrouter_model": args.model})
    preflight = await free_model_preflight(settings, args.max_calls * len(args.cases))
    model = FreeProgramModel(settings)
    workspace = LocalWorkspace(
        output_dir=(args.root / "runs").resolve(), physics_enabled=True, timeout_seconds=180
    )
    summaries = []
    try:
        for case in args.cases:
            result = await run_case(
                model,
                workspace,
                args.root / "inputs" / case,
                args.max_calls,
                args.native_config,
                args.geometry_profile,
            )
            summaries.append(result)
            save_json(
                args.root / "generation-summary.json",
                {"preflight": preflight, "results": summaries},
            )
            print(json.dumps({k: v for k, v in result.items() if k != "last_feedback"}), flush=True)
            if result["error"]:
                break
    finally:
        await model.close()


if __name__ == "__main__":
    asyncio.run(main())
