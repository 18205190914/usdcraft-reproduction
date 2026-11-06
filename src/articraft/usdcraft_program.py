"""Whole-program chat protocol using the existing compiler and geometric feedback.

No function-call arguments, category CAD templates, or silent provider fallback.
"""

from __future__ import annotations

import argparse
import ast
import asyncio
import hashlib
import json
import subprocess
from datetime import datetime
from pathlib import Path

import httpx

from articraft.agent.provider.openrouter import OpenRouterModel
from articraft.agent.workspace import LocalWorkspace
from articraft.box_workflow import free_model_preflight
from articraft.errors import ModelError
from articraft.settings import Settings
from articraft.usdcraft_benchmark import save_json
from articraft.usdcraft_evaluation import load_parts
from articraft.usdcraft_workflow import ModelingTools

API_CONTRACT = """Author ONE complete Python file that reconstructs the observed articulated object.
Return only Python, optionally inside one ```python fence. No prose or tool-call JSON.
Infer parts and their geometry from the measured source. Do not import the source mesh.
Use meters, Z up, build123d algebraic shapes; Box(w,d,h) is centered at its local origin.
Move a shape with Pos(x,y,z) * Box(...); combine shapes with +; cut cavities with -.
Use build123d primitives, profiles, transforms, booleans, sweeps or lofts required by the observations.
Import only symbols you use. Curved and inclined surfaces must preserve the measured source shape.
Import RigidBodyAssembly, JointAxis, JointDOF, Material, TestContext from articraft.sdk.
Material belongs to articraft.sdk. Use material=Material.ABS_PLASTIC.
API signatures (descriptions, NOT a program to copy):
- RigidBodyAssembly(string) constructs an assembly; its rigid_body(string) returns a body.
- A body has add(shape, name=string, material=Material.ABS_PLASTIC).
- Axis takes TWO positional tuples: Axis((x,y,z), (dx,dy,dz)); no 'anchor' keyword, no Pos/Axis object as tuple.
- body.at(axis_object) returns an endpoint bound to that actual body.
- assembly.joint(string, endpoint0, endpoint1, dofs=tuple_of_JointDOF).
- JointDOF(JointAxis.ROT_Z, limits=(lower_radians,upper_radians)) gives a hinge.
- JointDOF(JointAxis.TRANS_Z, limits=(lower_meters,upper_meters)) gives a slide.
- assembly.articulation(string, root=actual_base_body, joints=list_of_joint_name_strings).
Every body variable must be assigned by YOUR code before use. 'parent', 'child', 'anchor'
and 'direction' are role descriptions, not predefined Python globals.
Keep geometry of every body in the common observed world frame; identical endpoint anchors
then preserve its zero pose. Limits must include zero. Use a physical revolute/prismatic joint.
The file MUST set object_model = model at module level and define:
  def run_tests():
      return TestContext(object_model).report()
Use hollow bodies where observed. Preserve thin walls, separate doors/lids, handles and interior.
Cells without surface evidence are UNKNOWN, not empty/solid. Inference of hidden function is allowed.
No file access, network, subprocess or introspection. Keep code concise and explicit.
Read local surface/normal errors and source measurements before revising geometry.
After feedback, return the entire corrected file. The host compiler and geometry checker decide
observational acceptance; your own tests cannot override it. A geometry pass is not physics acceptance.
"""


class ProgramModel(OpenRouterModel):
    async def _send_with_retries(self, request: dict) -> httpx.Response:
        return await super()._send_with_retries(
            request | {"max_tokens": 4096, "reasoning": {"enabled": False}, "temperature": 0.2}
        )


def program_source(text: str) -> str:
    """Remove a single Markdown fence; never repair or complete model code."""
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines[0] not in ("```python", "```") or lines[-1] != "```":
            raise ValueError("Expected one complete Python code fence")
        text = "\n".join(lines[1:-1])
    if not text:
        raise ValueError("No visible Python program returned")
    return text + "\n"


def audit_program(source: str) -> list[dict]:
    """Report compile-contract errors without rewriting the generated program."""
    lint = subprocess.run(
        ["ruff", "check", "--select", "F821", "--output-format", "json", "-"],
        input=source,
        text=True,
        capture_output=True,
        check=False,
    )
    notes = [
        {"line": item["location"]["row"], "message": item["message"]}
        for item in json.loads(lint.stdout)
    ]
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return notes
    stored = {
        node.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store)
    }
    if "object_model" not in stored:
        notes.append(
            {
                "message": "No assignment to required global object_model appears in source; assign your completed assembly"
            }
        )
    notes.extend(
        {
            "line": node.lineno,
            "message": "Axis has no anchor keyword; use Axis(origin_xyz_tuple, direction_xyz_tuple)",
        }
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "Axis"
        and any(keyword.arg == "anchor" for keyword in node.keywords)
    )
    return notes


async def run_program(
    model: OpenRouterModel,
    workspace: LocalWorkspace,
    public: Path,
    max_calls: int,
    run: Path | None = None,
    native_config: Path | None = None,
    geometry_profile: Path | None = None,
) -> dict:
    run = run or workspace.create_run(f"program-{public.name}-{datetime.now():%Y%m%d-%H%M%S}")
    host = ModelingTools(public, workspace, run, native_config, geometry_profile)
    host.attempt = max(
        (int(p.name.split("-")[1]) for p in (run / "result").glob("attempt-*")), default=0
    )
    messages_file, events_file = run / "messages.json", run / "events.json"
    messages = (
        json.loads(messages_file.read_text())
        if messages_file.exists()
        else [
            {"role": "system", "content": API_CONTRACT},
            {
                "role": "user",
                "content": host.input["request"]
                + "\nSOURCE OBSERVATIONS:\n"
                + json.dumps(host.analysis),
            },
        ]
    )
    events = json.loads(events_file.read_text()) if events_file.exists() else []
    if messages[0]["content"] != API_CONTRACT:
        digest = hashlib.sha256(messages[0]["content"].encode()).hexdigest()[:12]
        (run / f"previous-contract-{digest}.txt").write_text(
            messages[0]["content"], encoding="utf-8"
        )
        messages[0]["content"] = API_CONTRACT
    # A resumed repair receives all current static findings in one feedback packet.
    if host.attempt:
        last = run / "result" / f"attempt-{host.attempt:02d}"
        source = (last / "main.py").read_text()
        messages.append(
            {
                "role": "user",
                "content": "STATIC CONTRACT AUDIT:\n" + json.dumps(audit_program(source)),
            }
        )
        if (last / "parts/assembly.json").exists():
            assembly = json.loads((last / "parts/assembly.json").read_text())
            parts = load_parts(last / "parts", assembly)
            measurements = {
                "source_bounds": host.source.bounds.tolist(),
                "candidate_part_bounds": {
                    name: mesh.bounds.tolist() for name, mesh in parts.items()
                },
            }
            messages.append(
                {
                    "role": "user",
                    "content": "CANDIDATE MEASUREMENTS, in the same source coordinates:\n"
                    + json.dumps(measurements),
                }
            )
    feedback = next((event["feedback"] for event in reversed(events) if "feedback" in event), {})
    if host.attempt and (
        feedback.get("status", "completed") != "completed"
        or not host.validation_matches(feedback)
        or feedback.get("native_physics", {}).get("status") == "execution_error"
    ):
        feedback = host.evaluate(json.loads((last / "compile.json").read_text()), last)
        feedback["source_sha256"] = hashlib.sha256(source.encode()).hexdigest()
        messages.append(
            {"role": "user", "content": "REFRESHED VALIDATION:\n" + json.dumps(feedback)}
        )
    error = ""
    for _ in range(max_calls):
        if feedback.get("combined_gate_passed"):
            break
        if feedback.get("native_physics", {}).get("status") == "execution_error":
            error = "Native execution failed; repair infrastructure before changing CAD"
            break
        call = len(events) + 1
        save_json(messages_file, messages)
        print(json.dumps({"case": public.name, "call": call, "phase": "program_query"}), flush=True)
        try:
            response = await model.query(messages)
        except ModelError as exc:
            error = str(exc)
            events.append({"call": call, "error": error})
            break
        text = response["text"]
        (run / f"response-{call:02d}.txt").write_text(text, encoding="utf-8")
        finish = response["response"]["choices"][0].get("finish_reason")
        try:
            if finish == "length":
                raise ValueError("Output budget exhausted; return a complete shorter program")
            source = program_source(text)
            source_hash = hashlib.sha256(source.encode()).hexdigest()
            if source_hash == feedback.get("source_sha256"):
                feedback = {
                    **feedback,
                    "source_unchanged": True,
                    "compiled_this_call": False,
                    "stop_reason": "identical program; previous unresolved checks retained",
                }
            else:
                feedback = await asyncio.to_thread(host.submit, source)
                feedback["static_audit"] = audit_program(source)
                feedback["source_sha256"] = source_hash
                feedback["compiled_this_call"] = True
        except ValueError as exc:
            feedback = {
                "compiler_status": "not_run",
                "geometry_gate_passed": False,
                "error": str(exc),
            }
        events.append(
            {
                "call": call,
                "finish_reason": finish,
                "visible_characters": len(text),
                "tokens": response.get("token_usage"),
                "cost_usd": response.get("cost"),
                "feedback": feedback,
            }
        )
        if text:
            messages.append({"role": "assistant", "content": text})
        messages.append(
            {"role": "user", "content": "INDEPENDENT FEEDBACK:\n" + json.dumps(feedback)}
        )
        save_json(messages_file, messages)
        save_json(events_file, events)
        print(
            json.dumps(
                {
                    "case": public.name,
                    "call": call,
                    "compiler": feedback["compiler_status"],
                    "geometry_gate_passed": feedback.get("geometry_gate_passed"),
                    "error": feedback.get("error"),
                }
            ),
            flush=True,
        )
        if feedback.get("combined_gate_passed") or feedback.get("source_unchanged"):
            break
        if feedback.get("native_physics", {}).get("status") == "execution_error":
            error = "Native execution failed; repair infrastructure before changing CAD"
            break
    summary = {
        "case": public.name,
        "run": str(run),
        "model": model.config.openrouter_model,
        "protocol": "whole Python via chat; host compiles and returns feedback",
        "model_calls": len(events),
        "submissions": host.attempt,
        "last_feedback": feedback,
        "error": error,
        "cost_usd": sum(float(e.get("cost_usd") or 0) for e in events),
        "source_code_template_supplied": False,
        "private_truth_sent_to_model": False,
        "pixels_sent_to_model": False,
    }
    save_json(events_file, events)
    save_json(run / "summary.json", summary)
    return summary


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--cases", nargs="+", default=["hinged_box", "drawer", "cabinet"])
    parser.add_argument("--max-calls", type=int, choices=range(1, 5), default=1)
    parser.add_argument("--resume-run", type=Path)
    parser.add_argument("--native-config", type=Path)
    parser.add_argument("--geometry-profile", type=Path)
    parser.add_argument("--model", default="cohere/north-mini-code:free")
    args = parser.parse_args()
    if args.resume_run and len(args.cases) != 1:
        parser.error("--resume-run requires exactly one --cases value")
    settings = Settings().model_copy(
        update={
            "openrouter_model": args.model,
            "openrouter_max_attempts": 1,
            "openrouter_request_timeout_seconds": 180,
        }
    )
    try:
        preflight = await free_model_preflight(settings, args.max_calls * len(args.cases))
    except (httpx.HTTPError, ValueError) as exc:
        failure = {"phase": "preflight", "error": f"{type(exc).__name__}: {exc}"}
        save_json(args.root / "preflight-failure.json", failure)
        print(json.dumps(failure), flush=True)
        return 1
    model = ProgramModel(settings)
    workspace = LocalWorkspace(
        output_dir=args.root / "runs", physics_enabled=True, timeout_seconds=180
    )
    results = []
    try:
        for case in args.cases:
            summary = await run_program(
                model,
                workspace,
                args.root / "inputs" / case,
                args.max_calls,
                args.resume_run,
                args.native_config,
                args.geometry_profile,
            )
            results.append(summary)
            save_json(
                args.root / f"batch-{datetime.now():%Y%m%d-%H%M%S}.json",
                {"preflight": preflight, "results": results},
            )
            if summary["error"]:
                break
    finally:
        await model.close()
    return (
        0
        if len(results) == len(args.cases)
        and all(not r["error"] and r["last_feedback"].get("combined_gate_passed") for r in results)
        else 1
    )


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
