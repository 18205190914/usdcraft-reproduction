"""A bounded, reference-assisted text -> box -> check -> repair example.

Run from the repository root: python -m articraft.box_workflow --case small
Only the explicitly selected zero-price OpenRouter model is permitted.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import re
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

import httpx

from articraft.agent.provider.openrouter import OpenRouterModel
from articraft.agent.record import Record, append_conversation
from articraft.agent.workspace import LocalWorkspace
from articraft.box_validation import BoxSpec, validate_box
from articraft.settings import Settings
from articraft.simulate import simulate_usdz

CASES = {
    "small": BoxSpec(0.24, 0.16, 0.12, 0.002, 0.006, 100),
    "large": BoxSpec(0.40, 0.25, 0.20, 0.003, 0.008, 110),
    "compact": BoxSpec(0.18, 0.12, 0.09, 0.0015, 0.004, 95),
}
SUBMIT_TOOL = {
    "type": "function",
    "name": "submit_program",
    "description": "Compile a complete main.py, independently measure the USDZ, and simulate it. Repair failures with another submission.",
    "parameters": {
        "type": "object",
        "properties": {
            "source": {
                "type": "string",
                "description": "Complete Python source, without Markdown fences.",
            }
        },
        "required": ["source"],
        "additionalProperties": False,
    },
    "strict": True,
}


BUILD_TOOL = {
    "type": "function",
    "name": "build_box",
    "description": "Build and independently verify a hollow articulated ABS box. All lengths in meters, angle in degrees. Correct parameters and call again if rejected.",
    "parameters": {
        "type": "object",
        "properties": {
            "width": {"type": "number", "description": "Outer X dimension in meters."},
            "depth": {"type": "number", "description": "Outer Y dimension in meters."},
            "height": {
                "type": "number",
                "description": "TOTAL CLOSED HEIGHT including lid, meters. Never subtract lid thickness.",
            },
            "wall": {
                "type": "number",
                "description": "Uniform side wall and bottom thickness, meters.",
            },
            "lid_height": {"type": "number", "description": "Solid lid thickness in meters."},
            "opening_degrees": {
                "type": "number",
                "description": "Maximum positive opening angle in degrees.",
            },
        },
        "required": ["width", "depth", "height", "wall", "lid_height", "opening_degrees"],
        "additionalProperties": False,
    },
    "strict": True,
}


def render_box(reference: str, parameters: BoxSpec) -> str:
    """A category-specific CAD tool; the model chooses parameters, not SDK methods."""
    constants = dict(
        zip(
            ("WIDTH", "DEPTH", "HEIGHT", "WALL", "LID_HEIGHT", "OPENING_DEGREES"),
            asdict(parameters).values(),
            strict=True,
        )
    )
    for name, value in constants.items():
        reference, count = re.subn(rf"(?m)^{name} = [^\n]+$", f"{name} = {value!r}", reference)
        if count != 1:
            raise ValueError(f"Reference must have exactly one {name} constant")
    return reference


class FreeBoxModel(OpenRouterModel):
    """Keep per-request bounds local to this example rather than changing the adapter."""

    def __init__(self, settings: Settings, *, mode: str = "parameters"):
        super().__init__(settings)
        self.mode = mode

    async def _send_with_retries(self, request: dict[str, Any]) -> httpx.Response:
        return await super()._send_with_retries(
            request
            | {
                "max_tokens": 2048 if self.mode == "parameters" else 8192,
                "reasoning": {"effort": "low"},
                "tool_choice": {
                    "type": "function",
                    "function": {
                        "name": "build_box" if self.mode == "parameters" else "submit_program"
                    },
                },
                "parallel_tool_calls": False,
            }
        )


def _save(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
    )


def assess_submission(
    workspace: LocalWorkspace, run: Path, source: str, spec: BoxSpec
) -> dict[str, Any]:
    """Only this trusted host function decides acceptance; model tests cannot override it."""
    (run / "workspace/main.py").write_text(source, encoding="utf-8")
    payload = workspace.compile_path(run)
    attempt = Record.load(run / "record.json").attempts
    prefix = run / "result" / f"attempt-{attempt:02d}"
    prefix.with_suffix(".py").write_text(source, encoding="utf-8")
    _save(prefix.with_suffix(".compile.json"), payload)
    compiler_tests = payload.get("test_report") or {}
    feedback: dict[str, Any] = {
        "accepted": False,
        "attempt": attempt,
        "source_sha256": hashlib.sha256(source.encode()).hexdigest(),
        "compiler_status": payload["status"],
        "compiler_checks": compiler_tests.get("checks_run"),
        "compiler_failures": compiler_tests.get("failures", []),
        "compiler_error": payload.get("error", ""),
        "compiler_traceback": payload.get("traceback", "")[-6000:],
    }
    if payload["status"] == "success" and payload.get("usdz"):
        asset = Path(payload["usdz"])
        if not asset.is_absolute():
            asset = run / asset
        feedback["asset"] = str(asset)
        geometry = validate_box(asset, spec)
        feedback["geometry"] = geometry
        if geometry["passed"]:
            simulations = {}
            for scenario in ("drop", "release"):
                result = simulate_usdz(
                    asset, run / "result/simulation" / scenario, seconds=3, scenario=scenario
                )
                report = asdict(result)
                report.pop("trajectory", None)
                report.update(
                    passed=result.stood_up,
                    fell_through_floor=result.fell_through_floor,
                    parts_stayed_together=result.parts_stayed_together,
                )
                simulations[scenario] = report
                if scenario == "release" and result.trajectory is not None:
                    _save(
                        run / "result/simulation" / f"{asset.stem}.trajectory.json",
                        result.trajectory.to_payload(),
                    )
            feedback["simulations"] = simulations
            feedback["accepted"] = all(item["passed"] for item in simulations.values())
    _save(prefix.with_suffix(".feedback.json"), feedback)
    return feedback


async def run_box(
    model: Any,
    workspace: LocalWorkspace,
    run: Path,
    spec: BoxSpec,
    reference: str,
    *,
    max_calls: int = 4,
    mode: str = "program",
) -> dict[str, Any]:
    """One submit tool, bounded calls, persisted feedback, no human source edits."""
    messages = [
        {
            "role": "system",
            "content": (
                "You author Articraft articulated CAD programs. Call submit_program with complete Python. "
                "The reference below is verified for a DIFFERENT size. Adapt all dimensions, wall/lid "
                "thickness, joint limits, and sampled test angles to the user's request. Reuse its actual "
                "SDK API: do not invent new methods. object_model and run_tests() must exist. "
                "Use meters, build123d Align/Axis/Box/Pos, Material.ABS_PLASTIC, two rigid bodies, "
                "body.at(Axis(...)), JointDOF(JointAxis.ROT_Z, limits=(0, radians(...))). "
                "Local ROT_Z follows the attachment frame's physical world +X axis. "
                "A hollow base is outer box minus a cavity starting at WALL, cutting past the top. "
                "A lid's local box uses (CENTER, MIN, MIN). The assembly joint places it at the rim. "
                "Expect hinge contact; test solid interpenetration, not absence of contact. "
                "The external validator measures the USDZ independently; changing self-tests cannot bypass it. "
                "On feedback failure, repair the source and resubmit. No file, environment, network, "
                "or subprocess access in generated programs.\nVERIFIED REFERENCE:\n" + reference
            ),
        },
        {"role": "user", "content": spec.prompt()},
    ]
    if mode == "parameters":
        messages[0]["content"] = (
            "Use build_box to create the user's articulated box. Extract the six numerical parameters "
            "from the request, preserving units (meters/degrees). The tool constructs a verified "
            "CAD reference program, independently measures its USDZ, and simulates it. If feedback "
            "rejects the parameters, correct them and call again. Do not write Python or commentary."
        )
    tool = BUILD_TOOL if mode == "parameters" else SUBMIT_TOOL
    start = time.monotonic()
    _save(
        run / "request.json",
        {"spec": asdict(spec), "prompt": spec.prompt(), "reference_assisted": True, "mode": mode},
    )
    cost = 0.0
    token_usage: dict[str, int] = {}
    feedback: dict[str, Any] = {"accepted": False}
    calls = 0
    error = ""
    try:
        for calls in range(1, max_calls + 1):
            print(json.dumps({"run": run.name, "model_call": calls, "phase": "query"}), flush=True)
            response = await model.query(messages, tools=[tool])
            cost += float(response.get("cost") or 0)
            for name, value in (response.get("token_usage") or {}).items():
                token_usage[name] = token_usage.get(name, 0) + int(value or 0)
            assistant = {
                "role": "assistant",
                "content": response.get("text", ""),
                "tool_calls": response.get("tool_calls", []),
                "provider_content": response.get("provider_content", []),
            }
            messages.append(assistant)
            # Do not record provider reasoning; source and tool results suffice for review.
            append_conversation(
                run / "conversation.jsonl",
                {k: v for k, v in assistant.items() if k != "provider_content"},
            )
            calls_out = response.get("tool_calls") or []
            if not calls_out:
                messages.append(
                    {
                        "role": "user",
                        "content": "Call the provided build tool with the required parameters.",
                    }
                )
                continue
            for call in calls_out:
                try:
                    arguments = json.loads(call["arguments"])
                    if mode == "parameters":
                        if call["name"] != "build_box":
                            raise ValueError("Expected build_box")
                        parameters = BoxSpec(**arguments)
                        source = render_box(reference, parameters)
                    else:
                        if call["name"] != "submit_program" or not isinstance(
                            arguments.get("source"), str
                        ):
                            raise ValueError("Expected submit_program(source: string)")
                        source = arguments["source"]
                    feedback = await asyncio.to_thread(
                        assess_submission, workspace, run, source, spec
                    )

                except Exception as exc:
                    feedback = {"accepted": False, "error": f"{type(exc).__name__}: {exc}"}
                tool_output = {
                    "type": "function_call_output",
                    "call_id": call["id"],
                    "output": json.dumps(feedback, default=str),
                }
                messages.append(tool_output)
                append_conversation(run / "conversation.jsonl", tool_output)
                print(
                    json.dumps(
                        {
                            "run": run.name,
                            "model_call": calls,
                            "phase": "checked",
                            "accepted": feedback["accepted"],
                            "compiler": feedback.get("compiler_status"),
                            "failed_checks": feedback.get("geometry", {}).get("failed_checks"),
                            "error": feedback.get("error") or feedback.get("compiler_error"),
                        }
                    ),
                    flush=True,
                )
            if feedback["accepted"]:
                break
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    summary = {
        "run_dir": str(run),
        "reference_assisted": True,
        "mode": mode,
        "spec": asdict(spec),
        "model_calls": calls,
        "accepted": feedback["accepted"],
        "cost_usd": cost,
        "token_usage": token_usage,
        "elapsed_seconds": round(time.monotonic() - start, 2),
        "error": error,
        "last_feedback": feedback,
    }
    _save(run / "result/box-summary.json", summary)
    record = Record.load(run / "record.json")
    record.status = "success" if feedback["accepted"] else "error"
    record.error = error or (
        "" if feedback["accepted"] else "No accepted submission within the call limit"
    )
    record.result = feedback.get("asset", "") if feedback["accepted"] else ""
    record.cost, record.token_usage = cost, token_usage
    record.save(run / "record.json")
    return summary


async def free_model_preflight(settings: Settings, max_calls: int) -> dict[str, Any]:
    """Check current price and remaining quota without consuming an inference call."""
    async with httpx.AsyncClient(timeout=30) as client:
        catalog = await client.get("https://openrouter.ai/api/v1/models")
        catalog.raise_for_status()
        entry = next(
            (m for m in catalog.json()["data"] if m["id"] == settings.openrouter_model), None
        )
        if not entry or not settings.openrouter_model.endswith(":free"):
            raise ValueError("This example requires an explicitly selected :free model")
        if not all(
            float(entry.get("pricing", {}).get(k, "nan")) == 0 for k in ("prompt", "completion")
        ):
            raise ValueError("The selected model is not currently zero-price")
        if "tools" not in entry.get("supported_parameters", []):
            raise ValueError("The selected model does not advertise tool support")
        auth = await client.get(
            "https://openrouter.ai/api/v1/key",
            headers={"Authorization": f"Bearer {(settings.openrouter_api_key or '').strip()}"},
        )
        auth.raise_for_status()
        quota = auth.json()["data"].get("free_model_daily_requests") or {}
        if quota.get("remaining") is not None and quota["remaining"] < max_calls:
            raise ValueError(
                f"Need a budget of {max_calls} free requests; quota remaining: {quota['remaining']}"
            )
        return {
            "model": entry["id"],
            "pricing": entry["pricing"],
            "free_model_daily_requests": quota,
        }


async def _main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=[*CASES, "all"], default="all")
    parser.add_argument("--max-calls", type=int, choices=range(1, 5), default=4)
    parser.add_argument("--mode", choices=["parameters", "program"], default="parameters")
    args = parser.parse_args()
    cases = CASES if args.case == "all" else {args.case: CASES[args.case]}
    settings = Settings().model_copy(
        update={"openrouter_max_attempts": 1, "openrouter_request_timeout_seconds": 300}
    )
    try:
        preflight = await free_model_preflight(settings, args.max_calls * len(cases))
    except (httpx.HTTPError, ValueError) as exc:
        print(
            json.dumps({"phase": "preflight", "error": f"{type(exc).__name__}: {exc}"}), flush=True
        )
        return 1
    print(json.dumps({"preflight": preflight}), flush=True)
    reference = Path("examples/storage_box.py").read_text(encoding="utf-8")
    model = FreeBoxModel(settings, mode=args.mode)
    workspace = LocalWorkspace(
        output_dir=Path("runs").resolve(), physics_enabled=True, timeout_seconds=180
    )
    summaries = []
    try:
        for name, spec in cases.items():
            run = workspace.create_run(f"box-auto-{name}-{datetime.now():%Y%m%d-%H%M%S}")
            _save(run / "preflight.json", preflight)
            summary = await run_box(
                model, workspace, run, spec, reference, max_calls=args.max_calls, mode=args.mode
            )
            summaries.append(summary)
            print(
                json.dumps({k: v for k, v in summary.items() if k != "last_feedback"}), flush=True
            )
            if summary["error"]:
                break
    finally:
        await model.close()
    return 0 if len(summaries) == len(cases) and all(s["accepted"] for s in summaries) else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(_main()))
