"""Validate three box requests with one tool call per batch, sharing a small API budget."""

from __future__ import annotations

import argparse
import asyncio
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import httpx

from articraft.agent.provider.openrouter import OpenRouterModel
from articraft.agent.record import Record
from articraft.agent.workspace import LocalWorkspace
from articraft.box_validation import BoxSpec
from articraft.box_workflow import (
    BUILD_TOOL,
    CASES,
    assess_submission,
    free_model_preflight,
    render_box,
)
from articraft.settings import Settings

PARAMETERS = BUILD_TOOL["parameters"]

TOOL = {
    "type": "function",
    "name": "build_boxes",
    "description": "Build, independently measure and simulate all three requested boxes. Adjust rejected parameters and resubmit.",
    "parameters": {
        "type": "object",
        "properties": dict.fromkeys(CASES, PARAMETERS),
        "required": list(CASES),
        "additionalProperties": False,
    },
    "strict": True,
}


class BatchModel(OpenRouterModel):
    async def _send_with_retries(self, request: dict[str, Any]) -> httpx.Response:
        return await super()._send_with_retries(
            request
            | {
                "max_tokens": 4096,
                "reasoning": {"enabled": False},
                "tool_choice": {"type": "function", "function": {"name": "build_boxes"}},
                "parallel_tool_calls": False,
            }
        )


def save(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="cohere/north-mini-code:free")
    parser.add_argument("--max-calls", type=int, choices=range(1, 4), default=3)
    args = parser.parse_args()
    settings = Settings().model_copy(
        update={
            "openrouter_model": args.model,
            "openrouter_max_attempts": 1,
            "openrouter_request_timeout_seconds": 180,
        }
    )
    try:
        preflight = await free_model_preflight(settings, args.max_calls)
    except (httpx.HTTPError, ValueError) as exc:
        print(
            json.dumps({"phase": "preflight", "error": f"{type(exc).__name__}: {exc}"}), flush=True
        )
        return 1
    print(json.dumps({"preflight": preflight}), flush=True)
    batch_id = f"box-batch-{datetime.now():%Y%m%d-%H%M%S}"
    directory = Path("runs", batch_id).resolve()
    directory.mkdir(parents=True)
    save(directory / "preflight.json", preflight)
    workspace = LocalWorkspace(
        output_dir=Path("runs").resolve(), physics_enabled=True, timeout_seconds=180
    )
    runs = {name: workspace.create_run(f"{batch_id}-{name}") for name in CASES}
    reference = Path("examples/storage_box.py").read_text()
    messages = [
        {
            "role": "system",
            "content": (
                "Call build_boxes once with all three named cases. Extract lengths exactly in meters and angles "
                "in degrees. Height is TOTAL CLOSED HEIGHT including the lid; never subtract lid thickness. "
                "Use the user's dimensions exactly. The CAD tool handles geometry and masses. If the tool "
                "rejects a case, correct the parameters and call again. No Python or explanations needed."
            ),
        },
        {
            "role": "user",
            "content": "\n\n".join(f"{name}: {spec.prompt()}" for name, spec in CASES.items()),
        },
    ]
    save(
        directory / "request.json",
        {
            "model": args.model,
            "mode": "parameter_tools_batch",
            "reference_assisted": True,
            "messages": messages,
        },
    )
    feedback = {}
    cost = 0.0
    error = ""
    model = BatchModel(settings)
    try:
        for number in range(1, args.max_calls + 1):
            print(
                json.dumps({"batch": batch_id, "api_request": number, "phase": "query"}), flush=True
            )
            response = await model.query(messages, tools=[TOOL])
            cost += float(response.get("cost") or 0)
            calls = response.get("tool_calls") or []
            assistant = {
                "role": "assistant",
                "content": response.get("text", ""),
                "tool_calls": calls,
                "provider_content": response.get("provider_content", []),
            }
            messages.append(assistant)
            save(
                directory / f"response-{number:02d}.json",
                {
                    "tool_calls": calls,
                    "usage": response.get("token_usage"),
                    "cost": response.get("cost"),
                    "finish_reason": (response.get("response", {}).get("choices") or [{}])[0].get(
                        "finish_reason"
                    ),
                },
            )
            if not calls:
                messages.append(
                    {
                        "role": "user",
                        "content": "Call build_boxes with the six requested parameters per case.",
                    }
                )
                continue
            for call in calls:
                try:
                    if call["name"] != "build_boxes":
                        raise ValueError("Expected build_boxes")
                    arguments = json.loads(call["arguments"])
                    for name, spec in CASES.items():
                        if feedback.get(name, {}).get("accepted"):
                            continue
                        parameters = BoxSpec(**arguments[name])
                        source = render_box(reference, parameters)
                        feedback[name] = await asyncio.to_thread(
                            assess_submission, workspace, runs[name], source, spec
                        )
                        save(
                            runs[name] / "request.json",
                            {
                                "prompt": spec.prompt(),
                                "batch": batch_id,
                                "reference_assisted": True,
                            },
                        )
                        print(
                            json.dumps(
                                {
                                    "case": name,
                                    "accepted": feedback[name]["accepted"],
                                    "failed_checks": feedback[name]
                                    .get("geometry", {})
                                    .get("failed_checks"),
                                }
                            ),
                            flush=True,
                        )
                    compact = {
                        name: {
                            "accepted": item["accepted"],
                            "failed_checks": item.get("geometry", {}).get("failed_checks"),
                            "actual_closed_size_m": item.get("geometry", {}).get("closed_size_m"),
                            "compiler_error": item.get("compiler_error"),
                            "expected_parameters": vars(CASES[name]),
                        }
                        for name, item in feedback.items()
                    }
                except Exception as exc:
                    compact = {"accepted": False, "error": f"{type(exc).__name__}: {exc}"}
                messages.append(
                    {
                        "type": "function_call_output",
                        "call_id": call["id"],
                        "output": json.dumps(compact),
                    }
                )
                save(directory / f"feedback-{number:02d}.json", feedback)
            if len(feedback) == len(CASES) and all(item["accepted"] for item in feedback.values()):
                break
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    finally:
        await model.close()
    summary = {
        "batch": batch_id,
        "model": args.model,
        "mode": "parameter_tools_batch",
        "reference_assisted": True,
        "api_requests": number,
        "cost_usd": cost,
        "error": error,
        "runs": {name: str(path) for name, path in runs.items()},
        "feedback": feedback,
        "passed": len(feedback) == len(CASES)
        and all(item["accepted"] for item in feedback.values()),
    }
    for name, run in runs.items():
        record = Record.load(run / "record.json")
        item = feedback.get(name, {})
        record.status = "success" if item.get("accepted") else "error"
        record.result = item.get("asset", "") if item.get("accepted") else ""
        record.error = (
            ""
            if item.get("accepted")
            else error or "No accepted submission within the batch budget"
        )
        record.save(run / "record.json")
    save(directory / "summary.json", summary)
    print(json.dumps({k: v for k, v in summary.items() if k != "feedback"}), flush=True)
    return 0 if summary["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
