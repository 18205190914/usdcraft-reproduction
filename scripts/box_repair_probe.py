"""Explicit live fault-injection test, separate from the successful three-case batch."""

from __future__ import annotations

import asyncio
import json
from dataclasses import asdict, replace
from datetime import datetime
from pathlib import Path
from typing import Any

import httpx

from articraft.agent.provider.openrouter import OpenRouterModel
from articraft.agent.workspace import LocalWorkspace
from articraft.box_workflow import (
    CASES,
    assess_submission,
    free_model_preflight,
    render_box,
    run_box,
)
from articraft.settings import Settings


class RepairModel(OpenRouterModel):
    initial_feedback: dict[str, Any]

    async def _send_with_retries(self, request: dict[str, Any]) -> httpx.Response:
        return await super()._send_with_retries(
            request
            | {
                "max_tokens": 2048,
                "reasoning": {"enabled": False},
                "parallel_tool_calls": False,
                "tool_choice": {"type": "function", "function": {"name": "build_box"}},
            }
        )

    async def query(self, messages, *, tools=None):
        if len(messages) == 2:
            messages = [
                *messages,
                {
                    "role": "user",
                    "content": (
                        "A deliberately injected tool fault created an incorrect box. Correct its parameters "
                        "using this independent USDZ feedback and rebuild. Height means TOTAL CLOSED height.\n"
                        + json.dumps(self.initial_feedback)
                    ),
                },
            ]
        return await super().query(messages, tools=tools)


async def main() -> None:
    settings = Settings().model_copy(
        update={
            "openrouter_model": "cohere/north-mini-code:free",
            "openrouter_max_attempts": 1,
            "openrouter_request_timeout_seconds": 180,
        }
    )
    preflight = await free_model_preflight(settings, 2)
    workspace = LocalWorkspace(
        output_dir=Path("runs").resolve(), physics_enabled=True, timeout_seconds=180
    )
    run = workspace.create_run(f"box-live-repair-probe-{datetime.now():%Y%m%d-%H%M%S}")
    reference = Path("examples/storage_box.py").read_text()
    spec = CASES["small"]
    wrong = replace(spec, height=spec.height - spec.lid_height)
    initial = assess_submission(workspace, run, render_box(reference, wrong), spec)
    assert initial["compiler_status"] == "success" and not initial["accepted"]
    injected = {
        "fault_injected": True,
        "initial_parameters": asdict(wrong),
        "expected_parameters": asdict(spec),
        "failed_checks": initial["geometry"]["failed_checks"],
        "actual_closed_size_m": initial["geometry"]["closed_size_m"],
    }
    (run / "fault-injection.json").write_text(json.dumps(injected, indent=2))
    (run / "preflight.json").write_text(json.dumps(preflight, indent=2))
    model = RepairModel(settings)
    model.initial_feedback = injected
    print(
        json.dumps({"phase": "fault_injected_and_rejected", "run": str(run), **injected}),
        flush=True,
    )
    try:
        summary = await run_box(
            model, workspace, run, spec, reference, max_calls=2, mode="parameters"
        )
    finally:
        await model.close()
    summary.update(fault_injected=True, model=settings.openrouter_model)
    (run / "result/repair-probe-summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: v for k, v in summary.items() if k != "last_feedback"}), flush=True)
    raise SystemExit(0 if summary["accepted"] else 1)


if __name__ == "__main__":
    asyncio.run(main())
