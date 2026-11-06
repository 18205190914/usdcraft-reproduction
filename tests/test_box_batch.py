from __future__ import annotations

import asyncio
import importlib.util
import json
import shutil
from dataclasses import asdict
from pathlib import Path

from articraft.agent.record import Record


def test_batch_repairs_only_rejected_case_with_two_scripted_requests(tmp_path, monkeypatch):
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location("box_batch_test", root / "scripts/box_batch.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    (tmp_path / "examples").mkdir()
    shutil.copyfile(root / "examples/storage_box.py", tmp_path / "examples/storage_box.py")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("sys.argv", ["box_batch.py", "--max-calls", "2"])

    async def preflight(settings, max_calls):
        return {"model": settings.openrouter_model, "pricing": {"prompt": "0", "completion": "0"}}

    class Model:
        calls = 0

        def __init__(self, settings):
            pass

        async def query(self, messages, *, tools):
            self.calls += 1
            parameters = {name: asdict(case) for name, case in module.CASES.items()}
            if self.calls == 1:
                parameters["small"]["height"] -= 0.006
            else:
                feedback = json.loads(messages[-1]["output"])
                assert not feedback["small"]["accepted"]
                assert feedback["large"]["accepted"]
                assert feedback["compact"]["accepted"]
            return {
                "tool_calls": [
                    {
                        "id": f"batch-{self.calls}",
                        "name": "build_boxes",
                        "arguments": json.dumps(parameters),
                    }
                ],
                "cost": 0,
            }

        async def close(self):
            pass

    monkeypatch.setattr(module, "free_model_preflight", preflight)
    monkeypatch.setattr(module, "BatchModel", Model)
    assert asyncio.run(module.main()) == 0
    batch = next((tmp_path / "runs").glob("*/summary.json"))
    summary = json.loads(batch.read_text())
    assert summary["passed"]
    assert summary["api_requests"] == 2
    for name, path in summary["runs"].items():
        record = Record.load(Path(path) / "record.json")
        assert record.status == "success"
        assert record.attempts == (2 if name == "small" else 1)
