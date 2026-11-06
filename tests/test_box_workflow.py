from __future__ import annotations

import asyncio
import json
from dataclasses import replace
from pathlib import Path

import httpx
import pytest
from pxr import Usd, UsdGeom, UsdPhysics

from articraft.agent.workspace import LocalWorkspace
from articraft.box_validation import BoxSpec, validate_box
from articraft.box_workflow import free_model_preflight, run_box
from articraft.settings import Settings

SPEC = BoxSpec(0.30, 0.20, 0.15, 0.002, 0.006, 100)
REFERENCE = Path(__file__).resolve().parents[1] / "examples/storage_box.py"


@pytest.fixture(scope="module")
def exported_box(tmp_path_factory):
    workspace = LocalWorkspace(output_dir=tmp_path_factory.mktemp("box"), physics_enabled=True)
    run = workspace.create_run("reference")
    (run / "workspace/main.py").write_text(REFERENCE.read_text())
    compiled = workspace.compile_path(run)
    assert compiled["status"] == "success", compiled
    return Path(compiled["usdz"])


def test_accepts_real_hinge_contact_and_measures_all_walls(exported_box):
    report = validate_box(exported_box, SPEC)
    assert report["passed"], report
    assert len(report["wall_thickness_samples_m"]) == 8
    assert len(report["bottom_thickness_samples_m"]) == 9
    assert len(report["sweep_intersection_volumes_m3"]) == 9


@pytest.mark.parametrize(
    "field,value,check",
    [
        ("width", 0.31, "closed_dimensions"),
        ("wall", 0.003, "four_wall_thicknesses"),
        ("opening_degrees", 110, "joint_limits"),
    ],
)
def test_rejects_wrong_external_contract(exported_box, field, value, check):
    report = validate_box(exported_box, replace(SPEC, **{field: value}))
    assert not report["passed"]
    assert not report["checks"][check]


@pytest.mark.parametrize("fault", ["axis", "penetration"])
def test_rejects_corrupted_export_even_without_python_self_tests(exported_box, tmp_path, fault):
    stage = Usd.Stage.Open(str(exported_box))
    path = tmp_path / "corrupted.usda"
    stage.GetRootLayer().Export(str(path))
    stage = Usd.Stage.Open(str(path))
    if fault == "axis":
        joint = next(p for p in stage.Traverse() if p.IsA(UsdPhysics.RevoluteJoint))
        UsdPhysics.RevoluteJoint(joint).GetAxisAttr().Set("X")
        expected_check = "rear_hinge_frame"
    else:
        lid_mesh = next(
            p for p in stage.Traverse() if p.IsA(UsdGeom.Mesh) and "/lid/" in str(p.GetPath())
        )
        mesh = UsdGeom.Mesh(lid_mesh)
        mesh.GetPointsAttr().Set([(p[0], p[1], p[2] - 0.001) for p in mesh.GetPointsAttr().Get()])
        expected_check = "nine_pose_no_solid_interpenetration"
    stage.GetRootLayer().Save()
    report = validate_box(path, SPEC)
    assert not report["passed"], report
    assert not report["checks"][expected_check], report


def test_feedback_repairs_a_compilable_wrong_size_before_accepting(tmp_path):
    reference = REFERENCE.read_text()

    class ScriptedModel:
        def __init__(self):
            self.calls = 0

        async def query(self, messages, *, tools):
            self.calls += 1
            if self.calls == 2:
                feedback = json.loads(messages[-1]["output"])
                assert feedback["compiler_status"] == "success"
                assert not feedback["accepted"]
                assert "closed_dimensions" in feedback["geometry"]["failed_checks"]
            source = (
                reference.replace("WIDTH = 0.30", "WIDTH = 0.31") if self.calls == 1 else reference
            )
            return {
                "tool_calls": [
                    {
                        "id": f"call-{self.calls}",
                        "name": "submit_program",
                        "arguments": json.dumps({"source": source}),
                    }
                ],
                "cost": 0,
            }

    model = ScriptedModel()
    workspace = LocalWorkspace(output_dir=tmp_path, physics_enabled=True)
    run = workspace.create_run("repair")
    summary = asyncio.run(run_box(model, workspace, run, SPEC, reference, max_calls=2))
    assert summary["accepted"], summary
    assert summary["model_calls"] == 2
    assert (run / "result/attempt-01.feedback.json").exists()
    assert (run / "result/attempt-02.feedback.json").exists()
    assert json.loads((run / "record.json").read_text())["status"] == "success"


def test_preflight_rejects_paid_price_before_inference(monkeypatch):
    requests = []

    def respond(request):
        requests.append(request.url.path)
        return httpx.Response(
            200,
            json={
                "data": [
                    {
                        "id": "test/model:free",
                        "pricing": {"prompt": "0.00001", "completion": "0"},
                        "supported_parameters": ["tools"],
                    }
                ]
            },
        )

    client = httpx.AsyncClient(transport=httpx.MockTransport(respond))
    monkeypatch.setattr("articraft.box_workflow.httpx.AsyncClient", lambda **kwargs: client)
    settings = Settings(openrouter_api_key="fake-test-only", openrouter_model="test/model:free")
    with pytest.raises(ValueError, match="not currently zero-price"):
        asyncio.run(free_model_preflight(settings, 4))
    assert requests == ["/api/v1/models"]


def test_parameter_tool_repairs_height_and_keeps_external_contract(tmp_path):
    from dataclasses import asdict

    reference = REFERENCE.read_text()

    class ParameterModel:
        calls = 0

        async def query(self, messages, *, tools):
            self.calls += 1
            assert tools[0]["name"] == "build_box"
            parameters = asdict(SPEC)
            if self.calls == 1:
                parameters["height"] -= SPEC.lid_height
            else:
                feedback = json.loads(messages[-1]["output"])
                assert feedback["compiler_status"] == "success"
                assert not feedback["accepted"]
                assert feedback["geometry"]["requested"]["height"] == SPEC.height
            return {
                "tool_calls": [
                    {
                        "id": f"parameters-{self.calls}",
                        "name": "build_box",
                        "arguments": json.dumps(parameters),
                    }
                ],
                "cost": 0,
            }

    workspace = LocalWorkspace(output_dir=tmp_path, physics_enabled=True)
    run = workspace.create_run("parameters")
    summary = asyncio.run(
        run_box(ParameterModel(), workspace, run, SPEC, reference, max_calls=2, mode="parameters")
    )
    assert summary["accepted"], summary
    assert summary["model_calls"] == 2
    assert summary["mode"] == "parameters"
