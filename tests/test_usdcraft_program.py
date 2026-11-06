from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import httpx
import numpy as np
import pytest
import trimesh

from articraft.agent.workspace import LocalWorkspace
from articraft.settings import Settings
from articraft.source_geometry import planar_observations
from articraft.usdcraft_program import ProgramModel, audit_program, program_source, run_program


def test_planar_observations_measure_inner_and_outer_walls():
    outer = trimesh.creation.box(extents=[0.24, 0.16, 0.12])
    inner = trimesh.creation.box(extents=[0.236, 0.156, 0.116])
    mesh = trimesh.boolean.difference([outer, inner], engine="manifold")
    report = planar_observations(mesh)
    xs = sorted(p["coordinate"] for p in report["patches"] if p["axis"] == "X")
    assert np.allclose(xs, [-0.12, -0.118, 0.118, 0.12])
    assert len(report["patches"]) == 12
    assert all(p["area"] > 0 for p in report["patches"])


def test_non_axial_faces_do_not_create_inferred_planes():
    mesh = trimesh.creation.box()
    mesh.apply_transform(trimesh.transformations.euler_matrix(0.2, 0.3, 0.4))
    assert planar_observations(mesh)["patches"] == []


@pytest.mark.parametrize("wrapper", ["{}", "```python\n{}\n```", "```\n{}\n```"])
def test_program_extraction_only_removes_outer_fence(wrapper):
    code = "object_model = None\n# preserve source exactly"
    assert program_source(wrapper.format(code)) == code + "\n"


@pytest.mark.parametrize("text", ["", "```python\nx = 1", "```javascript\nx = 1\n```"])
def test_empty_or_incomplete_program_is_not_submitted(text):
    with pytest.raises(ValueError):
        program_source(text)


def test_plain_program_request_does_not_send_function_call_controls():
    async def scenario():
        def respond(request):
            body = json.loads(request.content)
            assert "tools" not in body and "tool_choice" not in body
            assert not body["reasoning"]["enabled"]
            return httpx.Response(
                200, json={"choices": [{"message": {"content": "x = 1"}, "finish_reason": "stop"}]}
            )

        settings = Settings(openrouter_api_key="unit-test-key", openrouter_model="unit-model:free")
        model = ProgramModel(
            settings, client=httpx.AsyncClient(transport=httpx.MockTransport(respond))
        )
        try:
            result = await model.query([{"role": "user", "content": "write a program"}])
            assert result["text"] == "x = 1"
            assert result["tool_calls"] == []
        finally:
            await model.close()

    asyncio.run(scenario())


@pytest.mark.parametrize("repair", [False, True])
def test_resume_preserves_feedback_and_rechecks_geometry(tmp_path, repair):
    """Scripted integration test; not evidence of autonomous model repair."""
    public = tmp_path / "public"
    public.mkdir()
    a = trimesh.creation.box(extents=[0.1, 0.1, 0.01]).apply_translation([0, 0, 0.005])
    b = trimesh.creation.box(extents=[0.1, 0.08, 0.005]).apply_translation([0, 0.04, 0.0125])
    trimesh.util.concatenate([a, b]).export(public / "source.ply")
    (public / "input.json").write_text(
        json.dumps(
            {"source": "source.ply", "units": "metres", "request": "Rebuild the hinged panels"}
        )
    )
    code = """from build123d import Box, Pos, Axis
from articraft.sdk import JointAxis, JointDOF, Material, RigidBodyAssembly, TestContext
m = RigidBodyAssembly("panels")
a = m.rigid_body("base")
a.add(Pos(0,0,.005)*Box(.1,.1,.01), name="a", material=Material.ABS_PLASTIC)
b = m.rigid_body("lid")
b.add(Pos(0,.04,HEIGHT)*Box(.1,.08,.005), name="b", material=Material.ABS_PLASTIC)
axis = Axis((0,0,.01),(1,0,0))
m.joint("hinge",a.at(axis),b.at(axis),dofs=(JointDOF(JointAxis.ROT_Z,limits=(0,1.5)),))
m.articulation("main",root=a,joints=["hinge"])
object_model = m
def run_tests():
    return TestContext(object_model).report()
"""

    class ScriptedModel:
        config = SimpleNamespace(openrouter_model="scripted-no-network")
        calls = 0

        async def query(self, messages):
            self.calls += 1
            assert "planar_observations" in messages[1]["content"]
            if self.calls == 2:
                packet = next(
                    m
                    for m in reversed(messages)
                    if m["content"].startswith("INDEPENDENT FEEDBACK:")
                )
                feedback = json.loads(packet["content"].split("\n", 1)[1])
                assert feedback["compiler_status"] == "success"
                assert not feedback["geometry_gate_passed"]
            return {
                "text": code.replace("HEIGHT", ".0125" if self.calls == 2 and repair else ".025"),
                "response": {"choices": [{"finish_reason": "stop"}]},
            }

    model = ScriptedModel()
    workspace = LocalWorkspace(output_dir=tmp_path / "runs", physics_enabled=True)
    first = asyncio.run(run_program(model, workspace, public, 1))
    assert not first["last_feedback"]["geometry_gate_passed"]
    from pathlib import Path

    second = asyncio.run(run_program(model, workspace, public, 1, Path(first["run"])))
    assert second["model_calls"] == 2
    assert second["submissions"] == (2 if repair else 1)
    assert second["last_feedback"]["geometry_gate_passed"] == repair
    if not repair:
        assert second["last_feedback"]["source_unchanged"]
        assert not second["last_feedback"]["compiled_this_call"]


def test_contract_audit_reports_all_placeholders_without_rewriting():
    source = "from build123d import Axis\nparent.at(Axis(anchor=(0,0,0),direction=(1,0,0)))\nchild.at()\n"
    notes = audit_program(source)
    messages = "\n".join(n["message"] for n in notes)
    assert "parent" in messages and "child" in messages
    assert "object_model" in messages and "anchor keyword" in messages
    assert source.endswith("child.at()\n")
