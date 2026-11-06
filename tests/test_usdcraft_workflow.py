from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import numpy as np
import trimesh

from articraft.agent.workspace import LocalWorkspace
from articraft.usdcraft_io import read_usd_asset
from articraft.usdcraft_workflow import run_case


def test_compiled_usdz_recovers_world_axis_units_and_limits(tmp_path):
    workspace = LocalWorkspace(output_dir=tmp_path, physics_enabled=True)
    run = workspace.create_run("usdcraft-io-test")
    source = """from build123d import Box, Pos, Axis
from articraft.sdk import JointAxis, JointDOF, Material, RigidBodyAssembly, TestContext
model = RigidBodyAssembly("test")
base = model.rigid_body("base")
base.add(Pos(0,0,.005)*Box(.1,.1,.01), name="base", material=Material.ABS_PLASTIC)
lid = model.rigid_body("lid")
lid.add(Pos(0,.04,.0125)*Box(.1,.08,.005), name="lid", material=Material.ABS_PLASTIC)
axis = Axis((0,0,.01),(1,0,0))
model.joint("hinge", base.at(axis), lid.at(axis), dofs=(JointDOF(JointAxis.ROT_Z,limits=(0,1.5)),))
model.articulation("main", root=base, joints=["hinge"])
object_model = model
def run_tests():
    return TestContext(object_model).report()
"""
    (run / "workspace/main.py").write_text(source)
    result = workspace.compile_path(run)
    assert result["status"] == "success", result.get("error")
    assembly, parts = read_usd_asset(run / result["usdz"], tmp_path / "read")
    assert set(parts) == {"base", "lid"}
    joint = assembly["joints"][0]
    assert np.allclose(joint["axis"], [1, 0, 0], atol=1e-6)
    assert np.allclose(joint["origin"], [0, 0, 0.01])
    assert np.allclose(joint["limits"], [0, 1.5])
    assert np.allclose(parts["base"].extents, [0.1, 0.1, 0.01])


def test_source_program_feedback_loop_without_private_truth(tmp_path):
    """Integration/contract test only; scripted corrections are not model quality evidence."""
    public = tmp_path / "public"
    public.mkdir()
    a = trimesh.creation.box(extents=[0.1, 0.1, 0.01]).apply_translation([0, 0, 0.005])
    b = trimesh.creation.box(extents=[0.1, 0.08, 0.005]).apply_translation([0, 0.04, 0.0125])
    trimesh.util.concatenate([a, b]).export(public / "source.ply")
    (public / "input.json").write_text(
        json.dumps(
            {"source": "source.ply", "units": "metres", "request": "Rebuild the hinged panels."}
        )
    )
    program = """from build123d import Box, Pos, Axis
from articraft.sdk import JointAxis, JointDOF, Material, RigidBodyAssembly, TestContext
m = RigidBodyAssembly("panels")
a = m.rigid_body("base")
a.add(Pos(0,0,.005)*Box(.1,.1,.01), name="a", material=Material.ABS_PLASTIC)
b = m.rigid_body("lid")
b.add(Pos(0,.04,.0125)*Box(.1,.08,.005), name="b", material=Material.ABS_PLASTIC)
axis = Axis((0,0,.01),(1,0,0))
m.joint("hinge",a.at(axis),b.at(axis),dofs=(JointDOF(JointAxis.ROT_Z,limits=(0,1.5)),))
m.articulation("main",root=a,joints=["hinge"])
object_model = m
def run_tests():
    return TestContext(object_model).report()
"""

    class ScriptedModel:
        config = SimpleNamespace(openrouter_model="scripted-no-network")

        def __init__(self):
            self.calls = 0

        async def query(self, messages, *, tools):
            self.calls += 1
            if self.calls < 3:
                name, arguments = "measure_mesh", {}
            else:
                name = "submit_program"
                if self.calls == 4:
                    assert "SyntaxError" in messages[-1]["output"]
                arguments = {"source": "broken =" if self.calls == 3 else program}
            return {
                "text": "",
                "tool_calls": [
                    {"id": f"call-{self.calls}", "name": name, "arguments": json.dumps(arguments)}
                ],
                "response": {"choices": [{"finish_reason": "tool_calls"}]},
            }

    result = asyncio.run(
        run_case(
            ScriptedModel(),
            LocalWorkspace(output_dir=tmp_path / "runs", physics_enabled=True),
            public,
            max_calls=4,
        )
    )
    assert result["submissions"] == 2
    assert result["last_feedback"]["geometry_gate_passed"]
    assert not result["private_truth_sent_to_model"]
