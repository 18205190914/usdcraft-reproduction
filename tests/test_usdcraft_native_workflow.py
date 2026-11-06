"""Orchestration controls; mocked native calls are not physics evidence."""

from __future__ import annotations

import json

import pytest
import trimesh

from articraft.agent.workspace import LocalWorkspace
from articraft.usdcraft_session import prepare, submit
from articraft.usdcraft_workflow import ModelingTools


def public_input(directory):
    directory.mkdir()
    trimesh.creation.box([0.1, 0.08, 0.06]).export(directory / "source.ply")
    (directory / "input.json").write_text(json.dumps({"source": "source.ply", "request": "test"}))
    return directory


def native_config(directory):
    directory.mkdir()
    for name in [
        "apply_compatibility.py",
        "orthogonal_compound.py",
        "validate_asset.py",
        "launch_native.py",
    ]:
        (directory / name).write_text("# unit boundary fixture\n")
    (directory / "profile.json").write_text("{}")
    path = directory / "config.json"
    path.write_text(
        json.dumps({"script_dir": str(directory), "profile": str(directory / "profile.json")})
    )
    return path


def test_same_source_reuses_compile_but_changed_profile_revalidates(tmp_path, monkeypatch):
    public = public_input(tmp_path / "public")
    profile = tmp_path / "local.json"
    profile.write_text("{}")
    run = prepare(public, tmp_path / "runs", "unit-control", geometry_profile=profile)
    source, notes = tmp_path / "answer.py", tmp_path / "notes.json"
    source.write_text("broken =\n")
    notes.write_text("{}")
    first = submit(run, source, notes, "unit-control")
    assert first["compiler_status"] == "error"
    original = (run / "result/attempt-01/compile.json").read_bytes()

    def forbidden_compile(*args):
        raise AssertionError("Identical source must not compile again")

    monkeypatch.setattr(LocalWorkspace, "compile_path", forbidden_compile)
    cached = submit(run, source, notes, "unit-control")
    assert cached["source_unchanged"] and not cached["compiled_this_call"]
    (run / "geometry_profile.json").write_text('{"samples": 10000}')
    refreshed = submit(run, source, notes, "unit-control")
    assert refreshed["resumed_existing_submission"] and not refreshed["compiled_this_call"]
    assert (run / "result/attempt-01/compile.json").read_bytes() == original
    assert len(list((run / "result").glob("attempt-*"))) == 1


def test_execution_error_can_be_retried_without_a_cad_change(tmp_path, monkeypatch):
    public = public_input(tmp_path / "public")
    run = prepare(public, tmp_path / "runs", "unit-control")
    source, notes = tmp_path / "answer.py", tmp_path / "notes.json"
    source.write_text("broken =\n")
    notes.write_text("{}")
    submit(run, source, notes, "unit-control")
    feedback_file = run / "result/attempt-01/feedback.json"
    control = json.loads(feedback_file.read_text())
    control["native_physics"] = {"status": "execution_error", "passed": False}
    feedback_file.write_text(json.dumps(control))

    def forbidden_compile(*args):
        raise AssertionError("Infrastructure retry must reuse its compiled program")

    monkeypatch.setattr(LocalWorkspace, "compile_path", forbidden_compile)
    result = submit(run, source, notes, "unit-control")
    assert result["resumed_existing_submission"] and not result["compiled_this_call"]
    assert result["compiler_status"] == "error"
    assert len(list((run / "result").glob("attempt-*"))) == 1


def test_native_profile_change_refreshes_same_program_without_recompile(tmp_path, monkeypatch):
    public = public_input(tmp_path / "public")
    native = native_config(tmp_path / "native")
    run = prepare(public, tmp_path / "runs", "unit-control", native_config=native)
    source, notes = tmp_path / "answer.py", tmp_path / "notes.json"
    source.write_text("broken =\n")
    notes.write_text("{}")
    submit(run, source, notes, "unit-control")

    def forbidden_compile(*args):
        raise AssertionError("Solver profile change must revalidate without compiling")

    monkeypatch.setattr(LocalWorkspace, "compile_path", forbidden_compile)
    (native.parent / "profile.json").write_text('{"version": "changed-unit-control"}')
    refreshed = submit(run, source, notes, "unit-control")
    assert refreshed["resumed_existing_submission"] and not refreshed["compiled_this_call"]
    assert len(list((run / "result").glob("attempt-*"))) == 1


@pytest.mark.parametrize("local_pass,native_pass", [(False, True), (True, False), (True, True)])
def test_native_is_gated_by_local_geometry_and_controls_final_acceptance(
    tmp_path, monkeypatch, local_pass, native_pass
):
    import articraft.usdcraft_workflow as workflow

    public = public_input(tmp_path / "public")
    run = tmp_path / "run"
    run.mkdir()
    profile, native = tmp_path / "local.json", native_config(tmp_path / "native")
    profile.write_text("{}")
    host = ModelingTools(public, None, run, native, profile)
    host.attempt = 1
    output = run / "attempt-01"
    output.mkdir()
    monkeypatch.setattr(
        workflow, "read_usd_asset", lambda *a: ({"joints": [{}]}, {"body": host.source})
    )
    monkeypatch.setattr(workflow, "check_motion", lambda *a: {"passed": True})
    monkeypatch.setattr(workflow, "motion_summary", lambda value: value)
    monkeypatch.setattr(workflow, "diagnose_geometry", lambda *a, **kw: {"passed": local_pass})
    monkeypatch.setattr(workflow, "render_meshes", lambda *a: None)
    calls = []

    def native_control(*args):
        calls.append(args)
        return {"passed": native_pass, "scope": "mock transport control only"}

    monkeypatch.setattr(workflow, "validate_native", native_control)
    result = host.evaluate({"status": "success", "usdz": "fixture.usdz"}, output)
    assert bool(calls) == local_pass
    assert result["geometry_motion_gate_passed"]
    assert result["combined_gate_passed"] == (local_pass and native_pass)
    assert result["full_gate_passed"] == (local_pass and native_pass)
