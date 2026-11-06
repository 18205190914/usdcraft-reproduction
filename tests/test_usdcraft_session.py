from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import trimesh
from PIL import Image

from articraft.source_geometry import analyze_source
from articraft.usdcraft_benchmark import render_meshes
from articraft.usdcraft_session import evidence_views, plane_feedback, prepare, submit


def test_common_render_frame_preserves_translation(tmp_path):
    source = trimesh.creation.box(extents=[0.1, 0.08, 0.06])
    shifted = source.copy().apply_translation([0.06, 0, 0])
    frame = np.array([[-0.1, -0.1, -0.1], [0.15, 0.1, 0.1]])
    first, second = tmp_path / "first.png", tmp_path / "second.png"
    render_meshes([source], first, 100, frame_bounds=frame)
    render_meshes([shifted], second, 100, frame_bounds=frame)
    assert not np.array_equal(np.asarray(Image.open(first)), np.asarray(Image.open(second)))


def test_multiview_receipt_has_source_closed_and_open_states(tmp_path):
    base = trimesh.creation.box(extents=[0.1, 0.08, 0.06])
    lid = trimesh.creation.box(extents=[0.1, 0.08, 0.004]).apply_translation([0, 0, 0.032])
    assembly = {
        "joints": [
            {
                "name": "hinge",
                "parent": "base",
                "child": "lid",
                "type": "revolute",
                "axis": [1, 0, 0],
                "origin": [0, -0.04, 0.03],
                "limits": [0, 1.5],
            }
        ]
    }
    report = evidence_views(
        trimesh.util.concatenate([base, lid]), tmp_path, {"base": base, "lid": lid}, assembly
    )
    assert len(report["files"]) == 9
    assert report["part_color_order"] == ["base", "lid"]
    for item in report["files"]:
        assert hashlib.sha256(Path(item["path"]).read_bytes()).hexdigest() == item["sha256"]


def test_session_preserves_source_and_records_actual_compile_failure(tmp_path):
    public = tmp_path / "public"
    public.mkdir()
    trimesh.creation.box(extents=[0.1, 0.08, 0.06]).export(public / "source.ply")
    (public / "input.json").write_text(
        json.dumps(
            {
                "source": "source.ply",
                "units": "metres",
                "request": "Rebuild from public observations",
            }
        )
    )
    run = prepare(public, tmp_path / "runs", "scripted-unit-control")
    request = json.loads((run / "model-request.json").read_text())
    assert "private_truth" not in request["source_observations"]
    source = tmp_path / "answer.py"
    source.write_text("```python\nbroken =\n```")
    notes = tmp_path / "notes.json"
    notes.write_text(json.dumps({"scope": "unit control only"}))
    result = submit(run, source, notes, "scripted-unit-control")
    assert result["compiler_status"] == "error"
    assert "SyntaxError" in result["error"]
    compiled = run / "result/attempt-01/main.py"
    assert compiled.read_text() == "broken =\n"
    receipt = json.loads((compiled.parent / "agent-receipt.json").read_text())
    assert receipt["source_sha256"] == hashlib.sha256(compiled.read_bytes()).hexdigest()


def test_plane_feedback_separates_correct_and_shifted_candidates():
    mesh = trimesh.creation.box(extents=[0.1, 0.08, 0.06])
    _, _, analysis = analyze_source(mesh, 48)
    assert plane_feedback(analysis, mesh)["total_missing_plane_groups"] == 0
    shifted = mesh.copy().apply_translation([0, 0, 0.02])
    result = plane_feedback(analysis, shifted)
    assert result["total_missing_plane_groups"] >= 2
    assert any(p["axis"] == "Z" for p in result["missing_or_displaced_observed_planes"])
