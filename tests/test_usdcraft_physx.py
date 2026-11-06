"""Evidence transport tests only; these fixtures are not physics experiments."""

import json

import pytest

from articraft.usdcraft_physx import attach_native_feedback, sha256, summarize_native


@pytest.fixture
def evidence(tmp_path):
    asset = tmp_path / "candidate.usdz"
    asset.write_bytes(b"transport fixture, not USD")
    profile = tmp_path / "profile.json"
    profile.write_text(json.dumps({"scenarios": ["drop", "drive_cycle"], "repeats": 1}))
    rows = []
    for scenario in ["drop", "drive_cycle"]:
        detail = tmp_path / (scenario + ".json")
        detail.write_text(
            json.dumps(
                {
                    "scenario": scenario,
                    "asset_sha256": sha256(asset),
                    "profile_sha256": sha256(profile),
                    "profile": json.loads(profile.read_text()),
                    "engine": "omni.physx native CPU solver",
                    "steps": 2,
                    "passed": True,
                    "full_native_gate_passed": True,
                    "errors": [],
                    "adjacent_collision_policy": "standard-joints-enabled",
                    "peak_floor_penetration_m": 0,
                    "resting_floor_penetration_m": 0,
                    "max_joint_anchor_error_m": 0,
                }
            )
        )
        rows.append(
            {
                "case": "fixture",
                "scenario": scenario,
                "repeat": 1,
                "passed": True,
                "result": detail.name,
            }
        )
    index = tmp_path / "index.json"
    index.write_text(json.dumps(rows))
    return index, asset, profile


def test_rejects_stale_asset_version(evidence):
    index, asset, profile = evidence
    asset.write_bytes(b"changed after simulation")
    with pytest.raises(ValueError, match="another asset"):
        summarize_native(index, asset, profile, "fixture")


def test_rejects_solver_profile_change(evidence):
    index, asset, profile = evidence
    profile.write_text(profile.read_text() + "\n")
    with pytest.raises(ValueError, match="another solver profile"):
        summarize_native(index, asset, profile, "fixture")


def test_rejects_missing_repeat_and_duplicate(evidence):
    index, asset, profile = evidence
    rows = json.loads(index.read_text())
    index.write_text(json.dumps([rows[0], rows[0]]))
    with pytest.raises(ValueError, match="incomplete or duplicated"):
        summarize_native(index, asset, profile, "fixture")


def test_failed_collision_policy_is_not_full_acceptance(evidence):
    index, asset, profile = evidence
    path = index.parent / "drop.json"
    report = json.loads(path.read_text())
    report.update(
        full_native_gate_passed=False,
        adjacent_collision_policy="articulation-parent-child-collision-ignored",
    )
    path.write_text(json.dumps(report))
    summary = summarize_native(index, asset, profile, "fixture")
    assert summary["dynamics_passed"]
    assert not summary["passed"]


def test_attach_requires_lineage_and_preserves_compile_feedback(evidence):
    index, asset, profile = evidence
    run = index.parent / "run"
    folder = run / "result/attempt-01"
    folder.mkdir(parents=True)
    original = index.parent / "original.usdz"
    original.write_bytes(b"original fixture")
    (folder / "compile.json").write_text(json.dumps({"usdz": str(original)}))
    feedback = folder / "feedback.json"
    feedback.write_text(json.dumps({"combined_gate_passed": False}))
    before = feedback.read_bytes()
    with pytest.raises(ValueError, match="lineage"):
        attach_native_feedback(run, 1, index, asset, profile, "fixture")
    manifest = index.parent / "manifest.json"
    manifest.write_text(
        json.dumps([{"source_sha256": sha256(original), "output_sha256": sha256(asset)}])
    )
    result = attach_native_feedback(run, 1, index, asset, profile, "fixture", manifest)
    assert not result["full_gate_passed"]
    assert result["native_physics"]["passed"]
    assert feedback.read_bytes() == before
