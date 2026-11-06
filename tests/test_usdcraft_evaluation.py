from __future__ import annotations

import numpy as np

from articraft.usdcraft_benchmark import reference_cases
from articraft.usdcraft_evaluation import axis_error, axis_line_distance, evaluate_assembly


def test_axis_sign_and_parallel_line_distance():
    assert axis_error(np.array([1, 0, 0]), np.array([-1, 0, 0])) == 0
    assert np.isclose(
        axis_line_distance(
            np.zeros(3), np.array([1, 0, 0]), np.array([0, 0.02, 0]), np.array([1, 0, 0])
        ),
        0.02,
    )


def fixture():
    case = reference_cases()["hinged_box"]
    return {"parts": [{"name": n} for n, _ in case["parts"]], "joints": case["joints"]}, dict(
        case["parts"]
    )


def test_identity_and_part_order_do_not_change_metrics():
    gt, parts = fixture()
    prediction = {**gt, "parts": list(reversed(gt["parts"]))}
    report = evaluate_assembly(
        gt, prediction, parts, dict(reversed(list(parts.items()))), samples=2000
    )
    assert report["F1_percent"] == 100
    assert report["AE_degrees"] == 0
    assert report["LE"] == 0
    assert np.isclose(report["static_gIoU"], 1)


def test_failed_output_is_not_discarded():
    gt, parts = fixture()
    report = evaluate_assembly(gt, {"parts": [], "joints": []}, parts, {}, samples=1000)
    assert report["F1_percent"] == 0
    assert report["AE_degrees"] == 90
    assert report["LE"] == 0.5


def test_wrong_joint_type_gets_axis_penalty():
    gt, parts = fixture()
    prediction = {**gt, "joints": [{**gt["joints"][0], "type": "prismatic"}]}
    report = evaluate_assembly(gt, prediction, parts, parts, samples=1000)
    assert report["F1_percent"] == 0
    assert report["AE_degrees"] == 90
    assert report["LE"] == 0.5


def test_f1_angle_threshold_does_not_replace_measured_axis_error():
    gt, parts = fixture()
    prediction = {**gt, "joints": [{**gt["joints"][0], "axis": [np.cos(0.4), np.sin(0.4), 0]}]}
    report = evaluate_assembly(gt, prediction, parts, parts, samples=1000)
    assert report["F1_percent"] == 0
    assert np.isclose(report["AE_degrees"], np.degrees(0.4))


def test_reference_frame_retains_global_translation_error():
    gt, parts = fixture()
    moved = {n: m.copy().apply_translation([0.03, 0, 0]) for n, m in parts.items()}
    prediction = {
        **gt,
        "joints": [
            {
                **gt["joints"][0],
                "origin": [
                    x + 0.03 if i == 0 else x for i, x in enumerate(gt["joints"][0]["origin"])
                ],
            }
        ],
    }
    independent = evaluate_assembly(gt, prediction, parts, moved, samples=1000)
    anchored = evaluate_assembly(gt, prediction, parts, moved, samples=1000, reference_frame=True)
    assert np.isclose(independent["static_gIoU"], 1)
    assert anchored["static_gIoU"] < independent["static_gIoU"]
