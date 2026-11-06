from __future__ import annotations

import warnings

import numpy as np
import trimesh

from articraft.usdcraft_motion import solid_overlaps


def test_zero_volume_boolean_sheet_does_not_request_mass_centroid(monkeypatch):
    first = trimesh.creation.box()
    second = trimesh.creation.box().apply_translation([0.2, 0, 0])
    sheet = trimesh.Trimesh(
        vertices=[[0, 0, 0], [0, 1, 0], [0, 1, 1], [0, 0, 1]],
        faces=[[0, 1, 2], [0, 2, 3], [2, 1, 0], [3, 2, 0]],
        process=False,
    )
    monkeypatch.setattr(trimesh.boolean, "intersection", lambda *args, **kwargs: sheet)
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        assert solid_overlaps({"first": first, "second": second}) == []


def test_overlap_volume_is_stable_far_from_origin():
    first = trimesh.creation.box().apply_translation([10000, 10000, 10000])
    second = first.copy().apply_translation([0.5, 0, 0])
    report = solid_overlaps({"first": first, "second": second})
    assert np.isclose(report[0]["intersection_volume"], 0.5)
    assert np.isclose(report[0]["fraction_of_smaller_part"], 0.5)
