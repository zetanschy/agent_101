"""The built assets: present, consistent with look.json, labels valid."""
import json

import numpy as np

from real2sim import paths
from real2sim.look import LABELS


def test_look_json_has_every_section(built):
    for k in ("cameras", "lights", "materials", "textures", "plates", "samples", "baselines", "inputs_hash"):
        assert k in built, k
    key = built["lights"]["key"]
    assert 0 < key["elevation_deg"] < 90 and 0 < key["angular_radius_deg"] < 30
    for name, m in built["materials"].items():
        assert "source" in m, name


def test_files_exist(built):
    out = paths.out_dir(built["dataset"], "look")
    for sub, files in (("table", built["textures"]["table"]["files"]), ("backdrop", built["textures"]["backdrop_cylinder"]["files"]),
                       ("plates", built["plates"]["files"])):
        for f in files.values():
            assert (out / sub / f).exists(), f"{sub}/{f}"


def test_samples_index_and_labels(built):
    import cv2

    d = paths.out_dir(built["dataset"], "look", "samples")
    ix = json.loads((d / "index.json").read_text())
    assert ix["labels"] == LABELS and len(ix["samples"]) == 24
    ids = [s["id"] for s in ix["samples"]]
    assert len(set(ids)) == len(ids)
    for s in ix["samples"][::5]:
        for cam in ("front", "grip"):
            lab = cv2.imread(str(d / s[cam]["labels"]), cv2.IMREAD_UNCHANGED)
            img = cv2.imread(str(d / s[cam]["image"]))
            assert lab.shape == img.shape[:2] == (480, 640)
            assert set(np.unique(lab)) <= set(LABELS.values())
            assert s[cam]["label_px"]["background"] == int((lab == 0).sum())
