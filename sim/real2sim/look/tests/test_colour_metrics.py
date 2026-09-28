"""Colour transfer, the wrist colour model, and the image metrics against the core's."""
import numpy as np
import pytest

from real2sim import metrics
from real2sim.look import LABELS, colour, imagemetrics


def test_srgb_round_trip():
    u8 = np.arange(256, dtype=np.uint8)
    assert np.array_equal(colour.linear_to_srgb(colour.srgb_to_linear(u8)), u8)
    assert colour.srgb_to_linear(np.uint8(255)) == pytest.approx(1.0)


def test_lab_of_white_and_black():
    assert np.allclose(colour.linear_to_lab([1.0, 1.0, 1.0]), [100, 0, 0], atol=1e-3)
    assert np.allclose(colour.linear_to_lab([0.0, 0.0, 0.0]), [0, 0, 0], atol=1e-6)


def test_grip_front_model_inverts_and_keeps_luminance():
    rng = np.random.default_rng(0)
    L = np.array([0.606, 0.691, 0.603])
    wb, s, sigma = np.array([0.93, 1.0, 0.81]), 1.36, 1.14
    front = rng.uniform(0.01, 0.6, (500, 3))
    grip = colour.front_to_grip(front, wb, s, sigma, L)
    assert np.allclose(colour.grip_to_front(grip, wb, s, sigma, L), front, atol=1e-5)
    # the white reference maps to itself up to exposure; saturation leaves white alone
    assert np.allclose(colour.saturate(np.ones(3), 2.5), 1.0)
    a = front / L
    assert np.allclose(colour.saturate(a, sigma) @ colour.LUMA, a @ colour.LUMA)


def test_ssim_map_matches_core():
    rng = np.random.default_rng(1)
    a = rng.integers(0, 256, (60, 80, 3)).astype(np.uint8)
    b = np.clip(a.astype(int) + rng.integers(-30, 31, a.shape), 0, 255).astype(np.uint8)
    m = np.zeros((60, 80), bool)
    m[10:50, 5:70] = True
    assert imagemetrics.ssim_map(a, b)[m].mean() == pytest.approx(metrics.masked_ssim(a, b, m), abs=1e-6)


def test_compare_identity_and_regions():
    rng = np.random.default_rng(2)
    img = rng.integers(0, 256, (48, 64, 3)).astype(np.uint8)
    lab = np.zeros((48, 64), np.uint8)
    lab[5:30, 5:30] = LABELS["arm"]
    r = imagemetrics.compare(img, img, lab, erode_px=2)
    assert r["all"]["psnr"] == imagemetrics.PSNR_CAP and r["all"]["ssim"] == pytest.approx(1.0)
    assert r["arm"]["n"] == (30 - 5 - 4) ** 2 and r["cap"] is None  # eroded 2 px each side; no cap pixels
    brighter = np.clip(img.astype(int) + 20, 0, 255).astype(np.uint8)
    assert np.all(np.array(imagemetrics.compare(img, brighter, lab)["all"]["rgb_bias"]) > 0)
