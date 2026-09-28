"""Image metrics of renders against the real sample frames, per camera and per region.

    ./robot real2sim look score <render_dir> [--pattern '{id}_{cam}.png'] [--erode 2] [--out f.json]

A renderer writes one 640x480 8-bit sRGB PNG per sample and camera, named by the
sample id (samples/index.json), and gets back, per camera and per region
(background, arm, fingers, cap, mug, plus 'all'):

    psnr        dB over the region's pixels (real2sim.metrics.masked_psnr), capped at 99
    ssim        mean SSIM over the region: Gaussian window sigma 1.5, 11x11, K 0.01/0.03,
                as real2sim.metrics.masked_ssim. The map is computed once with cv2
                (tests check it against the core to 1e-6), then averaged per region.
    rgb_bias    mean(render - real) in LINEAR light per channel: exposure / white balance
    rgb_mae     mean |render - real| in linear light
    delta_e     mean CIE76 dE between the region's MEAN colours (a colour-calibration
                error, blind to texture), and the per-pixel mean dE
    lpips       only if the `lpips` package imports (not installed here; nothing is
                installed on its behalf)

Each region is eroded by `erode` px first, so sub-pixel boundary misregistration does
not dominate. A region with fewer than 30 pixels after erosion is reported as None.
Renders are compared as delivered. To judge textures apart from the camera's
white-balance drift, pass --wb, which divides the real front image by its recorded
gain and multiplies back the reference, i.e. it scores against the LOOK reference
state.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from .. import paths
from . import LABELS, colour

MIN_PX = 30
PSNR_CAP = 99.0  # dB: identical pixels give infinity, which would poison every mean


def load_index(ds=None) -> tuple[dict, Path]:
    d = paths.out_dir(ds, "look", "samples")
    p = d / "index.json"
    if not p.exists():
        raise FileNotFoundError(f"{p} missing: run ./robot real2sim look build")
    return json.loads(p.read_text()), d


def read_rgb(path) -> np.ndarray:
    import cv2

    im = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if im is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(im, cv2.COLOR_BGR2RGB)


def region_masks(labels, erode_px: int = 2) -> dict:
    from .segment import erode

    out = {name: erode(labels == v, erode_px) for name, v in LABELS.items()}
    out["all"] = np.ones(labels.shape, bool)
    return out


def ssim_map(a, b, peak: float = 255.0, sigma: float = 1.5, k1: float = 0.01, k2: float = 0.03) -> np.ndarray:
    """Per-pixel SSIM averaged over channels; 11x11 Gaussian window with reflect borders,
    the same statistics as real2sim.metrics.masked_ssim."""
    import cv2

    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    if a.ndim == 2:
        a, b = a[..., None], b[..., None]
    blur = lambda x: cv2.GaussianBlur(x, (11, 11), sigma, borderType=cv2.BORDER_REFLECT)  # noqa: E731
    c1, c2 = (k1 * peak) ** 2, (k2 * peak) ** 2
    maps = []
    for ch in range(a.shape[2]):
        x, y = a[..., ch], b[..., ch]
        mx, my = blur(x), blur(y)
        sxx, syy, sxy = blur(x * x) - mx * mx, blur(y * y) - my * my, blur(x * y) - mx * my
        maps.append(((2 * mx * my + c1) * (2 * sxy + c2)) / ((mx * mx + my * my + c1) * (sxx + syy + c2)))
    return np.mean(maps, 0)


def _lpips():
    """LPIPS (AlexNet) if the lpips package and torch import; None otherwise. Nothing
    is installed or downloaded on its behalf."""
    try:
        import lpips
    except ImportError:
        return None
    return lpips.LPIPS(net="alex", verbose=False)


def compare(real, render, labels, erode_px: int = 2, lpips_model=None) -> dict:
    """{region: {psnr, ssim, rgb_bias, rgb_mae, delta_e_mean_colour, delta_e_pixels, n}}"""
    from ..metrics import masked_psnr

    real, render = np.asarray(real), np.asarray(render)
    if real.shape != render.shape:
        raise ValueError(f"render {render.shape} != real {real.shape}")
    sm = ssim_map(real, render)
    lr, lo = colour.srgb_to_linear(real), colour.srgb_to_linear(render)
    lab_r, lab_o = colour.linear_to_lab(lr), colour.linear_to_lab(lo)
    out = {}
    for name, m in region_masks(labels, erode_px).items():
        n = int(m.sum())
        if n < MIN_PX:
            out[name] = None
            continue
        d = lo[m] - lr[m]
        mr, mo = lr[m].mean(0), lo[m].mean(0)
        out[name] = {"n": n, "psnr": round(min(masked_psnr(real, render, m), PSNR_CAP), 3),
                     "ssim": round(float(sm[m].mean()), 4),
                     "rgb_bias": np.round(d.mean(0), 5).tolist(), "rgb_mae": np.round(np.abs(d).mean(0), 5).tolist(),
                     "delta_e_mean_colour": round(float(colour.delta_e76(colour.linear_to_lab(mr), colour.linear_to_lab(mo))), 3),
                     "delta_e_pixels": round(float(colour.delta_e76(lab_r[m], lab_o[m]).mean()), 3)}
    if lpips_model is not None:
        import torch

        t = lambda x: torch.from_numpy(x.astype(np.float32) / 127.5 - 1).permute(2, 0, 1)[None]  # noqa: E731
        with torch.no_grad():
            out["all"]["lpips"] = round(float(lpips_model(t(real), t(render)).item()), 4)
    return out


def _mean_rows(rows: list) -> dict:
    keys = ("psnr", "ssim", "delta_e_mean_colour", "delta_e_pixels")
    res = {}
    for region in list(LABELS) + ["all"]:
        vals = [r[region] for r in rows if r.get(region)]
        if not vals:
            res[region] = None
            continue
        res[region] = {k: round(float(np.mean([v[k] for v in vals])), 4) for k in keys}
        res[region]["rgb_bias"] = np.round(np.mean([v["rgb_bias"] for v in vals], 0), 5).tolist()
        res[region]["samples"] = len(vals)
    return res


def score(render_dir, ds=None, pattern: str = "{id}_{cam}.png", erode_px: int = 2, wb: bool = False,
          renders=None) -> dict:
    """Score a directory of renders (or a callable renders(sample, cam) -> RGB) against
    the sample set. Missing renders are listed, not fatal."""
    index, sdir = load_index(ds)
    model = _lpips()
    per, missing = {"front": [], "grip": []}, []
    rows = []
    for s in index["samples"]:
        for cam in ("front", "grip"):
            real = read_rgb(sdir / s[cam]["image"])
            if wb and cam == "front":
                g = np.asarray(s[cam]["white_balance_gain"], np.float32)
                real = colour.linear_to_srgb(colour.srgb_to_linear(real) / g)
            labels = read_rgb(sdir / s[cam]["labels"])[..., 0]
            if renders is not None:
                ren = renders(s, cam)
            else:
                p = Path(render_dir) / pattern.format(id=s["id"], cam=cam)
                if not p.exists():
                    missing.append(p.name)
                    continue
                ren = read_rgb(p)
            if ren is None:
                missing.append(f"{s['id']}_{cam}")
                continue
            r = compare(real, ren, labels, erode_px, model)
            per[cam].append(r)
            rows.append({"id": s["id"], "cam": cam, "phase": s["phase"], **r})
    return {"dataset": index["dataset"], "scene_hash_of_samples": index["scene_hash"], "erode_px": erode_px,
            "white_balance_normalised": wb, "lpips": model is not None,
            "summary": {cam: _mean_rows(v) for cam, v in per.items()}, "missing": missing, "per_sample": rows}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="./robot real2sim look score", description=__doc__.split("\n\n")[0])
    ap.add_argument("render_dir")
    ap.add_argument("--pattern", default="{id}_{cam}.png")
    ap.add_argument("--erode", type=int, default=2)
    ap.add_argument("--wb", action="store_true", help="score against the white-balance reference state")
    ap.add_argument("--out", help="write the full JSON here (default: <render_dir>/look_score.json)")
    ap.add_argument("--ds")
    a = ap.parse_args(argv)
    res = score(a.render_dir, a.ds, a.pattern, a.erode, a.wb)
    out = Path(a.out or Path(a.render_dir) / "look_score.json")
    out.write_text(json.dumps(res, indent=1))
    for cam, summ in res["summary"].items():
        print(f"{cam}:")
        for region, v in summ.items():
            if v:
                print(f"  {region:10s} psnr {v['psnr']:6.2f}  ssim {v['ssim']:.3f}  dE(mean colour) "
                      f"{v['delta_e_mean_colour']:6.2f}  dE(px) {v['delta_e_pixels']:6.2f}  n={v['samples']}")
    if res["missing"]:
        print(f"missing {len(res['missing'])} renders, e.g. {res['missing'][:3]}", file=sys.stderr)
    print(f"-> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
