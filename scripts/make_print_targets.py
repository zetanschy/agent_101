#!/usr/bin/env python3
"""Generate everything that has to be printed, at exact scale.

    ./robot print-targets

Three sheets, all A4, all with a 100 mm scale-check bar:

  checkerboard   camera INTRINSICS (./robot sim-calibrate). 9x6 inner corners.
  charuco        camera EXTRINSICS (./robot calib-capture). Survives partial
                 occlusion and gives a pose from one view, which a plain
                 checkerboard cannot.
  push-t goal    the task target for the real mat, drawn from the same STL the sim
                 uses -- 80 x 80 mm outline, 20 mm bar and stem.

PRINT AT 100%. Every dialog defaults to "fit to page", which silently scales by a few
percent; that error goes straight into the calibration as a scale error on every
translation. Then measure the 100 mm bar and, if it is not 100 mm, pass what you
actually measured to --square so the solver knows the truth.
"""

import argparse
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages  # noqa: E402
from matplotlib.patches import Polygon, Rectangle  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT = ROOT / "sim" / "outputs" / "print"


def _tblock():
    """The same geometry the sim uses. Loaded by path because the package __init__
    pulls in isaaclab, which is not present on the host."""
    import importlib.util
    import sys as _sys
    path = ROOT / "sim" / "sim_agent101" / "tblock.py"
    spec = importlib.util.spec_from_file_location("_sim_agent101_tblock", path)
    mod = importlib.util.module_from_spec(spec)
    _sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod.T_BLOCK_GEOMETRY
MM = 1 / 25.4  # matplotlib works in inches


A4 = (210.0, 297.0)


def _sheet(landscape: bool = False, need=None):
    """Always a real A4 page.

    Sizing the page to the content was the bug: it produced 230x325 and 280x297 mm
    sheets, which no printer can output at 100% on A4, so every print silently
    scaled and every calibrated distance would have come out wrong.
    """
    w_mm, h_mm = (A4[1], A4[0]) if landscape else A4
    if need is not None:
        nw, nh = need
        if nw > w_mm - 14 or nh > h_mm - 32:
            raise SystemExit(
                f"content {nw:.0f}x{nh:.0f} mm does not fit A4"
                f"{' landscape' if landscape else ''} ({w_mm:.0f}x{h_mm:.0f}) with margins "
                f"and a scale bar -- reduce --square or the square count")
    fig = plt.figure(figsize=(w_mm * MM, h_mm * MM))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, w_mm); ax.set_ylim(0, h_mm)
    ax.set_aspect("equal"); ax.axis("off")
    return fig, ax


def _scalebar(ax, x, y, label):
    ax.plot([x, x + 100], [y, y], color="black", lw=1.2)
    for t in (x, x + 100):
        ax.plot([t, t], [y - 2, y + 2], color="black", lw=1.2)
    ax.text(x + 50, y + 4, "100 mm — measure this. If it is not 100 mm, the print is scaled.",
            ha="center", va="bottom", fontsize=7)
    ax.text(x, y - 8, label, ha="left", va="top", fontsize=7, color="0.35")


def checkerboard(cols_inner=9, rows_inner=6, square=25.0):
    nx, ny = cols_inner + 1, rows_inner + 1
    w, h = nx * square, ny * square
    # 10x7 squares at 25 mm is 250x175 -- lands on A4 only in landscape
    fig, ax = _sheet(landscape=True, need=(w, h))
    x0 = (ax.get_xlim()[1] - w) / 2
    y0 = ax.get_ylim()[1] - h - 30
    for i in range(nx):
        for j in range(ny):
            if (i + j) % 2 == 0:
                ax.add_patch(Rectangle((x0 + i * square, y0 + j * square), square, square,
                                       facecolor="black", edgecolor="none"))
    ax.add_patch(Rectangle((x0, y0), w, h, fill=False, edgecolor="0.6", lw=0.4))
    _scalebar(ax, x0, y0 - 18,
              f"checkerboard {cols_inner}x{rows_inner} inner corners, {square:g} mm squares  "
              f"->  ./robot sim-calibrate --camera front --cols {cols_inner} --rows {rows_inner} --square {square:g}")
    return fig


def push_t():
    """The T's footprint, straight from the geometry the simulation uses."""
    g = _tblock()
    bar_w = g.bar_width * 1000
    bar_d = g.bar_depth * 1000
    stem_w = g.stem_width * 1000
    stem_l = g.stem_length * 1000
    fig, ax = _sheet()
    cx, cy = 105, 155
    pts = [(-bar_w/2, bar_d/2), (bar_w/2, bar_d/2), (bar_w/2, -bar_d/2),
           (stem_w/2, -bar_d/2), (stem_w/2, -bar_d/2 - stem_l),
           (-stem_w/2, -bar_d/2 - stem_l), (-stem_w/2, -bar_d/2), (-bar_w/2, -bar_d/2)]
    off_y = (bar_d + stem_l) / 2 - bar_d / 2
    poly = [(cx + x, cy + y + off_y) for x, y in pts]
    # Solid red: the overhead camera has to segment this against a black mat and a
    # grey printed T, and an outline with a pale fill gives it almost nothing to
    # threshold on.
    RED = (0.83, 0.05, 0.05)
    ax.add_patch(Polygon(poly, closed=True, facecolor=RED, edgecolor=RED, lw=1.0,
                         joinstyle="miter"))
    # Captions clear of the shape: at 100 mm tall the T reaches further up the sheet
    # than the 80 mm one did, and the subtitle used to land on top of the crossbar.
    top = cy + off_y + bar_d / 2
    ax.text(cx, top + 22, "PUSH-T GOAL", ha="center", fontsize=11, weight="bold")
    ax.text(cx, top + 13,
            f"{bar_w:g} x {bar_d + stem_l:g} mm footprint, {bar_d:g} mm bar, {stem_w:g} mm stem",
            ha="center", fontsize=8, color="0.35")
    _scalebar(ax, cx - 50, 60, "tape flat on the mat where the T should end up")
    return fig


def charuco(cols=7, rows=10, square=25.0, marker=18.0):
    import cv2
    import numpy as np
    d = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_100)
    board = cv2.aruco.CharucoBoard((cols, rows), square / 1000.0, marker / 1000.0, d)
    img = board.generateImage((int(cols * square * 10), int(rows * square * 10)), marginSize=0)
    w, h = cols * square, rows * square
    fig, ax = _sheet(need=(w, h))
    x0 = (ax.get_xlim()[1] - w) / 2
    y0 = ax.get_ylim()[1] - h - 30
    ax.imshow(np.flipud(img), cmap="gray", vmin=0, vmax=255,
              extent=(x0, x0 + w, y0, y0 + h), origin="lower", interpolation="nearest")
    _scalebar(ax, x0, y0 - 18,
              f"ChArUco {cols}x{rows}, {square:g} mm square / {marker:g} mm marker, DICT_4X4_100  "
              f"->  ./robot calib-capture --cols {cols} --rows {rows} --square {square:g} --marker {marker:g}")
    return fig


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--square", type=float, default=25.0)
    p.add_argument("--marker", type=float, default=18.0)
    p.add_argument("--cols", type=int, default=7, help="charuco columns (7x10 at 25 mm fits A4)")
    p.add_argument("--rows", type=int, default=10)
    a = p.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    made = []
    for name, fig in (("checkerboard_intrinsics", checkerboard(square=a.square)),
                      ("charuco_extrinsics", charuco(cols=a.cols, rows=a.rows, square=a.square, marker=a.marker)),
                      ("push_t_goal", push_t())):
        pdf = OUT / f"{name}.pdf"
        with PdfPages(pdf) as pp:
            pp.savefig(fig)
        fig.savefig(OUT / f"{name}.png", dpi=200)
        plt.close(fig)
        made.append(pdf)
    print("print these at 100% scale (NOT 'fit to page'):\n")
    for m in made:
        print(f"  {m.relative_to(ROOT)}   (+ .png)")
    print("\nthen measure the 100 mm bar on each sheet before calibrating.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
