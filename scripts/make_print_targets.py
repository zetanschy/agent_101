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
MM = 1 / 25.4  # matplotlib works in inches


def _sheet(w_mm=210, h_mm=297):
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
    fig, ax = _sheet(max(210, w + 30), max(297, h + 50))
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


def push_t(bar_w=80.0, bar_d=20.0, stem_w=20.0, stem_l=60.0):
    """The T's footprint, from sim_agent101.assets.objects.TBlockGeometry."""
    fig, ax = _sheet()
    cx, cy = 105, 170
    # mesh frame: bar spans y -10..10, stem hangs to -70; centre it on the sheet
    pts = [(-bar_w/2, bar_d/2), (bar_w/2, bar_d/2), (bar_w/2, -bar_d/2),
           (stem_w/2, -bar_d/2), (stem_w/2, -bar_d/2 - stem_l),
           (-stem_w/2, -bar_d/2 - stem_l), (-stem_w/2, -bar_d/2), (-bar_w/2, -bar_d/2)]
    off_y = (bar_d/2 + (bar_d/2 + stem_l)) / 2 - bar_d/2
    poly = [(cx + x, cy + y + off_y) for x, y in pts]
    ax.add_patch(Polygon(poly, closed=True, facecolor=(1, 0.85, 0.85),
                         edgecolor=(0.85, 0.05, 0.05), lw=3.0, joinstyle="miter"))
    ax.plot([cx], [cy + off_y - (bar_d/2 + stem_l)/2 + stem_l/2], marker="+", ms=8,
            color=(0.85, 0.05, 0.05), lw=1)
    ax.text(cx, cy + 70, "PUSH-T GOAL", ha="center", fontsize=11, weight="bold")
    ax.text(cx, cy + 62,
            f"{bar_w:g} x {bar_d + stem_l:g} mm footprint, {bar_d:g} mm bar, {stem_w:g} mm stem",
            ha="center", fontsize=8, color="0.35")
    _scalebar(ax, cx - 50, 60, "tape flat on the mat where the T should end up")
    return fig


def charuco(cols=8, rows=11, square=25.0, marker=18.0):
    import cv2
    import numpy as np
    d = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_100)
    board = cv2.aruco.CharucoBoard((cols, rows), square / 1000.0, marker / 1000.0, d)
    img = board.generateImage((int(cols * square * 10), int(rows * square * 10)), marginSize=0)
    w, h = cols * square, rows * square
    fig, ax = _sheet(max(210, w + 30), max(297, h + 50))
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
    a = p.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    made = []
    for name, fig in (("checkerboard_intrinsics", checkerboard(square=a.square)),
                      ("charuco_extrinsics", charuco(square=a.square, marker=a.marker)),
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
