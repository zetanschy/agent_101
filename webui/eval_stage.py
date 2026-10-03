"""Stages for the web UI's Eval page: where to put the cap and the mug for trial i.

Trial i of a run uses seed (start + i), and the layout is drawn by the live sim's own
sampler (real2sim.live.layouts.random, one cap) from that seed. So stage i on the real
arm is the same stage i that `./robot sim-eval` scores in MuJoCo, and a real run pairs
with a sim run trial by trial (`./robot sim-eval --compare`).

The sim draws the layout and renders the overhead camera with the objects in place:
that render is the picture to copy. The same positions are projected through the
overhead camera's calibration (fitted on the real dataset, sim/real2sim) and drawn as
outlines on the LIVE real camera, so you slide the real objects until they sit inside
them. The outlines are the parts the overhead camera sees: the mug's rim and the cap's
up-facing end, at their real heights.

    real arm: `./robot real-eval` starts a MuJoCo sim of its own just for this (lockstep,
              idle unless asked, on stage.sock) and the stages come from it
    sim     : `./robot openpi-webui --sim mujoco` -- preparing a stage resets the robot's
              own sim with the seed, which is the placement
"""

from __future__ import annotations

import math
import os
import pathlib

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
RUNS = ROOT / "outputs" / "real_eval"
CAUSES = ["never picked up", "pushed the cap away", "dropped on the way", "missed the mug",
          "still holding at timeout", "mug knocked over", "cap fell off the table", "other"]
CAP_COL, MUG_COL = (214, 120, 42), (52, 104, 235)  # BGR: blue cap, orange mug (as the reports)

_scene = None


def scene():
    global _scene
    if _scene is None:
        from real2sim import scene as scene_mod

        _scene = scene_mod.load()
    return _scene


def camera():
    return scene().camera("front")


def sim_socket() -> str | None:
    """The sim that draws stages: the eval's own (real arm) or the robot's (sim mode)."""
    if os.environ.get("STAGE_SOCK"):
        return os.environ["STAGE_SOCK"]
    if os.environ.get("ROBOT_TYPE") == "real2sim":
        return os.environ.get("ROBOT_PORT")
    return None


def prepare(seed: int) -> tuple[dict, np.ndarray]:
    """Reset the stage sim to seed's layout and render the overhead camera (RGB, distorted
    and softened exactly as the policy's frames are)."""
    from real2sim.live import protocol, watch

    path = sim_socket()
    if not path:
        raise RuntimeError("no stage sim: start the page with `./robot real-eval` (or --sim mujoco)")
    sock = protocol.connect(path, 60.0)
    try:
        warp = watch._warper(protocol.call(sock, "hello", warp=True))
        layout = protocol.call(sock, "reset", layout="random:1", seed=int(seed))["layout"]
        r = protocol.call(sock, "observe", images=["front"], pinhole=warp is not None)
        img = r["images"]["front"]
        return layout, (warp("front", img) if (r.get("pinhole") and warp) else img)
    finally:
        sock.close()


def sim_caps_in_mug() -> bool | None:
    """Sim mode only: the sim's own grade of the stage as it stands."""
    if os.environ.get("ROBOT_TYPE") != "real2sim":
        return None
    from real2sim.live import protocol

    sock = protocol.connect(os.environ["ROBOT_PORT"], 30.0)
    try:
        cim = protocol.call(sock, "status").get("caps_in_mug") or []
        return bool(cim) and all(cim)
    finally:
        sock.close()


# --- geometry --------------------------------------------------------------------------
def object_points(layout: dict) -> dict:
    """World points (base frame, m) of what the overhead camera sees of each object."""
    sc = scene()
    tz, cd, md = sc.table_z(), sc.cap_dims(), sc.mug_dims()
    cap, mug = layout["caps"][0], layout["mug"]
    th = np.linspace(0, 2 * np.pi, 48, endpoint=False)
    rc = (cd["diameter_open_end"] if cap["up"] == "open" else cd["diameter"]) / 2
    cz = tz + cd["height"]
    R = md["outer_diameter"] / 2
    mz = tz + md["height"]
    yaw = math.radians(mug["handle_yaw_deg"])  # the handle is the mug's local +x
    h = md["handle"]
    hz = tz + (h["z_bottom"] + h["z_top"]) / 2
    return {
        "cap_ring": np.c_[cap["xy"][0] + rc * np.cos(th), cap["xy"][1] + rc * np.sin(th), np.full_like(th, cz)],
        "cap_centre": np.array([*cap["xy"], cz]),
        "mug_rim": np.c_[mug["xy"][0] + R * np.cos(th), mug["xy"][1] + R * np.sin(th), np.full_like(th, mz)],
        "mug_centre": np.array([*mug["xy"], mz]),
        "handle": np.array([[mug["xy"][0] + R * math.cos(yaw), mug["xy"][1] + R * math.sin(yaw), hz],
                            [mug["xy"][0] + (R + h["protrusion"]) * math.cos(yaw),
                             mug["xy"][1] + (R + h["protrusion"]) * math.sin(yaw), hz]]),
    }


def draw_targets(img_bgr: np.ndarray, layout: dict, label: bool = True, flip: bool = False) -> np.ndarray:
    """The stage's outlines on an overhead frame (BGR, 640x480). flip turns the frame 180°
    (display only: the bench is worked from the far side), with the labels drawn after
    the turn so they still read."""
    import cv2

    cam = camera()
    P = {k: cam.project(v) for k, v in object_points(layout).items()}
    sx, sy = img_bgr.shape[1] / cam.width, img_bgr.shape[0] / cam.height
    px = lambda a: np.round(np.asarray(a) * [sx, sy]).astype(np.int32)  # noqa: E731
    for ring, col in (("mug_rim", MUG_COL), ("cap_ring", CAP_COL)):
        pts = px(P[ring])
        cv2.polylines(img_bgr, [pts], True, (16, 16, 16), 4, cv2.LINE_AA)  # dark halo: readable on any mat
        cv2.polylines(img_bgr, [pts], True, col, 2, cv2.LINE_AA)
    a, b = px(P["handle"])
    cv2.line(img_bgr, tuple(a), tuple(b), (16, 16, 16), 6, cv2.LINE_AA)
    cv2.line(img_bgr, tuple(a), tuple(b), MUG_COL, 3, cv2.LINE_AA)
    H, W = img_bgr.shape[:2]
    if flip:
        img_bgr = np.ascontiguousarray(img_bgr[::-1, ::-1])
    if label:
        for k, txt, col in (("cap_centre", "cap", CAP_COL), ("mug_centre", "mug", MUG_COL)):
            u, v = px(P[k])
            if flip:
                u, v = W - 1 - u, H - 1 - v
            r = 22 if k == "cap_centre" else 48
            cv2.putText(img_bgr, txt, (u + r, v + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (16, 16, 16), 4, cv2.LINE_AA)
            cv2.putText(img_bgr, txt, (u + r, v + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.55, col, 1, cv2.LINE_AA)
    return img_bgr


def _image_direction(cam, frm, to) -> str:
    (u0, v0), (u1, v1) = cam.project(np.asarray(frm)), cam.project(np.asarray(to))
    ang = math.degrees(math.atan2(-(v1 - v0), u1 - u0)) % 360  # 0 = image right, 90 = image up
    names = ["right", "up-right", "up", "up-left", "left", "down-left", "down", "down-right"]
    return names[int(((ang + 22.5) % 360) // 45)]


_FLIPPED = {"left": "right", "right": "left", "up": "down", "down": "up", "up-left": "down-right",
            "down-right": "up-left", "up-right": "down-left", "down-left": "up-right"}


def describe(layout: dict, flip: bool = False) -> dict:
    """Where each object goes, in words a person placing them can follow. With flip the
    directions are the turned view's, matching what the page shows."""
    turn = (lambda d: _FLIPPED[d]) if flip else (lambda d: d)  # noqa: E731
    cam, tz = camera(), scene().table_z()
    out = {}
    for key, xy in (("cap", layout["caps"][0]["xy"]), ("mug", layout["mug"]["xy"])):
        x, y = xy
        r, az = math.hypot(x, y), math.degrees(math.atan2(x, -y))
        # which side of straight ahead it lands on in the overhead image
        side = "right" if cam.project(np.array([x, y, tz]))[0] > cam.project(np.array([0.0, -r, tz]))[0] else "left"
        out[key] = (f"{r * 100:.0f} cm from the base, " + ("straight ahead" if abs(az) < 3 else
                    f"{abs(az):.0f}° off straight ahead, to the image {turn(side)}"))
    out["cap"] += ", " + ("OPEN side up (cavity facing up)" if layout["caps"][0]["up"] == "open"
                          else "closed side up (rim on the mat)")
    pts = object_points(layout)["handle"]
    out["mug"] += f", handle toward image {turn(_image_direction(cam, pts[0], pts[1]))}"
    return out


def record_fields(layout: dict) -> dict:
    """The stage as sim_eval's trials.jsonl records it, so its report and map read real runs."""
    sc = scene()
    cap, mug = layout["caps"][0], layout["mug"]
    cz = sc.table_z() + (sc.cap_dims()["height"] if cap["up"] == "open" else 0.0)
    return {"cap_xy": cap["xy"], "cap_up": cap["up"], "mug_xy": mug["xy"], "handle_yaw_deg": mug["handle_yaw_deg"],
            "cap_xyz0": [*cap["xy"], round(cz, 4)], "mug_xyz0": [*mug["xy"], round(sc.table_z(), 4)]}


def auto_check(end_jpg: pathlib.Path) -> dict:
    """A hint for grading, from the last overhead frame: is there teal cap inside the red
    mug's opening? Only inside it: the mat itself has blue-green patches (thousands of
    "teal" pixels in a frame), but where the mug stands it hides the mat, so teal inside
    its outline is the cap. Colour thresholds can still be fooled (glare, the gripper
    over the mug); the operator's grade is what counts."""
    import cv2

    img = cv2.imread(str(end_jpg))
    if img is None:
        return {"verdict": "unknown", "detail": "no final frame"}
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    teal = cv2.inRange(hsv, (78, 60, 70), (100, 255, 255)) > 0
    red = (cv2.inRange(hsv, (0, 120, 35), (10, 255, 255)) | cv2.inRange(hsv, (170, 120, 35), (180, 255, 255)))
    n, lab, st, _ = cv2.connectedComponentsWithStats(red)
    if n < 2:
        return {"verdict": "unknown", "detail": "mug not found in the last frame"}
    i = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))
    (cx, cy), rad = cv2.minEnclosingCircle(np.column_stack(np.nonzero(lab == i))[:, ::-1].astype(np.float32))
    yy, xx = np.nonzero(teal)
    inside = int((np.hypot(xx - cx, yy - cy) < 0.8 * rad).sum())  # the opening, not the shadow around it
    verdict = "in" if inside >= 120 else "out" if inside < 25 else "unknown"
    return {"verdict": verdict, "detail": f"{inside} teal pixels inside the mug's opening (in >= 120, out < 25)"}
