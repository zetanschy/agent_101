"""`./robot real2sim live watch`: see what the policy sees, from another terminal.

Connects to the running live sim as one more client (the server takes several) and shows
both fitted cameras side by side. These are the frames the robot's get_observation
returns, distortion and lens softness included, with the sim time, the layout and the
caps in the mug drawn on top. It only OBSERVES: it never sends a command, so it cannot
disturb a policy, a teleop session or a recording, and in lockstep it does not advance
the sim.

    r        a new object layout (as 'r' in the lerobot terminal)
    q / Esc  close the window (the sim keeps running)

--snapshot PATH writes one frame and exits (no window), e.g. to check a headless box.
For the 3D scene instead of the cameras, start the sim with --sim-viewer (MuJoCo's viewer,
free orbit; R = new layout there too).
"""

from __future__ import annotations

import argparse
import time

import numpy as np

from . import protocol
from .server import default_socket

CAMS = ("front", "grip")


def _warper(info: dict):
    """The plugin's client-side warp (cv2), for an engine that sends pinhole renders."""
    import cv2

    w = info.get("warp")
    if not w:
        return None
    maps = {k: (*cv2.convertMaps(v["map_x"], v["map_y"], cv2.CV_16SC2), float(v.get("sigma_px") or 0.0))
            for k, v in w.items()}

    def warp(cam, img):
        m1, m2, sigma = maps[cam]
        out = cv2.remap(img, m1, m2, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        return cv2.GaussianBlur(out, (0, 0), sigma) if sigma > 0 else out

    return warp


def _frame(sock, warp, st: dict) -> np.ndarray:
    import cv2

    r = protocol.call(sock, "observe", images=list(CAMS), pinhole=warp is not None)
    tiles = []
    for c in CAMS:
        img = r["images"][c]
        if r.get("pinhole") and warp is not None:
            img = warp(c, img)
        tiles.append(np.ascontiguousarray(img))
    out = cv2.cvtColor(np.concatenate(tiles, axis=1), cv2.COLOR_RGB2BGR)
    cim = st.get("caps_in_mug") or []
    lay = (st.get("layout") or {}).get("kind", "?")
    text = f"{st.get('engine', '?')}  t={r['t']:6.1f}s  layout {lay}  caps in mug {sum(cim)}/{len(cim)}  (r: new layout, q: close)"
    cv2.rectangle(out, (0, 0), (out.shape[1], 26), (0, 0, 0), -1)
    cv2.putText(out, text, (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="real2sim live watch")
    ap.add_argument("--socket", default=None)
    ap.add_argument("--hz", type=float, default=15.0, help="refresh rate of the window")
    ap.add_argument("--snapshot", metavar="PATH", help="write one frame to PATH and exit (no window)")
    a = ap.parse_args(argv)
    import cv2

    path = str(a.socket or default_socket())
    try:
        sock = protocol.connect(path, 10.0)
    except OSError as e:
        print(f"no live sim at {path} ({e}): start one with './robot <cmd> --sim mujoco|isaac'")
        return 1
    info = protocol.call(sock, "hello", warp=True)
    warp = _warper(info)
    st = protocol.call(sock, "status")
    if a.snapshot:
        cv2.imwrite(a.snapshot, _frame(sock, warp, st))
        print(f"-> {a.snapshot}")
        return 0
    name = f"real2sim live ({info['engine']})"
    cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    last_status = 0.0
    try:
        while True:
            t0 = time.monotonic()
            if t0 - last_status > 0.5:
                st, last_status = protocol.call(sock, "status"), t0
            cv2.imshow(name, _frame(sock, warp, st))
            key = cv2.waitKey(max(1, int(1000 / a.hz - 1000 * (time.monotonic() - t0)))) & 0xFF
            if key in (ord("q"), 27) or cv2.getWindowProperty(name, cv2.WND_PROP_VISIBLE) < 1:
                break
            if key == ord("r"):
                print("new layout:", protocol.call(sock, "reset")["layout"].get("kind"), flush=True)
    except (ConnectionError, OSError):
        print("the sim stopped")
    finally:
        cv2.destroyAllWindows()
        sock.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
