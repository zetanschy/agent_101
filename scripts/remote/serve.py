#!/usr/bin/env python3
"""Serve a policy over gRPC from a GPU that is not attached to the robot.

    ./robot policy-serve --policy molmoact2 --port 8080
    ./robot policy-serve --list                     # what this box can serve

This is the GPU half. The robot half is scripts/remote/client.py, and the wire
between them is lerobot's own async-inference protocol, so a stock
`lerobot.async_inference.RobotClient` can drive this server and this can serve a
stock lerobot policy. Nothing here is a private protocol.

WHAT IS ACTUALLY ADDED over `python -m lerobot.async_inference.policy_server` is
adapters.install(), which widens the server's policy allow-list. That list is
hand-maintained rather than derived from the policy registry, so lerobot can fully
implement a policy and the server still refuse it: **molmoact2 is exactly that
case** -- `get_policy_class("molmoact2")` returns MolmoAct2Policy today, and an
unpatched server answers "policy type not supported". install() also covers models
lerobot has never heard of, through the adapter registry in adapters.py.

SECURITY, because this listens on a network port. gRPC here is unauthenticated and
unencrypted: anything that can reach the port can drive the robot and read its
cameras. The default bind is 127.0.0.1 for that reason, and the intended way to
cross machines is an SSH tunnel:

    ssh -N -L 8080:127.0.0.1:8080 gpubox        # from the robot

which needs no open port on the GPU box at all. `--host 0.0.0.0` exists for a
trusted LAN and says so at startup.
"""

from __future__ import annotations

import argparse
import logging
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import adapters  # noqa: E402


def load_adapters(names: list[str] | None = None) -> list[str]:
    """Import the adapter modules, so their `register` calls run.

    Import failures are reported and skipped rather than fatal: a GPU box that can
    serve MolmoAct2 has no reason to also have openpi installed, and refusing to
    start because an unrelated adapter's dependency is missing would make every box
    need every model.
    """
    import importlib

    available = names or []      # no adapters shipped yet; molmoact2 is native
    for module in available:
        try:
            importlib.import_module(module)
        except Exception as exc:  # noqa: BLE001 - a missing model is not fatal
            print(f"  adapter {module!r} unavailable: {type(exc).__name__}: {exc}")
    return adapters.install()


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--host", default="127.0.0.1",
                   help="bind address (default 127.0.0.1; see the note on tunnelling)")
    p.add_argument("--port", type=int, default=8080)
    p.add_argument("--fps", type=int, default=30, help="control rate the client runs at")
    p.add_argument("--inference-latency", type=float, default=0.033,
                   help="seconds the server tells the client to budget per inference")
    p.add_argument("--obs-queue-timeout", type=float, default=2.0)
    p.add_argument("--list", action="store_true", help="print the servable policies and exit")
    a = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    served = load_adapters()
    print(f"adapters: {', '.join(served) if served else '(none)'}")
    if a.list:
        from lerobot.async_inference import constants

        print(f"servable policy types: {', '.join(constants.SUPPORTED_POLICIES)}")
        return 0

    # No warning on 0.0.0.0 by itself: inside a container that is the only bind that
    # a published port can reach, and `./robot policy-serve` maps it to the host's
    # loopback. What matters is what the HOST publishes, and that is ./robot's call.
    if a.host not in ("127.0.0.1", "localhost", "0.0.0.0"):
        print(f"WARNING: binding {a.host} — this port is unauthenticated and "
              f"unencrypted; anything that reaches it can drive the robot.")

    from lerobot.async_inference.configs import PolicyServerConfig
    from lerobot.async_inference.policy_server import serve

    cfg = PolicyServerConfig(host=a.host, port=a.port, fps=a.fps,
                             inference_latency=a.inference_latency,
                             obs_queue_timeout=a.obs_queue_timeout)
    print(f"serving on {a.host}:{a.port} (fps={a.fps})", flush=True)
    serve(cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
