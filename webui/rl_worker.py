#!/usr/bin/env python3
"""Persistent worker for the mjlab-trained RL policies (Push-T on the SO-ARM101).

Same protocol as webui/infer_worker.py -- run / stop / home / quit on stdin, markers
on stdout -- so webui/app.py drives all three stacks identically. What differs is
everything below the protocol: there is no lerobot policy, no task string and no
action chunking. This is a 50 Hz closed loop around a 516 KB network that was trained
in mjlab against a simulated arm.

    python webui/rl_worker.py --policy <run>.onnx --goal-x 0.23 --goal-y 0.0 --goal-yaw 0

THE CONTRACT COMES FROM THE MODEL, NOT FROM HERE. mjlab's exporter writes the joint
names, the default joint pose, and the per-joint action scale into the ONNX metadata,
and this reads all three. Retrain with a different home pose or action scale and the
export carries it; nothing in this file needs to change. The two inputs are matched by
RANK, not by name -- (1, 22) is the state vector, (1, 3, H, W) is the camera -- so an
exporter that renames them still loads.

    obs = [ joint_pos - default (6, rad), joint_vel (6, rad/s),
            last raw action (6), goal (x, y, sin yaw, cos yaw) ]

Three things about that vector are worth knowing before trusting a run:

  * JOINT VELOCITY IS DIFFERENTIATED, NOT MEASURED. The SO-101 bus reports
    Present_Position and nothing else, so joint_vel is a finite difference over the
    control period, low-passed. This is the largest sim/real gap in the loop. It is
    survivable because the policy was trained with +/-1.5 rad/s of uniform noise on
    that term -- a band far wider than the differentiator's error -- so the network
    cannot have learned to depend on it precisely.
  * THE GOAL IS OPERATOR-SUPPLIED. The footprint is 1 mm tall and lives in a geom
    group the wrist camera never rendered, so even in simulation the policy was told
    where the goal was rather than seeing it. On the real table that has to come from
    a person measuring: goal_pose is (x, y) metres and yaw in the ROBOT BASE frame.
    Get it wrong and the policy will push the T, accurately, to the wrong place.
  * LAST ACTION IS THE RAW NETWORK OUTPUT, pre-scale, and is zeroed on every `run`.
    Feeding back the scaled target instead is a silent 2x error in one sixth of the
    observation.

SAFETY, in the order it matters:

  * --dry-run computes everything and sends NOTHING. The log still shows the targets,
    so a checkpoint can be watched before it is trusted. This is the honest first run
    for any policy that has only ever driven a simulated arm.
  * RATE LIMIT. Commands move at most --max-deg-per-s per joint, limited against the
    last COMMAND rather than the measurement -- limiting against where the arm got to
    turns a servo lagging under load into a command that creeps.
  * The loop stops on `stop`, and torque is released on the way out.
"""
import argparse
import importlib.util
import json
import math
import os
import pathlib
import sys
import threading
import time

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
HOME_FILE = pathlib.Path(os.environ.get("HOME_POSE_FILE", "/workspace/config/home_pose.json"))


def _load(name: str):
    """Import one module out of sim/sim_agent101 without importing the package.

    The package __init__ pulls in isaaclab, which does not exist in this container.
    Same trick scripts/robot/policy_bridge.py uses, for the same reason.
    """
    path = ROOT / "sim" / "sim_agent101" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_kin = _load("kinematics")
URDF_TO_LEROBOT = _kin.URDF_TO_LEROBOT
urdf_deg_to_lerobot, lerobot_to_urdf_deg = _kin.urdf_deg_to_lerobot, _kin.lerobot_to_urdf_deg


class Policy:
    """The exported network plus the contract its metadata carries."""

    def __init__(self, path: str):
        import onnxruntime as ort

        self.sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
        meta = self.sess.get_modelmeta().custom_metadata_map

        def _floats(key, n):
            v = [float(x) for x in meta[key].split(",")]
            if len(v) != n:
                raise ValueError(f"{key}: expected {n} values, got {len(v)}")
            return np.asarray(v, np.float32)

        self.joints = [s.strip() for s in meta["joint_names"].split(",")]
        self.n = len(self.joints)
        self.default = _floats("default_joint_pos", self.n)   # radians
        self.scale = _floats("action_scale", self.n)
        self.obs_names = [s.strip() for s in meta.get("observation_names", "").split(",")]

        # Match inputs by RANK, not name: rank 2 is the state vector, rank 4 the image.
        self.state_in = self.cam_in = None
        for inp in self.sess.get_inputs():
            if len(inp.shape) == 2:
                self.state_in, self.obs_dim = inp.name, int(inp.shape[1])
            elif len(inp.shape) == 4:
                self.cam_in = inp.name
                _, self.cam_c, self.cam_h, self.cam_w = (int(x) for x in inp.shape)
        if self.state_in is None or self.cam_in is None:
            raise ValueError(f"expected a rank-2 and a rank-4 input, got "
                             f"{[(i.name, i.shape) for i in self.sess.get_inputs()]}")
        self.out = self.sess.get_outputs()[0].name

    def __call__(self, state: np.ndarray, cam: np.ndarray) -> np.ndarray:
        out = self.sess.run([self.out], {self.state_in: state[None, :].astype(np.float32),
                                         self.cam_in: cam[None].astype(np.float32)})
        return np.asarray(out[0][0], np.float32)

    def describe(self) -> str:
        return (f"obs {self.obs_dim} [{', '.join(self.obs_names)}] + "
                f"camera {self.cam_c}x{self.cam_h}x{self.cam_w} -> {self.n} actions; "
                f"scale {np.array2string(self.scale, precision=3)}")


def positions_urdf_deg(follower) -> dict:
    """Measured joint positions as {URDF joint: degrees}, gripper rescaled from percent."""
    obs = follower.get_observation()
    return lerobot_to_urdf_deg({n: obs[f"{n}.pos"] for n in URDF_TO_LEROBOT.values()
                                if f"{n}.pos" in obs})


def do_home(follower):
    """Move to the saved home pose. Closed-loop, same shape as webui/home.py and
    infer_worker.do_home -- coarse ramp, then integral correction on the residual."""
    obs = follower.get_observation()
    motors = sorted(k[:-4] for k in obs if k.endswith(".pos"))
    if HOME_FILE.exists():
        saved = json.loads(HOME_FILE.read_text())
        target = {m: float(saved.get(m, 0.0)) for m in motors}
    else:
        target = {m: 0.0 for m in motors}

    def _pos():
        o = follower.get_observation()
        return {m: float(o[f"{m}.pos"]) for m in motors}

    print("HOME_START", flush=True)
    start = _pos()
    for i in range(1, 61):
        f = i / 60
        follower.send_action({f"{m}.pos": start[m] + (target[m] - start[m]) * f for m in motors})
        time.sleep(0.033)
    cmd, worst = dict(target), 999.0
    for _ in range(40):
        err = {m: target[m] - v for m, v in _pos().items()}
        worst = max(abs(v) for v in err.values())
        if worst < 1.5:
            break
        for m in motors:
            lo, hi = target[m] - 45.0, target[m] + 45.0
            cmd[m] = max(lo, min(hi, cmd[m] + 0.6 * err[m]))
        follower.send_action({f"{m}.pos": cmd[m] for m in motors})
        time.sleep(0.12)
    print(f"HOME_DONE (residual {worst:.1f} deg)", flush=True)


def grab_frame(follower, cam_key, w, h):
    """One camera frame as float32 CHW in [0, 1], resized to the network's input.

    lerobot's OpenCVCamera defaults to ColorMode.RGB, which is the order mjlab's
    renderer produced, so there is no channel swap here -- and adding one 'to be safe'
    would silently swap the T's grey against the footprint's red.
    """
    import cv2

    frame = follower.cameras[cam_key].read_latest()
    if frame is None:
        return None
    if frame.shape[1] != w or frame.shape[0] != h:
        frame = cv2.resize(frame, (w, h), interpolation=cv2.INTER_AREA)
    return np.transpose(frame.astype(np.float32) / 255.0, (2, 0, 1))


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--policy", required=True, help="path to the exported .onnx")
    p.add_argument("--goal-x", type=float, default=0.23, help="goal x in the base frame (m)")
    p.add_argument("--goal-y", type=float, default=0.0, help="goal y in the base frame (m)")
    p.add_argument("--goal-yaw", type=float, default=0.0, help="goal yaw in the base frame (deg)")
    p.add_argument("--hz", type=float, default=50.0, help="control rate (mjlab trained at 50)")
    p.add_argument("--max-deg-per-s", type=float, default=60.0, help="per-joint rate limit")
    p.add_argument("--vel-filter", type=float, default=0.3,
                   help="low-pass alpha on the differentiated joint velocity (1 = raw)")
    p.add_argument("--dry-run", action="store_true", help="compute everything, send nothing")
    p.add_argument("--cam-index", default=os.environ.get("CAM_GRIP_INDEX", "6"))
    a = p.parse_args()

    goal = np.array([a.goal_x, a.goal_y,
                     math.sin(math.radians(a.goal_yaw)),
                     math.cos(math.radians(a.goal_yaw))], np.float32)

    print("LOADING", flush=True)
    policy = Policy(a.policy)
    print(f"  policy: {policy.describe()}", flush=True)
    print(f"  joints: {', '.join(policy.joints)}", flush=True)
    if policy.obs_dim != 3 * policy.n + 4:
        print(f"  NOTE: obs is {policy.obs_dim}, not 3x{policy.n}+4 -- the layout below "
              f"assumes joint_pos/joint_vel/actions/goal and may not match", flush=True)

    # so_follower, NOT so101_follower: the pinned lerobot in this container names the
    # follower after the family and the leader after the model. policy_bridge.py has
    # the long version of this note.
    from lerobot.cameras.opencv import OpenCVCameraConfig
    from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig

    idx = a.cam_index
    cam_cfg = OpenCVCameraConfig(
        index_or_path=int(idx) if str(idx).isdigit() else pathlib.Path(idx),
        fps=int(os.environ.get("CAM_FPS", "30")),
        width=int(os.environ.get("CAM_WIDTH", "640")),
        height=int(os.environ.get("CAM_HEIGHT", "480")),
        fourcc=os.environ.get("CAM_FOURCC", "MJPG"),
    )
    kwargs = dict(port=os.environ.get("ROBOT_PORT", "/dev/ttyACM1"),
                  id=os.environ.get("ROBOT_ID", "zetans_follower"),
                  cameras={"grip": cam_cfg}, use_degrees=True)
    if "disable_torque_on_disconnect" in SO101FollowerConfig.__dataclass_fields__:
        kwargs["disable_torque_on_disconnect"] = True
    follower = SO101Follower(SO101FollowerConfig(**kwargs))
    follower.connect()
    print(f"  robot on {kwargs['port']}, wrist camera index {idx}", flush=True)
    print(f"  goal (base frame): x={a.goal_x:.3f} m  y={a.goal_y:.3f} m  yaw={a.goal_yaw:.1f} deg",
          flush=True)
    if a.dry_run:
        print("  DRY RUN: targets are computed and logged, nothing is sent", flush=True)
    print("MODEL_LOADED", flush=True)

    shutdown = threading.Event()
    shutdown.set()
    state = {"goal": goal}
    run_thread = None

    def loop():
        print("RUN_START", flush=True)
        dt = 1.0 / a.hz
        max_step = a.max_deg_per_s * dt
        last_action = np.zeros(policy.n, np.float32)   # RAW output, pre-scale
        qd = np.zeros(policy.n, np.float32)
        q_prev = None
        commanded = None
        ticks = 0
        try:
            while not shutdown.is_set():
                started = time.time()
                measured = positions_urdf_deg(follower)
                if len(measured) < policy.n:
                    print(f"RUN_ERROR: bus reported {len(measured)} joints, need {policy.n}",
                          flush=True)
                    return
                q = np.array([math.radians(measured[j]) for j in policy.joints], np.float32)

                # No velocity on this bus -- differentiate, then low-pass. See the
                # module docstring: the policy trained through +/-1.5 rad/s of noise
                # on this term, which is why the crude estimate is usable at all.
                if q_prev is not None:
                    raw = (q - q_prev) / max(dt, 1e-6)
                    qd = a.vel_filter * raw + (1.0 - a.vel_filter) * qd
                q_prev = q

                cam = grab_frame(follower, "grip", policy.cam_w, policy.cam_h)
                if cam is None:
                    time.sleep(dt)
                    continue

                obs = np.concatenate([q - policy.default, qd, last_action, state["goal"]])
                t0 = time.time()
                action = policy(obs, cam)
                infer_ms = (time.time() - t0) * 1000.0
                last_action = action

                # use_default_offset=True in the trained cfg: the action is an offset
                # on the DEFAULT pose, not on the current one.
                target_rad = policy.default + policy.scale * action
                target_deg = {j: math.degrees(v) for j, v in zip(policy.joints, target_rad)}

                if not a.dry_run:
                    if commanded is None:
                        commanded = dict(measured)
                    nxt = dict(measured)     # joints the policy does not drive hold still
                    for j, want in target_deg.items():
                        if j not in commanded:
                            continue
                        delta = max(-max_step, min(max_step, want - commanded[j]))
                        nxt[j] = commanded[j] + delta
                    follower.send_action({f"{k}.pos": v
                                          for k, v in urdf_deg_to_lerobot(nxt).items()})
                    commanded = nxt

                ticks += 1
                if ticks % 10 == 0:        # 5 Hz of log at the 50 Hz control rate
                    print(f"inference {infer_ms:.0f} ms | "
                          + " ".join(f"{j}={target_deg[j]:+.1f}" for j in policy.joints),
                          flush=True)

                remaining = dt - (time.time() - started)
                if remaining > 0:
                    time.sleep(remaining)
        except Exception as e:   # noqa: BLE001 - keep the worker alive on a run error
            print(f"RUN_ERROR: {e}", flush=True)
        finally:
            print("RUN_STOP", flush=True)

    try:
        for line in sys.stdin:
            cmd = line.strip()
            if cmd == "run":
                if run_thread and run_thread.is_alive():
                    continue
                shutdown.clear()
                run_thread = threading.Thread(target=loop, daemon=True)
                run_thread.start()
            elif cmd == "stop":
                shutdown.set()
            elif cmd == "home":
                shutdown.set()
                if run_thread and run_thread.is_alive():
                    run_thread.join(timeout=20)
                try:
                    do_home(follower)
                except Exception as e:  # noqa: BLE001
                    print(f"HOME_ERROR: {e}", flush=True)
            elif cmd.startswith("goal "):
                # Live, like the other workers' steps/actionsteps: the operator moves
                # the printed footprint and retypes where it went, without a reload.
                try:
                    x, y, yaw = (float(v) for v in cmd.split()[1:4])
                    state["goal"] = np.array([x, y, math.sin(math.radians(yaw)),
                                              math.cos(math.radians(yaw))], np.float32)
                    print(f"GOAL_SET x={x:.3f} y={y:.3f} yaw={yaw:.1f}", flush=True)
                except Exception as e:  # noqa: BLE001
                    print(f"GOAL_ERROR: {e}", flush=True)
            elif cmd == "quit":
                shutdown.set()
                break
    finally:
        shutdown.set()
        if run_thread and run_thread.is_alive():
            run_thread.join(timeout=10)
        try:
            follower.disconnect()
        except Exception as e:  # noqa: BLE001
            print(f"teardown note: {e}", flush=True)
        print("UNLOADED", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
