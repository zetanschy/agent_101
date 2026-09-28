"""Pull one LeRobot v3.0 dataset apart into arrays every track can read directly.

    ./robot real2sim core extract [--ds NAME] [--force]

writes, under sim/outputs/real2sim/<ds>/:

    episodes.npz        joints and episode bounds (schema below), from the parquet
    video/<cam>/chunk-XXX/file-YYY.mp4   byte copies of the dataset videos
    frames/<cam>.npy    every decoded frame, uint8 (N, H, W, 3), in dataset row order

WHY DOCKER. The dataset's videos are root-owned, mode 600 (the recorder runs as root
in the container), and the dataset must not be chmod'ed. So a root container copies
them out and decodes them, then hands the outputs to the calling user. The decoder
is pyav's `frame.to_ndarray(format="rgb24")`, the call LeRobot's pyav backend makes,
so these are the pixels a policy trained on this dataset saw. cv2 cannot decode AV1
on this box at all (tooling report). The copies are readable, so host ffmpeg can
also decode them later.

ROW ORDER. frames/<cam>.npy row i is the image of parquet row i (global `index`),
whatever the file layout: each episode's frames are read from its own video file
between from_timestamp and to_timestamp and written to rows [from, to). Every
decoded frame's pts must sit on the 1/fps grid (measured exact here), and every row
must be written exactly once, or extraction fails.

episodes.npz, all arrays, no pickles (np.load(..., allow_pickle=False) works):
    action, state    (N, 6) float32  LeRobot units: arm DEGREES, gripper RANGE_0_100 %
    episode_index, frame_index, index, task_index  (N,) int64
    timestamp        (N,) float32    SYNTHETIC: frame_index / fps (lerobot's writer)
    fps              ()   int64
    joint_names      (6,) str  lerobot motor names: shoulder_pan ... gripper
    feature_names    (6,) str  parquet column names: shoulder_pan.pos ...
    urdf_joint_names (6,) str  Rotation ... Jaw (same order)
    episodes         (E,) int64 episode ids;  ep_from, ep_to (E,) global rows [from, to)
    ep_length        (E,) int64;  tasks (E,) str
    cams             (C,) str  short names: front, grip
    video_keys       (C,) str  observation.images.front ...
    video_from_ts, video_to_ts  (E, C) float64  seconds inside that episode's file
    video_file       (E, C) str  readable copy, relative to the dataset's out_dir
    frames_file      (C,) str   frames/<cam>.npy, relative to out_dir
    meta             ()   str   JSON: dataset, repo_id, source root, info.json subset,
                                extraction time, git describe
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from . import paths


def _read_tables(root: Path):
    import pandas as pd  # lazy: not every interpreter has pandas

    info = json.loads((root / "meta" / "info.json").read_text())
    eps = pd.concat([pd.read_parquet(f) for f in sorted((root / "meta" / "episodes").glob("*/*.parquet"))])
    eps = eps.sort_values("episode_index").reset_index(drop=True)
    data = pd.concat([pd.read_parquet(f) for f in sorted((root / "data").glob("*/*.parquet"))])
    data = data.sort_values("index").reset_index(drop=True)
    return info, eps, data


def _git_describe() -> str:
    try:
        return subprocess.run(["git", "-C", str(paths.REPO), "describe", "--always", "--dirty"],
                              capture_output=True, text=True, timeout=10).stdout.strip()
    except Exception:  # git missing: provenance degrades, extraction does not
        return "unknown"


def _copy_name(cam: str, rel: str) -> str:
    """videos/observation.images.front/chunk-000/file-000.mp4 -> video/front/chunk-000/file-000.mp4"""
    return f"video/{cam}/{rel.split('/', 2)[2]}"


def build_arrays(root: Path) -> tuple[dict, dict]:
    """Parquet + meta -> the episodes.npz arrays, plus the per-camera decode plan."""
    info, eps, data = _read_tables(root)
    fps = int(info["fps"])
    n = len(data)
    if n != int(info["total_frames"]):
        raise RuntimeError(f"parquet has {n} rows, info.json says total_frames={info['total_frames']}")
    if not np.array_equal(data["index"].to_numpy(), np.arange(n)):
        raise RuntimeError("parquet `index` is not 0..N-1 contiguous")

    feats = info["features"]
    feature_names = list(feats["action"]["names"])
    if list(feats["observation.state"]["names"]) != feature_names:
        raise RuntimeError("action and observation.state name their joints differently")
    joint_names = [f.removesuffix(".pos") for f in feature_names]
    from .units import LEROBOT_JOINTS, URDF_JOINTS
    if tuple(joint_names) != LEROBOT_JOINTS:
        raise RuntimeError(f"unexpected joints {joint_names}, units.py assumes {LEROBOT_JOINTS}")

    action = np.stack(data["action"].to_numpy()).astype(np.float32)
    state = np.stack(data["observation.state"].to_numpy()).astype(np.float32)
    ep_idx = data["episode_index"].to_numpy(np.int64)
    frame_idx = data["frame_index"].to_numpy(np.int64)

    episodes = eps["episode_index"].to_numpy(np.int64)
    ep_from = eps["dataset_from_index"].to_numpy(np.int64)
    ep_to = eps["dataset_to_index"].to_numpy(np.int64)
    for e, a, b in zip(episodes, ep_from, ep_to):
        if not (np.all(ep_idx[a:b] == e) and np.array_equal(frame_idx[a:b], np.arange(b - a))):
            raise RuntimeError(f"episode {e}: rows [{a},{b}) are not its frames 0..{b - a - 1}")
    if int((ep_to - ep_from).sum()) != n:
        raise RuntimeError("episode bounds do not tile the dataset")

    video_keys = [k for k, v in feats.items() if v.get("dtype") == "video"]
    cams = [k.split(".")[-1] for k in video_keys]
    E, C = len(episodes), len(cams)
    vfrom, vto = np.zeros((E, C)), np.zeros((E, C))
    vfile = np.empty((E, C), dtype=object)
    plan: dict[str, dict] = {}
    for c, (key, cam) in enumerate(zip(video_keys, cams)):
        shape = feats[key]["shape"]  # (H, W, 3)
        segs: dict[str, list] = {}
        for i in range(E):
            ci = int(eps[f"videos/{key}/chunk_index"][i])
            fi = int(eps[f"videos/{key}/file_index"][i])
            rel = info["video_path"].format(video_key=key, chunk_index=ci, file_index=fi)
            vfrom[i, c] = float(eps[f"videos/{key}/from_timestamp"][i])
            vto[i, c] = float(eps[f"videos/{key}/to_timestamp"][i])
            f0 = int(round(vfrom[i, c] * fps))
            if abs(vfrom[i, c] * fps - f0) > 1e-3 or int(round(vto[i, c] * fps)) - f0 != ep_to[i] - ep_from[i]:
                raise RuntimeError(f"{cam} ep {episodes[i]}: video span {vfrom[i, c]}..{vto[i, c]} s "
                                   f"is not {ep_to[i] - ep_from[i]} frames on the 1/{fps} s grid")
            vfile[i, c] = _copy_name(cam, rel)
            segs.setdefault(rel, []).append([f0, f0 + int(ep_to[i] - ep_from[i]), int(ep_from[i])])
        plan[cam] = {"key": key, "shape": [n, *shape], "fps": fps,
                     "files": [{"src": rel, "copy": _copy_name(cam, rel), "segments": s}
                               for rel, s in segs.items()]}

    meta = {
        "dataset": root.name, "repo_id": f"{root.parent.name}/{root.name}", "source_root": str(root),
        "codebase_version": info.get("codebase_version"), "robot_type": info.get("robot_type"),
        "fps": fps, "total_frames": n, "total_episodes": E, "video_codec": {
            k: feats[k]["info"].get("video.codec") for k in video_keys},
        "extracted_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "git": _git_describe(),
        "decoder": "pyav frame.to_ndarray(format='rgb24') in " + paths.DOCKER_IMAGE,
    }
    arrays = {
        "action": action, "state": state, "episode_index": ep_idx, "frame_index": frame_idx,
        "index": data["index"].to_numpy(np.int64), "task_index": data["task_index"].to_numpy(np.int64),
        "timestamp": data["timestamp"].to_numpy(np.float32), "fps": np.int64(fps),
        "joint_names": np.array(joint_names), "feature_names": np.array(feature_names),
        "urdf_joint_names": np.array(URDF_JOINTS),
        "episodes": episodes, "ep_from": ep_from, "ep_to": ep_to, "ep_length": ep_to - ep_from,
        "tasks": np.array([", ".join(t) if not isinstance(t, str) else t for t in eps["tasks"]]),
        "cams": np.array(cams), "video_keys": np.array(video_keys),
        "video_from_ts": vfrom, "video_to_ts": vto, "video_file": vfile.astype(str),
        "frames_file": np.array([f"frames/{c}.npy" for c in cams]),
        "meta": np.array(json.dumps(meta)),
    }
    return arrays, plan


# --- inside the container ---------------------------------------------------------

def _decode_plan(plan_path: str, dataset_root: str, out_root: str) -> None:
    """Runs as root in DOCKER_IMAGE: copy the videos out, decode them into the caches."""
    import shutil

    import av  # only the container has pyav

    plan = json.loads(Path(plan_path).read_text())
    report = {}
    for cam, p in plan.items():
        n = p["shape"][0]
        out = Path(out_root) / "frames" / f"{cam}.npy"
        tmp = out.with_suffix(".tmp.npy")
        out.parent.mkdir(parents=True, exist_ok=True)
        mm = np.lib.format.open_memmap(tmp, mode="w+", dtype=np.uint8, shape=tuple(p["shape"]))
        written = np.zeros(n, dtype=np.int32)
        for f in p["files"]:
            src = Path(dataset_root) / f["src"]
            dst = Path(out_root) / f["copy"]
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src, dst)
            os.chmod(dst, 0o644)
            with av.open(str(src)) as container:
                stream = container.streams.video[0]
                k = -1
                for k, frame in enumerate(container.decode(stream)):
                    t = float(frame.pts * stream.time_base)
                    if abs(t * p["fps"] - k) > 1e-3:
                        raise RuntimeError(f"{src}: frame {k} has pts {t} s, off the 1/{p['fps']} grid")
                    for a, b, row0 in f["segments"]:
                        if a <= k < b:
                            mm[row0 + k - a] = frame.to_ndarray(format="rgb24")
                            written[row0 + k - a] += 1
            report.setdefault(cam, []).append({"file": f["src"], "decoded": k + 1})
        mm.flush()
        del mm
        if not np.all(written == 1):
            raise RuntimeError(f"{cam}: {int((written == 0).sum())} rows never written, "
                               f"{int((written > 1).sum())} written twice")
        os.replace(tmp, out)
        os.chmod(out, 0o644)
    (Path(out_root) / "frames" / "decode_report.json").write_text(json.dumps(report, indent=1))


# --- on the host ------------------------------------------------------------------

def run(ds: str | None = None, force: bool = False) -> Path:
    root = paths.hf_root(ds)
    out = paths.out_dir(ds, create=True)
    if not (root / "meta" / "info.json").exists():
        raise SystemExit(f"no LeRobot dataset at {root}")

    arrays, plan = build_arrays(root)
    npz = paths.episodes_npz(ds)
    tmp = npz.with_suffix(".tmp.npz")
    np.savez(tmp, **arrays)
    os.replace(tmp, npz)
    print(f"wrote {npz}  ({len(arrays['index'])} rows, episodes {arrays['episodes'].tolist()}, "
          f"fps {int(arrays['fps'])})")

    todo = {c: p for c, p in plan.items() if force or not _cache_ok(paths.frames_npy(ds, c), p["shape"])}
    if not todo:
        print("frame caches already complete (use --force to redo)")
    else:
        (out / "frames").mkdir(exist_ok=True)
        plan_file = out / "frames" / "decode_plan.json"
        plan_file.write_text(json.dumps(todo, indent=1))
        uid, gid = os.getuid(), os.getgid()
        inner = (f"trap 'chown -R {uid}:{gid} /out' EXIT; "
                 "python -c 'import sys; from real2sim.extract import _decode_plan; _decode_plan(*sys.argv[1:])' "
                 "/out/frames/decode_plan.json /d /out")
        cmd = ["docker", "run", "--rm", "--user", "root",
               "-v", f"{root}:/d:ro", "-v", f"{out}:/out", "-v", f"{paths.PKG}:/r2s/real2sim:ro",
               "-e", "PYTHONPATH=/r2s", "-e", "PYTHONDONTWRITEBYTECODE=1", "--entrypoint", "bash",
               paths.DOCKER_IMAGE, "-c", inner]
        t0 = time.time()
        print("decoding in docker:", ", ".join(todo), "...", flush=True)
        subprocess.run(cmd, check=True)
        print(f"decoded in {time.time() - t0:.1f} s")
    verify(ds)
    return npz


def _cache_ok(path: Path, shape) -> bool:
    try:
        a = np.load(path, mmap_mode="r")
        return a.shape == tuple(shape) and a.dtype == np.uint8
    except (OSError, ValueError):
        return False


def verify(ds: str | None = None) -> dict:
    """Frame counts, ownership and the per-episode slicing, with numbers."""
    z = np.load(paths.episodes_npz(ds), allow_pickle=False)
    n, out = len(z["index"]), paths.out_dir(ds)
    res = {"rows": n}
    for c, cam in enumerate(z["cams"]):
        a = np.load(out / str(z["frames_file"][c]), mmap_mode="r")
        if a.shape[0] != n:
            raise RuntimeError(f"{cam}: {a.shape[0]} cached frames, {n} parquet rows")
        for f in set(z["video_file"][:, c]):
            if not os.access(out / f, os.R_OK):
                raise RuntimeError(f"{out / f} is not readable")
        # a frame that decoded to all-black would still have the right shape
        means = [float(a[int(z["ep_from"][e])].mean()) for e in range(len(z["episodes"]))]
        res[str(cam)] = {"shape": list(a.shape), "first_frame_mean": [round(m, 1) for m in means],
                         "bytes": (out / str(z["frames_file"][c])).stat().st_size}
        if min(means) < 1.0:
            raise RuntimeError(f"{cam}: an episode's first frame is black")
    print(json.dumps(res))
    return res


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--ds", default=None, help=f"dataset name (default {paths.dataset()})")
    ap.add_argument("--force", action="store_true", help="re-decode frames even if the caches look complete")
    ap.add_argument("--verify", action="store_true", help="only check existing outputs")
    a = ap.parse_args(argv)
    verify(a.ds) if a.verify else run(a.ds, a.force)


if __name__ == "__main__":
    sys.exit(main())
