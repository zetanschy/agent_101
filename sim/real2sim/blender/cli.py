"""`./robot real2sim blender ...`: render pose logs / kinematic replays and score them.

    render --episode N (--kinematic | --poselog PATH|mujoco) [--engine eevee|cycles] [--samples-only]
           [--frames A:B[:S]] [--cams front,grip] [--out DIR] [--png] [--keep-pinhole] [--no-video]
           [--exposure constant|look] [--wb frame|episode|reference] [--quality key=value ...]
    samples [--engine eevee|cycles] [--poselog mujoco|PATTERN] [...]
                                            the 24 LOOK samples (episodes 0-2) + score; kinematic,
                                            or the nominal MuJoCo replays / one log per {ep}
    score DIR [--wb]                        LOOK's image metrics of DIR/samples/*.png

Every run writes to sim/outputs/real2sim/<ds>/blender/<tag>/ (tag = ep<N>_<source>_<engine>
[_samples]): front.mp4, grip.mp4, side_by_side.mp4, samples/<id>_<cam>.png for LOOK's
sample frames inside the rendered range, manifest.json (what was rendered, from which
scene / look / log, the camera model, timings, warnings) and score.json when samples
were rendered. Blender runs under the shared GPU lock (paths.gpu_locked).
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from .. import episodes as E
from .. import paths
from ..scene import load as load_scene
from . import camera_model as CM
from .job import JobSpec, Source, build_job, load_look, load_source, sample_frames
from .post import Consumer, write_json

BL_RENDER = Path(__file__).with_name("bl_render.py")
MUJOCO_NOMINAL = "mujoco/logs/ep{ep}_action_seed0.npz"  # mujoco/README.md: the nominal action replays
DEFAULT_QUALITY = {
    # EEVEE: 64 TAA samples converge the jittered soft shadow of the 30 deg key (measured
    # 0.32 s/frame here); Cycles: 64 spp + OptiX denoise (0.44 s/frame at 640x480,
    # tooling report), adaptive threshold 0.01
    "eevee": {"samples": 64, "shadow_rays": 2, "shadow_steps": 8, "raytracing": True, "fast_gi": True},
    "cycles": {"samples": 64, "adaptive_threshold": 0.01, "denoise": True, "max_bounces": 6, "diffuse_bounces": 3,
               "glossy_bounces": 4, "clamp_indirect": 10.0},
}


def parse_frames(s: str | None, n: int) -> list[int] | None:
    if not s:
        return None
    parts = [int(p) if p else None for p in s.split(":")]
    a, b = parts[0] or 0, parts[1] if len(parts) > 1 and parts[1] is not None else n
    step = parts[2] if len(parts) > 2 and parts[2] else 1
    return list(range(a, min(b, n), step))


def parse_quality(items) -> dict:
    out = {}
    for it in items or []:
        k, v = it.split("=", 1)
        try:
            out[k] = json.loads(v)
        except json.JSONDecodeError:
            out[k] = v
    return out


def run_blender(job_dir: Path, pin_dir: Path, cams, consumer: Consumer | None, mode: str = "beauty",
                log=print) -> dict:
    """Blender under the GPU lock, streaming its R2S_FRAME lines into the consumer."""
    cmd = paths.gpu_locked((*paths.BLENDER, str(BL_RENDER), "--", "--job", str(job_dir), "--out", str(pin_dir),
                            "--cams", ",".join(cams), "--mode", mode))
    t0 = time.time()
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    done, build, tail = None, None, []
    t_first = None
    for line in p.stdout:
        tail = (tail + [line.rstrip()])[-40:]
        if line.startswith("R2S_FRAME"):
            _, cam, i, dt, _path = line.split(maxsplit=4)
            if t_first is None:
                t_first = time.time() - t0
            if consumer is not None:
                consumer.frame_done(cam, int(i))
        elif line.startswith("R2S_BUILD"):
            build = line.split(maxsplit=2)[2]
            log(f"  blender scene built in {line.split()[1]} s ({time.time() - t0:.1f} s after launch incl. the "
                f"GPU-lock wait): {build.strip()}")
        elif line.startswith("R2S_DONE"):
            done = json.loads(line.split(maxsplit=1)[1])
    p.wait()
    if p.returncode != 0 or done is None:
        raise RuntimeError(f"blender failed (exit {p.returncode}):\n" + "\n".join(tail))
    done["wall_s"] = round(time.time() - t0, 1)
    return done


def render(a, log=print) -> dict:
    ds = paths.dataset(a.ds)
    if a.poselog == "mujoco":
        a.poselog = str(paths.out_dir(ds, MUJOCO_NOMINAL.format(ep=a.episode)))
    source = Source.kinematic() if a.kinematic else Source.log(a.poselog)
    eps = E.load(ds, include_excluded=True)
    ep = eps[a.episode]
    samples = {s["frame"]: s["id"] for s in sample_frames(ds, a.episode)}
    if a.samples_only:
        frames = sorted(samples)
    else:
        frames = parse_frames(a.frames, len(ep))
    tag = a.tag or f"ep{a.episode}_{source.tag}_{a.engine}" + ("_samples" if a.samples_only else "")
    out = Path(a.out) if a.out else paths.out_dir(ds, "blender", tag, create=True)
    out.mkdir(parents=True, exist_ok=True)
    job_dir, pin_dir = out / "_job", out / "_pinhole"
    quality = {**DEFAULT_QUALITY[a.engine], **parse_quality(a.quality)}
    cams = tuple(a.cams.split(",")) if a.cams else ("front", "grip")
    log(f"blender render: {ds} episode {a.episode}, {source.label}, engine {a.engine} -> {out}")
    t0 = time.time()
    job = build_job(JobSpec(ds, a.episode, source, a.engine, frames, cams, quality), job_dir, log)
    t_job = time.time() - t0
    base = load_scene(ds)
    _, scene, _ = load_source(base, a.episode, source)
    look, _ = load_look(ds)
    resp = CM.responses(scene, look, job, ds, wb=a.wb, exposure=a.exposure)
    consumer = Consumer(job, job_dir, pin_dir, out, resp, samples, E.fps(ds), ep.start, video=not a.no_video,
                        png_all=a.png, keep_pinhole=a.keep_pinhole)
    timing = run_blender(job_dir, pin_dir, cams, consumer, log=log)
    stats = consumer.close(source.label)
    manifest = {
        "schema": "real2sim.blender.render/1", "dataset": ds, "episode": a.episode, "tag": tag,
        "source": job["source"], "mode_label": source.label, "engine": a.engine, "quality": quality,
        "frames": {"count": len(job["frames"]), "first": job["frames"][0], "last": job["frames"][-1],
                   "unique_renders": {c: job["cameras"][c]["renders"] for c in cams}},
        "scene_hash": job["scene_hash"], "scene_overrides_applied": job["scene_overrides_applied"],
        "look": job["look"], "job_hash": job["job_hash"],
        "cameras": {c: {"pinhole": job["cameras"][c]["pinhole"], "latency_frames": job["cameras"][c]["latency_frames"],
                        "latency_source": job["cameras"][c]["latency_source"], "model": resp[c].describe()}
                    for c in cams},
        "dataset_derived_inputs": [
            "table texture: look/table/albedo_lin.npy (front clean plate + wrist mosaic, de-lit), declared background",
            "backdrop cylinder + environment: look/backdrop (wrist frames beyond the mat), declared background",
            "camera states: front white-balance gain per row, wrist white balance per frame (LOOK, global numbers)"],
        "timing": {"job_s": round(t_job, 2), "blender": timing, "post": stats,
                   "total_s": round(time.time() - t0, 1)},
        "warnings": job["warnings"],
    }
    if any(fr in samples for fr in job["frames"]):
        from ..look.imagemetrics import score

        res = score(out / "samples", ds, erode_px=2)
        res["per_sample"] = [r for r in res["per_sample"]]
        write_json(out / "score.json", res)
        manifest["score"] = {cam: {k: v for k, v in (res["summary"][cam] or {}).items()} for cam in res["summary"]}
        _print_score(res, log)
    write_json(out / "manifest.json", manifest)
    # the textures are ~70 MB per job and a pure function of the look assets (job.json
    # records their inputs hash): job.json + job.npz stay for inspection, a re-render rebuilds
    for t in job_dir.glob("tex_*.npy"):
        t.unlink()
    if not a.keep_pinhole:
        for c in cams:
            d = pin_dir / c
            if d.exists() and not any(d.iterdir()):
                d.rmdir()
        if pin_dir.exists() and not any(pin_dir.iterdir()):
            pin_dir.rmdir()
    per = {c: (timing["cameras"][c]["mean_s_after_first"], timing["cameras"][c]["first_s"]) for c in cams}
    log(f"done in {manifest['timing']['total_s']} s; blender s/frame after the first (first): {per}; -> {out}")
    return manifest


def _print_score(res, log=print):
    for cam, summ in res["summary"].items():
        if not summ:
            continue
        for region, v in summ.items():
            if v:
                log(f"  {cam:5s} {region:10s} psnr {v['psnr']:6.2f}  ssim {v['ssim']:.3f}  dE(mean) "
                    f"{v['delta_e_mean_colour']:6.2f}  dE(px) {v['delta_e_pixels']:6.2f}  n={v['samples']}")


def samples_cmd(a, log=print) -> dict:
    """Render the 24 LOOK samples (episodes 0-2) into ONE directory and score it: kinematic
    by default, or from one pose log per episode (--poselog with an {ep} placeholder;
    'mujoco' = the nominal MuJoCo action replays), where the caps are where physics put
    them instead of static at reset."""
    ds = paths.dataset(a.ds)
    pattern = str(paths.out_dir(ds, MUJOCO_NOMINAL)) if a.poselog == "mujoco" else a.poselog
    src = "kinematic" if not pattern else ("mujoco-nominal" if a.poselog == "mujoco" else "poselog")
    tag = a.tag or f"samples_{src}_{a.engine}"
    out = paths.out_dir(ds, "blender", tag, create=True)
    res = {}
    for ep in (0, 1, 2):
        b = argparse.Namespace(**{**vars(a), "episode": ep, "kinematic": not pattern,
                                  "poselog": pattern.format(ep=ep) if pattern else None, "samples_only": True,
                                  "frames": None, "out": str(out / f"ep{ep}"), "tag": f"{tag}/ep{ep}",
                                  "no_video": True})
        res[ep] = render(b, log)
    (out / "samples").mkdir(exist_ok=True)
    for ep in (0, 1, 2):
        for p in (out / f"ep{ep}" / "samples").glob("*.png"):
            os.replace(p, out / "samples" / p.name)
    from ..look.imagemetrics import score

    sc = score(out / "samples", ds, erode_px=2)
    write_json(out / "score.json", sc)
    log(f"all samples ({a.engine}, {src}):")
    _print_score(sc, log)
    timing = {ep: m["timing"] for ep, m in res.items()}
    write_json(out / "manifest.json", {"source": src, "poselog_pattern": pattern,
                                       "scene_hash": {ep: m["scene_hash"] for ep, m in res.items()},
                                       "episodes": {ep: m for ep, m in res.items()}, "score": sc["summary"],
                                       "timing": timing})
    return sc


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="./robot real2sim blender", description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("render", "samples"):
        p = sub.add_parser(name)
        p.add_argument("--ds")
        p.add_argument("--engine", choices=("eevee", "cycles"), default="eevee")
        p.add_argument("--cams", help="comma list (default front,grip)")
        p.add_argument("--quality", action="append", metavar="KEY=VALUE", help="engine setting override, repeatable")
        p.add_argument("--exposure", choices=("constant", "look"), default="constant",
                       help="wrist exposure model (camera_model.py)")
        p.add_argument("--wb", choices=("frame", "episode", "reference"), default="frame",
                       help="camera white balance: the real row's (LOOK), the episode mean, or none")
        p.add_argument("--keep-pinhole", action="store_true", help="keep the linear pinhole EXRs")
        p.add_argument("--png", action="store_true", help="also write every frame as PNG")
        p.add_argument("--tag", help="output directory name under blender/")
        if name == "samples":
            p.add_argument("--poselog", help="pose log per episode, '{ep}' in the path; 'mujoco' = "
                           "sim/outputs/real2sim/<ds>/" + MUJOCO_NOMINAL + " (default: kinematic)")
        if name == "render":
            p.add_argument("--episode", type=int, required=True)
            g = p.add_mutually_exclusive_group(required=True)
            g.add_argument("--kinematic", action="store_true", help="arm = FK(observation.state), objects static")
            g.add_argument("--poselog", help="a real2sim pose log (.npz) of this episode; 'mujoco' = the nominal "
                           "MuJoCo action replay, sim/outputs/real2sim/<ds>/" + MUJOCO_NOMINAL)
            p.add_argument("--samples-only", action="store_true", help="only LOOK's sample frames of the episode")
            p.add_argument("--frames", help="A:B[:S] episode-local frames (default: all)")
            p.add_argument("--out", help="output directory (default sim/outputs/real2sim/<ds>/blender/<tag>)")
            p.add_argument("--no-video", action="store_true")
    p = sub.add_parser("score")
    p.add_argument("dir")
    p.add_argument("--ds")
    p.add_argument("--wb", action="store_true")
    a = ap.parse_args(argv)
    if a.cmd == "render":
        render(a)
    elif a.cmd == "samples":
        samples_cmd(a)
    elif a.cmd == "score":
        from ..look.imagemetrics import score

        d = Path(a.dir)
        res = score(d / "samples" if (d / "samples").is_dir() else d, a.ds, erode_px=2, wb=a.wb)
        write_json((d / "score.json"), res)
        _print_score(res)
    return 0


if __name__ == "__main__":
    sys.exit(main())
