"""`./robot real2sim calib fit [--quick]`: the whole calibration, end to end.

    holdout   fit episodes 0 + 1, score episode 2 (evaluate.py)
    loeo_1    fit 0 + 2, score 1          } full mode only: the leave-one-episode-out
    loeo_0    fit 1 + 2, score 0          } spread of every global parameter
    final     fit 0 + 1 + 2 (the values that go into the config)      } full mode only

The folds run as parallel subprocesses (`python -m real2sim.calib.fit`), each with its
own EGL renderer and a share of the Jacobian workers, then report.build() writes
report.json, the overlays and contact sheets, and -- full mode only, and only if the
held-out gates in report.GATES pass -- config/<ds>.calib.json.

Outputs: sim/outputs/real2sim/<ds>/calib/ (quick: calib/quick/, never the config).
MEASURED wall time on this box (16 cores, RTX 3060, shared with the other tracks):
quick (hold-out fold only, fewer frames) ~3.5 min, full ~20 min.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time

from .. import paths

FOLDS_FULL = {"holdout": ((0, 1), (2,)), "loeo_1": ((0, 2), (1,)), "loeo_0": ((1, 2), (0,)), "final": ((0, 1, 2), ())}
FOLDS_QUICK = {"holdout": ((0, 1), (2,))}  # a smoke test: the hold-out fold alone, report only


def out_dir(ds=None, quick=False, create=False):
    return paths.out_dir(ds, "calib", *(["quick"] if quick else []), create=create)


def run(ds=None, quick=False, verbose=None, jobs=None) -> dict:
    verbose = verbose or (lambda msg: print(msg, flush=True))
    ds = paths.dataset(ds)
    out = out_dir(ds, quick, create=True)
    folds = jobs or (FOLDS_QUICK if quick else FOLDS_FULL)
    cpu = os.cpu_count() or 4
    workers = max(2, min(12, cpu - 2) // len(folds))
    t0 = time.time()
    procs = {}
    for name, (train, test) in folds.items():
        cmd = [*paths.HOST_PY, "-m", "real2sim.calib.fit", "--train", ",".join(map(str, train)),
               "--test", ",".join(map(str, test)), "--out", str(out / name), "--ds", ds] + (["--quick"] if quick else [])
        env = dict(os.environ, MUJOCO_GL="egl", R2S_CALIB_WORKERS=str(workers), PYTHONDONTWRITEBYTECODE="1",
                   OMP_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2", MKL_NUM_THREADS="2")
        log = open(out / f"{name}.stderr.txt", "w")
        procs[name] = (subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT), log)
        verbose(f"calib: {name}: train {train} test {test or '-'} ({workers} Jacobian workers)")
    failed = []
    for name, (p, log) in procs.items():
        rc = p.wait()
        log.close()
        if rc != 0 or not (out / name / "eval.json").exists():
            failed.append(name)
        verbose(f"calib: {name} done (rc {rc}, {time.time() - t0:.0f} s)")
    if failed:
        raise RuntimeError(f"folds failed: {failed}; see {out}/<fold>.stderr.txt and <fold>/log.txt")
    from . import report

    summary = report.build(ds, out, list(folds), quick=quick, wall=time.time() - t0, verbose=verbose)
    if summary.get("config"):  # the cap identities of the observations follow the new calibration
        from . import observations

        observations.label(ds, verbose=False)
        observations.jaw_angles(ds, verbose=False)
    verbose(f"calib: {time.time() - t0:.0f} s total; report {out / 'report.json'}")
    return summary


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="real2sim CALIB: hold-out + LOEO + final fit, report, config")
    ap.add_argument("--quick", action="store_true", help="2 folds on fewer frames, < 5 min; never writes the config")
    ap.add_argument("--ds")
    a = ap.parse_args()
    s = run(a.ds, a.quick)
    print(json.dumps(s.get("gates", {}), indent=1))
    sys.stdout.flush()
    os._exit(0)
