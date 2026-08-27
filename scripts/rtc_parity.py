#!/usr/bin/env python3
"""Pin openpi's RTC port to LeRobot's implementation, numerically.

The two stacks cannot share a process — le101 needs CUDA torch, openpi needs
JAX — so this runs in two passes over a JSON file:

    ./robot rtc-parity            # both passes, in their own containers
    python scripts/rtc_parity.py --dump  out.json    # in agent101/lerobot
    python scripts/rtc_parity.py --check out.json    # in agent101/openpi

The dump exercises le101's RTCProcessor directly: prefix weights over a grid of
(d, s, H, schedule), and a full guided denoise step against a linear denoiser
whose Jacobian is known, so the check can tell a faithful port from a plausible
one. openpi.models.rtc.rtc_test covers the same ground against a transcription
of the reference; this is the version that runs the reference itself.
"""

import argparse
import json
import pathlib

# H is openpi's default action horizon; 10 is le101's default execution_horizon.
GRID = [
    # (inference_delay d, prefix_attention_horizon s, action_horizon H)
    (0, 10, 50), (1, 10, 50), (4, 10, 50), (9, 10, 50), (10, 10, 50),
    (14, 29, 50), (25, 25, 50), (0, 50, 50), (49, 50, 50), (60, 50, 50),
    (3, 0, 50), (0, 0, 50), (2, 6, 16), (5, 20, 32),
]
SCHEDULES = ["zeros", "ones", "linear", "exp"]
TIMES = [1.0, 0.9, 0.75, 0.5, 0.25, 0.1, 0.0]
MAX_GUIDANCE_WEIGHT = 10.0  # le101's RTCConfig default

# Denoise-step case: a linear denoiser v(x) = x @ A^T + b, small enough to write
# down and with an exactly known Jacobian.
CASE = {"batch": 1, "horizon": 16, "dim": 4, "delay": 4, "prefix_horizon": 10, "time": 0.6, "seed": 0}


def _case_arrays(np):
    rng = np.random.default_rng(CASE["seed"])
    b, h, d = CASE["batch"], CASE["horizon"], CASE["dim"]
    return (
        rng.normal(size=(d, d)).astype("float32"),          # A
        rng.normal(size=(d,)).astype("float32"),            # bias
        rng.normal(size=(b, h, d)).astype("float32"),       # x_t
        rng.normal(size=(b, h, d)).astype("float32"),       # previous chunk
    )


def dump(path: pathlib.Path) -> int:
    """Run le101's RTCProcessor and write its outputs. Needs the lerobot image."""
    import numpy as np
    import torch

    from lerobot.configs import RTCAttentionSchedule
    from lerobot.policies.rtc.configuration_rtc import RTCConfig
    from lerobot.policies.rtc.modeling_rtc import RTCProcessor

    out: dict = {"grid": [], "guidance": [], "case": dict(CASE)}

    for name in SCHEDULES:
        proc = RTCProcessor(
            RTCConfig(
                prefix_attention_schedule=RTCAttentionSchedule[name.upper()],
                max_guidance_weight=MAX_GUIDANCE_WEIGHT,
            )
        )
        for d, s, h in GRID:
            weights = proc.get_prefix_weights(d, s, h)
            out["grid"].append({"schedule": name, "d": d, "s": s, "h": h, "weights": weights.tolist()})

    # beta(tau) is not exposed on its own, so recover it from a denoise step whose
    # correction is known: with a zero previous chunk and all-ones weights, the
    # step returns v - beta * correction.
    a, bias, x_t, prev = _case_arrays(np)
    a_t, bias_t = torch.from_numpy(a), torch.from_numpy(bias)
    denoise = lambda x: torch.einsum("ij,bhj->bhi", a_t, x) + bias_t  # noqa: E731

    proc = RTCProcessor(
        RTCConfig(
            prefix_attention_schedule=RTCAttentionSchedule[SCHEDULES[-1].upper()],
            max_guidance_weight=MAX_GUIDANCE_WEIGHT,
        )
    )
    for time in TIMES:
        v = proc.denoise_step(
            torch.from_numpy(x_t),
            torch.from_numpy(prev),
            CASE["delay"],
            time,
            denoise,
            execution_horizon=CASE["prefix_horizon"],
        )
        out["guidance"].append({"time": time, "v_guided": v.detach().numpy().tolist()})

    # Unguided reference for the same inputs, so the check can isolate the guidance.
    out["v_plain"] = denoise(torch.from_numpy(x_t)).detach().numpy().tolist()
    out["max_guidance_weight"] = MAX_GUIDANCE_WEIGHT
    path.write_text(json.dumps(out))
    print(f"wrote {path}  ({len(out['grid'])} weight cases, {len(out['guidance'])} denoise steps)")
    return 0


def check(path: pathlib.Path) -> int:
    """Recompute everything with openpi's RTC. Needs the openpi image."""
    import jax.numpy as jnp
    import numpy as np

    from openpi.models import rtc as _rtc

    ref = json.loads(path.read_text())
    failures: list[str] = []

    for case in ref["grid"]:
        got = np.asarray(_rtc.prefix_weights(case["d"], case["s"], case["h"], _rtc.schedule_code(case["schedule"])))
        want = np.asarray(case["weights"], dtype=np.float32)
        if got.shape != want.shape or not np.allclose(got, want, atol=1e-6):
            failures.append(
                f"weights {case['schedule']} d={case['d']} s={case['s']} h={case['h']}: "
                f"max |diff| {np.abs(got[: len(want)] - want[: len(got)]).max() if got.shape == want.shape else 'shape'}"
            )
    print(f"prefix weights: {len(ref['grid']) - len(failures)}/{len(ref['grid'])} match")

    a, bias, x_t, prev = _case_arrays(np)
    a_j, bias_j, x_j, prev_j = (jnp.asarray(v) for v in (a, bias, x_t, prev))
    velocity = lambda x: jnp.einsum("ij,bhj->bhi", a_j, x) + bias_j  # noqa: E731
    weights = _rtc.prefix_weights(CASE["delay"], CASE["prefix_horizon"], CASE["horizon"], _rtc.EXP)

    np.testing.assert_allclose(np.asarray(velocity(x_j)), np.asarray(ref["v_plain"], dtype=np.float32), atol=1e-5)

    n_guidance = len(ref["guidance"])
    guidance_failures = 0
    for entry in ref["guidance"]:
        want = np.asarray(entry["v_guided"], dtype=np.float32)
        for jacobian in ("identity", "full"):
            got = np.asarray(
                _rtc.guided_velocity(
                    velocity, x_j, entry["time"], prev_j, weights, ref["max_guidance_weight"], jacobian=jacobian
                )
            )
            if np.allclose(got, want, atol=1e-4):
                break
        else:
            guidance_failures += 1
            failures.append(f"denoise step time={entry['time']}: no jacobian mode matches (max |diff| {np.abs(got - want).max():.3e})")
            continue
        print(f"  denoise step time={entry['time']:<4} matches jacobian={jacobian}")
    print(f"guided velocity: {n_guidance - guidance_failures}/{n_guidance} match")

    if failures:
        print("\nFAILURES:")
        for f in failures:
            print(" ", f)
        return 1
    print("\nopenpi RTC matches the LeRobot reference.")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--dump", metavar="JSON", help="run the le101 reference and write it (lerobot image)")
    g.add_argument("--check", metavar="JSON", help="recompute with openpi and compare (openpi image)")
    args = p.parse_args()
    return dump(pathlib.Path(args.dump)) if args.dump else check(pathlib.Path(args.check))


if __name__ == "__main__":
    raise SystemExit(main())
