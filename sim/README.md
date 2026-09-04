# sim2real: push the printed T, in Isaac Sim

Trains a push-T policy in simulation against a scene built to match *this*
workspace's hardware, so the policy has a chance of working on the real arm.

Based on NVIDIA's [Sim-to-Real-SO-101-Workshop](https://github.com/isaac-sim/Sim-to-Real-SO-101-Workshop),
vendored at `thirdparty/sim2real_so101`. We take its **USD assets** — the SO-ARM101
arm, the mat, the light box — and its tuned actuator gains. We do **not** import its
Python: it is pinned to Isaac Lab 2.3 APIs and this box runs 2.1. Depending on the
submodule for assets and not for code keeps the Isaac version a one-line choice.

## What is modelled from the real setup

| Real thing | In sim | Fidelity |
|---|---|---|
| Printed T, `t_20_factor_0.5_scaled.stl` | `assets/usd/t_block.usd` | Exact mesh. 80x80 mm footprint, 20 mm bar and stem, 16 mm thick, 44.8 cm³. Convex **decomposition** collision. |
| Printed camera mount, `klip_support-1.stl` | `assets/usd/klip_support.usd` | Exact mesh, visual only, riding the wrist. |
| Logitech C270, overhead | `camera_front` | Intrinsics from measured crop behaviour + spec dFOV. Extrinsics estimated. |
| Klip Xtreme KWC-500 FHD, wrist | `camera_grip` | Same. |
| SO-ARM101 | workshop USD + gains | The workshop's, tuned on a real SO-101. |

### The T's collision shape is the load-bearing detail

A T is non-convex. Its convex **hull** fills in both notches and turns a shape that
catches a gripper into a pentagon that slides off one. `sim_convert_assets.py` forces
`collision_approximation="convexDecomposition"`; if a push policy trained in sim
slides off the real T, check this first.

### Cameras

Two claims, with different confidence:

**Measured** — both sensors are natively 16:9 and produce 4:3 by cropping the sides,
not by adding height. Template-matching a live 4:3 frame inside a live 16:9 frame
scores 0.987 (C270) and 0.998 (KWC-500) for "centred horizontal crop", against 0.402
and 0.860 for the alternative. So 640x480 — what `.env` records and infers at — is
*narrower* than the sensor. Assume otherwise and the sim gets ~11 degrees of extra
horizontal view on the C270 and ~15 on the KWC-500.

    ./robot sim-camera-check       # reproduces this against the live cameras

**Spec, not measured** — absolute focal length comes from the manufacturers' diagonal
FOV (C270 55°, KWC-500 80°). That is a spec sheet, not this unit. Replace it:

    ./robot sim-calibrate --camera front    # OpenCV checkerboard
    ./robot sim-calibrate --camera grip

which writes `fx/fy/cx/cy` into `sim_agent101/config/cameras.json` with
`source="calibrated"`. Isaac Lab builds the camera through
`PinholeCameraCfg.from_intrinsic_matrix`, so nothing else changes.

**Estimated, and the weakest link** — where the cameras *sit*. The overhead height
(0.59 m above the mat) is back-solved from the real frame spanning ~0.40 m across a
37.6° horizontal FOV, which is better than a guess but not a measurement. The wrist
offset is eyeballed from the real wrist frame. Both are constants at the top of
`tasks/push_t_env_cfg.py` (`OVERHEAD_POS`, `WRIST_POS`, `WRIST_PITCH_DEG`). A wrist
view 2 cm out will not transfer however good the intrinsics are — tune these against
real captures before collecting data.

## Running it

Isaac Sim here is a **native conda install, not Docker**, unlike everything else in
this repo: it needs the GPU plus a 2.3 GB shader cache and 37 GB of Omniverse data in
`$HOME`. `scripts/sim/sim.sh` selects the environment from `SIM_CONDA_ENV` in `.env`.

    ./robot sim-assets            # STL -> USD, once (and after editing the CAD)
    ./robot sim-play              # build the scene, step it, render both cameras
    ./robot sim-play --gui        # ...and watch
    ./robot sim-teleop            # drive the scene from the real leader arm
    ./robot sim-record            # ...and write a LeRobot dataset while you do
    ./robot sim-train             # train reach with PPO (headless, 4096 envs)
    ./robot sim-policy            # watch a checkpoint and score its tracking

`sim-play` writes renders to `outputs/sim/` and checks the things that fail silently:
the T resting on the mat rather than sinking or drifting, both cameras returning
non-black frames at the right resolution, the goal error being finite.

### Which Isaac environment

`SIM_CONDA_ENV=45pysaac` — Isaac Sim 4.5.0 + Isaac Lab 2.1.0.

`env_isaaclab` (Isaac Sim 5.1.0 + Isaac Lab 2.3.0) is also installed and would match
the workshop's own pinning, but it hangs during Kit startup on this box — main thread
spinning at 100%, log frozen at ~700 ms, with or without cameras. Its `isaaclab` core
package was also missing from the env, which suggests that install was never
finished. If you repair it, `SIM_CONDA_ENV` is the only thing that should change.

## Layout

    sim_agent101/
      cameras.py              real camera model; the measurement lives in its docstring
      config/cameras.json     intrinsics, spec or calibrated
      assets/cad/             the printed parts, as printed
      assets/usd/             generated by sim-assets, not committed
      assets/objects.py       T block, camera mount, camera factory
      assets/so101.py         the arm, pointed at the workshop USD
      mdp/push_t.py           where the T is, where it should go, when that is done
      mdp/randomize.py        lighting, camera and robot-colour randomization
      mdp/reach.py            distance to a commanded gripper pose
      tasks/push_t_env_cfg.py the push-T scene
      tasks/reach_env_cfg.py  the reach task, which is the one with rewards
      tasks/agents/           PPO hyperparameters, keyed to a task by gym.register
    scripts/
      sim.sh                  runs a script inside the Isaac environment
      sim_convert_assets.py   STL -> USD
      sim_play.py             build + validate the scene
      sim_teleop.py           leader arm -> the Isaac scene
      sim_record.py           ...and out to a LeRobot dataset
      sim_train.py            rsl_rl PPO
      sim_policy.py           run a checkpoint, report tracking error
      sim_camera_check.py     sim camera assumptions vs the real cameras
      sim_calibrate_cameras.py checkerboard intrinsics

## Task definition

`Agent101-So101-Push-T` — teleop and data collection. The T starts anywhere in a
16x16 cm patch at any yaw; the goal is a pose in a 12x12 cm patch at any yaw, drawn
as a translucent green T. 60 s episodes, no rewards, no success termination.

`Agent101-So101-Push-T-Eval` — adds success: centroid within 2 cm and yaw within 15°
of the goal, *and* the T has stopped moving. 30 s episodes.

Both measure the T at its **centroid**, not its mesh origin — the origin sits at the
middle of the crossbar, so scoring there would reward parking the bar on the goal
with the stem pointing anywhere. Yaw error is wrapped to (-π, π] and deliberately not
folded by symmetry: a T has none, and upside down is a different outcome on the mat.

`Agent101-So101-Push-T-DR` — the same scene with domain randomization on every
reset: friction, mass, actuator gains, lighting, camera pose and the robot's shade.
A separate task id rather than a flag, so *which scene was this recorded in* is
answerable from a dataset's task name alone.

## Reach: the RL task

Push-T is an imitation task — the scene exists to be teleoped into a dataset, and
`PushTEnvCfg` has `rewards = None`. Reach is the other half, and the reason it is
here is narrow: nothing in this repo otherwise exercises **training** in Isaac Lab.
It needs no object and no cameras, it is the standard first manipulation task, and
it trains in minutes, so it is the cheapest available proof that the arm model, the
workshop's actuator gains and the rsl_rl stack all work before something harder
depends on them.

Built from Isaac Lab's own `manager_based/manipulation/reach` and the SO-ARM101
recipe on the [Seeed wiki](https://wiki.seeedstudio.com/es/training_soarm101_policy_with_isaacLab/)
(`MuammerBay/isaac_so_arm101`, BSD-3), with four changes:

| | here | the reference |
|---|---|---|
| arm | `SO101_CFG`, workshop USD + gains, joints `Rotation`/`Pitch`/… | its own SO-ARM USD, lerobot joint names |
| target box | derived from this bench: 0.10–0.25 m ahead of the root, ±0.10 m across, 0.05–0.20 m over the mat | a fixed box |
| control rate | 30 Hz on a 1/120 s step, matching push-T and `DATASET_FPS` | 30 Hz on 1/60 |
| orientation reward | weight 0 | weight 0 for SO-101, unexplained |

    ./robot sim-train                                  # 4096 envs, headless
    ./robot sim-train --task Agent101-So101-Reach-DR   # randomized actuator gains
    ./robot sim-train --max-iterations 250 --resume    # continue the newest run
    ./robot sim-train --name reach-v2                  # names the log dir and the run
    tensorboard --logdir sim/outputs/rsl_rl

**wandb is on by default**, the same rule `./robot train` follows: a `WANDB_API_KEY`
in `.env.local` or a `wandb login` in `~/.netrc`, project from `WANDB_PROJECT`. With
no credentials it says so and logs to tensorboard alone rather than failing a run
over a logger. `--no-wandb` and `--wandb-offline` override; `--wandb` forces it on
and fails fast if there is nothing to authenticate with.

Two things worth knowing, both of which look like wandb working when it is not.
`scripts/sim/sim.sh` now sources `.env.local` as well as `.env` — the sim commands are
the ones that skip Docker, so nothing else would ever export the key for them. And
`sim_train.py` calls `wandb.finish()` explicitly, because Kit ends the process inside
`app.close()` without running atexit handlers: without it the run directory appears,
the config is written, and the transaction log stays at zero bytes.

    ./robot sim-policy                                 # newest checkpoint, windowed
    ./robot sim-policy --headless --steps 600          # just the error numbers
    ./robot sim-policy --export                        # policy.pt + policy.onnx

    ./robot sim-policy --goal manual                   # ONE arm, and you move the goal
    ./robot sim-policy --goal auto                     # the goal walks a circle itself

`--goal` picks **who chooses the target**, and the policy is identical in all three.
`task` is the environment's own sampler — a new pose every 4 s, the distribution the
policy trained on, and the setting to score a checkpoint in. `manual` is the one to
show someone: a single arm and `/World/GoalHandle`, an orange ball that *is* the
goal, draggable with the translate gizmo while the sim runs, or driven with W/S
(forward/back), A/D (left/right), Q/E (up/down), R to recentre. `auto` walks it round
a circle hands-free, which is how the manual path gets tested headless.

### Trying it on the real arm

    ./robot sim-policy --real --goal manual            # nothing moves; watch first
    ./robot sim-policy --real --goal manual --engage   # ...now the arm follows

The scene gains a **ghost arm**: the white one is what the policy commanded, the
translucent cyan one beside it is where the real robot actually is, snapped from its
encoders every step. Drag the goal and you watch both at once. `./robot` starts
`scripts/robot/policy_bridge.py` in the container and stops it after; that bridge is
the only process in this repo that writes to a motor. The two halves talk through
`sim/outputs/policy/{targets,state}.json`, atomically renamed, the same way
`leader-publish` and `sim-teleop` already do.

**Nothing moves without `--engage`.** `--real` alone runs the policy, draws both arms
and prints the worst per-joint gap while the bridge stays read-only — which is the
honest way to look at a checkpoint before trusting it. With `--engage`: commands are
rate-limited to 60 °/s per joint, a target frame older than 100 ms stops the
commanding, the gripper is never commanded at all (the policy does not drive it), and
torque is released on the way out.

Know what this checkpoint does not know before you engage it. It was trained with a
**massless wrist** — the printed mount and the webcam are visual-only in the sim, so
their ~100 g on the last link is unmodelled — with **no bench collision geometry**,
and with five joints. It can ask for a pose that puts the gripper through the table.

The ghost sits 0.6 m to the side rather than overlaid, and that is a workaround, not
a preference: two arms in the same place collide, and with the ghost coincident the
policy's own tracking error goes from 11 mm to 158 mm. Both config-level fixes were
tried — see the note on `GHOST_OFFSET_Y` in `tasks/reach_env_cfg.py`.

manual and auto disable the 4 s resampling — otherwise the goal jumps away mid-drag —
stretch the episode so a time-out cannot reset the arm on you, and clamp the goal to
the training box, so dragging past the edge stops at the edge instead of asking for
something unreachable and making the checkpoint look broken. Measured chasing the
`auto` circle at 12 cm/s: 7–18 mm, against 7 mm for a target standing still.

Watch `Metrics/ee_pose/position_error` — mean distance from the gripper to the
commanded point, in metres. Measured here, 250 iterations (24.5M steps, 4m21s on an
RTX 3060, 4096 envs) takes it from 0.185 m — what an untrained policy scores on this
target box — to 0.007 m. The agent config's default is 1000 iterations, which is where the
reference leaves it; this task was already at 7 mm by 250, so watch the curve rather
than the iteration count.

`sim-policy` on that checkpoint reports **mean 9.6 mm, p90 8.0 mm, worst 121 mm**
over 300 steps. The mean sitting *above* the p90 is the expected shape, not a bug:
the target resamples every 4 s and the first steps afterwards are travel, so a few
large values pull the mean up while the settled error is what the p90 shows.

### Three things about this task that are not the reference's

**The target box stops short of where push-T plays.** The T is sampled 0.19–0.31 m
from the root; reach targets stop at 0.25 m. Past ~0.27 m the SO-101 is near full
extension, the set of orientations it can achieve at a point collapses, and the
tracking error cannot go to zero however good the policy is. The arm *does* get out
to the T at 0.31 m — lying nearly flat on the mat, which is one configuration rather
than a region worth sampling free-space targets in.

**The action space is not push-T's.** Reach drives the five arm joints as a scaled
delta on the rest pose (`target = REST_POSE + 0.5 * action`), because an untrained
policy emitting absolute radians flails through the whole joint range on step one.
push-T sends absolute targets for all six joints, because a teleop recording *is*
absolute joint angles. A reach checkpoint therefore does not drop into the push-T
action pipeline unchanged; anything deploying one on the real arm has to apply the
same affine map.

**The scene has no bench.** Mat, table and light box are visual-only in push-T, so
they change nothing physical — but this scene is cloned thousands of times, where
"visual only" still costs a prim and a draw call per environment. The bench survives
where it matters, in the target box. What it costs is that the arm can sweep below
the bench plane, which the real one cannot; targets are all above the mat so it has
no reason to, but watch a checkpoint in `./robot sim-policy` before trusting it near
real hardware.
