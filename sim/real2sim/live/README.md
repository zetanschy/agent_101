# real2sim live: teleop it, record with it, run policies on it

The simulated follower as a lerobot robot. Every command that drives the arm takes
`--sim mujoco|isaac` and drives the sim instead. The leader stays the real one.

```
./robot teleop --sim mujoco                                   # the real leader drives the sim
./robot record --sim mujoco --name cap_to_mug --task "Put the cap into the red cup" --episodes 20
./robot infer  --sim mujoco --policy <repo> --task "..." [--rtc|--async]      # lerobot policies
./robot openpi-eval --sim mujoco --policy /checkpoints/openpi_pi05_lora_cap_to_cup_200 \
        --task "Put the cap into the red cup"                  # the openpi (JAX) pi0.5 stack
./robot infer-remote --sim isaac ...                           # the robot half of remote inference
```

| option | default | |
|---|---|---|
| `--sim mujoco\|isaac` | — | the engine |
| `--sim-layout real:N\|random\|random:K` | `random` | where the caps and the mug start: episode N's calibrated layout, or K (1-3) random caps and the mug |
| `--sim-viewer` | off | the engine's 3D window; `R` there = a new layout |
| `--sim-clock realtime\|lockstep` | `realtime` | see below |
| `--sim-seed S` | — | reproducible random layouts |

**A new layout** comes from pressing `r` in the terminal that runs lerobot (the plugin
listens the way lerobot-record listens for its arrow keys), `R` in the viewer, or
`./robot real2sim live reset [--layout ...]` from another terminal. lerobot-record gives a
robot no hook between episodes, and a reset on a timer could teleport objects in the
middle of a recorded episode. So you reset during the reset phase, as you would on the bench.

**Recorded datasets** come out of lerobot 0.6.1 itself, through `record.sh`. They have the
real dataset's features (same keys, shapes and names), `robot_type: real2sim`, and a
`sim_<engine>_` prefix on the name, so a sim dataset is never filed under a real one's.

## How it works

```
real leader ── lerobot (Docker) ── lerobot_robot_real2sim ──unix socket── live.server (native) ── mujoco | isaac
               teleoperate/record/                            sim/outputs/real2sim/live/sim.sock
               rollout/openpi eval
```

- `./robot ... --sim ENGINE` starts `live.server` in the engine's interpreter: the
  mjlab uv env for MuJoCo, the Isaac env for Isaac, which holds the GPU lock for the session.
- It then runs the usual script in Docker with `ROBOT_TYPE=real2sim`,
  `ROBOT_PORT=<socket>`, and `sim/real2sim/lerobot_plugin` on `PYTHONPATH`.
- lerobot's `register_third_party_plugins()` finds the plugin's dist-info there, so
  `--robot.type=real2sim` exists with no pip install.
- The plugin's `get_observation` / `send_action` are one socket round trip each
  (`protocol.py`: JSON header + raw array buffers, not pickle, because the two ends run
  different numpy majors).
- **A command is the replay's servo, live.** `goals.OnlineGoals` is the online twin of
  the replay's `GoalStream`: the identified servo model, the firmware clamp, and one frame
  of dead time. Fed episode 0's recorded actions in lockstep, the live MuJoCo follower
  reproduces the evaluated replay bitwise (`tests/test_live.py`).
- **The per-pixel work runs in the plugin.** The server is single-threaded (EGL and Kit
  belong to the thread that made them), and warping plus blurring both cameras there cost
  18 of the 33 ms tick. That starved the next request (act: 27.6 ms median). So the
  engine sends pinhole renders, the server hands the remap maps and blur over once at
  hello, and the plugin applies them with cv2. The frames match the server's own warp to
  0.06 / 0.15 grey levels on average (front / wrist). A client without cv2 still gets
  finished frames.
- **An observation is what the bus returns**: joint readings through the scene's fitted
  offsets and gripper map. They are in degrees (`use_degrees=True`, this bench) or lerobot's
  -100..100 range normalisation (`use_degrees=False`, which the openpi checkpoints use;
  converted through the follower calibration's ticks exactly as the bus does). Each
  fitted camera is rendered with its distortion.
- **Clock.** `realtime` advances 1/30 s of sim per 1/30 s of wall whether or not a command
  arrived, because the real servo keeps holding its goal while a policy thinks.
  `lockstep` advances one step per command, deterministic, for an engine or renderer
  slower than real time.

## What the MuJoCo engine measures (RTX 3060 box)

| | |
|---|---|
| physics | 9.8x real time on one core |
| reset (build + settle) | 1.1-1.2 s |
| both cameras, server side (pinhole render) | ~3 ms; the distortion warp + lens blur run in the plugin (cv2, ~1 ms a camera) |
| observation round trip from the container, both cameras | median 11.5 ms, p90 14.6 ms; an action 0.2 ms |
| teleop loop with the real leader | 59 Hz mean (lerobot-teleoperate's 60 Hz target; sim at 1.00x real time) |
| openpi pi0.5 cap-to-cup, closed loop | 372 ms per 50-action chunk, sim at 1.00x real time |

The images are MuJoCo's rasterizer, dressed with what LOOK derived from the dataset that
a rasterizer can carry (`look_mujoco.py`):
- the recorded mat texture, at 1 mm on the table plane;
- the key light fitted to the mug shadows;
- the webcams' fitted softness.
They have no reflections in the steel mug and no room behind the arm, and a policy
trained on real frames sees that gap. For photoreal frames of a physics run, render its
pose log with Blender (`blender/`); Isaac renders RTX live.

## Known limits

- **lerobot's pi0.5 does not fit this 12 GB card.** It runs out of memory while loading, at
  8.5 GB, on the sim exactly as it would on the arm. Run it through `policy-serve` on a
  bigger GPU with `./robot infer-remote --sim ...`, or use the openpi stack, which peaks at
  6.9 GB.
- **Inference is only as good as the images.** The openpi cap-to-cup checkpoint, trained on
  real frames, drove the MuJoCo sim for 40 s without placing the cap. The MuJoCo images
  are a rasterizer's.
- **No episode hook in lerobot-record:** reset the layout yourself between episodes (above).

## Files

| file | what |
|---|---|
| `server.py` | the socket server: clock, requests, bus units, layouts, status |
| `engine.py` | the interface an engine implements |
| `engine_mujoco.py` | MuJoCo: the replay's model and servo stepped live, EGL render + torch warp |
| `../isaac/live.py` | Isaac: the Isaac track's scene and servo stepped live, RTX render |
| `goals.py` | the servo's goal stream, online |
| `layouts.py` | real and random object layouts, the rest pose |
| `look_mujoco.py` | the dataset mat texture and fitted light on the MuJoCo scene |
| `protocol.py` | the wire format |
| `../lerobot_plugin/` | `lerobot_robot_real2sim`: the `real2sim` robot (config + class + dist-info) |
| `run.sh` | `./robot real2sim live serve\|reset\|status\|ping\|test` |
