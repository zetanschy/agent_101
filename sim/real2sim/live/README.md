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
./robot webui --sim mujoco          # the browser panel (lerobot + RL backends) on the sim
./robot openpi-webui --sim mujoco   # the openpi backend: Load, Run, Stop, Home as on the arm
```

On this PC the openpi webui's default port 8001 is taken by Omniverse's thumbnail service
(and 8002 by its tagging service). Set `OPENPI_WEBUI_PORT` (e.g. 8011, in `.env.local`) or
stop those services. That applies to the real arm as much as to the sim.

| option | default | |
|---|---|---|
| `--sim mujoco\|isaac` | — | the engine |
| `--sim-layout real:N\|random:K\|random` | `random:1` | where the caps and the mug start: one cap and the mug at random reachable spots, drawn afresh at every reset; `random:K` K caps (1-3); `random` 1-3; `real:N` episode N's calibrated layout (`real:0` is one cap) |
| `--sim-viewer` | off | the engine's 3D window; `R` or Reset there = a new layout |
| `--sim-clock auto\|realtime\|lockstep` | `auto` | realtime for MuJoCo, lockstep for Isaac (below) |
| `--sim-seed S` | — | reproducible random layouts |

**Watching it.** Two views, usable together:
- `./robot real2sim live watch` in a second terminal, while the webui, a teleop or a
  recording runs, shows both sim cameras exactly as the policy receives them, with sim
  time, layout and caps in the mug on top. It is observe-only: it never commands the arm
  and does not advance a lockstep sim. `r` gives a new layout, `f` turns both cameras
  180 deg (display only: the policy and recordings get it as mounted; `--flip`
  starts that way), `q` closes the window; `--snapshot out.png` writes one frame without a
  window.
- `--sim-viewer` on the command itself opens the engine's 3D window with a free camera
  (`R` or Reset = new layout). For example: `OPENPI_WEBUI_PORT=8011 ./robot openpi-webui --sim
  mujoco --sim-viewer`.

**A new layout** (cap and mug moved to fresh random spots, arm back at rest) comes from any
of these:
- **By itself before every recorded episode after the first:** once the previous episode
  is saved (its video encoded) and before the new one records a frame. lerobot-record
  announces "Recording episode N" through logging, and the plugin, running in the same
  process, re-places at that line, so the episode's first frame already shows the new
  positions. `R2S_NEW_LAYOUT_EACH_EPISODE=0` turns it off.
- `l` in the terminal that runs lerobot. Not `r`, which lerobot-record binds to re-record.
- In MuJoCo's viewer: the **Reset** button, Backspace, or `R`. The viewer's own Reset
  would put everything back where the model started and snap the arm to its zero pose.
  The engine sees sim time run backwards, never integrates that state, and draws a new
  layout instead.
- `./robot real2sim live reset [--layout ...]` from another terminal.

A layout with the same bodies as the current one (the default `random:1` always is) is
re-placed in the running model in ~0.02 s, so the viewer window stays open. A different
cap count rebuilds the model (~1-3 s). lerobot-record gives a
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

## What each engine measures (RTX 3060 box)

| | MuJoCo | Isaac |
|---|---|---|
| replay over the socket, `--layout real:0 --start recorded` | bitwise (0.0 deg) | bitwise (0.0 deg) |
| physics | 9.8x real time, one CPU core | **0.19x real time** (the evaluated PhysX: 16 substeps, 64 solver iterations, one env) |
| default clock (`auto`) | realtime | lockstep |
| server-side images, both cameras | pinhole render ~3 ms; warp + blur in the plugin (cv2) | RTX real-time 34 ms + camera model on the GPU 2 ms |
| observation / action round trip | 11.5 ms / 0.2 ms | 1.5 ms (pre-rendered) / ~200 ms (the step) |
| real-leader teleop loop | 59 Hz, sim at 1.00x real time | 5.1 Hz, one 1/30 s step per command |
| reset (new layout) | 1.1-1.2 s | 1.7-2.0 s |
| start-up to serving | ~5 s | ~18 s (Kit) |
| GPU memory | EGL only | 4.8-5.2 GB, and the GPU lock for the session |
| images | a rasterizer with LOOK's mat, light and lens softness | RTX with the full LOOK: steel, room backdrop, key + dome light |

**Why Isaac runs in lockstep.** It cannot hold real time with the physics the comparison
evaluated, and changing that physics would make a different sim. In lockstep every
command advances exactly 1/30 s of sim time, so every recorded frame is one physics step
and a dataset is physically consistent, however long each step takes in wall time.
Two consequences:
- The arm moves at about a fifth of the leader's speed in wall time. To record a demo at
  natural sim speed, move the leader slowly.
- lerobot-record counts `episode_time_s` / `reset_time_s` in wall seconds. At ~5 frames a
  second, 30 s of wall time records ~150 frames, 5 s of sim. Raise
  `--dataset.episode_time_s` about 6x for Isaac.
MEASURED: a 10 s episode recorded 51 frames.

**Isaac next to a policy on this card.** With a local policy, Isaac's 5.2 GB, the openpi
checkpoint's 6.9 GB peak and the desktop's ~3 GB do not fit in 12 GB. Put the policy on
another GPU (`./robot policy-serve` there, `./robot infer-remote --sim isaac` here), or
run the policy against MuJoCo.

The MuJoCo images are a rasterizer, dressed with what LOOK derived from the dataset that
a rasterizer can carry (`look_mujoco.py`):
- the recorded mat texture, at 1 mm on the table plane;
- the key light fitted to the mug shadows;
- the webcams' fitted softness.
They have no reflections in the steel mug and no room behind the arm. Isaac renders those.

## Known limits

- **lerobot's pi0.5 does not fit this 12 GB card.** It runs out of memory while loading, at
  8.5 GB, on the sim exactly as it would on the arm. Run it through `policy-serve` on a
  bigger GPU with `./robot infer-remote --sim ...`, or use the openpi stack, which peaks at
  6.9 GB.
- **Inference: the openpi cap-to-cup checkpoint puts the cap in the mug on MuJoCo.** MEASURED
  through `./robot openpi-webui --sim mujoco --sim-layout real:0`: RTC mode, degrees, 75
  inferences at 389 ms. The cap entered the mug ~19 s after Run (the real episode 0 took
  ~12 s), with the look-dressed MuJoCo images. An earlier 40 s `openpi-eval --sim mujoco`
  run did not place it, but it differed in three ways at once: plain MuJoCo images,
  synchronous chunks, and normalized joint units (evaluate.py's default). One run is not a
  success rate; this says the loop works end to end, not how often the policy succeeds.
- **No episode hook in lerobot-record:** reset the layout yourself between episodes (above).
- **Isaac's viewer** is dark: exposure and grading would also rescale the frames the cameras
  record, so it is left alone. Closing its window ends the session, because Kit's shutdown
  hangs on this box.
- **Isaac resets after the first** re-place a pool of three caps on one stage (Replicator
  cannot tear a stage down and rebuild it in-process). They are near-exact rather than
  bitwise: episode 0 replayed after a reset matches for 214 frames, then within 0.09 deg.

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
| `watch.py` | `live watch`: both sim cameras in a window, observe-only |
| `../lerobot_plugin/` | `lerobot_robot_real2sim`: the `real2sim` robot (config + class + dist-info) |
| `run.sh` | `./robot real2sim live serve\|reset\|status\|ping\|test` |
