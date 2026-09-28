# real2sim / mujoco — the real episodes, replayed physically in MuJoCo

This track replays every real pick of `soarm101/testingt_real2sim_20260927_152936` in
MuJoCo 3.11 (CPU, the mjlab uv env).

- The arm is driven by the recorded **action**, through an identified model of its
  STS3215 servos.
- The caps are placed once, on the table, and move only through contact. Nothing is
  welded, scripted or pushed by an added force.
- The gripper squeezes no harder than the real servo's identified torque limit. The arm
  pushes no harder than the STS3215's stall torque.

All numbers below were measured on this box. The results (section 5) are on the fitted
scene, hash `6f282e812cc585f5`: CALIB's dataset-only calibration merged over the
provisional config and the servo model.

## Commands

```
./robot real2sim mujoco fingers                  # CoACD of the finger parts (astribot_simu env, ~60 s, cached)
./robot real2sim mujoco servo-fit [--iters 25]   # servo model -> config/<ds>.servo.json (~15 min, 8 processes)
./robot real2sim mujoco replay --episode 0 --mode action|track|kinematic [--seed S] [--ensemble K]
                               [--render] [--gui] [--frames A:B] [--tag T]
./robot real2sim mujoco eval [--episodes 0,1,2] [--modes action,track] [--ensemble 20] [--render] [--tag T]
./robot real2sim mujoco render --log <pose log .npz> [--every N]   # mp4: real | sim, front over wrist
./robot real2sim mujoco view --log <pose log .npz>                 # mujoco.viewer (needs DISPLAY=:0)
./robot real2sim mujoco info                     # servo model, physics, scene feasibility checks
./robot real2sim mujoco test                     # 25 tests, ~30 s
```

The official evaluation is `./robot real2sim mujoco eval --ensemble 20`: 9 min on
6 processes, written to `eval.json`.

`replay` and `eval` also take what-if options. All of them are recorded in each log's
`meta` and in `eval.json`, and `--set` changes the scene hash. **A what-if run needs
`--tag T`.** The untagged logs and `eval.json` are the nominal replays that other tracks
read.

| option | what it does |
|---|---|
| `--placement grasp` | puts each cap where both real fingers held it antipodally at the real blocked frame (`model.grasp_placements`, the Isaac track's estimator) |
| `--set PATH=VALUE` | overrides any merged-scene value, e.g. `table.z=0.02`, `objects.cap.diameter=0.02932`, `table.tilt=[dx,dy]` |
| `--sigma cap_xy=M,table_z=M,cap_d=LO:HI,cap_h=M` | ensemble uncertainties instead of the scene's |
| `--cap-shape taper` | cap collision as CALIB's fitted frustum instead of the mid-height cylinder (section 1) |

The scene is `real2sim.scene.load()`, so CALIB's `config/<ds>.calib.json` is picked up
automatically.

Outputs go to `sim/outputs/real2sim/<ds>/mujoco/`:

| path | contents |
|---|---|
| `logs/ep<N>_action_seed0.npz` | **the nominal action replays of the official eval**, one per used episode (the Blender track renders ep0's). The scene hash is in `meta.scene_hash`. |
| `logs/ep<N>_<mode>_seed<S>[_<tag>].npz` | `real2sim.poselog` pose logs, with `.metrics.json` beside `replay` logs; ensemble logs are in `logs/[<tag>/]` |
| `video/ep<N>_action_seed0.mp4` | real \| sim, front over wrist, for the nominal action replays |
| `eval.json`, `eval_<tag>.json` | evaluations: `eval.json` is the fitted scene, `eval_sens_cap_closed_end.json` the sensitivity run |
| `eval_provisional.json`, `logs/provisional/`, `video/provisional/` | the provisional-scene evaluation, for the history (section 7) |
| `runs/` | stdout of the comparison runs quoted below |
| `servo_fit.json` | the servo identification report |
| `coacd/` | the finger pieces |

## 1. The model (`model.py`)

**Arm.** `so101_calib.xml` with the base at identity.

- Collision hulls are in group 3. There is no self-collision.
- Joint ranges are the firmware range plus 5°. The data exceed the URDF on 4 joints.
- **The Jaw's lower limit is −11.5°**, where the finger meshes touch. This is the real
  shut stop (2049 ticks in all 4 episodes), and it takes the place of finger–finger
  collision.

**Fingers (`fingers.py`).** Both finger hulls are replaced by their CoACD pieces
(threshold 0.05; 48–51 and 22 pieces).

- The real inner faces are stepped planes at x = ∓7.9 / 9.9 / 11.6–11.9 mm, and CoACD
  cuts exactly at those steps.
- The single hull of `wrist_roll_follower` fills the grasp opening from above. Its volume
  is 2.58× the part's; the CoACD pieces total 1.09×.

**Mount.**

- `klip_support`: 18 g (READ, mjlab).
- KWC-500 body: 20 g (ESTIMATED: a 30×26×16 mm block plus lens and cable, range 10–40 g).
- Both are welded to the gripper and are visual only.

**Cameras.** `front` sits in the world and `grip` on `gripper`, each as an oversized
pinhole (`camera.render_spec`). `render.py` warps each render through the real distortion.

- Test: marker spheres render where `Camera.project` puts them, median < 0.2 px, for
  both cameras. The grip camera was checked at a recorded arm pose.

**Table.** A box whose top is at `table_top(x, y)` = z plus `table.tilt` when the scene
has one.

**Objects** come from `real2sim.objects`.

- Cap: primitive-cylinder collision of the scene's `diameter`, hollow-cap visual and
  hollow-cap inertia.
  - CALIB fits a taper, 29.32 mm at the closed end to 32.32 mm at the open end, and puts
    the mid-height 30.82 mm in `diameter`.
  - `--cap-shape taper` collides with that frustum instead. The table below shows it
    fits the grasps worse.
- Mug: a floor cylinder, 32 wall boxes whose inner faces lie at the inner radius, and
  6 handle capsules. Mesh inertia, 0.30 kg, free body.

**Solver.**

- dt = 1/1800 s (60 substeps per frame), implicitfast, elliptic cone, impratio 100,
  50 iterations.
- Object and finger contacts use solref (2 dt, 1), critically damped, with solimp
  (0.99, 0.999, 0.5 mm) and condim 4.
- The joints' dry friction is a hard constraint.

**Why those contact numbers.** MuJoCo's contact stiffness is an *acceleration* per metre
of penetration. Force per metre therefore scales with the effective mass, and a 2.5 g cap
has almost none.

- MuJoCo's defaults sink a squeezed cap 7.6 mm per newton, which matches the 5–6.6 mm the
  understand phase measured.
- These settings: a 12–16 N squeeze sinks it 0.03–0.09 mm.
- A cap dropped into the mug from 15 cm penetrates ≤ 0.26 mm on impact, against 0.9–1.4 mm
  at 30 substeps. It bounces less than 5 % of the drop.

**Cap collision shape on the fitted scene.** Action mode, nominal, all six picks. Jaw hold
error is sim − real in degrees; + means the sim jaw is more open (`runs/taper_ep*.log`).

| collision | ep0 A (closed up) | ep1 A (closed) | ep1 B (open) | ep1 C (open) | ep2 A (open) | ep2 B (open) | ep0 released |
|---|---|---|---|---|---|---|---|
| **cylinder, 30.82 mm** (default) | +0.53 | +0.73 | +0.81 | +0.52 | +1.17 | +0.65 | no |
| taper, 29.32 → 32.32 mm | +0.30 | +0.78 | +1.41 | +1.33 | +1.79 | +1.40 | no |

- The fingers hold the closed-up caps near their closed top (contacts 9–13 mm above the
  rim plane). They hold the open-up caps on their rims (0–5 mm), the frustum's wide end.
- The taper therefore opens the four rim grasps by another 0.6–0.8° and still does not
  release ep0.
- Its finger–cap penetration is 0.08–0.13 mm, and every other pick ends in the mug as
  with the cylinder.

**Measured alternatives (what-if geometry, section 7).** Track mode, ep2 caps A / B:

| variant | outcome | jaw hold − real | slip | finger–cap penetration |
|---|---|---|---|---|
| **default** (CoACD, flex, condim 4, impratio 100, 60 substeps) | both in the mug, drop −1 / −2 frames | +0.08 / −0.26° | 0.06 / 0.03 mm | 0.08 mm |
| mjlab single finger hulls | both missed (jaw closes past) | −5.3 / −13.8° | — | 0.2 mm |
| rigid jaw (no flex) | in the mug | **+2.7 / +3.3°** | 0.05 / 0.50 mm | 0.07 mm |
| condim 3 | in the mug | +0.13 / −0.26° | **2.27** / 0.20 mm | 0.08 mm |
| condim 6 | as default | +0.18 / −0.26° | 0.07 / 0.03 mm | 0.08 mm |
| impratio 10 | in the mug | +0.10 / −0.26° | 0.38 / 0.08 mm | 0.08 mm |
| 30 substeps | in the mug | +0.05 / −0.40° | 0.11 / 0.09 mm | **0.22 mm** |
| cap 12 mm tall | both missed (the rim grasps do not reach) | — | — | 0.1 mm |
| cap 29 mm (provisional) | in the mug | **+1.41 / +1.35°** | 0.06 / 0.03 mm | 0.61 mm |

## 2. The servo model (`servo.py`, `servo_fit.py`, `grip_hold.py`)

Per joint, in SI units:

```
tau = clip(kp (u - q) - kd qdot, ±effort) - damping qdot - frictionloss sign(qdot)
```

- `u` is the recorded action. It is clamped to the firmware limits, held for one frame,
  delayed by the dead time, and slewed at ≤ 5.29 rad/s (Rhoban BAM, STS3215 m1).
- `kd` is inside the clamp (the firmware's D term). `damping` is outside it (the
  motor's back-EMF).
- **The gripper adds a series flex.** `Jaw_flex` is a hinge coaxial with the horn in the
  jaw body. The encoder reads the horn; the finger is held back by the cap through PLA
  flex, cap squash and horn give.

### The kv discrepancy, resolved

Test conditions: mjlab's constants (kp 17.8, kv 2.0, armature 0.1), ep0, no dead time.

| kv as | force clamp | mean arm RMSE | lag (real 4.7 frames) | peak speed (real 146°/s) |
|---|---|---|---|---|
| joint damping (outside the clamp) | 1.5 N·m | 7.93° | 9.4 frames | 55°/s |
| joint damping (outside the clamp) | 3.0 N·m | 1.63° | 4.9 | 97°/s |
| actuator velocity bias (inside the clamp) | 1.5 or 3.0 N·m | 1.16° | 4.0 | 154–156°/s |

- Euler and implicitfast agree to 0.01° at both dt 1/480 and 1/900.
- The cause is the clamp: damping b outside a clamp F caps the speed at F/b = 0.75 rad/s.

### Identification

- Input and target: action → observation.state, with contacts off.
- Train on episodes 0 and 1; episode 2 is held out.
- A separable (1+λ)-ES, 931 s on 8 processes.
- The five arm joints share one STS3215 model (kp, kd, damping, armature), with a
  per-joint frictionloss.
- The Jaw keeps the shared kp and fits the rest of its parameters on free frames. An
  unloaded joint only fixes ratios: a free per-joint fit gave kp 3.8.

| | kp N·m/rad | kd | damping | armature kg·m² | frictionloss N·m | clamp N·m |
|---|---|---|---|---|---|---|
| arm (shared) | **9.354** | 0.083 | 0.770 | 0.0090 | 0.063 / 0.275 / 0.288 / 0.106 / 0.039 | 3.41 |
| Jaw | 9.354 | 0.766 | 0.061 | 0.0065 | 0.067 | 1.155 |

- BAM's STS3215 model predicts kp = 9.34 at P = 16.
- The dead time is 33.3 ms (1 frame), on top of the slew.
- Jaw flex: 19.23 N·m/rad. Its damping, 0.0176 N·m·s/rad, is ASSUMED (ζ 0.5 of the
  ~120 Hz mode).

| RMSE vs observation.state (°) | pan | lift | elbow | wrist flex | wrist roll | Jaw (free frames) |
|---|---|---|---|---|---|---|
| ep0 (train) | 0.25 | 0.96 | 0.79 | 0.54 | 0.41 | 0.19 |
| ep1 (train) | 0.29 | 0.99 | 0.71 | 0.34 | 0.22 | 0.17 |
| **ep2 (held out)** | **0.27** | **1.23** | **0.84** | **0.44** | **0.32** | **0.89** |

**Peak speeds.** On held-out ep2 the sim peaks at 163°/s on lift (real 171°/s) and
165°/s on elbow (real 166°/s).

**The arm clamp.** The recorded ep2 motion needs up to 3.16 N·m on Pitch by inverse
dynamics. That is above the vendor ratings (1.91 N·m at 7.4 V, 2.94 N·m at 12 V), so the
clamp is BAM's full-duty stall of 3.41 N·m. With the clamp at 2.22 N·m, ep2 lift and
elbow peaked at only 141 / 154°/s.

**Hard joint friction.** With MuJoCo's default soft friction, a joint the real servo
holds still crept 0.74° in 1.5 s (0.30° with the hard constraint). Held-out elbow RMSE
went from 1.25° to 0.84°.

### Gripper holds (`grip_hold.py`)

Over 6 squeeze plateaus the encoder falls linearly with the commanded squeeze:
`encoder = 16.62 % − 0.486 × min(err, 5.30 %)`, residual 0.14 % RMS (max 0.22 %). A
rigid jaw cannot produce that line; a spring in series with a P servo does. The ep2 B
hold, at 10.8 % error, sits on the clamp.

- Flex stiffness = kp / 0.486 = 19.2 N·m/rad.
- Clamp = kp × 5.30 % = **1.155 N·m**. That is inside the 0.95–1.76 bound, and 34 % of
  the stall rather than the 50 % of Max_Torque_Limit. The gripper's Overload_Torque 25
  protection is the likely reason (UNVERIFIED).
- **The fingers first touch the caps at 16.62 % on the encoder.** Under CALIB's gripper
  map that is Jaw 10.81°, where the finger-mesh gap in the contact band is
  **29.54 mm**. (It was 8.94° and 27.15 mm under the provisional tick_physical map.)
- In ep0 the real cap drops as the encoder settles at 16.64 %, with 16.90 % commanded.
  The horn stops 0.26 % short of the command, inside the Jaw's 0.41° friction deadband.

## 3. Replay modes (`replay.py`)

**action** is the physically honest default: all six servos get the recorded action.

**track** is a hybrid.

- The arm servos get the goal that makes the servo model reproduce the recorded state:
  `u = q_ref + (τ + kd q̇)/kp`, with τ from inverse dynamics along a spline of
  observation.state.
- The clamp and the compliance stay, so a contact the recording knew nothing about still
  pushes the arm.
- The Jaw stays on the action.

**kinematic** is NOT physical (`meta.physical = false`). Joints are set to the state and
objects stay put; use it for camera and render checks only.

All caps of a multi-cap episode are present from the start, in one continuous simulation.

## 4. What is measured (`evaluate.py`)

**Per pick:**

- grasped / lifted (≥ 20 mm) / carried (both fingers on the cap for ≥ 90 % of the
  transport)
- slip in the gripper frame
- sim separation and drop frames vs the real drop (the wrist camera's teal drop)
- in the mug at the end
- the sim encoder vs observation.state over the hold
- squeeze (normal force per finger, summed over contacts)
- finger–cap penetration, max and p99
- fingertip error at grasp: the grasp site under the sim joints vs FK(observation.state)
  at the real blocked frame, as a distance and as dz

**Per episode:**

- joint RMSE (lag 0 and best lag) and grasp-site error
- outcome agreement and mug displacement
- spurious arm–object force
- actuator force vs clamp
- arm–table clearance, sim and recorded
- speed
- cap pixel error (`cap_pixels`) against CALIB's real teal blobs
  (`observations/detections.npz` + `labels.npz`):
  - per camera, per cap, per phase: rest, carried and in the mug (front view only)
  - the sim cap centre is projected with the scene camera; the wrist camera uses the sim
    gripper pose
  - the sim is sampled at the image's exposure, row − `cameras.<cam>.latency.frames`
    (2.27 front, 1.42 wrist), linearly between frames; `cap_pixels_no_latency` is the
    same comparison without that shift
  - identity comes from CALIB's labels, which are gated around the calibrated prediction
    and never the sim's, so a phase CALIB could not label is reported with n = 0
  - in-mug blobs carry no identity; per frame they are paired with the sim caps whose
    real drop is over, closest pairs first

**Ensembles** perturb, per seed:

- cap xy, table z, and the cap's diameter and height, by the scene's sigmas or `--sigma`
  - The cap uses CALIB's σ of 0.89 mm (D) and 2.28 mm (h). D is clipped to the
    provisional 23–32 mm `diameter_range`.
  - D used to be drawn uniformly from that range. On the fitted scene the range swamps
    the fitted σ: ep0's ensemble rose from 5 % to 65 % in the mug for the wrong reason.
- the four frictions, uniformly within their material ranges
- cap mass, 2–3 g

## 5. Results on the fitted scene: `eval.json`, hash `6f282e812cc585f5`

The scene is CALIB's dataset-only calibration:

- table 23.99 ± 5.47 mm
- joint offsets (0.65, 1.13, −4.43, 0.29, −2.09)°
- gripper map −11.13° + 1.3208°/% × pct
- cap 30.82 mm (mid-height) × 13.88 mm; mug 86.8 × 121 mm
- both cameras, with their latencies

The physics is section 1's, with the cylinder cap.

**Feasibility** (`scene_checks`):

- The recorded arm clears the table by +0.05 / +2.98 / −1.60 mm in ep0 / ep1 / ep2. ep2's
  −1.6 mm is within the table's σ; the sim arm rests on the table there (−0.06 mm).
- The fingers' zero-force gap is 29.54 mm; the scene's cap is 30.82 mm (+1.28 mm).

### Per pick, nominal (seed 0)

Action mode; track-mode values in parentheses where they differ. Jaw hold is the sim encoder
vs observation.state over the hold. The fingertip error is at the real blocked frame. The
last column is the 20-seed ensemble, in the mug, action / track.

| pick (up end) | lifted | in the mug | drop − real | jaw hold sim / real | squeeze per finger | slip | finger–cap pen. max / p99 | fingertip at grasp (dz) | ensemble in the mug |
|---|---|---|---|---|---|---|---|---|---|
| ep0 A (closed) | 181 mm | **no: never released** | — | 8.96 / 8.43° (+0.53) | 13.6 N | 0.03 mm | 0.03 / 0.03 mm (0.15 / 0.03) | 5.6 mm (−0.1) (0.4) | 5 / 5 % |
| ep1 A (closed) | 165 mm | yes | −1 | 10.46 / 9.73° (+0.73) | 6.4 N | 0.01 mm | 0.06 / 0.02 mm (0.62 / 0.61) | 5.7 mm (−3.7) (0.2) | 100 / 70 % |
| ep1 B (open) | 161 mm | yes | 0 | 8.91 / 8.10° (+0.81) | 14.3 N | 0.02 mm | 0.04 / 0.04 mm | 4.8 mm (−4.4) (0.3) | 95 / 80 % |
| ep1 C (open) | 160 mm | yes | −1 | 8.19 / 7.68° (+0.52) | 14.6 N | 0.05 mm | 0.03 / 0.03 mm | 1.8 mm (−1.1) (0.2) | 65 / 65 % |
| ep2 A (open) | 197 mm | yes | 0 | 9.30 / 8.13° (+1.17) | 15.8 N | 0.06 mm | 0.38 / 0.03 mm (0.08 / 0.06) | 4.3 mm (+1.1) (2.4) | 85 / 90 % |
| ep2 B (open) | 170 mm | yes | −2 | 8.14 / 7.48° (+0.65) | 15.9 N | 0.06 mm | 0.08 / 0.08 mm (0.03 / 0.03) | 5.8 mm (−5.0) (0.3) | 95 / 85 % |

- **5 of 6 picks end in the mug, in both modes.** Every drop is within 2 frames of the
  real one, and every jaw hold is 0.5–1.2° too open (section 6).
- **ep0 A is grasped, lifted and carried, but never released** (diagnosis in section 6).
- Track mode keeps each pick's outcome and drop frame. Its jaw holds are within 0.2° of
  action mode.
- **Fingertip error at grasp** is 1.8–5.8 mm in action mode. The sim grasp site sits up to
  5 mm lower than FK(state), which is the servo model's open-loop error. Track mode:
  0.2–2.4 mm.

### Per episode

| | joint RMSE vs state, ° (pan lift elbow w.flex w.roll Jaw) | grasp-site error mean / p95 | 20-seed success | ensemble finger–cap pen. max / p99 median |
|---|---|---|---|---|
| ep0 action | 0.33 0.81 0.63 0.43 0.40 **7.87** | 3.2 / 7.7 mm | 5 % | 1.51 / 0.04 mm |
| ep0 track | 0.02 0.04 0.04 0.04 0.01 7.87 | 0.2 / 0.6 mm | 5 % | 0.86 / 0.06 mm |
| ep1 action | 0.30 0.95 0.69 0.34 0.22 0.56 | 4.7 / 9.0 mm | 65 % | 0.63 / 0.07 mm |
| ep1 track | 0.03 0.02 0.05 0.09 0.01 0.53 | 0.2 / 0.4 mm | 65 % | 0.69 / 0.39 mm |
| ep2 action | 0.34 1.00 0.87 0.45 0.32 1.04 | 5.2 / 11.0 mm | 85 % | 1.67 / 0.09 mm |
| ep2 track | 0.07 0.05 0.06 0.03 0.01 1.04 | 0.3 / 1.4 mm | 85 % | 0.73 / 0.08 mm |

- **ep0's Jaw RMSE of 7.87°** comes from the unreleased cap: the sim jaw stays blocked on
  it while the real one shuts.
- **Arm and servos.** No arm link touched an object in any nominal run (0.0 N). The
  actuator peaks stay under the clamps (Pitch 2.71 of 3.41 N·m, ep2 track). The Jaw
  squeezes on its 1.155 N·m clamp.
- **The rim graze.** The mug moved 0.0 mm in the action nominals and 0.5 mm in ep1 track.
  - There the carried cap A grazes the mug's rim near frame 275. That also slides it
    ~7 mm across the fingers.
  - It is the only nominal finger–cap penetration above 0.4 mm: 0.62 mm over 9 frames.
  - In 18 of 40 ep1 ensemble seeds the same graze moves the mug by 1.4–19 mm.
- **Ensemble penetration.** 4 of 120 seeds exceed 1 mm, each on a single frame: 4 of
  27,340 touched frames, max 1.67 mm.
  - All four drew a cap of ≥ 31.2 mm on a table raised ≥ 2.8 mm.
  - The p99 medians are 0.04–0.09 mm; ep1 track's 0.39 mm is the rim graze.
- **Speed:** 3.8–4.8× real time per process with one eval running, 2.0–4.2× with two.

**What the ensembles lose.** These are the per-seed perturbations of the failed picks.

- **ep0 A fails on the drawn cap width.** The seed that released it drew 28.7 mm; the
  held ones averaged 31.3 mm.
- **Almost every other failure is a seed whose fingers stop above the cap.**
  - The failed seeds' tables averaged 1.5–13.9 mm below 24.0 mm, and their caps
    9.4–13.3 mm tall.
  - The successful seeds averaged +0.2..+2.0 mm and 14.0–14.4 mm.
- **The table's σ of 5.47 mm (CALIB's LOEO spread) sets most of these rates** (section 6).

### The cap in both cameras, sim vs real (`cap_pixels`)

Pixels are median / p90 (frames), latency-corrected (section 4). The wrist and in-mug
columns give action / track medians.

| pick | front rest | front carried | front in the mug | wrist rest | wrist carried |
|---|---|---|---|---|---|
| ep0 A | 6 / 6 (119) | — (0) | 190 / 188 (98): not released | 24 / 14 (147) | 32 / 16 (12) |
| ep1 A | 4 / 7 (113) | — (0) | 15 / 18 (457) | 12 / 6 (182) | 10 / 10 (78) |
| ep1 B | 3 / 7 (411) | — (0) | 16 / 32 (285) | 27 / 18 (148) | 28 / 22 (46) |
| ep1 C | 8 / 11 (8) | 15 / 17 (2) | 3 / 9 (44) | 26 / 13 (167) | 29 / 14 (46) |
| ep2 A | 7 / 7 (124) | 11 / 15 (67) | 34 / 34 (305) | 18 / 14 (179) | 18 / 12 (98) |
| ep2 B | 4 / 6 (538) | — (0) | 28 / 40 (152) | 24 / 8 (143) | 27 / 16 (65) |

- **Rest, front view: 3–8 px.** The caps start where CALIB put them. ep1 C is the
  exception (section 6).
- **Rest, wrist view: 6–18 px in track mode**, where the arm is at the recorded pose. In
  action mode it is 12–27 px, because the arm is up to 5 mm off at the approach.
- **Carried.** The wrist view (10–32 px action, 10–22 px track) measures where the cap
  sits in the gripper. The front view sees a carried cap under the gripper, so CALIB
  labels it on only 2–67 frames.
- **In the mug, front view: 3–40 px.** This is where the cap came to rest on the mug
  floor, which is chaotic after a drop onto polished steel. CALIB marks as in-mug any teal
  blob inside the rim circle, which can include the cap's reflections.
- **The latency correction matters where something moves.**
  - Wrist rest: the camera sweeps over the resting cap during the approach, and the
    correction cuts the p90 by up to 22 px (ep1 C track 63 → 41 px, ep2 A action 41 → 27).
  - The front carried p90 of ep2 A goes from 17 to 15 px.
  - A resting cap under the still front camera is unchanged.

**Videos:** `video/ep<N>_action_seed0.mp4`, real | sim, front over wrist. The real rows are
shifted by the fitted latency (2 / 1 frames) and labelled with their rows.

### 5b. SENSITIVITY: the cap at CALIB's closed-end diameter (`eval_sens_cap_closed_end.json`, hash `cbc4f64bd8c0b258`)

This is a sensitivity run, not a result. It sets `--set objects.cap.diameter=0.02932`: the
fitted closed-end diameter as the collision cylinder, −1.7σ from the mid-height 30.82 mm
(σ 0.89 mm). Everything else is as fitted.

| pick | in the mug, action / track | drop − real | jaw hold − real, action / track | squeeze | finger–cap pen. max / p99, action | ensemble in the mug, action / track |
|---|---|---|---|---|---|---|
| ep0 A | **yes** / no | +5 / — | −0.26 / −0.26° | 11.8 N | 0.08 / 0.02 mm | 35 / 20 % |
| ep1 A | yes / yes | −1 | −0.05 / −0.10° | 4.5 N | 0.05 / 0.04 mm (track 0.46 / 0.44) | 100 / 75 % |
| ep1 B | yes / yes | −1 | +0.02 / −0.18° | 12.5 N | 0.05 / 0.04 mm | 95 / 80 % |
| ep1 C | yes / yes | −1 | −0.25 / −0.30° | 13.0 N | 0.12 / 0.03 mm | 70 / 70 % |
| ep2 A | yes / yes | −1 | +0.14 / +0.20° | 15.6 N | 0.42 / 0.13 mm | 80 / 90 % |
| ep2 B | yes / yes | −2 | −0.53 / −0.54° | 16.2 N | 0.08 / 0.08 mm | 95 / 85 % |

- The mean |jaw hold error| is 0.21° (action), against 0.74° at 30.82 mm.
- ep0's Jaw RMSE falls from 7.87° to 0.57° in action mode.
- 20-seed episode success: ep0 35 / 20 %, ep1 70 / 70 %, ep2 80 / 85 %.
- **ep0's release is still a knife edge. In track mode the cap stays in the fingers**:
  - The horn stops at 10.77°, 0.42° short of the 11.19° command; the Jaw's friction
    deadband is 0.41°.
  - The finger still touches the cap at 0.05–0.08 N, which is enough friction for 2.5 g.
  - 0.1 mm of gap decides ep0. Its ensemble rate is the number to use, not its nominal.

## 6. What the scene needs (for CALIB)

1. **Where the fingers hold it, the cap is 29.3–29.6 mm wide, not 30.82 mm.**
   - All six sim holds are too open, by +0.52 to +1.17° (mean +0.74°). At 1.28 mm per
     degree in the contact band, that is 0.7–1.5 mm of cap.
   - The six real holds lie on one line (residual 0.14 % RMS), closed-up and open-up caps
     alike. Their zero-force contact is at a finger gap of 29.54 mm.
   - At ep0's release the real encoder settles at 16.64 %, just past that contact angle
     (16.62 %), and the cap falls. So the real cap is ≤ 29.6 mm where it is held.
   - The sim finger stays blocked at 17.44 % by the 30.82 mm cylinder, squeezing 1.75 N
     per finger. The command is 16.90 %.
   - The fingers do not feel the fitted lip. The open-up caps are held on their rims, the
     frustum's 32.3 mm end, and the taper makes those holds worse (section 1).
   - The closed-end 29.32 mm cylinder brings all six holds to −0.53..+0.14° (5b).
2. **Table height.** The fitted 23.99 mm is consistent with the arm: the recorded
   clearance is +0.05 / +2.98 / −1.60 mm.
   - Its σ, 5.47 mm, decides the ensembles. Almost every failed seed has the table
     1.5–14 mm low, where the recorded fingers stop above the cap.
   - A tighter table would let the ensemble rates measure the physics rather than the
     table.
3. **ep1 C's resting xy.** Its front blob is about 15 px (≈ 17 mm) from the scene's C.
   - The nearest-blob metric of the first fitted eval gave 15.5 px median over 687 frames.
   - CALIB's own labels match only 8 frames within their 12 px gate.
   - CALIB's leave-one-episode-out cross-view ray error for 1.C was 16.5 mm.
4. **Mug height, σ 5.3 mm.** In track mode the carried ep1 A grazes the rim (section 5),
   and the rim height is only known to that σ.

## 7. History: the provisional and what-if scenes

**Provisional scene** (`eval_provisional.json`, logs in `logs/provisional/`, hash `f561bc4f68b5faeb`):

- table 20 mm, zero joint offsets, cap 29 × 12 mm, initial-guess wrist camera
- not physically feasible: the recorded arm was 8.8 / 0.7 / 6.1 mm inside the table
- nominal: 1/6 picks in the mug in action mode, 2/6 in track mode
- 20-seed success: 0–15 % per episode

**What-if** (`eval_whatif_h16.json`, hash `8adbc0d9e6e2b231`): a hypothesis, run to
separate physics from calibration.

- fit9 joint offsets, table 20 mm, cap 27.1 × 16 mm, `--placement grasp`
- nominal: 4/6 picks in the mug (action) and 5/6 (track), jaw holds within ±0.3°
- ep0 was not released; ensembles 30–85 %
- The measured alternatives in section 1 ran on this geometry.
- The command below reproduces it on top of CALIB's layer, so its numbers are history:

```
./robot real2sim mujoco eval --tag whatif_h16 --placement grasp \
  --set "robot.joint_offsets.deg=[0.0, 0.19, -2.8, -0.47, 0.0]" --set table.z=0.0200 \
  --set objects.cap.diameter=0.0271 --set objects.cap.height=0.016 \
  --sigma cap_xy=0.003,table_z=0.001,cap_d=0.0265:0.0277,cap_h=0 --ensemble 20
```

## Files

| file | contents |
|---|---|
| `servo.py` | the servo model (engine-agnostic) and the goal stream |
| `servo_fit.py` | identification, kv cross-check, servo.json writer (validates the merge) |
| `grip_hold.py` | hold plateaus: flex stiffness and torque limit |
| `fingers.py` | CoACD finger pieces (cached) |
| `model.py` | MjSpec scene builder, contact physics, table tilt, grasp placement |
| `replay.py` | replay driver, three modes, contact scan, pose log |
| `evaluate.py` | metrics (per pick, per episode, latency-corrected cap pixels), ensembles, scene checks, `--set` overrides |
| `render.py` | EGL renders through the real distortion; mp4 beside the latency-shifted real frames |
| `viewer.py` | mujoco.viewer playback |
| `cli.py`, `run.sh` | commands |
| `tests/` | 25 tests (mjlab env) |
