# real2sim ISAAC: the real episodes replayed in Isaac Sim 4.5 / Isaac Lab 2.1

The same replay as the MuJoCo track (`../mujoco`), in PhysX with RTX rendering. The two
share the scene config, the servo model and its parameters, the replay modes and the
metrics code, so their numbers can be compared line by line.

The replay is physically feasible by construction:
- Objects are placed once at reset and afterwards move only through contact.
- Nothing is welded to the fingers, no object is scripted, and no force is hacked in.
- The servos are clamped at the real STS3215 limits.

Every number below was measured in this track. The section "What the numbers depend on"
says which scene each number was measured on.

## Commands

```
./robot real2sim isaac replay --episode 0 --mode action   # or track | kinematic
        [--num-envs 16 --perturb mujoco[:SIGMA] [--seed 1000]]  # the MuJoCo track's ensemble draws
        [--num-envs 8 --perturb cap_xy=0.002,table_z=0.001,cap_d=0.0005,friction=0.2,mass=0.3 --seed 1]
        [--render] [--gui] [--placement config|grasp] [--physics hz=960] [--set table.z=0.011] [--tag T]
./robot real2sim isaac eval <run dir>...      # eval.json next to the pose log (host python3, no GPU)
./robot real2sim isaac all [replay args]      # every used episode, action + track, then eval
./robot real2sim isaac calibrate              # RTX light units + dome orientation, cached per look build
./robot real2sim isaac render --episode 0 (--log <run dir> | --kinematic) [--modes rt,pt] [--samples-only]
./robot real2sim isaac samples [--modes rt,pt] # LOOK's 24 sample frames in each mode, scored
./robot real2sim isaac camcheck               # marker render: pixel error of both cameras
./robot real2sim isaac test                   # 13 tests, no Kit (Isaac env python), ~3 s
```

- Every Kit process runs under `flock $R2S_GPU_LOCK` and a `timeout` of `R2S_ISAAC_TIMEOUT`
  seconds (default 7200). Kit can hang at exit on this box, and a hung process would hold
  the GPU lock for every track.
- Runs are written to `sim/outputs/real2sim/<ds>/isaac/ep<N>_<mode>[_<tag>]/`:
  `poselog.npz`, `ensemble/env_KKK.npz`, `run.json`, `eval.json`, and `front.mp4` /
  `grip.mp4` with `--render`. `all --tag T` names them `ep<N>_<mode>_T`.
- `--perturb mujoco` gives env e >= 1 exactly the MuJoCo track's ensemble scene of seed
  `--seed` (default 1000) + e - 1: `real2sim.mujoco.evaluate.draw_perturbation`, imported,
  not copied (test_ensemble_look.py checks it draw by draw). Env 0 stays nominal, so
  `--num-envs 16` replays the MuJoCo eval's seeds 1000-1014. `SIGMA` is the MuJoCo
  `--sigma` string (`cap_xy=M,table_z=M,cap_d=LO:HI,cap_h=M`).
- `--set path=value` overrides a merged-scene value, for sensitivity runs only. The override
  and the resulting scene hash are recorded in `run.json` and in eval.
- `--physics key=value` overrides a value from `params.py`.
- `--placement grasp` puts each cap where both fingers would hold it at the recorded
  blocked-jaw frame (`placement.py`). It is a diagnostic, not the default.
- `eval` pins numpy to one thread (run.sh): a 16-env eval took 221 s with the default BLAS
  threads and 75 s with one.

## Files

| module | needs Kit | what |
|---|---|---|
| `scene.py` | yes | the Isaac Lab scene of one episode, built in one place; `Look` hooks for part 2 |
| `usd_assets.py` | yes (pxr) | cap and mug USD: visual mesh, convex collision pieces, bound physics material |
| `contact.py` | yes | PhysX contact views, one per link and per cap: separations, forces, the MuJoCo-compatible pose-log extras |
| `render.py` | yes | RTX pinhole frames warped into the real distorted cameras, encoded to mp4 by ffmpeg |
| `look.py` | yes (host: textures) | LOOK's assets on the scene: materials, table texture, backdrop, key light, domes, calibration cache |
| `lookrender.py` | yes | calibration, dome probe, and playback renders (samples, episodes) in scene radiance through `blender.camera_model` |
| `replay.py` | yes | the entry point: modes, settle, goal streams, ensembles, pose logs |
| `live.py` | yes | the live sim's Isaac engine (`../live`): this replay stepped on demand, the look rendered per observation |
| `camcheck.py` | yes | marker spheres at known points, checked against `real2sim.camera` |
| `servo.py` | no (torch) | the MuJoCo track's servo model expressed for PhysX, and its goal stream |
| `track.py` | no | track-mode goals by inverse dynamics on the Isaac inertia |
| `params.py` | no | every Isaac-only number, with provenance and the measurement behind it |
| `geometry.py` | no | the convex pieces, exact containment depths, mass sums, table tilt |
| `placement.py` | no | reset-pose estimates: `config`, or `grasp` (antipodal, from FK) |
| `penetration.py` | no | offline overlaps from the pose log plus meshes |
| `evaluate.py` | no | eval.json: the MuJoCo track's `episode_metrics`, plus Isaac extras |

## The scene (`scene.py`)

**Frame**
- The robot root is at identity in each env. MEASURED (probe2): with the root at identity,
  the PhysX link frames of all 7 bodies equal `real2sim.kinematics`' URDF frames.
- In every eval, the logged link poses match FK of the logged joints to 0.0005 mm
  (`link_frames_fk_mm`).
- The understand-phase note "USD = URDF rotated +90°" described the push-T placement, not
  the asset.

**Arm**
- The workshop USD, whose finger collisions are SDF.
- MEASURED: its collision meshes equal the MJCF meshes that `kinematics` and
  `penetration.py` use, to 1.5e-5 mm nearest-neighbour distance. So the offline checks run
  on the geometry PhysX cooked.
- Joint limits are written into the USD before PhysX parses it, so no reset can restore
  the URDF limits. They are then read back from PhysX into `run.json`.
  - Range: the firmware range ±5° (MuJoCo's `LIMIT_MARGIN_DEG`). The firmware clamps the
    goal, not the joint.
  - Jaw lower stop: -11.5°, the angle where the fingers touch, which is the real shut stop.
- Self-collision is off. PhysX filters parent-child links anyway, so the touch limit is
  what stops the fingers.
- The printed parts are painted white. The workshop USD ships them yellow.

**Payload**
- The printed `klip_support` is a collider (PhysX `convexDecomposition` of the repo's
  baked `klip_support.usd`).
- The KWC-500 head is a box collider.
- Their mass (18 g + 20 g, the MuJoCo track's values) is folded into the gripper link's
  `MassAPI` by the parallel-axis theorem. MEASURED read-back: gripper 0.087 → 0.125 kg,
  with the centre of mass moved about 30 mm.
- With `track`, `track.py` builds its inverse dynamics on exactly this inertial.

**Table and objects**
- The table is a static box with its top at `scene.table_z()` (`table.tilt` is honoured if
  CALIB fits one). A ground plane sits 0.75 m below it.
- Contacts between the table and the fixed base are filtered out.
- Caps and the mug come from `usd_assets.py`. Their collision is explicit convex pieces
  (`geometry.py`), because the tooling run measured that a convex hull lids the mug and
  SDF-against-SDF is unstable:
  - cap: a top disk plus 16 skirt sectors
  - mug: a floor, 32 wall sectors and 8 handle pieces
  - The pieces' chords sit inside the real wall, 0.26 mm at most.
  - Mass is explicit. PhysX derives the centre of mass and inertia from the pieces.

**Materials**
- These are the MuJoCo track's coefficients, and PhysX combines them with `max`, as MuJoCo
  does. Every pair therefore has the same coefficient in both engines:

  | pair | coefficient |
  |---|---|
  | finger-cap | 0.40 |
  | cap-mat | 0.60 |
  | cap-mug | 0.30 |
  | mug-mat | 0.60 |

- Static friction equals dynamic friction, and restitution is 0.

**Cameras**
- Both cameras are built from the scene config as the square-pixel oversized pinholes of
  `camera.render_spec(cam, square=True)`: front 638×484, grip 771×606.
- They go through sim_agent101's `camera_cfg`, imported rather than copied, with cx and cy
  passed as +0.5 (see Cameras below).
- `render.py` warps every frame into the real K and distortion.

## The servo (`servo.py`, `track.py`)

The model and its parameters come from the MuJoCo track: `ServoModel` is read from the
scene's `actuator` group, which is MUJOCO's `servo.json`.

    tau = clip(kp (u - q) - kd qd, ±effort) - damping qd - frictionloss sign(qd)

PhysX cannot express this directly: its drive clips spring and damper together, and it
has no dry friction and no second hinge in the jaw link. So, on every physics step:

- **Arm joints.** The implicit PhysX drive carries stiffness `kp` and damping `kd + damping`.
  Its target is set to `q + (clip(kp (u-q) - kd qd) + kd qd) / kp`. The result is the clip
  inside and the motor damping outside, integrated implicitly. PhysX `maxForce` is only a
  backstop and never binds.
- **Dry friction.** An elasto-plastic (bristle) feed-forward effort:
  - It holds static loads up to `frictionloss`, as MuJoCo's friction constraint does. The
    fitted Pitch and Elbow frictionloss of 0.28-0.29 N·m amounts to a 1.8° deadband at kp 9.
  - The bristle stiffness is 200 N·m/rad, so the deflection is at most 0.08°.
  - It is explicit-stable: k dt²/J = 0.10 and c dt/J = 0.62 at the fitted armature 0.009.
- **Jaw.** The horn is a state integrated in `servo.py`, with the armature as its inertia and
  exact Coulomb friction. The PhysX Jaw drive is the series spring (the MuJoCo track's flex
  hinge): stiffness `flex.stiffness`, target equal to the horn. The pose log's `q` holds the
  horn, which is what the encoder reads, and `x_finger` holds the finger.
- **Goal stream.** The MuJoCo `GoalStream` semantics (dead time, zero-order hold, firmware
  clamp, slew), vectorised in torch.
  - MEASURED equal to `real2sim.mujoco.servo.GoalStream` to 1e-9 rad over 944 substeps
    (`tests/test_servo_params.py`).
- **Track mode.** `u = q_ref + (tau_ID + kd qd_ref) / kp`.
  - `tau_ID` comes from `mj_inverse` along a cubic spline through `observation.state`,
    with the Isaac gripper inertial, the servo armature and damping, and
    `frictionloss·tanh(qd/0.02)`. This is the MuJoCo track's definition.
  - The Jaw stays on the action.

**Units and semantics, VERIFIED on a contact-free replay** (`--set table.z=-0.2`). Numbers are
per-joint RMSE of the arm against `observation.state`, in degrees.

| run | Rotation | Pitch | Elbow | Wrist_Pitch | Wrist_Roll |
|---|---|---|---|---|---|
| Isaac ep0 (servo.json of 13:31) | 0.27 | 0.99 | 0.96 | 0.61 | 0.41 |
| MuJoCo fit ep0, same servo.json | 0.26 | 0.85 | 1.12 | 0.57 | 0.45 |
| Isaac ep1 | 0.30 | 0.97 | 0.91 | 0.32 | 0.21 |
| MuJoCo ep1 | 0.31 | 1.05 | 0.74 | 0.33 | 0.23 |
| Isaac ep2 (held out) | 0.27 | 1.42 | 1.02 | 0.50 | 0.34 |
| MuJoCo ep2 | 0.30 | 1.31 | 1.25 | 0.49 | 0.38 |

Earlier probe (provisional model, the same model replayed in both engines, ep0):

| configuration | per-joint RMSE vs MuJoCo (°) |
|---|---|
| this emulation | 0.06 / 0.23 / 0.40 / 0.05 / 0.04 |
| literal port (PhysX `maxForce = effort`, no friction) | 0.31 / 0.30 / 0.46 / 0.30 / 0.30 |

A degree-based stiffness (off by 57×) could not land within 0.1°, so the units are SI.

## Physics settings (`params.py`), each one measured

Common conditions for every row:
- ep0, action mode, `--placement grasp`
- dev scene: `--set table.z=0.011 --set objects.cap.diameter=0.0273`
- servo.json of 13:31 with flex
- average-combine materials, depenetration 0.5 m/s, 32 iterations, unless the row says otherwise

"Pen." is the finger-cap penetration in mm: offline max and PhysX max. "Slip" is the cap's
slip in the grasp.

| variant | loop s | cap in mug | pen. offline / PhysX (mm) | slip | jaw hold − real |
|---|---|---|---|---|---|
| 480 Hz, 32 iterations | 66 | yes | 0.77 / 1.93 | 4.1 mm, 16.5° | -0.54° |
| 960 Hz | 136 | yes | 0.08 / 0.07 | 0.04 mm | -0.14° |
| 240 Hz | 32 | **no** (dropped 74 frames early) | 0.79 / 3.75 | 6.2 mm | -3.84° |
| **64 iterations** | 83 | yes | 0.10 / 0.13 | 0.08 mm | -0.14° |
| contact offset 2 mm | 123 | yes | 0.78 / 2.40 | 1.4 mm | -1.37° |
| sweep CCD | 116 | yes | identical to the 32-iteration run | | |
| **depenetration 5 m/s** | 63 | yes | 0.18 / 0.21 | 0.07 mm | -0.25° |
| rigid jaw (no flex) | 70 | **no** (dropped 98 frames early) | 0.76 / 2.68 | 13.9 mm | +0.55° |

Chosen:
- 480 Hz, TGS, 64 position iterations, 1 velocity iteration
- PhysX's own contact offsets, rest offset 0
- `maxDepenetrationVelocity` 5 m/s
- no CCD
- a settle of 0.3 s before t_0

Why:
- With 32 iterations the solver never converges on the 12 N squeeze of a 2.5 g cap between
  two SDF fingers. The cap sinks in and turns in the grasp.
- A low depenetration velocity stops the solver pushing the fingers back out.

**Drop into the mug** (`probe_drop.py`, a cap dropped 6 cm above the rim):
- Measured PhysX separation at impact:

  | setting | separation at impact | rebound | final position |
  |---|---|---|---|
  | default contact offset | -1.48 mm | none | on the floor |
  | speculative CCD | -1.48 mm (identical) | | |
  | sweep CCD | -1.48 mm (identical) | | |
  | 3 mm cap contact offset | -0.0001 mm | | |

- The 3 mm offset was rejected. On ep0 it kept the fingers holding the cap after the release
  command, so the cap never dropped: speculative contacts span the 0.4 mm the fingers open.
  It also cost 1.8× the time.
- So a drop impact shows up as a 1.5-3 mm PhysX separation for 1-2 physics steps. At the
  logged instants the offline overlap is 0.00 mm.

## Cameras (`camcheck.py`)

Method:
- Emissive markers at known points, with lights off.
- The table, the floor, the objects and the robot visuals are hidden, and anti-aliasing is
  off.
- Each marker's intensity centroid is compared with `Camera.project`.

Results:

| camera | production (cx, cy + 0.5) | without +0.5 | distorted 640×480 after remap |
|---|---|---|---|
| grip, 15 markers | mean (-0.007, 0.003) px, RMS 0.07, max 0.08 | mean (-0.44, -0.47) px, RMS 0.64 | RMS 0.037 px, max 0.058 |
| front, 20 markers | mean (0.09, -0.03) px, RMS 0.31, max 1.27 (one marker) | mean (-0.44, -0.47) px, RMS 0.71 | RMS 0.26 px |

sim_agent101's `camera_cfg` centres the principal point on W/2 (pixel edges), while
`real2sim.camera` centres on (W-1)/2. Passing +0.5 is the verified fix.

## Results

eval.json uses the MuJoCo track's `episode_metrics` on the Isaac pose log, so the two
tracks share key names and definitions. The Isaac extras sit under `isaac`.

### Fitted scene, hash `6f282e812cc585f5` (the official eval)

CALIB's dataset-only calibration merged over the core config and MUJOCO's servo.json;
placement `config` (CALIB's cap and mug poses), 480 Hz, 64 iterations, flex jaw.

```
./robot real2sim isaac all                                                  # nominal, 1 env each
./robot real2sim isaac all --num-envs 16 --perturb mujoco --tag ens16         # + the MuJoCo seeds 1000-1014
./robot real2sim isaac replay --episode 0 --mode action --num-envs 16 --perturb mujoco --tag ens16_rerun
./robot real2sim isaac all --set objects.cap.diameter=0.02932 --tag sens_cap_closed_end   # sensitivity
```

**Per pick, nominal.** "sep" is the last both-finger frame, "drop" the frame the cap has
left the fingers by 15 mm, both minus the real drop (the wrist camera's teal drop). The
jaw hold is the sim ENCODER minus observation.state over the real hold. Squeeze is the
normal force per finger (fixed, moving), median over the transport. Penetration is
PhysX's finger-cap separation over every physics step while touching.

| pick (up) | mode | lifted | in mug | sep / drop − real (frames) | jaw hold − real | squeeze (N) | slip | pen. max / p99 (mm) |
|---|---|---|---|---|---|---|---|---|
| ep0 A (closed) | action | 181 mm | **no** | +51 / never | +0.49° | 13.6, 13.5 | 0.03 mm | 0.14 / 0.10 |
| ep0 A | track | 178 mm | **no** | +51 / never | +0.54° | 13.6, 13.6 | 0.00 mm | 0.08 / 0.05 |
| ep1 A (closed) | action | 164 mm | yes | −3 / +1 | +0.52° | 6.4, 6.5 | 0.01 mm | 0.10 / 0.09 |
| ep1 A | track | 157 mm | yes | −3 / +1 | +0.63° | 6.2, 6.7 | 0.02 mm | 0.06 / 0.05 |
| ep1 B (open) | action | 161 mm | yes | −2 / +1 | +0.55° | 13.8, 13.6 | 0.05 mm | 0.09 / 0.09 |
| ep1 B | track | 155 mm | yes | −3 / +1 | +0.38° | 13.3, 13.4 | 0.01 mm | 0.11 / 0.11 |
| ep1 C (open) | action | 160 mm | yes | −3 / +1 | +0.40° | 14.4, 14.3 | 0.07 mm | 0.13 / 0.13 |
| ep1 C | track | 153 mm | yes | −3 / −1 | +0.37° | 14.3, 14.0 | 0.04 mm | 0.09 / 0.08 |
| ep2 A (open) | action | 196 mm | yes | −2 / +3 | +0.77° | 15.5, 16.6 | 0.11 mm | 0.21 / 0.10 |
| ep2 A | track | 199 mm | yes | −2 / +4 | +1.17° | 15.7, 15.6 | 0.20 mm | 0.47 / 0.08 |
| ep2 B (open) | action | 170 mm | yes | −4 / +1 | +0.40° | 16.0, 16.0 | 0.02 mm | 0.07 / 0.07 |
| ep2 B | track | 160 mm | yes | −4 / +1 | +0.56° | 15.3, 15.6 | 0.05 mm | 0.17 / 0.14 |

5 of the 6 real picks end in the mug in both modes, each dropped 1-4 frames after the real
drop; ep0 A is lifted and carried but never released (next section).

The Jaw servo's torque never exceeds its identified 1.155 N·m clamp. It saturates in both
ep2 holds, as the real ep2 B hold does (grip_hold). The finger forces, up to 16.6 N, sit
0-9 % above clamp / cap-centre lever arm (13.8-15.2 N), because the contacts are nearer
the jaw axis than the cap centre.

MuJoCo's jaw holds on the same scene (its README, cylinder cap, action) are +0.53 / +0.73 /
+0.81 / +0.52 / +1.17 / +0.65°. Isaac's are 0.04-0.40° closer to the data.

**Per episode, nominal.** Joint RMSE against observation.state at lag 0 (the best lag is 0
in every run), fingertip = grasp-site error.

| run | arm RMSE (°) | Jaw RMSE | fingertip mean / p95 | offline finger-cap pen. max / p99 | mug moved | arm-object force | loop | ms / frame | env-steps/s |
|---|---|---|---|---|---|---|---|---|---|
| ep0 action | 0.40 0.79 0.62 0.37 0.41 | 7.83° | 3.3 / 7.1 mm | 0.08 / 0.07 mm | 0.0 mm | 0 N | 94 s | 193 | 83 |
| ep0 track | 0.06 0.08 0.10 0.05 0.04 | 7.87° | 0.5 / 0.9 mm | 0.04 / 0.03 mm | 0.0 mm | 0 N | 90 s | 185 | 86 |
| ep1 action | 0.30 0.94 0.70 0.35 0.22 | 0.51° | 4.7 / 9.0 mm | 0.35 / 0.35 mm | 0.0 mm | 0 N | 145 s | 162 | 99 |
| ep1 track | 0.04 0.06 0.09 0.05 0.03 | 0.50° | 0.4 / 0.9 mm | 0.38 / 0.37 mm | 3.6 mm | 0 N | 138 s | 154 | 104 |
| ep2 action | 0.30 1.01 0.92 0.46 0.32 | 0.99° | 5.3 / 11.0 mm | 0.36 / 0.35 mm | 0.0 mm | 0 N | 119 s | 152 | 105 |
| ep2 track | 0.08 0.10 0.12 0.04 0.03 | 1.04° | 0.5 / 1.7 mm | 0.24 / 0.17 mm | 0.0 mm | 0 N | 165 s | 211 | 76 |

- Every finger-cap penetration is under the 1 mm target: PhysX at most 0.47 mm (one step
  of ep2 A in track mode, p99 0.08 mm), offline at most 0.38 mm.
- ep0's Jaw RMSE of 7.8° is the unreleased cap: the jaw cannot close past it after the
  real release.
- ep1 track moved the mug 3.6 mm: at frames 274-299 the carried cap A scrapes the rim
  (up to 2.3 N cap-mug force, no arm contact). The action replay does not touch it. The
  rim height has a 5.3 mm fitted sigma.
- No non-finger link touched an object in any run.

### Why ep0 A is not released (evidence)

The real release is a knife edge: the recorded command opens to 16.90 %, and the real
encoder stops at 16.64 % (10.85° under the fitted gripper map), where the finger gap is
29.6 mm. The real cap dropped as the encoder passed 16.51 %: a gap of 29.37 mm.

The fitted scene's cap is a straight cylinder at the calibration's mid-height diameter,
30.82 mm, 1.4 mm wider than that. At the sim's release (frames 364-380) the finger stands
at 11.91° with a 30.93 mm gap, the horn at 11.54° pushed open past the command, and the
cap is still squeezed with 1.75 N per finger. Then the real gripper closes again and the
sim squeezes the cap harder, 16.2 N; it falls 51 frames late, outside the mug.

Three independent measurements put the width the fingers hold at 29.3-29.6 mm, not 30.8:
- MUJOCO's grip_hold θc: the zero-force contact at 16.615 % on the encoder, over all 7
  hold plateaus of both cap orientations (0.14 % RMS). Under the fitted map that is a
  29.55 mm gap.
- CALIB's own closed-end diameter: 29.32 ± 0.85 mm.
- The 15-seed ensemble: only the draw with a 28.71 mm cap was released (drop +0 frames).
  Caps of 29.85 mm and wider stayed in the fingers.

The jaw holds say the same from the other side. All six sim holds are 0.37-1.17° more
open than the data, which is 0.5-1.7 mm of extra width at 1.47 mm/°.

This is a calibration inconsistency, not a PhysX one, so nothing was changed for it. The
MuJoCo track measured the same: with the cylinder ep0 is not released. CALIB's taper does
not release ep0 either, and it holds the open-up caps on their wide rim, 0.6-0.8° more
open still (mujoco README section 1).

**Sensitivity: the cap at CALIB's closed-end diameter** (`--set objects.cap.diameter=0.02932`,
hash `cbc4f64bd8c0b258`, nominal, 1 env):

| pick | action: in mug, drop − real, jaw hold − real | track: in mug, drop − real, jaw hold − real |
|---|---|---|
| ep0 A | **yes**, +13, −0.29° | **yes**, +17, −0.25° |
| ep1 A | yes, +1, −0.01° | yes, 0, −0.16° |
| ep1 B | yes, +1, −0.16° | yes, +1, −0.38° |
| ep1 C | yes, +1, −0.41° | yes, −1, −0.50° |
| ep2 A | yes, +3, −0.10° | yes, +3, +0.17° |
| ep2 B | yes, +1, −0.68° | yes, +1, −0.58° |

- All six picks end in the mug in both modes. ep0's drop is still 13-17 frames late.
- The holds swing to 0.0-0.7° too closed, so the real width sits between 29.3 and 30.8 mm.
- Penetration stays at or under 0.60 mm PhysX and 0.32 mm offline. The mug moved 0.7 mm
  in ep1 track.
- ep2 B in action mode squeezes 32 N on the fixed finger during the carry (median): the
  narrower cap sits against the gripper body. Slip is 0.06 mm.

### Ensembles: the MuJoCo track's draws

`--perturb mujoco`, 16 envs, env 0 nominal, envs 1-15 = the MuJoCo seeds 1000-1014 of
`real2sim.mujoco.evaluate.draw_perturbation` (its version of 17:33):
- cap xy ~ N(0, CALIB's per-cap sigma, 0.13-0.15 mm)
- table z ~ N(0, 5.47 mm), clipped to 7.6-40.4 mm
- cap diameter ~ N(30.82, 0.89 mm) and height ~ N(13.88, 2.28 mm)
- the four frictions ~ U(MATERIAL_RANGES), cap mass ~ U(2, 3) g

The table friction is authored per env in USD; the other materials go through the tensor
API. Statistics are over the 15 perturbed envs, with the MuJoCo ensemble's keys (eval.json
`ensemble`).

| run | episode success | in mug per pick | drop − real, median (frames) | jaw hold − real, median | finger-cap pen. PhysX max / p99 median | offline max | mug moved max |
|---|---|---|---|---|---|---|---|
| ep0 action | 0.07 | A 0.07 | A +0 | A +0.56° | 0.14 / 0.09 mm | 0.26 mm | 0.0 mm |
| ep0 track | 0.13 | A 0.13 | A +6 | A +0.46° | 0.57 / 0.06 mm | 0.27 mm | 0.0 mm |
| ep1 action | 0.67 | A 1.00, B 0.93, C 0.67 | +1, +1, +1 | +0.66°, +0.53°, −0.04° | 0.16 / 0.10 mm | 0.38 mm | 5.3 mm |
| ep1 track | 0.53 | A 0.93, B 0.87, C 0.53 | +1, +1, 0 | +0.57°, +0.33°, −1.90° | 0.18 / 0.07 mm | 0.41 mm | 8.5 mm |
| ep2 action | 0.67 | A 0.67, B 0.93 | +4, +2 | +1.03°, +0.31° | 0.29 / 0.09 mm | 0.36 mm | 0.8 mm |
| ep2 track | 0.80 | A 0.87, B 0.80 | +3, +2 | +0.90°, +0.78° | 0.56 / 0.10 mm | 0.60 mm | 0.0 mm |

- The penetration target holds in all 90 perturbed replays: at most 0.57 mm in PhysX and
  0.60 mm offline.
- ep0 is released only when the cap is narrow: one draw, 28.71 mm (see the evidence above).
- 16 of the 20 failing draws of ep1 and ep2 have the table drawn 2.4-13.9 mm low, and
  most of those misses are open-up caps never lifted. The fitted table's 5.5 mm sigma is as
  wide as the depth of the rim grasps.
- Every env's drawn scene is in eval.json (`ensemble.seeds[].perturbation.mujoco`), so the
  two engines' rows pair up seed by seed.

### Determinism

The same command twice (ep0 action, 16 envs, seed 1000; `ens16` against `ens16_rerun`):
- 14 of 16 envs are bitwise identical in every logged array.
- The other two differ only in `x_fc_force`, the mean finger-cap force, by at most 1.9e-6 N:
  the summation order of PhysX's contact-force reduction.
- Joint angles, link and object poses, finger angle, servo torques and penetrations are
  bitwise identical in all 16 envs. The max divergence of the state is 0.
- `enhanced_determinism` is not needed. Earlier, on the dev scene with 1 env, it gave the
  same bits as without it.

The env count is part of the configuration. The 1-env nominal run and env 0 of the 16-env
run (same scene, same inputs) differ:
- joints up to 0.14° apart, objects up to 0.51 mm, from frame 1 on: GPU PhysX batches its
  solver differently for a different number of envs
- the same outcomes, per-joint RMSE within 0.03°, jaw holds within 0.13°
- slip 0.05 mm against 0.20 mm on ep2 A in track mode

### Cost

RTX 3060 12 GB, Ryzen 7 5700X; physics 480 Hz, 16 steps per frame.

| run | loop | per frame | env-steps/s | VRAM (Kit process) |
|---|---|---|---|---|
| 1 env, ep0 / ep1 / ep2 (488 / 898 / 782 frames) | 90-94 / 138-145 / 119-165 s | 152-211 ms (0.16-0.22× real time) | 76-105 | 2.5 GB |
| 16 envs, ep0 / ep1 / ep2 | 122-135 / 240-281 / 159-166 s | 203-313 ms | 817-1260 | 2.7 GB |

- Around each loop, Kit start-up, scene and reset take about 25 s (nvidia-smi process
  lifetime minus the loop). The reset alone is 0.8-11 s, the longest with per-env parsing.
- An `all` run (6 replays, 1 env) took 19 min with its eval (then multi-threaded).
- The eval takes 75 s single-threaded for a 16-env run.
- VRAM was sampled by nvidia-smi every 2 s. The desktop holds another 1.9 GB.
- A 16-env frame costs 1.3-1.9x a 1-env frame, so 16 envs give 10-12x the env-steps. The
  16-env runs read one contact view per link and per cap, which PhysX needs for more than
  one env.

### Dev scene (history: provisional scene, hash before CALIB)

Settings:
- `--placement grasp`
- cap 27.3 mm: the gap at the MuJoCo grip_hold contact angle θc = 8.94°
- per-episode table 1 mm below the recorded arm's lowest finger point: ep0 11.0 mm,
  ep1 17.8 mm, ep2 12.9 mm

This isolates the contact physics from the bench-geometry and placement inconsistency that
CALIB is fitting. The servo.json is the 14:08 fit (dead time 33.3 ms), except ep0 (13:31
fit).

| run | picks in mug (sim / real) | per pick: drop frame sim − real / jaw hold − real | joint RMSE (°) arm, Jaw | pen. PhysX / offline (mm) |
|---|---|---|---|---|
| ep0 action | A / A | +14 / -0.13° | 0.28 0.96 0.63 0.46 0.40, 0.33 | 0.09 / 0.11 |
| ep0 track | A / A | +7 / -0.19° | 0.05 0.08 0.09 0.04 0.06, 0.33 | 0.11 / 0.20 |
| ep1 action | A, B / A, B, C | A 0 / -0.57°; B +1 / +0.11°; C missed | 0.32 0.96 0.68 0.34 0.22, 1.63 | 0.06 / 0.22 |
| ep1 track | **A, B, C** / A, B, C | A 0 / -0.39°; B +1 / +0.15°; C 0 / -0.43° | 0.05 0.10 0.10 0.05 0.03, 0.36 | 0.15 / 0.26 |
| ep2 action | B / A, B | A missed; B +1 / -0.13° | 0.27 1.11 0.84 0.41 0.32, 2.17 | 0.07 / 0.12 |
| ep2 track | B / A, B | A pushed away at frame 176; B +1 / -0.22° | 0.04 0.10 0.08 0.04 0.03, 2.17 | 0.13 / 0.21 |

In track mode 5 of the 6 real picks end in the mug. The miss, ep2 A, is a placement
failure, not a grasp failure: on the approach at frame 176 the fixed finger touched the
FK-placed cap with 0.04 N and pushed it 7 mm.

Other measured values:
- Grip force during transport: 11.6-16.6 N per finger (ep1 A: 3.4 N). The servo torque at
  the horn has a median of 0.80 N·m against the 1.11 N·m limit.
- The squeeze bound, limit / lever arm, is 13.9-14.2 N. The maximum finger force is
  12.3 N.
- Slip in the grasp: at most 0.04 mm.
- The mug never moved (0.0 mm).
- No non-finger link touched an object (`arm_object_force_max_N` 0).
- The fingers touch the table at the start pose: gripper link up to 13.5 N in ep0. The dev
  table is 1 mm below the recorded rest pose, and the arm sags onto it.

With the scene's own placement (`config`: the front-camera cap xy, provisional) ep0 action
misses the cap. The jaw closes to 1.0° where the real one holds at 6.5°. The config xy and
the FK grasp centre are 12 mm apart, the FK-vs-camera inconsistency of the understand
phase. Front-view cap pixel error: 2.4 px with config placement, 16.5 px with grasp
placement. The grip-view error is 113 px, because the grip camera is still the ChArUco
initial guess.

## What the numbers depend on, and open issues

- **Scene.** Every eval.json records `scene_hash`, the layers, the `--set` overrides and
  whether the hash still matches the config. The results above are on the fitted scene
  `6f282e812cc585f5`; the sensitivity rows on `cbc4f64bd8c0b258`; the dev-scene history on
  the provisional config with its overrides.
- **Cap width (for CALIB).** The fitted 30.82 mm mid-height cylinder is 1.3-1.5 mm wider
  than the width the real fingers hold (evidence above). Nothing in PhysX was changed for
  it. With CALIB's closed-end 29.32 mm all six picks succeed, but ep0 still drops 13-17
  frames late and the holds turn 0.0-0.7° too closed. Neither straight cylinder matches
  both the holds and ep0's release.
- **Table height (for the ensembles).** CALIB's 5.47 mm sigma dominates the ensemble
  outcomes: most misses have the table drawn low.
- **Env count.** GPU PhysX results depend on how many envs share the solver: 0.14° /
  0.51 mm between the 1-env and the 16-env nominal. Same seed and env count reproduce bit
  for bit.
- **Transient drop penetration.** 1.5-3 mm for 1-2 physics steps at the cap-mug impact
  (PhysX); 0.00 mm at the logged instants. See the drop probe above.
- **Friction.** With max-combine (MuJoCo's rule) lowering the cap's friction does not lower
  the cap-mat pair. `--perturb mujoco` scales each material group as MuJoCo does, the
  table's in each env's USD. The older `friction=` spec still leaves the table alone.
- **Kinematic mode and `--gui`** are implemented but only lightly exercised. Kinematic was
  smoke-tested; the GUI was never opened on this headless session. Renders (`lookrender.py`)
  do their own playback instead.
- **Torsional friction.** MuJoCo's 0.001 m torsion coefficient is not modelled; PhysX patch
  friction only.

## Rendering look (`look.py`, `lookrender.py`)

LOOK's assets (`look/look.json`, build of 17:17 on the fitted scene, inputs hash
`121e4b46ede6e02e`) on the Isaac scene, rendered by RTX as linear scene radiance.
Everything the camera does after that goes through `real2sim.blender.camera_model.Response`,
imported, so Blender and Isaac are scored through one camera model:
- lens distortion (the fitted K and k1, k2)
- LOOK's front white balance per dataset row
- the wrist colour model (white balance, sigma 1.15, exposure `constant` = Blender's
  `GRIP_EXPOSURE`)
- sRGB

```
./robot real2sim isaac calibrate                                   # once per look build (3 Kit runs, 6.5 min)
./robot real2sim isaac samples --modes rt,pt                       # 24 samples x 2 cameras x 2 modes, scored
./robot real2sim isaac render --episode 0 --log sim/outputs/real2sim/<ds>/isaac/ep0_action --modes rt
```

**What is on the scene** (`look.py` module doc has the reasons):
- **Materials**, all UsdPreviewSurface, F0 0.04 for the dielectrics:
  - white PLA (0.80, roughness 0.55) and servo black on the robot, edited in place: the
    workshop USD's meshes are instance proxies
  - mount black on the klip mount and the webcam box
  - teal cap and red enamel mug (baked into their USD)
  - polished stainless inside the mug and on its rim (face subsets)
  - the glossy black mat: a quad with LOOK's de-lit albedo texture, roughness 0.155
- **Light:** the key as a DistantLight (az -120.5°, el 49.4°, 24.7° across) plus ONE diffuse
  dome on `env.hdr`.
  - RTX real-time takes a single dome: a second, specular-only dome halved the grey check.
  - The reflections see the observed room through LOOK's backdrop cylinder, drawn twice:
    the seen wall (cutout elsewhere) for all rays, the full wall with the fill for the
    cameras only. This is Blender's "reflections see the observed room" rule.
  - The steel's floor and rim emit the room they mirror, as Blender's steel does: 0.5 × F0 ×
    the fill.
- **Units, MEASURED** (`calibrate`, grey Lambertian quad, `look_tex/<build>/calibration.json`):

  | render units per scene unit | key | dome | emission |
  |---|---|---|---|
  | RTX real-time | 0.00417 | 0.00083 | 1.00 |
  | path tracing | 0.00424 | 0.00028 | 1.00 |

  - The dome's units differ 2.9x between the modes, so each mode renders in its own process
    with its own intensities.
  - Every render re-checks the grey quad under the full lighting. Rendered / predicted was
    1.007-1.008 in real-time mode and 0.995 in path tracing, for every process.
- **Dome orientation, MEASURED** (`--probe-dome`): RTX puts LOOK's +x at -x and +y at +y.
  The dome uses the column-flipped map rotated 180°. Its `inputs:specular 0` is
  honoured.
- **Camera time:** each camera is rendered at row - CALIB's latency (front 2.27, wrist 1.42
  frames), as in the Blender job.

**LOOK's 24 samples** (kinematic: the recorded state, objects static at reset; fitted
scene `6f282e812cc585f5`; `look score`, erode 2 px, the frames' own white balance).
Each cell is PSNR dB / SSIM / ΔE of the mean colour.

| camera | region | RTX real-time | path tracing | Blender EEVEE (for reference) |
|---|---|---|---|---|
| front | background | 26.57 / 0.918 / 2.9 | **26.79 / 0.919 / 2.5** | 26.72 / 0.920 / 2.5 |
| front | arm | 16.60 / 0.616 / 6.1 | **16.74 / 0.630 / 4.8** | 16.50 / 0.615 / 5.0 |
| front | fingers | 15.00 / 0.394 / 6.6 | 15.24 / 0.407 / 6.3 | 15.02 / 0.398 / 6.7 |
| front | cap | 15.03 / 0.252 / 14.7 | **15.54 / 0.262 / 12.8** | 14.52 / 0.220 / 12.0 |
| front | mug | 16.74 / 0.406 / 6.8 | **17.08 / 0.427 / 6.5** | 16.57 / 0.417 / 6.8 |
| front | all | 23.16 / 0.875 / 3.0 | **23.30 / 0.877 / 2.1** | 23.14 / 0.877 / 2.0 |
| wrist | background | **21.88 / 0.752 / 7.0** | 20.97 / 0.736 / 8.0 | 21.11 / 0.736 / 7.6 |
| wrist | fingers | 22.04 / 0.904 / 3.1 | **23.17 / 0.902 / 2.6** | 22.41 / 0.894 / 2.9 |
| wrist | cap | 14.90 / 0.585 / 21.9 | 14.81 / 0.590 / 21.1 | **15.85 / 0.598 / 18.3** |
| wrist | mug | 15.80 / 0.528 / 9.0 | 15.82 / 0.532 / 8.1 | 15.40 / 0.518 / 7.8 |
| wrist | all | **20.15 / 0.762 / 2.8** | 19.80 / 0.751 / 2.8 | 19.83 / 0.748 / 2.7 |

| mode | s per image | renders per pose | VRAM (Kit process / whole GPU) | process wall (8 samples x 2 cameras) |
|---|---|---|---|---|
| RTX real-time | 0.26-0.28 s | 6 | 4.9 / 7.8 GB | 21 s, of which 16 s boot |
| path tracing (32 spp x 8 = 256 spp, OptiX denoiser) | 6.1-7.7 s | 10 | 5.3 / 8.2 GB | 120-145 s |

- Both render products render every frame, so a sample image costs both cameras' frames.
- The real-time renderer is within 0.02-1.13 dB of the path tracer in every region and
  23-30x faster.
  - The path tracer leads on every front region (0.1-0.5 dB) and on the wrist fingers
    (1.1 dB).
  - The real-time renderer leads on the wrist background (0.9 dB).
- LOOK's ceiling for a background is 29.1 dB front (the arm-free plate).
- The Blender column is that track's `samples_kinematic_eevee/score.json` of 18:42. It adds renderer-side fits that Isaac does not: the enamel's R x 2.1 and a
  per-camera cap colour (blender/materials.py), which is where it leads on the wrist cap.

**The ep0 episode**, the nominal action replay on the fitted scene
(`render/ep0_ep0_action/rt/`):
- `front.mp4`, `grip.mp4` and `side_by_side.mp4` (real | render, front over wrist, 1280x960,
  488 frames)
- real-time, 3 renders per frame, 0.149 s per camera image, 3 min for the whole run
- The sample frames on the physics log are in `samples_rt/` beside them. The unreleased
  cap is visible in the wrist view after frame 368, while the real one is in the mug.

**Findings on the way (MEASURED):**
- Isaac Lab's 'quality' preset turns on RTX's own ambient light
  (`/rtx/sceneDb/ambientLightIntensity 1`). It dominated every image until `set_mode` set
  it to 0.
- A render product whose updates were disabled came back with an empty HdrColor, so both
  cameras render every frame.
- Hiding the robot hides the wrist camera prim too. Its render product then showed the
  grey check quad for ~30 more frames, so the grey check runs last.
- `replay --render` still renders with the plain `scene.Look` (TiledCamera LDR at physics
  time), for quick previews. `render` is the look-accurate playback.

**Limits:**
- No cables, no camera body detail, no room geometry beyond LOOK's cylinder.
- The kinematic samples leave the carried cap on the table, as Blender's do.
- The wrist cap is 14-16 dB in every renderer: LOOK's one-sigma colour model (the cap alone
  wants sigma 1.59).
- The steel's room reflection is by face (floor and rim), not by mirror ray as in Blender.

## Live (`live.py`)

`IsaacEngine` is `../live`'s Isaac engine: the action-mode replay above, stepped on
demand, so the real leader arm, `lerobot-record` and policies drive it as they drive the
bench arm.

```
./robot real2sim live serve --engine isaac --layout real:0 --clock lockstep     # holds the GPU lock
$R2S_HOST_PY sim/real2sim/isaac/tests/live_replay.py --episode 0 --images       # lockstep vs the offline replay
$R2S_HOST_PY sim/real2sim/isaac/tests/live_replay.py --realtime 60 --resets real:0,real:1   # (server on the realtime clock)
```

- **Physics**: this README's, unchanged. The scene is `scene.py`'s, stepped as `params.py` says,
  with `servo.PhysxServo` on every substep. The goals come from `live.goals.OnlineGoals`
  (dead time, firmware clamp, zero-order hold, slew), not from a recorded array.
- **Stage**: built once, on the first layout. The spare cap bodies B and C are appended
  after the mug; a reset parks unused caps 3 m out, hidden. The stage is not rebuilt in
  process: replicator's annotator teardown raised, and the rebuild after it hung.
- **Images**: the look (`look.py`) in RTX real-time, read as HdrColor. `blender.camera_model`
  runs on the GPU: 2 ms for both cameras, against 17 + 58 ms on the host.
  - One RTX frame per observation. The server renders it right after each step.
  - The camera state is LOOK's reference (no per-row white balance).
  - The cameras' 2.3 / 1.4 frame latency is not emulated.

Measured on the fitted scene `eaa2d91c5133fa25` (RTX 3060, 1 env):

| | |
|---|---|
| ep0's 488 recorded actions over the socket, lockstep, started at the recording's start | encoders equal the offline `ep0_action` pose log **bit for bit** (max \|Δq\| 0.0°, both cameras rendered every frame); cap A in the mug |
| the same from the server's own rest pose (the median first frame: Wrist_Roll 28° and Pitch 4.4° off ep0's) | max 28.3° at frame 0; mean 0.002-0.10° per joint after frame 30; cap A in the mug |
| ep0 again after a re-placed reset | bitwise for 214 frames, then within 0.09°; in the mug |
| a step (16 physics steps at 480 Hz) | p50 117-204 ms by phase. In the reach, PhysX is 123 of 138 ms |
| a render (both cameras, one RTX frame) | p50 38 ms: RTX 33.8, annotators 0.8, camera model 2.1 |
| one render against a converged one (5 more), along ep0 | front 49.8-58.4 dB, wrist 42.6-57.3 dB PSNR |
| realtime clock, both images every frame | **0.19× real time** (0.16-0.21); observe round trip p50 163 ms |
| lockstep, both images every frame | 0.14× real time; observe 1.6 ms (the frames are pre-rendered), act 216 ms |
| start-up to serving | ~18 s: Kit ~12 s, first build 5.5-6.5 s |
| reset (re-place, settle 0.3 s, 3 renders), 1-3 caps | 1.5-1.7 s in the engine, 1.7-2.0 s over the socket |
| GPU memory | 4.8-5.2 GB, the server process |

- **Speed is the replay's physics.** 64 TGS iterations on the GPU for a single env are
  launch-bound. Anything faster (fewer iterations, CPU PhysX) is physics these numbers do
  not vouch for. The MuJoCo engine is the real-time one.
- **`--viewer`** opens the Kit window. Its camera sits inside LOOK's backdrop cylinder, since
  from outside the wall hides the scene; R starts a new layout.
  - The viewport is dark, and no render setting may brighten it. RTX applies exposure and
    colour grading before the HdrColor the cameras read (film ISO ×2^10 scaled HdrColor
    ×1019, auto exposure ×2.3), and it ignores a camera's own USD exposure.
  - Closing the window ends the session: exit 0, socket removed. Kit's quit is intercepted,
    because its shutdown hangs on this box and would keep the GPU lock.
  - The R key and the quit path were checked by injected events and `post_quit`. A real
    click on the window was not tested.
- **GPU lock**: `live/run.sh` runs the server under `flock`. Started any other way,
  `boot()` takes the lock itself, waiting if another job holds it.
