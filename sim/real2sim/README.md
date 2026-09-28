# real2sim: the real teleop episodes, replayed physically in simulation

Every real episode of a LeRobot dataset (default `soarm101/testingt_real2sim_20260927_152936`)
is replayed in two stacks that are compared with the recording and with each other:

| stack | physics | rendering |
|---|---|---|
| `mujoco` + `blender` | MuJoCo 3.11 (CPU) | Blender 5.2 (EEVEE / Cycles), offline from the pose log |
| `isaac` | Isaac Sim 4.5 / Isaac Lab 2.1 (PhysX, GPU) | RTX real-time or path tracing, in process |

In the data, the arm picks small teal bottle caps off a black glossy mat and drops them
into a red enamel mug: 1 cap in episode 0, 3 in episode 1 (one regrasp), 2 in episode 2.
Episode 3 is idle and excluded. All six real picks succeed.

**Physically feasible, or it does not count.** Objects are placed once at reset and move
only through simulated contact. Nothing is welded to the gripper, no object is scripted,
and no force is hacked in. The servos are limited to what the real STS3215 produces
(identified from the data: 3.41 N·m arm clamp, 1.155 N·m jaw). Finger-cap penetration stays
under 1 mm.

**Close to reality is measured, not asserted.** Every claim carries a number from
`metrics` against the dataset: joint RMSE and lag, fingertip error, grasp and release
frames, jaw angle while holding, cap positions in both camera views, outcome, image metrics.

**The cameras come from the dataset alone.** Both cameras' intrinsics, distortion and
poses, the wrist mount, the joint zero offsets, the gripper map, the table height and the
object poses and sizes are fitted jointly from this dataset's front and wrist frames plus
the recorded joints (`calib/`). No checkerboard, no live camera: earlier calibrations are
starting guesses only. Both views must agree on the same arm pose and object poses at the
same timesteps, and the fit is scored per view and across views on a held-out episode.

## The pipeline

```
./robot real2sim core extract            # dataset -> episodes.npz, readable videos, frame caches (Docker, ~11 s)
./robot real2sim calib fit               # dataset-only multi-view calibration -> config/<ds>.calib.json (~30 min)
./robot real2sim look build              # dataset-derived textures, lights, materials, sample set (~2 min)
./robot real2sim mujoco servo-fit        # servo model from action vs state -> config/<ds>.servo.json (~10 min)

./robot real2sim mujoco eval --ensemble 20                                    # MuJoCo: all picks, nominal + ensembles (~5 min)
./robot real2sim isaac all                                                    # Isaac: nominal, 1 env each (~15 min)
./robot real2sim isaac all --num-envs 16 --perturb mujoco --tag ens16         # the MuJoCo seeds, seed for seed (~30 min)

./robot real2sim blender render --episode 0 --poselog mujoco                  # MuJoCo replay -> Blender, both cameras (~6 min)
./robot real2sim isaac render --episode 0 --log <isaac run dir>               # Isaac replay -> RTX, both cameras
./robot real2sim blender samples && ./robot real2sim isaac samples            # the 24 scored sample timesteps

./robot real2sim compare summary         # every number above, scene hash checked -> compare/summary.md
./robot real2sim compare video --episode 0   # real | MuJoCo+Blender | Isaac RTX, front over wrist
```

**Live: teleop, record and run policies on the sim.** `./robot teleop|record|infer|
infer-remote|openpi-eval ... --sim mujoco|isaac` runs the usual command against the live sim
instead of the arm, with the real leader: `live/README.md`.

`./robot real2sim mujoco view --log <pose log>` and `./robot real2sim isaac replay ... --gui`
open a viewer. Each track's README explains its commands, options and measurements:
`calib/`, `mujoco/`, `isaac/`, `look/`, `blender/`.

## What the dataset gave (calib/, scene hash `eaa2d91c5133fa25`)

Fitted on episodes 0-2; the gates are scored on episode 2 with the fit made on 0 and 1.

| | fitted | sigma |
|---|---|---|
| joint zero offsets (pan, lift, elbow, wrist flex, roll) | 0.65, 1.13, −4.43, 0.29, −2.09° | 0.1-0.3° |
| gripper map | jaw° = −11.13 + 1.321 · pct | 0.19°, 0.005 |
| table height (base frame) | 24.0 mm | 5.5 mm, see below |
| cap | 29.32 mm (rigid closed end) × 13.9 mm; 3 mm tamper lip | 0.85, 2.3 mm |
| mug | rim radius 41.9 mm, 121 mm tall | 0.7, 5.3 mm |
| wrist camera | fx 695.5, fy 657.2, k1 −0.456, k2 0.272, on the gripper at (3.1, 73.5, −4.1) mm | |
| front camera | 1.111 m over the base, 10° tilt, k1 −0.043, k2 2.09 | |

Held out (episode 2): front arm silhouette IoU 0.755, wrist finger IoU 0.966, a cap seen by
the wrist camera lands within 8.6 mm of the front camera's ray, wrist mug rim 6.9 px.

Three judgement calls, with the numbers behind them:
- **Front intrinsics stay at the prior.** Freed, f and cy slide 30 and 38 px along a valley
  where the image mapping changes by under 3 px (a principal-point shift is a camera tilt
  to first order), and the held-out scores do not change. The data determine the mapping,
  not the split between K and pose.
- **The cap is its rigid closed end, 29.32 mm, not the tapered mean, 30.82 mm.** The real
  encoder settles at the fingers' zero-force contact, a 29.4-29.6 mm gap, and ep0's cap falls
  there. A 30.82 mm cylinder held all six picks 0.5-1.2° too open in both engines and never
  released ep0; 29.32 mm brings the holds to −0.7..+0.2° and releases it.
- **Two ensembles.** The table's 5.5 mm sigma is the leave-one-episode-out spread, and one
  fold (without episode 1) dominates it by trading a 13.5 mm table for an 18 mm cap; the
  other three folds agree within 2.5 mm. `stress` perturbs the table by 5.5 mm alone, which
  makes geometry no fold produced; `consistent` uses 1.2 mm, the spread of the three folds
  that agree. Both are reported.

## Isaac Sim vs MuJoCo + Blender

Same scene (`eaa2d91c5133fa25`), same servo model, same replay modes, same metrics code,
same perturbation draws seed for seed. `action` plays the recorded leader action through
the identified servo (the honest default); `track` makes the arm follow the recorded
follower state, so the arm is where the cameras saw it. Full tables: `compare/summary.md`.

### Physics

| nominal replay | MuJoCo | Isaac |
|---|---|---|
| caps in the mug, action / track | **6/6** / 5/6 | **6/6** / **6/6** |
| drop frame vs the real one, action (track) | −2..+5, ep0 +5 (−2..−1, ep0 held) | +1..+13, ep0 +13 (−1..+3, ep0 +17) |
| jaw angle while holding vs real, both modes | −0.54..+0.20° | −0.68..+0.17° |
| squeeze per finger | 4.5-16.2 N | 4.3-23.9 N |
| arm joint RMSE vs recorded state, action / track | 0.50-0.60° / 0.03-0.04° | 0.50-0.60° / 0.05-0.07° |
| fingertip error, action / track | 3.4-5.2 / 0.2-0.3 mm | 3.6-5.3 / 0.4-0.5 mm |
| finger-cap penetration, max over all frames | ≤ 0.46 mm | ≤ 0.60 mm |
| slip during the carry | ≤ 0.07 mm | ≤ 0.13 mm |

| episode success over the perturbed seeds | MuJoCo (n=20) | Isaac (n=15) |
|---|---|---|
| stress: ep0 / ep1 / ep2, action | 0.35 / 0.65 / 0.80 | 0.53 / 0.67 / 0.80 |
| stress, track | 0.25 / 0.65 / 0.85 | 0.53 / 0.60 / 0.80 |
| consistent: ep0 / ep1 / ep2, action | 0.40 / 0.85 / 0.90 | 0.47 / 0.87 / 0.87 |
| consistent, track | 0.30 / 0.70 / 0.95 | 0.40 / 0.67 / 0.93 |

The engines agree within sampling noise (±0.1 at these n) except on ep0, which is a knife
edge in both: the real gripper opens by 1.5 % to let go, and 0.1 mm of cap width decides
whether the sim cap falls on time, late, or not at all. Over the perturbed seeds the worst
single-frame finger-cap penetration is 1.7 mm in MuJoCo (seeds that draw a wide cap on a
raised table) and 0.84 mm in Isaac; every nominal replay stays under 0.6 mm.

### Rendering

LOOK's 24 sample timesteps × 2 cameras, arm at the recorded joints, every renderer fed the
same dataset-derived look (mat texture, backdrop, key light, materials) and warped into
the fitted distorted cameras, scored by the same scorer: PSNR dB / mean-colour ΔE.

| renderer | s/image | front: whole | arm | cap | mug | wrist: whole | fingers | cap | mug |
|---|---|---|---|---|---|---|---|---|---|
| Blender EEVEE | 0.31-0.34 | 23.17 / 2.0 | 16.5 / 5.0 | 14.3 / 12.8 | 16.6 / 6.8 | 19.86 / 2.7 | 22.4 / 2.9 | 15.2 / 18.2 | 15.3 / 8.0 |
| Blender EEVEE, shared look only | 0.31-0.34 | 23.15 / 2.0 | 16.5 / 5.0 | 14.3 / 12.8 | 16.4 / 7.0 | 19.61 / 2.8 | 22.4 / 2.9 | 14.1 / 21.0 | 15.2 / 9.3 |
| Blender Cycles | 0.67-1.19 | 23.27 / 1.9 | 16.7 / 4.9 | 15.0 / 13.6 | 16.4 / **4.0** | 19.54 / 2.6 | 22.8 / 2.9 | 15.4 / 17.9 | 14.7 / 8.3 |
| Isaac RTX real-time | 0.28 | 23.19 / 3.0 | 16.6 / 6.1 | 14.8 / 15.6 | 16.7 / 6.8 | **20.17** / 2.8 | 22.1 / 3.1 | 14.5 / 22.0 | 15.8 / 9.0 |
| Isaac RTX path-traced | 7.2 | **23.33** / 2.1 | 16.7 / 4.8 | 15.3 / 13.8 | 17.1 / 6.5 | 19.85 / 2.7 | 23.2 / 2.6 | 14.5 / 21.1 | 15.8 / 8.1 |

- For scale: a perfect static background (the real arm-free plate) scores 29.1 dB on the
  front background, where every renderer here reaches 26.6-26.9.
- On identical inputs (the shared-look row against Isaac) the front view agrees within
  0.2 dB and the wrist view within 0.6 dB, Isaac's real-time path ahead on the wrist
  background. Blender's two extra fits (`blender/materials.py`: a per-camera cap colour, a
  ×2.1 enamel red) buy it 0.25 dB on the wrist view.
- The low region scores (arm 16.5, cap 14-15 dB) are the same for every renderer, so they
  are the calibration's residual misalignment and the unmodelled servo cables, not the
  engine: Blender's silhouettes match the MuJoCo label renders to IoU 0.99-1.00
  (`blender check`), and Isaac's cameras put marker spheres within 0.26 px rms (front)
  and 0.04 px (wrist) of `camera.project` (`isaac camcheck`).
- `compare/ep0_triptych.mp4` shows episode 0 as real | MuJoCo + Blender | Isaac RTX, front
  over wrist, at the same instants (the fitted camera latency applied to the real rows).

### Cost on this box (RTX 3060 12 GB)

| | MuJoCo + Blender | Isaac |
|---|---|---|
| one replay | 3.3-5.2× real time on one CPU core | 0.17-0.19× real time (1 env, GPU) |
| 126 replays (nominal + 20-seed ensembles, 3 episodes × 2 modes) | 274 s on 6 processes | ~32 min for 96 (16 envs, 2.1× real time aggregate) |
| rendering, per camera image | EEVEE 0.30-0.35 s, Cycles 0.63-1.13 s (offline) | RTX real-time 0.15-0.28 s, path traced ~7 s (in process) |
| GPU memory | none for physics; Blender's not measured | 2.5-2.7 GB physics, 4.9-5.3 GB rendering |
| start-up | < 1 s (MuJoCo), 1-14 s (Blender shaders) | 11-18 s (Kit) |
| determinism | bitwise: two fresh ep1 runs and the 6-process eval's log are identical | bitwise for a fixed env count; changing the env count moves results by up to 0.14° / 0.51 mm |

### Which one

The two stacks are equally faithful here: same outcomes within noise, the same tracking
error, jaw holds within 0.7° of the data in both, penetration under 0.6 mm in both, and
images within 0.2 dB (front) and 0.6 dB (wrist) of each other when they render the same
look inputs. What separates them is cost and plumbing:

- **MuJoCo + Blender is the better replay harness on this machine.** A replay is ~25×
  faster than Isaac's and ensembles ~9× faster, on the CPU, leaving the GPU free, and it
  is bitwise reproducible whatever the process layout. The identified servo is native
  MuJoCo (a clamped position actuator, joint damping, armature, dry friction, and the
  finger's series compliance as one more hinge), and the robot model is mjlab's
  `so101_calib.xml`, the one this repo's mjlab RL tasks train on. Blender renders the pose
  log offline at 0.3 s per image with EEVEE; Cycles, at 2-3× that, has the lowest colour
  error on the polished mug (front ΔE 4.0 against 6.5-6.8 for every other renderer),
  because it traces the multi-bounce reflections of the steel.
- **Isaac is the stack for closed-loop visual evaluation.** It renders both cameras in the
  same process at 0.15-0.28 s per image, so a vision policy can be evaluated against its
  own images, which MuJoCo + Blender can only do at Blender's speed. It is also where the
  repo's Isaac tasks (push-T, reach, teleop, domain randomisation) already live. The price
  is plumbing: PhysX cannot express the identified servo directly, so the drive target is
  rewritten every physics step to put the torque clamp inside, dry friction is a bristle
  feed-forward, and the jaw's horn is a state integrated in Python (`isaac/servo.py`).

Path tracing (Isaac) and Cycles (Blender) cost 2-25× more than the real-time paths and
gain at most a few tenths of a dB on these metrics: the remaining image error is the look
model and the calibration, not the renderer.

## Known gaps

- **ep0's release is a knife edge** in both engines (5-17 frames late, or held by 0.1 N in
  MuJoCo track mode). The cap is a rigid cylinder; the real tamper lip is not modelled.
- **The cap height and the table height trade off** in one calibration fold. Measuring the
  cap with calipers (height, closed-end diameter) would pin both.
- **Open-loop drift:** in action mode the fingertip is 3-6 mm from where the real one was at
  the grasp, because the operator closed the loop visually and the replay cannot.
- **ep1 C** rests about 17 mm from its front-camera blob in the calibration.
- **Look:** the wrist camera's auto-exposure is not modelled; only 2.6 % of the room behind
  the mug was ever observed; the look was tuned on the same 24 samples it is scored on.

## Frames, units, conventions

- **World:** the URDF base frame, i.e. the `base` body of mjlab's `so101_calib.xml` at
  identity. It is not the Isaac push-T env frame and not the mjlab scene frame (both yaw
  the robot).
  - Base −y is forward, which is image up in the overhead camera.
  - The table is at `scene.table_z()`, not at z = 0.
- **Units:** SI, radians, quaternions (w, x, y, z).
  - Joint vectors follow `units.URDF_JOINTS`: Rotation, Pitch, Elbow, Wrist_Pitch,
    Wrist_Roll, Jaw.
  - Recorded values stay in LeRobot units: arm in degrees, gripper in percent.
    `scene.units()` converts them with the scene's joint offsets and gripper map.
- **Real joint ranges:** `units.limits_rad()`. They exceed the URDF on Pitch, Elbow,
  Wrist_Pitch and Wrist_Roll, and the data go there, so engines must widen their limits.
- **Cameras:** OpenCV model.
  - Pixel (0, 0) is the centre of the top-left pixel.
  - Distortion is (k1, k2, p1, p2, k3).
  - Renderers are pinholes. Render `camera.render_spec(cam)`, then warp the result into
    the real camera with `camera.remap(img, camera.remap_maps(cam, spec))`.
  - Conversions: `to_mujoco` (verified by a render test to < 0.5 px), `to_blender`,
    `to_usd` (square pixels only: use `render_spec(cam, square=True)`).
  - Images trail `observation.state` by the fitted latency (`cameras.<cam>.latency`: 2.3
    front, 1.4 wrist frames): compare a sim frame at state row t with real image t + lag.
- **Objects** (`objects.py`):
  - Cap origin is the centre of the rim plane, with +z towards the closed top. Open-up is
    a pose (180° about x), not a different mesh.
  - Mug origin is the centre of the bottom face, with +z up and the handle along +x.

## Files

Configuration lives in `sim/real2sim/config/`:
- `<ds>.json`: provisional values, each with its source.
- `<ds>.calib.json`: dataset-fitted cameras, table, offsets, object poses and sizes.
- `<ds>.servo.json`: the identified actuator model.

`scene.load()` deep-merges them in that order. The schema is documented in `scene.py`,
and every group carries `"source"`.

Outputs go to `sim/outputs/real2sim/<ds>/` (gitignored):
- `episodes.npz`: action and state in LeRobot units, plus episode bounds (`extract.py`).
- `frames/front.npy` and `frames/grip.npy`: uint8, N × 480 × 640 × 3, about 2 GB each.
  Open them with `episodes.frames()` (memmap).
- `video/<cam>/chunk-000/file-000.mp4`: readable copies of the root-owned dataset videos.
- `meshes/cap_<hash>.{obj,stl}` and `meshes/mug*_<hash>.{obj,stl}`.
- `<track>/…`: each track's own outputs. Engine replays write the common pose log
  (`poselog.py`); `compare/` holds the summary and the side-by-side videos.

| directory | what |
|---|---|
| `*.py`, `run.sh`, `env.sh`, `tests/` | the engine-agnostic core: dataset, units, scene, kinematics, camera model, objects, pose log, metrics |
| `calib/` | the dataset-only multi-view calibration, and the real per-frame observations (`observations.py`) |
| `mujoco/` | servo identification, the MuJoCo scene, replay, evaluation, EGL renders |
| `isaac/` | the Isaac Lab scene, the servo in PhysX, replay, evaluation, RTX rendering |
| `look/` | dataset-derived textures, lights, materials, the sample set and its scorer |
| `blender/` | the Blender renderer for pose logs and kinematic replays |
| `compare/` | the cross-stack summary and the side-by-side video |

`sim/real2sim/env.sh` exports the interpreters and paths from `paths.py`: `$R2S_HOST_PY`,
`$R2S_MJLAB_PY`, `$R2S_ISAAC_PY`, `$R2S_BLENDER`, `$R2S_COACD_PY`, `$R2S_GPU_LOCK`,
`$R2S_OUT` and the rest. Every Isaac run and every Blender GPU render runs under
`flock "$R2S_GPU_LOCK"`, so they never share the 12 GB card.
