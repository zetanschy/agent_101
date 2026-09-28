# real2sim LOOK, part 2: the Blender renderer

This track renders the real episodes in Blender 5.2 (EEVEE or Cycles on OptiX) the way
the two real webcams saw them, and scores every picture against the recording with
LOOK's shared scorer. It renders any engine's pose log, i.e. a physics replay (the MuJoCo
track's, or Isaac's), and a kinematic replay of the dataset for appearance work.
Every look input comes from LOOK part 1 (`look/look.json`); the cameras, table, joint
offsets and object poses come from the merged scene (`scene.load`, CALIB's fit).

```
./robot real2sim blender render --episode 0 --poselog mujoco            # the nominal MuJoCo replay, EEVEE
./robot real2sim blender render --episode 0 --poselog PATH.npz --engine cycles
./robot real2sim blender render --episode 1 --kinematic --samples-only   # arm = FK(state), objects static
./robot real2sim blender samples [--engine eevee|cycles] [--poselog mujoco|PATTERN]   # the 24 LOOK samples + score
./robot real2sim blender score DIR [--wb]                                # LOOK's metrics of DIR/samples/*.png
./robot real2sim blender check [--episode 0]                             # geometry: silhouettes vs LOOK's labels
./robot real2sim blender test                                            # 6 host tests, < 1 s, no GPU
```

`render` also takes `--frames A:B[:S]`, `--cams front,grip`, `--png` (every frame),
`--keep-pinhole` (the linear EXRs), `--no-video`, `--wb frame|episode|reference`,
`--exposure constant|look`, `--quality KEY=VALUE` (engine settings, e.g.
`samples=128`, and `room_reflection=false`) and `--tag`. `--poselog mujoco` is
`sim/outputs/real2sim/<ds>/mujoco/logs/ep<N>_action_seed0.npz` (mujoco/README.md).

Every Blender launch runs under the shared GPU lock (`paths.gpu_locked`, i.e.
`flock sim/outputs/real2sim/.gpu.lock`), so a render queues behind Isaac jobs and the
"blender scene built ... after launch" line reports the wait. No Blender process
outlives its command.

**Outputs** go to `sim/outputs/real2sim/<ds>/blender/<tag>/`. The tag is
`ep<N>_<source>_<engine>[_samples]` or `samples_<source>_<engine>`.
- `front.mp4` and `grip.mp4`: H.264 crf 18 at the dataset's 30 fps.
- `side_by_side.mp4`: real | Blender, front over wrist. Each render tile is captioned
  with its source, so a kinematic replay cannot pass for a physical one.
- `samples/<id>_<cam>.png`: LOOK's sample frames inside the rendered range.
- `score.json`: LOOK's `imagemetrics.score` of those samples.
- `manifest.json`: scene hash, look inputs hash, log metadata, camera model, engine
  settings, timings and warnings.
- `_job/job.json` and `job.npz`: exactly what was rendered. The textures are deleted
  after the render; a re-render rebuilds them.

## How a frame is made

Four modules, and the split is deliberate: Blender's Python has numpy but no mujoco,
cv2, trimesh or scipy, so every decision is taken on the host and written down.

| step | module | what |
|---|---|---|
| job | `job.py` | Poses at camera time, meshes, textures, lights and materials, into `_job/` |
| render | `bl_render.py` | Inside Blender: build the scene once, set 7 link and N object matrices per frame, write linear half-float EXR pinholes |
| camera | `camera_model.py` | Lens remap into the real distorted 640x480, optical blur, colour model, sRGB, uint8 |
| post | `post.py` | Stream the EXRs into mp4s and PNGs in order, delete each EXR after its last use, build the side-by-side |

**Camera time.** CALIB fitted a latency per camera: front 2.27 frames, wrist 1.42.
The image of row k shows the world at t = k - latency. Link and object poses are
interpolated between log rows there (positions linearly, rotations by nlerp).
Identical poses render once per camera: `dedupe` merges the rest phases.

**Pinholes.** Blender renders an oversized pinhole per camera (`camera.render_spec`:
front 635x481, wrist 771x579, since the wrist lens has strong barrel distortion).
`camera.remap_maps` warps it into the real pixels. The conversion is
`camera.to_blender`; `check` proves the whole chain against LOOK's MuJoCo label
renderer.

### What the scene contains

| element | content | provenance |
|---|---|---|
| arm | The 17 visual meshes of mjlab's `so101_calib.xml`, plus the printed klip bracket and the KWC-500 body box on the gripper (19 meshes) | Same as LOOK's label renderer (`look/segment.py`) |
| arm materials | Printed parts and fingers: white PLA (0.80, the white reference); STS3215 servos: black; mount: black | LOOK `materials` |
| caps | The core cap mesh at the scene's dimensions (29.3 x 13.9 mm), one per log object | `objects.cap_mesh`, CALIB |
| mug | Body with two slots (red enamel outside and bottom; steel inside and on the rim), plus the handle | `objects.mug_parts`, CALIB 86.8 x 121 mm |
| table | A quad at `table.z`, textured with LOOK's de-lit albedo; GGX F0 0.04, roughness 0.155 | **Dataset-derived texture** (front clean plate + wrist mosaic) |
| backdrop | An open cylinder (R 0.80 m, 0.60 m tall) showing the room; invisible to diffuse and shadow rays | **Dataset-derived texture** (wrist frames), 2.6 % observed |
| world | `env.hdr`, whose uniform fill (0.161) carries the ambient; reflections see only its observed part | LOOK `backdrop` |
| key | A sun with LOOK's fit: az -120.45 deg, el 49.43 deg, 24.7 deg disk, 0.652 normal irradiance. Blender's sun strength is normal irradiance (measured), so the value goes in unchanged | LOOK `lights.key` (the mug shadows) |

The two dataset-derived textures are background only. No real pixel is ever pasted
into an arm or object region. The per-frame camera states (front white-balance gain
per row; wrist white balance per frame) are LOOK's global numbers per frame, not
pixels. They replay the webcams' auto white balance the way the arm replays the
recorded joints, and the manifest lists them under `dataset_derived_inputs`.

### Colour

| camera | model |
|---|---|
| front | camera_lin = π L_PLA / ρ_PLA × radiance × g_row. g_row is the row's white-balance gain against LOOK's reference state |
| wrist | camera_lin = L_PLA × sat(π radiance / ρ_PLA, σ 1.15) / (s × wb_frame), with s = 1 (`--exposure constant`) |

After the colour model: clip, then the sRGB curve (assumed; no bracketed target
exists).

## Look fixes, measured

Each fix was measured on the 24 samples. Every number here is on the fitted scene
**6f282e812cc585f5** with look inputs **121e4b46ede6e02e** (LOOK rebuild of 17:17).
The baseline is the renderer as the previous build left it, re-run on this scene.

Baseline to final, kinematic EEVEE, 24 samples. Each cell is PSNR dB / SSIM / dE
(mean colour).

| camera | region | baseline | final |
|---|---|---|---|
| front | background | 20.9 / 0.74 / 9.0 | **26.7 / 0.92 / 2.5** |
| front | mug | 15.0 / 0.35 / 10.1 | **16.6 / 0.42 / 6.8** |
| front | arm | 16.5 / 0.61 / 4.3 | 16.5 / 0.62 / 5.0 |
| front | all | 19.8 / 0.71 / 3.2 | **23.1 / 0.88 / 2.0** |
| wrist | background | 18.2 / 0.60 / 11.0 | **21.1 / 0.74 / 7.6** |
| wrist | fingers | 21.3 / 0.84 / 2.9 | **22.4 / 0.89 / 2.9** |
| wrist | cap | 14.2 / 0.60 / 21.4 | **15.9 / 0.60 / 18.3** (physics log: 18.1 / 0.69 / 5.7) |
| wrist | mug | 14.5 / 0.46 / 8.8 | **15.4 / 0.52 / 7.8** |
| wrist | all | 17.6 / 0.64 / 3.5 | **19.8 / 0.75 / 2.7** |

The front arm's dE rises 4.3 -> 5.0: the white PLA lost the fill's second reflection,
and its shaded faces were already too grey (see Known gaps).

1. **Reflections see only the observed room** (`job.textures` REFLECTIONS).
   - LOOK's dielectric albedos are effective albedos: ρ_PLA L / L_PLA with only the
     key's specular lobe removed. The ambient's reflection is therefore already inside
     them.
   - Mirroring the uniform fill a second time adds F0 0.04 × 0.16 radiance. That is
     +0.015 camera-linear on a mat whose total is 0.006-0.008.
   - So diffuse and camera rays see `env.hdr`, and glossy rays see only its observed
     part (`tex_env_glossy.npy`). The backdrop's unseen texels are transparent to
     glossy rays.
   - Both engines honour a world switched on Light Path "Is Glossy Ray" (measured: a
     diffuse plane stays at 1.0 / 0.99, a metal plane goes to 0).
2. **The steel reflects the rest of the room at half the fill** (`materials.py` ROOM
   REFLECTION).
   - The steel's colour is a spec F0; nothing of the room is in it.
   - With a black room it renders black: front mug dE 30.9 even in Cycles, whose glossy
     bounces do trace key -> lit wall -> floor.
   - So it adds 0.5 × F0 × the unobserved environment in its mirror direction. The term
     applies only where that direction points up out of the mug; the walls' downward
     mirror rays are traced as usual.
   - The scale 0.5 is measured two ways:
     - At 1.0, the steel's median came out 1.75-2.07x the real one (EEVEE and Cycles).
     - The mat's glare fit (F0 0.04) caps the zenith it mirrors at 0.46 × fill.
3. **The cap's colour in the wrist camera** (`materials.py` PER-CAMERA COLOUR).
   - With the shared saturation factor (σ 1.15; the cap alone wants 1.59), the wrist
     render's cap was 3.9x too red and 16-30 % too bright in G/B.
   - The front cap at rest is within 3-8 % per channel.
   - The wrist pass therefore uses LOOK's measured wrist-camera albedo of the cap top
     (0.077, 0.388, 0.428). This is a camera property: the cap keeps one mesh and one
     roughness.
4. **Enamel red** (`materials.py` RENDER FIT).
   - Through this renderer the enamel came out too dark in R:
     - 2.1x on e0's wrist carry/release samples, which see the lit side (1.8-2.2 on
       e1/e2).
     - 4-5x in the front view, which sees the key-shadowed side.
   - LOOK's value rests on 416 lit front pixels. R is scaled by the wrist factor (2.1);
     the shadow-side remainder is a known gap.
5. **Optical softness** (`camera_model.py` OPTICS).
   - A Gaussian in linear light, σ = 0.8 px (front) and 2.5 px (wrist), fitted on e0's
     8 samples (all-region PSNR) and checked on e1-e2.
   - Wrist gain on e1-e2: +0.40 dB and SSIM +0.026. Front: +0.15 dB.
   - The real fingers, 2-8 cm from the lens, want more; defocus is depth-dependent.
6. **Wrist exposure** stays constant, s = 1.
   - The wrist fingers' linear bias is +0.008 on about 0.5.
   - LOOK's per-frame s is 1.35-1.89 at the rest samples, where s = 1 already nulls the
     background, and it scatters over 0.39-2.33 in carry frames.

## Measured results

Scene **6f282e812cc585f5**, look inputs **121e4b46ede6e02e**. LOOK's 24 samples are 8
phases x episodes 0-2 per camera, eroded 2 px; `n` is the samples in which the region
exists. `./robot real2sim look score` returns the same numbers (checked: max
difference 0.0).
- **kinematic**: the arm is FK(observation.state); the objects stay static at reset.
- **mujoco-nominal**: the nominal MuJoCo action replays (`--poselog mujoco`), whose
  logs carry the same scene hash.

Each cell is PSNR dB / SSIM / dE (mean colour) / dE (pixels), with `n` in brackets.

**Front camera**

| source, engine | arm | fingers | cap | mug | background | all |
|---|---|---|---|---|---|---|
| kinematic, EEVEE | 16.5 / 0.62 / 5.0 / 13.1 (24) | 15.0 / 0.40 / 6.7 / 15.5 (23) | 14.5 / 0.22 / 11.9 / 22.5 (16) | 16.6 / 0.42 / 6.8 / 15.4 (24) | 26.7 / 0.92 / 2.5 / 3.4 (24) | 23.1 / 0.88 / 2.0 / 4.6 (24) |
| kinematic, Cycles | 16.7 / 0.63 / 4.9 / 12.8 (24) | 15.3 / 0.41 / 6.2 / 15.0 (23) | 15.3 / 0.26 / 12.7 / 21.0 (16) | 16.4 / 0.39 / 4.0 / 15.2 (24) | 26.7 / 0.92 / 2.6 / 3.4 (24) | 23.2 / 0.88 / 1.8 / 4.7 (24) |
| mujoco-nominal, EEVEE | 16.2 / 0.60 / 5.1 / 13.4 (24) | 14.9 / 0.38 / 6.7 / 15.8 (23) | 15.1 / 0.24 / 11.3 / 20.9 (16) | 16.5 / 0.41 / 6.0 / 15.6 (24) | 27.4 / 0.92 / 2.6 / 3.3 (24) | 23.2 / 0.88 / 2.1 / 4.6 (24) |
| mujoco-nominal, Cycles | 16.4 / 0.62 / 5.0 / 13.1 (24) | 15.1 / 0.40 / 6.3 / 15.3 (23) | 15.3 / 0.26 / 12.8 / 20.9 (16) | 16.4 / 0.39 / 3.9 / 15.2 (24) | 27.4 / 0.92 / 2.7 / 3.3 (24) | 23.3 / 0.88 / 2.0 / 4.6 (24) |

**Wrist camera**

| source, engine | fingers | cap | mug | background | all |
|---|---|---|---|---|---|
| kinematic, EEVEE | 22.4 / 0.89 / 2.9 / 6.7 (24) | 15.9 / 0.60 / 18.3 / 24.4 (21) | 15.4 / 0.52 / 7.8 / 19.2 (11) | 21.1 / 0.74 / 7.6 / 9.7 (24) | 19.8 / 0.75 / 2.7 / 10.5 (24) |
| kinematic, Cycles | 22.8 / 0.91 / 2.9 / 6.4 (24) | 16.1 / 0.61 / 17.8 / 23.7 (21) | 14.7 / 0.51 / 8.3 / 19.9 (11) | 20.4 / 0.73 / 9.0 / 10.3 (24) | 19.5 / 0.75 / 2.7 / 10.8 (24) |
| mujoco-nominal, EEVEE | 21.3 / 0.87 / 3.1 / 7.3 (24) | 18.1 / 0.69 / 5.7 / 14.1 (21) | 15.0 / 0.49 / 8.0 / 20.8 (11) | 20.3 / 0.72 / 8.5 / 10.4 (24) | 19.5 / 0.74 / 2.5 / 10.5 (24) |
| mujoco-nominal, Cycles | 21.6 / 0.89 / 3.2 / 7.1 (24) | 18.5 / 0.69 / 6.4 / 13.7 (21) | 14.3 / 0.48 / 8.0 / 21.6 (11) | 19.7 / 0.72 / 9.7 / 11.0 (24) | 19.3 / 0.74 / 2.5 / 10.8 (24) |

The wrist camera never sees the arm region (the printed links behind the wrist).

For scale, LOOK's ceiling is a perfect static background: the arm-free episode plate at
the frame's white balance. It scores 29.1 dB / 0.947 / dE 0.92 on the front background.

**Speed** (RTX 3060, 640x480 out, per camera, after the first frame; ~3 s launch and
scene build once the lock is held):

| engine | settings | front s/frame | wrist s/frame |
|---|---|---|---|
| EEVEE | 64 TAA samples, ray tracing, fast GI, 2 shadow rays | 0.30 | 0.33-0.35 |
| Cycles, OptiX | 64 spp, adaptive 0.01, OptiX denoiser, 6 bounces (4 glossy) | 0.63 | 1.12 |

The host camera model adds 0.03-0.05 s/frame per camera, overlapped with the render.
A launch's first frame costs about 1.1 s with a warm shader cache and up to 14 s cold
(EEVEE compiles its shaders, e.g. on the first run from a new login).

**Physics replay** (`render --episode 0 --poselog mujoco`, EEVEE, both cameras):
- Output: `blender/ep0_mujoco-ep0_action_seed0_eevee/` with `side_by_side.mp4` (real |
  Blender), `front.mp4` and `grip.mp4`. The log is MuJoCo's nominal ep0 action replay
  on scene 6f282e812cc585f5.
- 488 frames: 486 / 487 unique renders (a simulated pose is never exactly static, so
  dedupe saves almost nothing).
- **346.6 s end to end** (351 s wall, including a 22 s GPU-lock wait):
  - Blender 318.7 s: front 148.7 s, wrist 168.1 s.
  - Side-by-side 5.3 s.
- Its 8 samples score:
  - front all 23.6 dB / 0.88; mug 17.2 / 0.45 / dE 5.3.
  - wrist all 19.5 / 0.74; cap 19.5 / 0.75 / dE 5.0.

## Geometry vs appearance

**The renderer adds no misregistration.** `check` (ep0's 8 samples, EEVEE) compares
Blender's silhouettes with LOOK's MuJoCo label renderer at the same camera time:
- **Front:** IoU printed 0.998, fingers 0.998, mug 1.000, servo 0.991, mount 1.000.
  0.11 % of the foreground pixels disagree.
- **Wrist:** fingers 1.000, mug 1.000.

Cameras, meshes, link poses and the lens remap therefore agree end to end
(`blender/check_ep0_eevee/check.json`).

**Against the real frames, the geometry is the calibration's.** LOOK's sample QA on the
same scene:

| check | median | range |
|---|---|---|
| Front robot, model vs image-refined, IoU | 0.859 | 0.796-0.894 |
| Wrist fingers, model vs data, IoU | 0.931 | 0.722-0.977 |
| Wrist mug, model vs image, IoU | 0.384 | 0.245-0.809 |

The wrist mug's image mask is anchored on the red enamel, so it misses much of the
steel. CALIB's held-out ep2 gives front arm IoU 0.755, wrist fingers 0.966, and mug rim
6.9 px median in the wrist view.

So on the front arm and fingers (16.5 / 15.0 dB), a sizeable share of every region sits
on the wrong surface: a model-vs-image IoU of 0.86 leaves ~14 % of it misregistered by
several px (visible as an offset of the whole arm in ep0 descend). What the model
doesn't carry adds to that: the red/white servo cables and the gold part at the base.
Both are geometry, not look.

The background (26.7 dB, 0.92), whose geometry is a plane, shows what the look alone
achieves. The ceiling there is 29.1 dB.

**Physics vs kinematic.**
- The physics replay tracks the recorded joints with MuJoCo's error, so the view-dependent
  regions lose a little: front arm 16.5 -> 16.2 dB, wrist fingers 22.4 -> 21.3, wrist
  background 21.1 -> 20.3.
- The picked cap is where physics put it, which is closer to the truth: wrist cap dE
  18.3 -> 5.7.

## Known gaps

1. **Enamel, shadowed side: still ~2.2x too dark in R** (front mug red pixels: real
   0.040-0.056 camera-linear).
   - This is what is left after RENDER FIT. It fits LOOK's uniform ambient
     under-lighting vertical faces: a cos^4(elevation) sky gives them 1.9x more at the
     same horizontal irradiance.
   - The front arm's shaded PLA faces rendering greyer is probably the same cause.
   - The enamel's R x 2.1 lives in `blender/materials.py`, not in `look.json`, so Isaac
     RTX does not get it.
2. **Steel in the wrist view is too bright and too flat** (wrist mug 15.4 dB / dE 7.8).
   The room term treats every upward mirror ray as seeing the open room. From the
   wrist, the floor's mirror rays see the gripper and the dark camera body. The 0.5
   scale matches the front view's median, not the wrist's.
3. **The wrist background in carry / over-mug / release frames is up to +0.07 linear too
   bright** (PSNR 15.8-19.5 there vs 20.7-27.6 at rest to grasp; kinematic EEVEE). Two
   causes:
   - The webcam's auto exposure, which darkens those frames, is not modelled: s is
     constant, and LOOK's per-frame s is not reliable there.
   - The room behind the mug is only 2.6 % observed; the rest is the uniform fill.
4. **Defocus is depth-dependent.** One Gaussian (front 0.8 px, wrist 2.5 px)
   under-blurs the fingers 2-8 cm from the wrist lens. They still improve at 4.5 px.
5. **Not modelled:** the servo cables, the gold base part, the webcam LED, and the room
   reflections in the front glare (LOOK: sharp dark shapes in episodes 1 and 3).
6. **Camera response.** The tone curve is assumed to be sRGB and ρ_PLA to be 0.80
   (LOOK). The wrist colour uses one saturation factor plus the cap's per-camera albedo;
   other saturated surfaces keep σ 1.15.
7. **EEVEE vs Cycles.**
   - EEVEE has no glossy multi-bounce, and its reflections of other objects are screen
     space: a cap in the steel appears only while on screen. Cycles traces them (front
     mug dE 6.8 EEVEE vs 4.0 Cycles).
   - Otherwise the two engines agree to within ±0.8 dB per region.
8. **Kinematic mode** leaves the caps static at reset, so every post-grasp cap region is
   wrong by construction. Use a pose log for anything involving objects.
9. **The ep0 physics replay shows MuJoCo's nominal outcome, which fails.**
   - The cap stays in the jaws: its distance from the gripper link is 105.8-106.2 mm
     from frame 230 to the end.
   - It rides back to the rest pose and ends on the table 205 mm from the mug, whereas
     the real cap drops into the mug at ~frame 370.
   - MuJoCo's own `eval.json` agrees: ep0 action nominal success False, ensemble 0.05.
   - When MUJOCO publishes a new log, re-render with the same command (~6 min).
10. **Not a held-out test.** The fixes were chosen by looking at all 24 samples, which are
    also the frames LOOK derived its assets from. What is split:
    - The blur is fitted on ep0 and checked on ep1-2.
    - The enamel factor is fitted on ep0's wrist samples (ep1-2 give 1.8-2.2).
    - The steel scale comes from rest/approach samples of all three episodes.
