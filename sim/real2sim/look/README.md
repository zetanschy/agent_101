# real2sim LOOK, part 1: look assets from the dataset

This track gives a renderer everything it needs to look like the real recording. It is
engine-agnostic, and every number is derived from this dataset's frames and joints. The
two consumers are Blender (LOOK part 2) and Isaac RTX (ISAAC part 2). Both read one index,
`sim/outputs/real2sim/<ds>/look/look.json`. Each number there carries a `source`.

```
./robot real2sim look build            # everything: 139 s idle, up to 280 s under load (host python3 + MuJoCo EGL, no GPU lock)
./robot real2sim look info             # summary; says STALE when cameras/table/robot/objects changed
./robot real2sim look score <dir> [--wb] [--pattern '{id}_{cam}.png'] [--erode 2] [--out f.json]
./robot real2sim look test             # 15 tests, ~6 s
```

The cameras, the table height, the joint offsets and the object poses all come from the
merged scene (`scene.load`). So when CALIB writes `<ds>.calib.json`, run `build` again.
`info` compares look.json's `inputs_hash` (the scene's cameras, table, robot, objects and
episodes) against the current scene. A new servo layer does not trigger a rebuild.

## What is derived, and how

| asset | method | module |
|---|---|---|
| front background plates | Per episode: the per-pixel median of every 3rd frame, with the FK arm (dilated 12 px) and the teal caps masked. Across episodes: a clean plate, where each pixel skips the episodes whose mug or mug shadow covers it. | `plates.py` |
| front white balance | The C270's auto white balance drifts. Per channel, each frame and episode gets a gain against the clean plate, as a ratio of linear sums over background pixels. | `plates.py` |
| key light, ambient | The four mug shadows (episode plate / clean plate), fitted with a distant disk source through a penumbra-wedge model (az, el, angular radius, per-episode umbra depth). | `lighting.py` |
| mat gloss | A GGX lobe for the front glare, given the fitted source. alpha comes from a 1-D search; diffuse and specular come from least squares. | `lighting.py` |
| white reference | The FK printed parts that are white in the image and face up (world normal from a depth render), in the rest frames: `L_PLA`, with rho_PLA 0.80 assumed. | `materials.py` |
| wrist colour model | White balance from the static finger core in each frame. One saturation factor sigma is fitted on the cap, the enamel and the fluorescent marks. Exposure comes from mat luminance against the front texture. | `colour.py`, `materials.py`, `table.py` |
| table texture | The front clean plate rectified onto z = table_z at 1 mm. It is extended by a mosaic of 530 wrist frames at 2 mm (FK + grip camera, colour model, robust mean), then de-lit into albedo. | `table.py` |
| far field | A proxy cylinder (R 0.80 m, 0.60 m tall) textured from the wrist frames, and an equirect environment map whose unseen directions carry the ambient share. | `backdrop.py` |
| materials | The measured colours of cap, enamel, steel, mat, PLA, servo, mount, desk and fluorescent marks, turned into linear albedo (PLA-referenced, irradiance-corrected). | `materials.py` |
| sample set | 24 timesteps x 2 cameras (8 phases x episodes 0-2), each with the real image and a label map. | `samples.py`, `segment.py` |
| image metrics | Masked PSNR/SSIM plus per-region colour error. LPIPS is used only if `lpips` imports, and it is not installed. | `imagemetrics.py` |

Labels (`samples/*_labels.png`): 0 background, 1 arm (including the klip mount and webcam
body), 2 fingers, 3 cap, 4 mug. Teal inside the mug opening counts as mug: those are
dropped caps and their reflections. Metrics erode each region by 2 px.

- **Front labels:** the FK silhouette (fit6r), re-decided in a 2-6 px band by "brighter than
  the episode plate". Shadows stay background.
- **Wrist fingers:** from the data. Each pixel's white-PLA frequency is taken over the 691
  dark-background wrist frames at the nearest openings.
- **Wrist mug:** from the image alone. Mug-coloured components are anchored on the red
  enamel.
- **Calibration diagnostics:** in each sample's `qa`, the IoU of the model fingers and mug
  against those image masks.

## Frames, units, layouts

- **Frame:** base frame, metres.
- **Textures:** `table/*`, row r and col c map to
  (x0 + (c + .5) res, y1 - (r + .5) res). Image up is base +y and image right is base +x.
  UV (0, 0) = (x0, y0) on a quad at z = table_z. Extent x -0.90..0.70, y -1.10..0.25.
- **Cylinder and equirect:** column = azimuth atan2(y - cy, x - cx) from -pi to +pi. Row 0
  is the top (the equirect row 0 is +z).
- **Image files:** 8-bit PNGs are sRGB as delivered.
- **Albedo:** `table/albedo.png` is 16-bit, sRGB-encoded.
- **Linear data:** `*_lin.npy` (float16) and `backdrop/env.hdr` (float32) are linear.
- **Scene units:** the total horizontal irradiance at the table is 1, so a horizontal
  Lambertian face of albedo rho has radiance rho / pi.
- **Front camera:** camera_lin = `exposure_scene_to_camera_lin` x radiance x the frame's
  white-balance gain.
- **Wrist camera:** `colour.front_to_grip(front_lin, wb, exposure, sigma, L_PLA)`. The
  per-sample values are in `samples/index.json`, `grip.colour_model`.
- **How to light a render:**
  - A distant disk source at `lights.key` (direction, angular radius; normal irradiance
    `normal_irradiance_scene_units`).
  - `backdrop/env.hdr` as the world. Its uniform fill carries the ambient; the key is not
    in the map.
  - Check that a horizontal white plane (rho 1) renders at 1/pi. Some engines' sun
    conventions include pi.

## Measured results (build of 2026-09-27, provisional scene; `look.json` has all)

**Plates**
- Valid pixels: 95-96 % per episode; the clean plate covers 96.1 %. The robot base is never
  visible.
- White-balance gains:
  - Per-episode R gain: 0.96-1.08.
  - Within episodes 1-2, the per-frame R gain has a std of 0.07-0.11 and G 0.02-0.03.
  - Episode 0 is stable (std 0.015).

**Key light**
- Direction: az -122.0 deg, el 43.7 deg. Angular radius 15.2 deg (a large soft source).
- Ambient fraction: 0.33 / 0.48 / 0.43 / 0.49 (median 0.46).
- Shadow ratio rms 0.075; shadow IoU at 0.8 is 0.62-0.69.
- A finite-distance lamp fits worse the nearer it is: rms 0.095 at 0.8 m vs 0.0745 at 9 m
  (build-phase probe, mug height free).
- Freeing the mug height sends it to 84 mm (el 39.2 deg): shadow length and elevation are
  degenerate.

**Mat gloss**
- GGX alpha 0.027 (ep0) / 0.021 (ep2), roughness 0.16 / 0.15.
- It explains the glare: rms 0.0089 vs 0.0256 for a flat mat. The top 3 % of the glare is
  predicted at 0.125 against an observed 0.127.

**White reference**
- L_PLA = (0.606, 0.691, 0.603) linear. Luminance across the 30 rest frames is p5/p50/p95
  0.635 / 0.659 / 0.699.

**Albedo** (linear, PLA-referenced)

| material | albedo |
|---|---|
| cap top | (0.172, 0.409, 0.484) |
| mat | (0.0060, 0.0068, 0.0086) |
| servo | (0.013, 0.015, 0.018) |
| desk | (0.18, 0.17, 0.17) |
| enamel (irradiance-corrected, 203 lit px) | (0.108, 0.016, 0.016) |

- The mat's PLA-free estimate from the glare (F0 0.04) is (0.0028, 0.0025, 0.0022), a
  factor ~2.6 lower.

**Wrist vs front**
- sigma = 1.14. Surfaces fitted alone give 1.59 (cap), 1.10 (enamel) and 0.93
  (fluorescent), so one saturation factor is only a first-order model.
- The wrist camera's exposure varies p5/p50/p95 0.44 / 1.36 / 2.06 against the front.
- Cap top seen by the wrist camera through the full model: (0.068, 0.378, 0.413).
  Luminance is 13 % below the front's; red is 2.5x lower.

**Coverage**
- Table grid: front 17.2 %, wrist only 22.8 %, unobserved 60.0 % (filled with the mat
  median).
- Cylinder wall: 5.5 %, from 42 frames.
- Equirect: 0.8 % of the upper hemisphere observed. The fill radiance is 0.146.

**Samples**
- Front robot, model vs refined: IoU 0.86 (median).
- Wrist fingers, model vs data: IoU 0.37 (0.25-0.41). This is the provisional wrist
  calibration: ChArUco fx 680 with a mount pose fitted at fx 487, so the model draws the
  jaws about 1.4x too large (estimated from that ratio and the overlays).

**Baseline**
- A perfect static background (the arm-free episode plate at the frame's white balance)
  scores 29.1 dB PSNR / 0.947 SSIM / dE 0.92 in the front background region. That is the
  ceiling for a background.
- The same render scores 4.7 dB on the arm region: that is what a missing arm costs.

## Limits (where the approximations break)

- **Camera response:** the tone curve is assumed to be sRGB (no bracketed target exists).
  rho_PLA 0.80 is assumed. Absolute albedos scale with both assumptions; ratios do not.
- **Wrist-derived assets need CALIB:** the mosaic, the cylinder, the wrist mug model QA and
  the wrist exposure use the scene's grip camera. That camera is still the provisional
  ChArUco K plus mjlab's mount pose, which is inconsistent by the dolly-zoom (fingers IoU
  0.37). They re-derive on `build`.
- **Proxies are flat:** the desk clutter (round object, pole) is smeared along the plane
  inside R and flattened onto the wall beyond. The textures are exact only from where the
  wrist camera was.
- **Front glare:** sharp dark shapes in the glare (episodes 1 and 3) are room reflections,
  and no light model here explains them. 27.6 % of the front-covered texels have
  specular above 40 % of L. There the albedo is the plain-mat median times the texel's
  own fine detail.
- **Albedo gradient:** the plain mat's albedo still varies smoothly (G p5/p50/p95 0.004 /
  0.007 / 0.014). A second, broader gloss lobe does not explain it, so it stays in as
  baked diffuse light.
- **Mug enamel:** only 203 lit pixels in the front view. Its albedo is the least certain.

## Files

Code lives in `sim/real2sim/look/`:
- `__init__.py`: label codes and conventions.
- Modules: `colour.py`, `segment.py`, `plates.py`, `lighting.py`, `materials.py`,
  `table.py`, `backdrop.py`, `samples.py`, `imagemetrics.py`, `build.py`, plus `run.sh`
  and `tests/`.

Outputs go to `sim/outputs/real2sim/<ds>/look/`:
- `look.json`
- `plates/`: per-episode and clean plates, `plates.npz` with the gains, masks and shadow
  ratios.
- `table/`: radiance, albedo, specular, source, `wrist_model.json`.
- `backdrop/`: `cylinder_lin.npy`, `env.hdr` plus previews and coverage masks.
- `samples/`: 24 x 2 images and label maps, `index.json`.
- `previews/`: contact sheets. The total is about 66 MB.
