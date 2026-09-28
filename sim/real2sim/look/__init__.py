"""real2sim LOOK, part 1: what a renderer needs to look like the real dataset, from the dataset.

Engine-agnostic assets. Blender (LOOK part 2) and Isaac RTX (ISAAC part 2) both read
them from one index, sim/outputs/real2sim/<ds>/look/look.json. Everything is derived
from the recorded episodes: the front and wrist frames plus the recorded joints. The
arm is placed by FK of observation.state and the cameras come from the merged scene,
so CALIB's <ds>.calib.json feeds straight in. `./robot real2sim look build` regenerates
everything in 2.5-5 minutes, depending on machine load (a budget of under 10). Re-run
it whenever `./robot real2sim look info` reports STALE. look.json records the hash of
the scene groups it read (cameras, table, robot, objects, episodes).

WHAT IS DERIVED, AND HOW (numbers in look.json; method details in each module):

    plates      front background plates. Per episode: the per-pixel median over
                frames, with the arm (FK silhouette, dilated) and the caps (teal)
                masked. Across episodes: a CLEAN plate, where each pixel skips the
                episodes whose mug or mug shadow covers it. The C270's
                auto-white-balance drifts, so every episode and frame gets
                per-channel gains against the clean plate.            plates.py
    lighting    the key light's DIRECTION and angular size from the four mug
                shadows (a mug in a new place each episode). They fit a distant
                disk source better than any lamp nearer than a few metres. The
                same fit gives the ambient fraction from the umbra depth. The mat's
                gloss (GGX roughness) comes from the front glare, given that
                light.                                                 lighting.py
    table       the table plane at scene.table_z(), textured in the base frame.
                The front clean plate is rectified at 1 mm/texel. Past the front
                view the texture is a mosaic of wrist frames at 2 mm (FK + wrist
                camera, through the wrist->front colour model). Outputs are the
                observed radiance, a de-lit ALBEDO texture, the front view's
                specular field, and the source map.                       table.py
    backdrop    the far field the wrist camera sees at carry height (desk,
                clutter, pole). The proxy is a vertical cylinder around the
                workspace standing on the table plane. The build projects wrist
                frames onto it, and also writes an equirectangular environment
                map for reflections (the steel mug).                   backdrop.py
    materials   measured colours of cap, mug enamel, steel, mat, white PLA, black
                servo, desk and fluorescent marks, turned into linear albedo. It
                assumes an sRGB camera response, referenced to white PLA (albedo
                0.80, assumed). The wrist camera gets its own model relative to the
                front: per-frame white balance and exposure plus a saturation
                factor. The cap, seen by both, is the cross-check. materials.py
    samples     24 timesteps x 2 cameras across episodes 0-2 (rest, approach,
                descend, grasp, lift, carry, over-mug, release). Each has the
                real image and a label map, which every renderer scores against.
                                                                       samples.py
    metrics     masked PSNR / SSIM (real2sim.metrics) plus per-region colour
                error in linear RGB and CIE Lab. LPIPS is used only if `lpips`
                imports; it is not installed on this box.        imagemetrics.py

LABELS (samples/*_labels.png, uint8; the same codes in every camera):
    0 background   mat, table, desk and the far field: everything not below
    1 arm          the robot except the two fingers: printed parts, servos, and the
                   klip mount with its webcam body
    2 fingers      wrist_roll_follower_so101_v1 (fixed finger) + moving_jaw_so101_v1
    3 cap          teal cap pixels OUTSIDE the mug. Teal inside the mug opening,
                   meaning caps already dropped or their reflections in the polished
                   steel, counts as mug: only a reflection-capable renderer can
                   reproduce it.
    4 mug          enamel wall, handle, rim, steel interior and everything seen in it
Metrics erode every region by `erode_px` (default 2) before averaging, so a
sub-pixel calibration error at a boundary does not dominate a region's score.

COLOUR CONVENTION. Images on disk are 8-bit sRGB exactly as the cameras delivered
them (decoded rgb24). Values in look.json and in the *.npy/.hdr assets are LINEAR.
They are named `_lin` (camera-linear: the sRGB decode of camera values) or
`albedo` (scene-linear: divided by the camera's white balance and exposure).

FRAMES. As everywhere in real2sim: the URDF base frame, metres. Texture layouts:
    table     row r, col c centre = (x0 + (c + .5) res, y1 - (r + .5) res). Image up
              = base +y (behind the robot), right = base +x. UV (0, 0) = (x0, y0),
              i.e. OpenGL/Blender UVs on a quad spanning [x0, x1] x [y0, y1] at z =
              table_z.
    cylinder  col = azimuth phi = atan2(y - cy, x - cx) from -pi (left) to +pi, row
              0 = top (z = table_z + height), last row = table_z.
    equirect  col = azimuth phi as above (0 = +x), row 0 = straight up (+z).
              Centre and radius are in look.json.

Interpreter: host python3 (cv2, scipy, mujoco 3.6 EGL). No module here imports
Isaac or Blender. The heavy imports (cv2, mujoco, scipy) happen inside functions,
as in the core.
"""

LOOK_VERSION = 1  # bump when an output's meaning changes; look.json records it

LABELS = {"background": 0, "arm": 1, "fingers": 2, "cap": 3, "mug": 4}
LABEL_NAMES = {v: k for k, v in LABELS.items()}
# Label-map preview colours (RGB), for contact sheets only.
LABEL_RGB = {0: (0, 0, 0), 1: (220, 60, 220), 2: (60, 220, 60), 3: (40, 200, 230), 4: (230, 60, 40)}
