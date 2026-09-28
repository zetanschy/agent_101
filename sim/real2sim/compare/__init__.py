"""Isaac Sim vs MuJoCo + Blender on the same real episodes, from what each track measured.

    ./robot real2sim compare summary              -> outputs/<ds>/compare/summary.{json,md}
    ./robot real2sim compare video --episode 0    -> outputs/<ds>/compare/ep0_triptych.mp4

Nothing is re-simulated or re-rendered here. `summary` reads the tracks' own outputs
(mujoco/eval*.json, isaac/ep*/eval.json, the renderers' sample scores and manifests) and
keeps the scene hash each number was measured on: a row measured on another scene than the
current one is marked STALE rather than mixed in silently. `video` puts the real frames next
to the MuJoCo+Blender and the Isaac RTX render of the same nominal replay.
"""
