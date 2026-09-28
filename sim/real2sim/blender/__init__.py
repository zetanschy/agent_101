"""real2sim LOOK, part 2: render pose logs and kinematic replays in Blender, like the real cameras.

Commands, the scene, the measured metrics per camera and engine, speed and known gaps:
README.md. Host side: job.py (what to render), camera_model.py (the webcams), post.py
(videos), cli.py / check.py (commands); Blender side: bl_render.py.
"""

JOB_SCHEMA = "real2sim.blender.job/1"
