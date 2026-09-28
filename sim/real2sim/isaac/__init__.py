"""real2sim's Isaac track: the real episodes replayed in Isaac Sim 4.5 / Isaac Lab 2.1.

Modules that need Kit (import them only after AppLauncher has started the app):
    scene       the Isaac Lab scene of one episode, built in one place, with Look hooks
    usd_assets  cap / mug USD files: visual mesh, convex collision pieces, material
    contact     PhysX contact views: finger-cap forces and separations
    render      RTX pinhole frames -> the real distorted cameras -> mp4
    replay      the entry point: ./robot real2sim isaac replay ...
Pure numpy / torch, importable anywhere:
    params      Isaac-only physics numbers with provenance
    servo       the STS3215 servo model re-expressed for PhysX
    geometry    convex pieces of the cap and mug, exact containment depths, mass sums
    placement   object placement estimates (config, or antipodal grasp from FK)
    evaluate    eval.json from a replay's pose logs (host python3)
    penetration offline finger/cap/table/mug overlap from pose logs + meshes

See README.md in this directory for the commands and the measured numbers.
"""
