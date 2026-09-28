"""Physics honesty of the object contacts: resting, dropping into the mug, squeezing."""
import numpy as np
import pytest

from real2sim.mujoco import model as mdl


def _step(m, d, seconds):
    import mujoco

    for _ in range(int(round(seconds / m.opt.timestep))):
        mujoco.mj_step(m, d)


def test_cap_rests_without_drift_or_energy(scene, servo):
    """A cap on the table for 2 s: no creep, no jitter (energy never grows)."""
    import mujoco

    b = mdl.build(scene, 0, servo, cameras=())
    m = b.model
    d = mujoco.MjData(m)
    mdl.set_arm_state(b, d, scene.units().to_urdf(np.array([0, -100, 95, 75, 0, 1.5])))  # parked, clear of the cap
    mujoco.mj_forward(m, d)
    cap = b.obj("cap_A")
    p0 = d.qpos[cap["qadr"]:cap["qadr"] + 3].copy()
    _step(m, d, 0.2)
    p1 = d.qpos[cap["qadr"]:cap["qadr"] + 3].copy()
    v = []
    for _ in range(18):
        _step(m, d, 0.1)
        v.append(np.linalg.norm(d.qvel[cap["dadr"]:cap["dadr"] + 6]))
    p2 = d.qpos[cap["qadr"]:cap["qadr"] + 3]
    assert np.linalg.norm(p1 - p0) < 5e-4  # settles within 0.5 mm of where it was placed
    assert np.linalg.norm(p2 - p1) < 1e-4  # then does not move (0.1 mm in 1.8 s)
    assert max(v) < 1e-3


def test_cap_dropped_into_the_mug_stays_in(scene, servo):
    """Dropped from 15 cm above the rim: lands inside (no tunnelling through the 5 mm floor)."""
    import mujoco

    from real2sim import metrics

    b = mdl.build(scene, 0, servo, cameras=())
    m = b.model
    d = mujoco.MjData(m)
    mdl.set_arm_state(b, d, scene.units().to_urdf(np.array([0, -100, 95, 75, 0, 1.5])))
    cap, mug = b.obj("cap_A"), b.obj("mug")
    mp = d.qpos[mug["qadr"]:mug["qadr"] + 3].copy()
    d.qpos[cap["qadr"]:cap["qadr"] + 3] = mp + [0.01, -0.005, scene.mug_dims()["height"] + 0.15]
    mujoco.mj_forward(m, d)
    pen, z, hit = 0.0, [], None
    for i in range(int(1.5 / m.opt.timestep)):
        mujoco.mj_step(m, d)
        z.append(d.qpos[cap["qadr"] + 2])
        n = d.ncon
        if n:
            k = np.isin(d.contact.geom1[:n], cap["geoms"]) | np.isin(d.contact.geom2[:n], cap["geoms"])
            if k.any():
                pen = max(pen, -d.contact.dist[:n][k].min())
                hit = i if hit is None else hit
    z = np.array(z)
    drop = z[0] - z[hit]
    bounce = z[hit:].max() - z[hit]
    assert bounce < 0.05 * drop  # contacts dissipate: no bounce energy from the solver
    cq = d.qpos[cap["qadr"]:cap["qadr"] + 7]
    mq = d.qpos[mug["qadr"]:mug["qadr"] + 7]
    assert metrics.cap_in_mug(cq[:3], cq[3:], mq[:3], mq[3:], scene.cap_dims(), scene.mug_dims())
    assert pen < 1e-3  # m: even the impact stays under the 1 mm target
    assert np.linalg.norm(mq[:2] - mp[:2]) < 1e-3  # the 300 g mug is not knocked about by a 2.5 g cap


def test_squeeze_penetration_and_force_limit(scene, servo):
    """The jaw closes on a cap placed between the fingers (arm held by its servos at a
    recorded grasp pose): penetration stays under 1 mm and the squeeze is bounded by
    the gripper's torque clamp."""
    import mujoco

    from real2sim import episodes, kinematics

    ep = episodes.load()[0]
    u = scene.units()
    q = u.to_urdf(ep.state[231])
    tips = kinematics.default().fingertips(q)
    table_dz = tips["fixed"][2] - 0.003 - scene.table_z()  # fingertips 3 mm above the table
    pert = mdl.Perturbation(table_dz=float(table_dz), cap_dxy={"A": tuple(tips["mid"][:2] - np.array(scene.episode(0)["caps"][0]["xy"]))})
    b = mdl.build(scene, 0, servo, perturb=pert, cameras=())
    m = b.model
    d = mujoco.MjData(m)
    q_open = q.copy()
    q_open[5] = u.jaw_rad(25.0)
    mdl.set_arm_state(b, d, q_open)
    d.ctrl[b.act] = q_open
    mujoco.mj_forward(m, d)
    _step(m, d, 0.3)
    d.ctrl[b.act[5]] = u.jaw_rad(10.9)  # ep0's commanded squeeze
    pen, fmax = 0.0, 0.0
    cap = b.obj("cap_A")
    cg = set(cap["geoms"])
    fingers = set(b.finger_geoms["fixed"]) | set(b.finger_geoms["moving"])
    for _ in range(int(1.0 / m.opt.timestep)):
        mujoco.mj_step(m, d)
        for i in range(d.ncon):
            c = d.contact[i]
            if {c.geom1, c.geom2} & cg and {c.geom1, c.geom2} & fingers:
                pen = max(pen, -c.dist)
                f = np.zeros(6)
                mujoco.mj_contactForce(m, d, i, f)
                fmax = max(fmax, f[0])
    assert pen < 1e-3
    jaw = servo.joint("Jaw")
    # the tip force can exceed tau/r only transiently; bound it by the clamp over the
    # shortest lever (60 mm) with a 2x margin for impact
    assert fmax < 2 * jaw.effort / 0.060
    blocked = np.degrees(d.qpos[b.qadr[5]])
    assert blocked > np.degrees(u.jaw_rad(10.9)) + 2  # the cap blocks the jaw well short of the goal
