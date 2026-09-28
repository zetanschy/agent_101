"""What PhysX says about the contacts that matter: every robot link against every
object and the table, and every cap against the table and the mug.

Uses the PhysX tensor API's RigidContactView (the one Isaac Lab's ContactSensor wraps)
because only get_contact_data() exposes per-contact SEPARATIONS, PhysX's own signed
distance at each contact point (negative = penetration). The views need the
ContactReportAPI on their sensor bodies (activate_contact_sensors=True on the robot and
the objects, scene.py). Filter pairing is per env: sensor i is paired with the matches
of every filter pattern in its own env (Isaac Lab's ContactSensor relies on the same).

One view per sensor body, every one read on every physics step:
    robot    one per link (7), E sensors each;  filters: the caps, the mug, the table
    objects  one per cap, E sensors each;       filters: the table, the mug
(Contacts says why a pattern may match only one body per env.)
Accumulation stays on the GPU: each step folds its minimum separation, its forces and
a touch flag into per-frame buffers with scatter-reduce, and the host reads them once
per frame (one sync), so a physics step costs a few kernel launches, not a sync.

The pose-log extras the MuJoCo track's evaluate.episode_metrics reads come from here
(x_touch, x_fc_force, x_pen_fc, x_pen_obj, x_arm_obj; see Contacts.frame_extras), so
both engines' eval.json share their definitions exactly.
"""

from __future__ import annotations

import numpy as np
import torch

SEP_NONE = 1.0  # m, "no contact" in the separation buffers
TOUCH_N = 1e-3  # N: a pair with more force than this during a frame is touching
FINGERS = ("gripper", "jaw")  # the links carrying the fixed and the moving finger
ARM_NON_FINGER = ("base", "shoulder", "upper_arm", "lower_arm", "wrist")
ROBOT_LINKS = ARM_NON_FINGER + FINGERS  # the 7 rigid bodies (kinematics.LINKS)


class PairView:
    """sensor bodies x filter patterns, all envs: per-frame min separation, max and mean
    |force|, and the values of the frame's last step."""

    def __init__(self, psv, sensor: str, filters: list[str], device, max_contacts: int = 8192):
        self.view = psv.create_rigid_contact_view(sensor, filter_patterns=filters, max_contact_data_count=max_contacts)
        self.S, self.F = self.view.sensor_count, self.view.filter_count
        if self.F != len(filters):
            raise RuntimeError(f"contact view {sensor}: {self.F} filters, expected {len(filters)}")
        self.paths = list(self.view.sensor_paths)
        self.device = device
        self._idx = torch.arange(max_contacts, device=device)
        self.new_frame()

    def new_frame(self) -> None:
        z = lambda v: torch.full((self.S, self.F), v, device=self.device)  # noqa: E731
        self.min_sep, self.last_sep = z(SEP_NONE), z(SEP_NONE)
        self.max_force, self.sum_force, self.last_force = z(0.0), z(0.0), z(0.0)
        self.steps = 0

    def accumulate(self, dt: float) -> None:
        f = self.view.get_contact_force_matrix(dt).reshape(self.S, self.F, 3).norm(dim=-1)
        self.last_force = f
        torch.maximum(self.max_force, f, out=self.max_force)
        self.sum_force += f
        self.steps += 1
        _, _, _, sep, cnt, start = self.view.get_contact_data(dt)
        sep = sep.reshape(-1)
        cnt, start = cnt.reshape(-1).to(torch.long), start.reshape(-1).to(torch.long)
        # owner pair of every contact slot, without assuming how PhysX packs the pairs:
        # slot e belongs to pair p when start[p] <= e < start[p] + cnt[p]
        e = self._idx[:, None]
        inside = (e >= start[None, :]) & (e < (start + cnt)[None, :])
        valid = inside.any(1)
        owner = inside.float().argmax(1)
        vals = torch.where(valid, sep, torch.full_like(sep, SEP_NONE))
        step_min = torch.full((self.S * self.F,), SEP_NONE, device=self.device)
        step_min.scatter_reduce_(0, owner, vals, reduce="amin")
        self.last_sep = step_min.reshape(self.S, self.F)
        torch.minimum(self.min_sep, self.last_sep, out=self.min_sep)

    def numpy(self) -> dict:
        n = max(self.steps, 1)
        return {"min_sep": self.min_sep.cpu().numpy(), "last_sep": self.last_sep.cpu().numpy(),
                "max_force": self.max_force.cpu().numpy(), "mean_force": (self.sum_force / n).cpu().numpy(),
                "last_force": self.last_force.cpu().numpy()}


def _env_of(path: str) -> int:
    return int(path.split("/env_")[1].split("/")[0])


class Contacts:
    """The replay's contact views (module doc), built after sim.reset().

    ONE VIEW PER SENSOR BODY. PhysX pairs sensor i with the i-th match of every filter
    pattern (or with a pattern's only match), so each sensor pattern must match exactly
    one body per env, as each filter does. A single Robot/* view matches 10 prims per env
    (the 7 links, joints, Looks, root_joint): fine with one env, and refused with 8
    ("filter pattern '.../cap_A' did not match the correct number of entries (expected
    80, found 8)", MEASURED on the first 8-env run). The same holds for cap_* with
    several caps. So: 7 link views and one view per cap, each with E sensors."""

    def __init__(self, sim, num_envs: int, cap_names: list[str], has_mug: bool, device, links=ROBOT_LINKS):
        psv = sim.physics_sim_view
        env = "/World/envs/env_*"
        self.E, self.caps, self.has_mug = num_envs, list(cap_names), has_mug
        table = f"{env}/Table/geometry/mesh"
        self.obj_filters = self.caps + (["mug"] if has_mug else [])
        self.links = sorted(links)  # the pose log's contact_links order (alphabetical, as before)
        filt = [f"{env}/{o}" for o in self.obj_filters] + [table]
        self.robot = {l: PairView(psv, f"{env}/Robot/{l}", filt, device) for l in self.links}
        self.objects = {c: PairView(psv, f"{env}/{c}", [table] + ([f"{env}/mug"] if has_mug else []), device)
                        for c in self.caps}
        # sensor s of a view -> its env; PhysX orders the sensors itself, so map by path
        self._env = {v: [_env_of(p) for p in v.paths] for v in [*self.robot.values(), *self.objects.values()]}
        for v, es in self._env.items():
            if sorted(es) != list(range(num_envs)):
                raise RuntimeError(f"contact view {v.paths[:2]}...: sensors in envs {es}, expected one per env")

    def _views(self):
        return [*self.robot.values(), *self.objects.values()]

    def new_frame(self) -> None:
        for v in self._views():
            v.new_frame()

    def accumulate(self, dt: float) -> None:
        for v in self._views():
            v.accumulate(dt)

    def _stack(self, views: dict, names: list[str]) -> dict:
        """{q: (E, len(names), F)} from one (E, F) view per name."""
        out = {}
        for i, n in enumerate(names):
            v = views[n]
            for q, a in v.numpy().items():
                buf = out.setdefault(q, np.full((self.E, len(names)) + a.shape[1:], np.nan))
                buf[self._env[v], i] = a
        return out

    def frame(self) -> dict:
        """Per-frame arrays, (E, ...) numpy:
            robot_<q>   (E, L, F) q in min_sep / last_sep / max_force / mean_force, links in
                        self.links order, filters = caps, mug, table
            objects_<q> (E, C, 2) the caps against (table, mug)"""
        out = {f"robot_{q}": v for q, v in self._stack(self.robot, self.links).items()}
        if self.caps:
            out.update({f"objects_{q}": v for q, v in self._stack(self.objects, self.caps).items()})
        return out

    def frame_extras(self, fr: dict) -> dict:
        """The MuJoCo track's pose-log extras from one frame() (module doc), (E, ...):
            x_touch    (E, C, 2) bool  cap c touched by the fixed / moving finger this frame
            x_fc_force (E, C, 2)       mean normal force of the fixed / moving finger on cap c (N)
            x_pen_fc   (E,)            deepest finger-cap penetration over the frame's steps (m)
            x_pen_obj  (E,)            deepest penetration of any contact involving an object (m)
            x_arm_obj  (E,)            largest force of a non-finger link on an object (N); the
                                       fixed finger shares the gripper link with the mount and
                                       the webcam, whose contacts count as the finger's here"""
        C = len(self.caps)
        li = {l: i for i, l in enumerate(self.links)}
        fing = [li[f] for f in FINGERS]
        ms, mx, mean = fr["robot_min_sep"], fr["robot_max_force"], fr["robot_mean_force"]
        touch = (mx[:, fing, :C] > TOUCH_N) | (ms[:, fing, :C] < 0.0)
        pen = lambda s: (-s).clip(min=0.0) * (s < SEP_NONE)  # noqa: E731
        ex = {"x_touch": touch.transpose(0, 2, 1), "x_fc_force": mean[:, fing, :C].transpose(0, 2, 1),
              "x_pen_fc": pen(ms[:, fing, :C]).reshape(self.E, -1).max(1, initial=0.0)}
        nobj = len(self.obj_filters)
        p_obj = pen(ms[:, :, :nobj]).reshape(self.E, -1).max(1, initial=0.0)
        if "objects_min_sep" in fr:
            p_obj = np.maximum(p_obj, pen(fr["objects_min_sep"]).reshape(self.E, -1).max(1, initial=0.0))
        ex["x_pen_obj"] = p_obj
        arm = [li[l] for l in ARM_NON_FINGER if l in li]
        ex["x_arm_obj"] = mx[:, arm, :nobj].reshape(self.E, -1).max(1, initial=0.0)
        return ex

