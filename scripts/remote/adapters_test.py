#!/usr/bin/env python3
"""Tests for the remote-inference policy seams.

    ./robot run python scripts/remote/adapters_test.py   # no GPU, no arm, no weights

Everything here guards a way the seam can be wrong WITHOUT ANYTHING LOOKING WRONG.
A patch that silently stops applying, an allow-list widened for a policy the factory
cannot build, an image pipeline that runs twice -- each still produces a server that
starts, connects and serves actions, and shows up only as a policy that performs
worse over the wire than it did on the robot.
"""

from __future__ import annotations

import pathlib
import sys
from dataclasses import dataclass

import numpy as np
import torch

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import adapters  # noqa: E402

# At import, as serve.py does at startup: the tests below run in name order and must
# not depend on which of them happened to install the seams first.
adapters.install()

OBS_STATE = "observation.state"
OBS_IMG = "observation.images"


def _obs(**images) -> dict:
    batch = {OBS_STATE: torch.zeros(1, 6), "task": "pick up the lemon"}
    for name, arr in images.items():
        batch[f"{OBS_IMG}.{name}"] = arr
    return batch


# --------------------------------------------------------------------------- #
# The allow-list
# --------------------------------------------------------------------------- #


def test_molmoact2_is_servable():
    """THE POINT OF THIS MODULE. lerobot implements MolmoAct2 in full, and the async
    server's SUPPORTED_POLICIES is a hand-maintained list that omits it -- so an
    unpatched server refuses a policy it is perfectly able to run."""
    from lerobot.async_inference import constants

    assert "molmoact2" in constants.SUPPORTED_POLICIES


def test_the_server_module_sees_the_patch_not_just_the_constants_module():
    """policy_server did `from .constants import SUPPORTED_POLICIES`, binding the
    list into its own namespace at import. Patching the constants module alone
    leaves the server checking the original and refusing everything added."""
    from lerobot.async_inference import policy_server as ps

    assert "molmoact2" in ps.SUPPORTED_POLICIES


def test_a_native_policy_keeps_its_own_processors():
    """MolmoAct2 is a first-class lerobot policy with real normalization. The
    pass-through pipeline is only for models that preprocess themselves; applied
    here it would silently feed the model unnormalized inputs."""
    from lerobot.policies.molmoact2.configuration_molmoact2 import MolmoAct2Config
    from lerobot.policies import factory

    for config_cls, _ in adapters.REGISTRY.values():
        assert not isinstance(MolmoAct2Config(), config_cls), \
            "a native policy must not be claimed by the adapter registry"
    assert factory.get_policy_class("molmoact2").__name__ == "MolmoAct2Policy"


def test_only_policies_lerobot_can_build_are_allow_listed():
    """Allow-listing a name the factory cannot resolve turns a clear 'not supported'
    at connect time into a stack trace at load time, on the robot, mid-session."""
    from lerobot.async_inference import constants
    from lerobot.policies import factory

    for name in constants.SUPPORTED_POLICIES:
        if name in adapters.REGISTRY:
            continue
        factory.get_policy_class(name)      # raises if the allow-list is lying


def test_install_is_idempotent():
    """serve.py calls it at import and a test may call it again; a second pass must
    not stack another wrapper or duplicate the allow-list."""
    from lerobot.async_inference import constants

    before = list(constants.SUPPORTED_POLICIES)
    adapters.install()
    adapters.install()
    assert constants.SUPPORTED_POLICIES == before


# --------------------------------------------------------------------------- #
# The adapter seam, for models lerobot does not implement
# --------------------------------------------------------------------------- #


@dataclass
class _FakeConfig(adapters.RemoteConfigBase):
    pass


class _FakePolicy(adapters.RemotePolicyBase):
    config_class = _FakeConfig
    name = "fake"

    def predict_action_chunk(self, batch, **kwargs):
        state, images, task = adapters.observation_to_numpy(batch)
        return torch.zeros(1, 4, state.shape[0])


def test_a_registered_adapter_reaches_all_three_seams():
    from lerobot.async_inference import constants
    from lerobot.policies import factory

    adapters.register("fake", _FakeConfig, _FakePolicy)
    adapters.install()
    assert "fake" in constants.SUPPORTED_POLICIES
    assert factory.get_policy_class("fake") is _FakePolicy
    pre, post = factory.make_pre_post_processors(_FakeConfig())
    probe = {"anything": object()}
    assert pre(probe) is probe and post(probe) is probe, \
        "an adapter must get a pass-through pipeline; it preprocesses itself"


def test_the_passthrough_does_not_swallow_a_foreign_config():
    """If the wrapper claimed every config, act and pi0 would lose their
    normalization and quietly degrade rather than fail."""
    from lerobot.policies import factory

    class NotOurs:
        pass

    reached = {}
    try:
        factory.make_pre_post_processors(NotOurs())
    except Exception:  # noqa: BLE001 - it must REACH lerobot rather than be claimed
        reached["lerobot"] = True
    assert reached.get("lerobot"), "a foreign config was swallowed by the pass-through"


# --------------------------------------------------------------------------- #
# Observation decoding
# --------------------------------------------------------------------------- #


def test_images_come_back_as_a_model_expects_them():
    """HWC uint8, because that is what a camera produced and what a model's own
    processor takes. A CHW float tensor handed to PIL raises; a [0,1] float that is
    not scaled back silently becomes a black image."""
    chw = torch.rand(1, 3, 8, 12)
    state, images, task = adapters.observation_to_numpy(_obs(front=chw))
    assert state.shape == (6,) and state.dtype == np.float32
    assert images[0].shape == (8, 12, 3) and images[0].dtype == np.uint8
    assert images[0].max() > 1, "a [0,1] float was not scaled back to 0..255"
    assert task == "pick up the lemon"


def test_uint8_images_pass_through_untouched():
    raw = np.full((4, 5, 3), 200, np.uint8)
    _, images, _ = adapters.observation_to_numpy(_obs(front=raw))
    assert images[0].dtype == np.uint8 and images[0].max() == 200


def test_camera_order_is_the_sorted_key_order_not_arrival_order():
    """A model that wants [scene, wrist] gets two arrays with nothing in them saying
    which is which. Hand them over swapped and it still returns a plausible chunk,
    pointed somewhere else. A dict off the wire has no inherent order, so the only
    stable one is by name -- and it must not depend on insertion."""
    a = np.full((2, 2, 3), 11, np.uint8)
    b = np.full((2, 2, 3), 22, np.uint8)
    batch = {OBS_STATE: torch.zeros(1, 6), f"{OBS_IMG}.zzz": b, f"{OBS_IMG}.aaa": a}
    _, images, _ = adapters.observation_to_numpy(batch)
    assert [int(i[0, 0, 0]) for i in images] == [11, 22]


def test_a_missing_state_says_what_arrived():
    try:
        adapters.observation_to_numpy({f"{OBS_IMG}.front": np.zeros((2, 2, 3), np.uint8)})
    except KeyError as e:
        assert "observation.state" in str(e)
    else:
        raise AssertionError("a missing state must not be inferred")


def test_the_chunk_shape_survives_the_server_s_truncation():
    """The server does chunk[:, :actions_per_chunk, :]. A (T, D) array would be
    sliced on the wrong axis and truncate the ACTION DIMENSION instead."""
    pol = _FakePolicy(_FakeConfig())
    out = pol.predict_action_chunk(_obs(front=np.zeros((2, 2, 3), np.uint8)))
    assert out.ndim == 3 and out.shape == (1, 4, 6)
    assert out[:, :2, :].shape == (1, 2, 6)


def main() -> int:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"ok  {t.__name__}")
    print(f"\n{len(tests)} passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
