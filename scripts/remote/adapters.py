#!/usr/bin/env python3
"""Dress an arbitrary model as a LeRobot policy, so the policy server can serve it.

    from adapters import register, install, observation_to_numpy

WHY THIS EXISTS. lerobot ships a gRPC pair -- `lerobot.async_inference.PolicyServer`
on the GPU box and `RobotClient` on the robot -- and it is the right transport for
"the model is on another machine": it already does action chunking, a client-side
action queue, re-querying before that queue drains, and aggregation across the seam
between chunks. None of that is worth reimplementing over a REST endpoint.

What it cannot do is serve a model lerobot has never heard of. The server resolves a
policy by name through three seams, and each one refuses an unknown name:

    lerobot.async_inference.constants.SUPPORTED_POLICIES   an allow-list
    lerobot.policies.factory.get_policy_class              name -> class
    lerobot.policies.factory.make_pre_post_processors      the I/O pipeline

`install()` widens all three. A model then needs only a small class pair -- a config
and a policy exposing `predict_action_chunk` -- and the whole server, transport and
client stack works unchanged.

THE PREPROCESSOR IS DELIBERATELY BYPASSED, and this is the subtle part. lerobot's
pipeline resizes images to the policy's declared input size and scales them to [0, 1]
before the policy sees them. A model that ships its own processor -- every VLA does;
it has its own resolution, its own padding rule, its own normalization -- then gets
an image that has been through two different resizes and possibly two normalizations.
Nothing errors. The policy simply performs worse over the wire than it does locally,
which is the hardest class of bug to notice and the easiest to blame on the network.
So a registered adapter gets a pass-through pipeline and does its own preprocessing,
exactly as it would if it were running on the robot.

WHAT AN ADAPTER MUST PROVIDE is one method, `predict_action_chunk(batch)`, returning
a (T, action_dim) or (B, T, action_dim) tensor. Everything else -- the training-only
half of lerobot's PreTrainedPolicy surface -- is stubbed here.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
import torch

from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.utils.constants import OBS_IMAGES, OBS_STATE

logger = logging.getLogger("remote.adapters")

#: name -> (config class, policy class). Written by `register`, read by `install`.
REGISTRY: dict[str, tuple[type, type]] = {}

#: Policy types lerobot IMPLEMENTS but the async server's allow-list omits.
#:
#: SUPPORTED_POLICIES is a hand-maintained list, not a view of the policy registry,
#: so a policy can be fully ported into lerobot -- config, processors, weights,
#: `predict_action_chunk` -- and still be refused at connect time with "policy type
#: not supported". molmoact2 is exactly that: `get_policy_class("molmoact2")` returns
#: MolmoAct2Policy today, and the server will not serve it.
#:
#: These are NOT adapters. They keep lerobot's own processors, because a first-class
#: policy has real normalization that must not be bypassed -- the pass-through below
#: is only for models that preprocess themselves.
NATIVE_EXTRA: tuple[str, ...] = ("molmoact2",)


@dataclass
class RemoteConfigBase(PreTrainedConfig):
    """The inference-only half of a PreTrainedConfig.

    PreTrainedConfig is shaped for training -- it wants optimizer and scheduler
    presets and delta-index windows. A served checkpoint needs none of that, but the
    abstract methods still have to exist or the class cannot be instantiated.
    """

    #: Where the weights are. Interpreted by the adapter, not by lerobot: a Hub id,
    #: a local directory, whatever that model loads from.
    checkpoint: str = ""
    device: str = "cuda"
    dtype: str = "bfloat16"
    #: Actions the model emits per inference. The server truncates to the client's
    #: `actions_per_chunk`, so this is an upper bound rather than a contract.
    chunk_size: int = 30
    #: Default task string, used when the client sends none.
    task: str = ""
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def observation_delta_indices(self) -> list | None:
        return None

    @property
    def action_delta_indices(self) -> list | None:
        return None

    @property
    def reward_delta_indices(self) -> list | None:
        return None

    def get_optimizer_preset(self):
        raise NotImplementedError("inference only")

    def get_scheduler_preset(self):
        return None

    def validate_features(self) -> None:
        """Nothing to validate: the adapter owns its own input contract.

        lerobot validates that a policy's declared input features match the dataset's.
        A served model has no dataset here, and the features it wants are whatever its
        own processor wants -- see the module docstring on why we do not let lerobot
        shape the inputs at all.
        """
        return None


def observation_to_numpy(batch: dict) -> tuple[np.ndarray, list[np.ndarray], str]:
    """(state, images, task) out of the server's observation dict.

    Images come back as HWC uint8 -- the layout a camera produced and every model's
    own processor expects -- rather than the CHW float lerobot would hand a native
    policy. Image ORDER is the sorted observation key order, which is the only stable
    ordering available: a dict from the wire has no inherent one, and a model that
    wants "scene then wrist" needs that to be the same on every tick.
    """
    state = batch.get(OBS_STATE)
    if state is None:
        raise KeyError(f"observation has no {OBS_STATE}; got {sorted(batch)}")
    if isinstance(state, torch.Tensor):
        state = state.detach().cpu().numpy()
    state = np.asarray(state, dtype=np.float32).reshape(-1)

    images: list[np.ndarray] = []
    for key in sorted(k for k in batch if k.startswith(OBS_IMAGES)):
        img = batch[key]
        if isinstance(img, torch.Tensor):
            img = img.detach().cpu()
            if img.ndim == 4:
                img = img[0]
            # CHW -> HWC when the batch arrived already permuted.
            if img.ndim == 3 and img.shape[0] in (1, 3) and img.shape[-1] not in (1, 3):
                img = img.permute(1, 2, 0)
            img = img.numpy()
        img = np.asarray(img)
        # Float images are in [0, 1] wherever they have been through a normalizer.
        if img.dtype != np.uint8:
            img = (np.clip(img, 0.0, 1.0) * 255.0).astype(np.uint8) if img.max() <= 1.0 \
                else img.astype(np.uint8)
        images.append(img)

    task = batch.get("task", "")
    if isinstance(task, (list, tuple)):
        task = task[0] if task else ""
    return state, images, str(task or "")


class RemotePolicyBase(PreTrainedPolicy):
    """The inference-only half of a PreTrainedPolicy.

    Subclasses set `config_class` and `name`, and implement `predict_action_chunk`.
    """

    config_class = RemoteConfigBase
    name = "remote_base"

    def __init__(self, config: RemoteConfigBase, **kwargs):
        super().__init__(config)
        self.config = config

    # -- training surface, present so the class can be instantiated ----------- #

    def get_optim_params(self) -> dict:
        raise NotImplementedError("inference only")

    def forward(self, batch):
        raise NotImplementedError("inference only")

    # -- inference ------------------------------------------------------------ #

    def reset(self) -> None:
        """Called between episodes. Stateless by default."""
        return None

    def predict_action_chunk(self, batch: dict, **kwargs) -> torch.Tensor:
        raise NotImplementedError

    def select_action(self, batch: dict, **kwargs) -> torch.Tensor:
        """One action, for callers that want a step at a time rather than a chunk.

        The policy server never calls this -- it takes whole chunks -- but
        PreTrainedPolicy declares it, and a caller that finds it raising would have
        no idea the chunk path was the supported one.
        """
        return self.predict_action_chunk(batch, **kwargs)[:, 0, :]


def register(name: str, config_cls: type, policy_cls: type) -> None:
    """Make `name` a policy type the server will accept."""
    if name in REGISTRY:
        logger.warning("adapter %s registered twice; keeping the first", name)
        return
    REGISTRY[name] = (config_cls, policy_cls)


class _PassThrough:
    """The pipeline for a model that preprocesses itself. See the module docstring."""

    def __call__(self, x):
        return x


def install() -> list[str]:
    """Widen lerobot's three policy seams to cover the registered adapters.

    Idempotent, and returns the names now served. Must run BEFORE the server builds
    a policy; `serve.py` does it at import time for that reason.
    """
    from lerobot.async_inference import constants as async_constants
    from lerobot.policies import factory as policy_factory


    supported = list(async_constants.SUPPORTED_POLICIES)

    # Native policies first, and only if lerobot really can build them: allow-listing
    # a name the factory cannot resolve turns a clear "not supported" at connect time
    # into a stack trace at load time.
    for name in NATIVE_EXTRA:
        if name in supported:
            continue
        try:
            policy_factory.get_policy_class(name)
        except Exception as exc:  # noqa: BLE001 - this lerobot simply lacks it
            logger.debug("not allow-listing %s: %s", name, exc)
            continue
        supported.append(name)

    for name in REGISTRY:
        if name not in supported:
            supported.append(name)
    async_constants.SUPPORTED_POLICIES = supported
    # policy_server.py did `from .constants import SUPPORTED_POLICIES`, binding the
    # list object into its own module namespace at import time. Rebinding the name on
    # the constants module alone would leave the server checking the original.
    try:
        from lerobot.async_inference import policy_server as _ps

        _ps.SUPPORTED_POLICIES = supported
    except ImportError:  # the server half is not installed on a robot-only box
        pass

    if not getattr(policy_factory.get_policy_class, "_remote_patched", False):
        original_get = policy_factory.get_policy_class

        def get_policy_class(name: str):
            if name in REGISTRY:
                return REGISTRY[name][1]
            return original_get(name)

        get_policy_class._remote_patched = True
        policy_factory.get_policy_class = get_policy_class
        try:
            from lerobot.async_inference import policy_server as _ps

            _ps.get_policy_class = get_policy_class
        except ImportError:
            pass

    if not getattr(policy_factory.make_pre_post_processors, "_remote_patched", False):
        original_make = policy_factory.make_pre_post_processors

        def make_pre_post_processors(policy_cfg, pretrained_path=None, **kwargs):
            for config_cls, _ in REGISTRY.values():
                if isinstance(policy_cfg, config_cls):
                    return _PassThrough(), _PassThrough()
            return original_make(policy_cfg, pretrained_path, **kwargs)

        make_pre_post_processors._remote_patched = True
        policy_factory.make_pre_post_processors = make_pre_post_processors
        try:
            from lerobot.async_inference import policy_server as _ps

            _ps.make_pre_post_processors = make_pre_post_processors
        except ImportError:
            pass

    return sorted(REGISTRY)
