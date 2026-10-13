"""A lerobot policy behind the interface sim_eval.py drives openpi through.

`openpi.policies.rtc.RealTimeChunker` is what the benchmark loop calls: infer(obs,
prefix_start=, inference_delay=) -> {"actions": (H, D) in robot units}, plus reset().
This is the same thing for a lerobot checkpoint, so a lerobot model and an openpi
model are scored by one loop: same stages, same 15-of-50 schedule, same seams.

The RTC bookkeeping mirrors RealTimeChunker exactly:
  - the prefix is the previous MODEL-SPACE chunk from `prefix_start` on (lerobot's
    RTC works on normalized actions, like openpi's);
  - s (lerobot's execution_horizon) is the whole real overlap, and d is clipped to it;
  - exp schedule, max guidance weight 5.0.
lerobot computes v_t before x_t requires grad, so its correction is the identity-
Jacobian one, the same `jacobian="identity"` the openpi runs use.

Loading follows le101's own rollout (rollout/context.py): a PEFT adapter goes on top
of the base policy its adapter config names, and the saved pre/post processors are
used with the camera rename the training used.

Runs in the lerobot image (CUDA torch); the openpi image's torch is CPU-only.
"""

from __future__ import annotations

import json
import pathlib

import numpy as np

# The rename every pi05 run in this repo trains with (scripts/robot/train.sh).
RENAME = {"observation.images.front": "observation.images.base_0_rgb",
          "observation.images.grip": "observation.images.left_wrist_0_rgb"}


def is_lerobot_checkpoint(policy: str) -> bool:
    """openpi checkpoints are /checkpoints/<run> dirs holding params/; anything else
    (a Hub repo id, or a dir with lerobot's config.json) is a lerobot policy."""
    p = pathlib.Path(policy)
    if p.is_dir():
        return not (p / "params").is_dir() and (p / "config.json").is_file()
    return not policy.startswith("/")


class LerobotChunker:
    def __init__(self, policy_path: str, robot, task: str, device: str = "cuda",
                 schedule: str = "exp", max_guidance_weight: float = 5.0):
        import torch
        from lerobot.configs.policies import PreTrainedConfig
        from lerobot.configs.types import RTCAttentionSchedule
        from lerobot.policies.factory import get_policy_class, make_pre_post_processors
        from lerobot.policies.rtc.configuration_rtc import RTCConfig
        from lerobot.utils.feature_utils import hw_to_dataset_features

        self._torch = torch
        cfg = PreTrainedConfig.from_pretrained(policy_path)
        cfg.pretrained_path = policy_path
        cfg.device = "cpu"  # built on the CPU, merged there, then moved (see below)
        # Set before construction, so the policy builds its RTCProcessor with it.
        cfg.rtc_config = RTCConfig(enabled=True, prefix_attention_schedule=RTCAttentionSchedule(schedule.upper()),
                                   max_guidance_weight=max_guidance_weight)
        cls = get_policy_class(cfg.type)
        self.trained = None
        if cfg.use_peft:
            from peft import PeftConfig, PeftModel
            peft_cfg = PeftConfig.from_pretrained(policy_path)
            policy = cls.from_pretrained(pretrained_name_or_path=peft_cfg.base_model_name_or_path, config=cfg)
            policy = PeftModel.from_pretrained(policy, policy_path, config=peft_cfg)
            self.trained = sum(p.numel() for n, p in policy.named_parameters()
                               if "lora_" in n or "modules_to_save" in n) or None
            # MERGED, the same network: LoRA folded into the weights, the fully trained
            # modules swapped in. Unmerged, PEFT keeps a frozen original of every fully
            # trained module beside its trained copy: with openpi's recipe (the whole
            # vision encoder trained) that is 9.3 GB, past a 12 GB card.
            policy = policy.merge_and_unload()
        else:
            policy = cls.from_pretrained(policy_path, config=cfg)
        cfg.device = device
        self.policy = policy.to(device).eval()
        self.pre, self.post = make_pre_post_processors(
            policy_cfg=cfg, pretrained_path=policy_path,
            preprocessor_overrides={"device_processor": {"device": device},
                                    "rename_observations_processor": {"rename_map": RENAME}})
        self.config, self.device, self.task, self.robot = cfg, device, task, robot
        self.hw = hw_to_dataset_features(robot.observation_features, "observation", use_video=True)
        self._prev = None
        self.last_delay = self.last_horizon = 0

    def describe(self) -> dict:
        return {"stack": "lerobot", "type": self.config.type, "use_peft": bool(self.config.use_peft),
                "rtc": {"schedule": self.config.rtc_config.prefix_attention_schedule.value,
                        "max_guidance_weight": self.config.rtc_config.max_guidance_weight},
                "trained_params": self.trained, "merged": bool(self.config.use_peft)}

    def reset(self) -> None:
        self._prev = None
        self.last_delay = self.last_horizon = 0

    def _batch(self, raw: dict):
        from lerobot.policies.utils import prepare_observation_for_inference
        from lerobot.utils.feature_utils import build_dataset_frame
        frame = build_dataset_frame(self.hw, raw, prefix="observation")
        batch = prepare_observation_for_inference(frame, self._torch.device(self.device), self.task,
                                                  getattr(self.robot, "robot_type", None) or self.robot.name)
        batch["task"] = [self.task]
        return self.pre(batch)

    def infer(self, raw: dict, *, prefix_start: int, inference_delay: int) -> dict:
        torch = self._torch
        with torch.inference_mode(False), torch.no_grad():
            batch = self._batch(dict(raw))
            tail = None if self._prev is None else self._prev[max(prefix_start, 0):]
            if tail is None or len(tail) == 0:
                self.last_delay = self.last_horizon = 0
                chunk = self.policy.predict_action_chunk(batch)
            else:
                s = len(tail)
                d = int(min(max(inference_delay, 0), s))
                self.last_delay, self.last_horizon = d, s
                chunk = self.policy.predict_action_chunk(batch, inference_delay=d,
                                                         prev_chunk_left_over=tail.unsqueeze(0),
                                                         execution_horizon=s)
            self._prev = chunk.squeeze(0).detach().clone()
            actions = self.post(chunk)
        actions = actions.squeeze(0).float().cpu().numpy() if hasattr(actions, "cpu") else np.asarray(actions)
        return {"actions": actions, "raw_actions": self._prev.float().cpu().numpy()}


if __name__ == "__main__":  # the one rule that matters: which stack a --policy is
    for p, want in [("/checkpoints/openpi_pi05_lora_cap_to_mug_sim_50", False),
                    ("zetanschy/lerobot_pi05_default_cap_to_mug_real_50", True)]:
        got = is_lerobot_checkpoint(p)
        print(f"{'ok ' if got == want else 'BAD'} {p}: {'lerobot' if got else 'openpi'}")
    print(json.dumps(RENAME))
