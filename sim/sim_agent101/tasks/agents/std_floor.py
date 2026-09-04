"""A Gaussian policy whose exploration noise cannot collapse.

mjlab floors sigma at 0.2 for every push-T run that worked, and its note on why is
worth repeating in full, because the failure it describes is silent:

    Side contact only became geometrically reachable once the action scale was
    widened, but reaching it is a specific descent the policy still has to find. At
    iteration 1000 of the first run with the wider box, the policy's median minimum
    end-effector height was 123 mm -- higher than run 7 managed with the NARROWER box
    -- and only 3.1% of episodes ever reached side-contact geometry, with mean_std
    down to 0.026. There was no exploration left to find it with.

    std_range floors sigma at 0.2 so the descent stays discoverable. Raising
    entropy_coef alone does not bound sigma; this does.

rsl_rl 2.3.3 has init_noise_std and noise_std_type and no bound at all, so the floor
is added here. The class is registered into the runner's own module namespace because
that is where rsl_rl resolves a policy by name -- `eval(self.policy_cfg.pop(
"class_name"))`, evaluated in on_policy_runner's globals.
"""

from __future__ import annotations

import math

import rsl_rl.runners.on_policy_runner as _on_policy_runner
import torch
from rsl_rl.modules import ActorCritic


class ActorCriticStdFloor(ActorCritic):
    """ActorCritic with sigma clamped into [std_min, std_max]."""

    def __init__(self, *args, std_min: float = 0.2, std_max: float = 1.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.std_min, self.std_max = float(std_min), float(std_max)

    def update_distribution(self, observations):
        # Clamp the PARAMETER, not the sampled distribution: the optimizer writes to
        # this tensor, so clamping here -- on every forward pass, before the parent
        # builds the Normal -- is what makes the bound survive the next gradient step
        # rather than being re-learned away between them.
        with torch.no_grad():
            if self.noise_std_type == "scalar":
                self.std.clamp_(self.std_min, self.std_max)
            else:
                self.log_std.clamp_(math.log(self.std_min), math.log(self.std_max))
        super().update_distribution(observations)


# Registered at import. sim_agent101.tasks.agents imports this module, so anything
# that has imported the package can name the class in a runner config.
_on_policy_runner.ActorCriticStdFloor = ActorCriticStdFloor
