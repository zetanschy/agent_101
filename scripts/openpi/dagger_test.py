#!/usr/bin/env python3
"""Parity tests for scripts/openpi/dagger.py against le101's own DAgger strategy.

    python3 scripts/openpi/dagger_test.py     # reads le101's source; no GPU, no arm

This file exists because dagger.py is a REIMPLEMENTATION of
`lerobot/rollout/strategies/dagger.py`, forced by the fact that lerobot's policy factory
cannot load an openpi orbax checkpoint. A reimplementation is allowed to differ where
the engine differs; it is not allowed to differ in the operator's contract. So the two
things a human's hands depend on are checked against their source rather than against my
memory of it:

  * the transition table -- and above all what is MISSING from it, since a
    (CORRECTING, pause_resume) entry would hand the arm back to the policy with a hand
    still on the leader;
  * the `intervention` feature, because a different dtype or shape makes the recorded
    dataset untrainable by the same pipeline.

Both are parsed out of le101's file, so bumping the submodule to a version that changed
either one fails here instead of on the arm.
"""

from __future__ import annotations

import importlib.util
import pathlib
import re
import sys

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent.parent
LEROBOT = ROOT / "thirdparty/le101/src/lerobot"


def _dagger():
  spec = importlib.util.spec_from_file_location("agent101_dagger", HERE / "dagger.py")
  module = importlib.util.module_from_spec(spec)
  sys.modules[spec.name] = module
  spec.loader.exec_module(module)
  return module


def _their_transitions() -> set[tuple[str, str, str]]:
  """(phase, event, next_phase) triples out of le101's _DAGGER_TRANSITIONS literal."""
  source = (LEROBOT / "rollout/strategies/dagger.py").read_text()
  block = re.search(r"_DAGGER_TRANSITIONS.*?=\s*\{(.*?)\n\}", source, re.S)
  assert block, "le101's _DAGGER_TRANSITIONS is gone; the parity check needs rewriting"
  return {
    (m[0].lower(), m[1], m[2].lower())
    for m in re.findall(
      r"\(DAggerPhase\.(\w+),\s*\"(\w+)\"\):\s*DAggerPhase\.(\w+)", block.group(1)
    )
  }


def test_the_transition_table_is_theirs():
  mine = {(p.value, e, n.value) for (p, e), n in _dagger().TRANSITIONS.items()}
  theirs = _their_transitions()
  assert mine == theirs, f"only in mine: {mine - theirs}   only in theirs: {theirs - mine}"


def test_a_correction_cannot_be_handed_straight_back_to_the_policy():
  """The absence that matters. `space` while correcting must not resume the policy."""
  dagger = _dagger()
  assert (dagger.Phase.CORRECTING, "pause_resume") not in dagger.TRANSITIONS
  # ... and the only way out of a correction is to stop it.
  out = {e for (p, e) in dagger.TRANSITIONS if p is dagger.Phase.CORRECTING}
  assert out == {"correction"}, out


def test_every_phase_can_be_left():
  dagger = _dagger()
  leavable = {p for (p, _) in dagger.TRANSITIONS}
  assert leavable == set(dagger.Phase), set(dagger.Phase) - leavable


def test_the_intervention_feature_matches_theirs():
  """A different dtype or shape makes the dataset untrainable by the same pipeline."""
  source = (LEROBOT / "rollout/context.py").read_text()
  block = re.search(r"dataset_features\[\"intervention\"\]\s*=\s*\{(.*?)\}", source, re.S)
  assert block, "le101 no longer adds an `intervention` feature the same way"
  theirs = block.group(1)
  mine = _dagger().INTERVENTION_FEATURE
  assert mine["dtype"] == re.search(r"\"dtype\":\s*\"(\w+)\"", theirs).group(1)
  assert mine["shape"] == (1,) and "(1,)" in theirs
  assert mine["names"] is None and "None" in theirs


def test_the_keys_are_the_documented_ones():
  """le101 binds these per input device; this is the keyboard half of its defaults."""
  dagger = _dagger()
  assert dagger.EVENTS == {"space": "pause_resume", "c": "correction"}
  keyboard = (LEROBOT / "rollout/configs.py").read_text()
  block = re.search(r"class DAggerKeyboardConfig.*?(?=\n@|\nclass )", keyboard, re.S)
  assert block, "DAggerKeyboardConfig is gone"
  # Their defaults are declared as fields; check the events line up with ours by name.
  for event in dagger.EVENTS.values():
    assert event in block.group(0), f"{event} is not a DAgger keyboard binding in le101"


def main() -> int:
  tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
  for test in tests:
    test()
    print(f"ok  {test.__name__}")
  print(f"\n{len(tests)} passed")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
