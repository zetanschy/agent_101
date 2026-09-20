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


def _their_keys() -> dict[str, str]:
  """event -> key, out of le101's DAggerKeyboardConfig field defaults."""
  source = (LEROBOT / "rollout/configs.py").read_text()
  block = re.search(r"class DAggerKeyboardConfig.*?(?=\n@|\nclass )", source, re.S)
  assert block, "DAggerKeyboardConfig is gone; the parity check needs rewriting"
  return dict(re.findall(r'(\w+):\s*str\s*=\s*"(\w+)"', block.group(0)))


def test_the_keys_are_theirs_not_mine():
  """Compare KEYS, not event names.

  An earlier version of this test only checked that "pause_resume" and "correction"
  appeared somewhere in their config, which passed happily while this file bound
  correction to `c` -- a key their docstring uses as a FORMAT EXAMPLE and never binds.
  Their default is `tab`.
  """
  dagger = _dagger()
  theirs = _their_keys()
  mine = {event: key for key, event in dagger.EVENTS.items()}
  for event, key in mine.items():
    assert theirs.get(event) == key, (
      f"{event}: this file binds {key!r}, le101 binds {theirs.get(event)!r}"
    )


def test_the_two_key_namespaces_do_not_collide():
  """Phase keys come from DAggerKeyboardConfig, episode keys from the recording loop.

  Two upstream sources, so nothing guarantees on its own that they stay disjoint -- and
  a key that both paused the policy and saved an episode would be the worst kind of
  surprise mid-session. An earlier version of this file invented `n` for saving
  precisely to avoid stealing `enter` (their `upload`); using lerobot's own recording
  keys removes the invention, and this is what keeps the two sets apart.
  """
  dagger = _dagger()
  overlap = set(dagger.EVENTS) & set(dagger.RECORDING_KEYS)
  assert not overlap, f"the same key means two things: {sorted(overlap)}"
  # `enter` is their upload and this file does not use it; leaving it unbound is
  # deliberate, so a finger that reaches for it does nothing rather than something else.
  assert "enter" not in dagger.EVENTS and "enter" not in dagger.RECORDING_KEYS


def _their_recording_controls() -> dict[str, str]:
  """key -> lerobot event, out of apply_recording_control's own branches."""
  source = (LEROBOT / "utils/keyboard_input.py").read_text()
  block = re.search(r"def apply_recording_control.*?(?=\ndef |\n@)", source, re.S)
  assert block, "apply_recording_control is gone; the parity check needs rewriting"
  body = block.group(0)
  controls = {}
  for key in re.findall(r'control == "(\w+)"', body):
    after = body.split(f'control == "{key}"', 1)[1].split("elif")[0]
    events = re.findall(r'events\["(\w+)"\]\s*=\s*True', after)
    controls[key] = [e for e in events if e != "exit_early"] or events
  return {k: v[0] for k, v in controls.items() if v}


def test_the_recording_keys_are_lerobots():
  """right saves, left discards, esc stops -- the keys every ./robot record uses.

  Episode control is not DAgger's invention: lerobot's apply_recording_control already
  binds these, and an operator who has recorded a dataset on this bench has the gesture
  in their fingers. Checked against their branches so a rebind upstream fails here.
  """
  dagger = _dagger()
  theirs = _their_recording_controls()
  assert set(dagger.RECORDING_KEYS) == set(theirs), (
    f"mine {sorted(dagger.RECORDING_KEYS)} vs lerobot {sorted(theirs)}"
  )
  # Their event names, and what each means here.
  meaning = {"exit_early": "save", "rerecord_episode": "discard", "stop_recording": "stop"}
  for key, event in theirs.items():
    assert dagger.RECORDING_KEYS[key] == meaning[event], (
      f"{key}: lerobot raises {event}, this file calls it {dagger.RECORDING_KEYS[key]!r}"
    )


def test_discarding_is_what_lerobot_calls_it():
  """A discard must clear the episode buffer, which is the API record uses for a
  re-record; anything else would leave the failed attempt in the parquet."""
  source = (LEROBOT / "scripts/lerobot_record.py").read_text()
  assert "clear_episode_buffer()" in source
  assert "clear_episode_buffer()" in (HERE / "dagger.py").read_text(), (
    "dagger.py discards an episode without clearing the buffer"
  )


def main() -> int:
  tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
  for test in tests:
    test()
    print(f"ok  {test.__name__}")
  print(f"\n{len(tests)} passed")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
