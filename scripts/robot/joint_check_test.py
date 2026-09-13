#!/usr/bin/env python3
"""Tests for the one thing joint-check has to get right: when it may draw a conclusion.

    python3 scripts/robot/joint_check_test.py     # no arm, no docker

The sweep measures a joint's slip as the midpoint between its two mechanical stops. The
midpoint of anything LESS than the full travel is not that number -- it is where the
operator's hand happened to be. The first version of this tool printed the warning and
the verdict together ("you did not reach both stops ... wrist_flex's zero is where it
always was"), which is worse than printing nothing: it reads as a green light on a
measurement that never happened, and on this bench a wrong green light means recording
a DAgger session in a frame the checkpoints do not share.
"""

from __future__ import annotations

import importlib.util
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
SPAN = 104.09  # wrist_flex's real calibrated half-range on this arm


def _mod():
  spec = importlib.util.spec_from_file_location("joint_check", HERE / "joint_check.py")
  module = importlib.util.module_from_spec(spec)
  sys.modules[spec.name] = module
  spec.loader.exec_module(module)
  return module


def _text(lo, hi, span=SPAN):
  return "\n".join(_mod().verdict("wrist_flex", lo, hi, span))


def test_no_sweep_at_all_gives_no_verdict():
  """The exact session that prompted this: Ctrl-C pressed immediately."""
  out = _text(0.53, 0.53)
  assert "NOT MEASURED" in out
  assert "is where it always was" not in out
  assert "Nothing to patch" not in out
  assert "joint-offset" not in out, "offered a fix from a measurement that did not happen"


def test_a_half_sweep_gives_no_verdict_either():
  """One stop reached, not the other: the midpoint looks plausible and is not."""
  out = _text(-104.0, 0.0)
  assert "NOT MEASURED" in out
  assert "MIDPOINT" not in out


def test_a_full_sweep_that_centres_says_so():
  out = _text(-104.0, 104.0)
  assert "NOT MEASURED" not in out
  assert "Nothing to patch" in out
  assert "joint-offset" not in out


def test_a_full_sweep_that_is_offset_gives_the_number_and_the_command():
  out = _text(-92.0, 116.0)  # same travel, shifted by +12
  assert "NOT MEASURED" not in out
  assert "MIDPOINT +12.00" in out
  assert "--degrees 12.00" in out
  assert "Sweep again afterwards" in out


def test_the_threshold_is_not_so_tight_that_a_real_sweep_fails():
  """Stops are stiff and nobody reaches the last degree; 95% of travel must pass."""
  edge = SPAN * 0.95
  out = _text(-edge, edge)
  assert "NOT MEASURED" not in out, "a 95% sweep was rejected; the bar is too high"


def test_without_a_stored_span_it_still_reports_rather_than_guesses():
  """No calibration to compare against: say the midpoint, claim no coverage."""
  out = "\n".join(_mod().verdict("wrist_flex", -10.0, 10.0, None))
  assert "MIDPOINT" in out
  assert "NOT MEASURED" not in out


def main() -> int:
  tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
  for test in tests:
    test()
    print(f"ok  {test.__name__}")
  print(f"\n{len(tests)} passed")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
