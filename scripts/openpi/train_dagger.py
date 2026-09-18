#!/usr/bin/env python3
"""openpi's trainer, with DAgger sampling installed before it builds a loader.

A thin shim, deliberately. It installs the sampling patch and then hands control
to openpi's own `cli()` and `main()`, so every `--data.repo-id`, `--exp-name`,
`--batch-size` and `--weight-loader.params-path` override keeps working -- it is
still tyro parsing openpi's own dataclasses, and this file adds no arguments to
that line. That is also WHY the DAgger settings arrive through the environment
instead: openpi's parser rejects any flag its dataclasses do not define, so
`--dagger-human-weight` there would be a parse error rather than an option.

Used exactly like openpi's scripts/train.py; scripts/openpi/train.sh invokes it
in place of that file when --dagger is passed, and not otherwise.

    python scripts/openpi/train_dagger.py pi05_soarm101_lora_cap_to_cup \
        --exp-name=dagger_r1 --data.repo-id=zetanschy/v1_cap_to_cup_dagger

It is a no-op without DAGGER_ENABLED=1, so running everything through it would
also work; train.sh keeps openpi's own entry point for ordinary runs so that an
ordinary run has nothing of ours between it and the trainer.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))

import dagger_weights  # noqa: E402


def main() -> None:
    # openpi's trainer lives in its `scripts` package, which is not installed --
    # it is importable only with the openpi checkout root on sys.path. Running
    # this file directly puts ITS OWN directory at sys.path[0], not openpi's, so
    # the checkout root has to be added explicitly. Derived from the imported
    # module rather than hardcoded: it is /opt/openpi in the image and
    # thirdparty/openpi on a bare box.
    import openpi

    openpi_root = Path(openpi.__file__).parents[2]
    if str(openpi_root) not in sys.path:
        sys.path.insert(0, str(openpi_root))

    from openpi.training import config as openpi_config
    from scripts.train import main as openpi_train_main

    # Before the trainer builds a loader, because that is what gets patched.
    summary = dagger_weights.install_if_configured()
    if summary is not None:
        print(dagger_weights.format_summary(summary, Path("(configured dataset)")),
              flush=True)

    openpi_train_main(openpi_config.cli())


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
