#!/usr/bin/env python3
"""Per-frame sampling weights for a DAgger round, from the `intervention` column.

Adapted from a private training pipeline this bench does not own. The weight
rule, the quantisation and the loader patch follow that design; what is written
here is where the labels come from and how they are read.

    ./robot openpi-dagger-stats --dataset zetanschy/v1_cap_to_cup_dagger

WHY WEIGHT AT ALL. A DAgger dataset is not a demonstration set. It is a complete
autonomous rollout with human takeovers spliced into it, so training on it
uniformly is wrong in two directions at once: the operator's corrective frames
are the entire point of the round and deserve more of the batch, while the
policy's own frames in the seconds BEFORE a takeover are the failure run-up and
must not be reinforced. scripts/openpi/dagger.py records `intervention` and
deliberately stops there -- "a training run can weight them, if you decide it
should". This is that decision.

WEIGHTS ARE APPLIED TO SAMPLING, NOT TO THE LOSS. Each frame is repeated in a
precomputed index a number of times proportional to its weight, and openpi
shuffles and batches that index as usual. A zero-weight frame then costs no video
decode, where a frame whose loss was scaled to zero would still pay a full
forward and backward pass. openpi's loss stays openpi's.

WHERE THE LABELS COME FROM, and this is the one real difference from the
design it follows. That one labels time SPANS, because its takeovers are
reconstructed after the fact from a control-source signal. This bench does not
have to infer anything: `./robot dagger` writes a per-frame boolean `intervention`
column into the dataset itself, so the label is exact and there is no span-to-
frame mapping to get wrong. The cost is that this opens the data shards -- but
only three scalar columns of them (`index`, `episode_index`, `intervention`),
never a video, so it stays a metadata-speed read.

THE DEFAULTS ARE NOT THE USUAL 2.0/0.3, which is tuned for SPARSE interventions.
Measured over this bench's six DAgger sets the operator share runs 25-50% of
frames, and at 2.0/0.3 that puts 78-86% of every batch on corrections with almost
no autonomous data left to anchor against forgetting. 1.0/0.5 lands near an even split. Tune against the
`operator share of draws` line `stats` prints, not against the ratio.
"""

from __future__ import annotations

import argparse
import dataclasses
import functools
import glob
import hashlib
import inspect
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger("dagger_weights")

#: Cached sampling index and its provenance.
INDEX_FILE = "dagger_index.npy"
INDEX_META_FILE = "dagger_index.json"

#: Where those two live. NOT inside the dataset directory, which is what the
#: reference implementation does -- there they sit beside a staging copy that is
#: thrown away, here the dataset directory is a LeRobotDataset that gets pushed to
#: the Hub, and `push_to_hub` uploads whatever it finds. Two derived files shipped
#: as part of a published dataset, where anyone pulling it would reasonably read
#: them as data.
INDEX_CACHE = Path(__file__).resolve().parents[2] / "outputs" / "dagger" / "index"

#: LeRobot layout this understands. v2.x has a different data layout and a
#: different episode bookkeeping; rather than guess at the mapping, refuse it.
SUPPORTED_CODEBASE_VERSIONS = ("v3.0",)

_installed = False


@dataclasses.dataclass(frozen=True)
class DaggerConfig:
    """The weight rule, and how finely it is quantised into a sampling index.

    Frames are weighted, then normalised into a repetition count:

      * operator frames (`intervention` true) get `human_weight`;
      * policy frames further than `pre_window_s` from the next operator frame
        get `auto_weight`;
      * policy frames inside that window ramp linearly from `auto_weight` at the
        window edge down to `pre_min_weight` at the takeover.
    """

    human_weight: float = 1.0
    auto_weight: float = 0.5
    pre_window_s: float = 5.0
    pre_min_weight: float = 0.0
    #: Index length as a multiple of the frame count. The index is built ONCE and
    #: reused for the whole run, so quantisation error is permanent exclusion
    #: rather than per-epoch noise: at scale 1, `auto_weight * k` rounds to well
    #: under 1 and a large share of frames are never drawn at all. 8 costs a few
    #: MB and leaves only the deep pre-intervention ramp undrawn, which is the
    #: intent rather than a loss.
    epoch_scale: int = 8
    seed: int = 42

    def __post_init__(self) -> None:
        if min(self.human_weight, self.auto_weight, self.pre_min_weight) < 0.0:
            raise ValueError("dagger weights must be non-negative")
        if self.human_weight <= 0.0 and self.auto_weight <= 0.0:
            raise ValueError("human_weight and auto_weight cannot both be zero")
        if self.pre_window_s < 0.0:
            raise ValueError("pre_window_s must be non-negative")
        if self.pre_min_weight > self.auto_weight:
            raise ValueError("pre_min_weight must not exceed auto_weight: the ramp "
                             "runs DOWN toward the takeover")
        if self.epoch_scale < 1:
            raise ValueError("epoch_scale must be at least 1")

    def fingerprint(self) -> str:
        return json.dumps(dataclasses.asdict(self), sort_keys=True)


@dataclasses.dataclass
class Summary:
    """What the index turned out to be, for the log and for `stats`."""

    frames_total: int = 0
    frames_human: int = 0
    episodes: int = 0
    fps: float = 0.0
    index_len: int = 0
    never_drawn: int = 0
    human_share_of_draws: float = 0.0

    @property
    def human_fraction(self) -> float:
        return self.frames_human / self.frames_total if self.frames_total else 0.0


def _read_info(root: Path) -> dict[str, Any]:
    info_path = root / "meta" / "info.json"
    if not info_path.is_file():
        raise FileNotFoundError(f"{info_path} not found — is {root} a LeRobot dataset?")
    info = json.loads(info_path.read_text())
    version = info.get("codebase_version")
    if version not in SUPPORTED_CODEBASE_VERSIONS:
        raise ValueError(
            f"{root} is codebase_version {version!r}; this understands "
            f"{SUPPORTED_CODEBASE_VERSIONS}. Convert it rather than have the frame "
            "indices mean something different than assumed."
        )
    if "intervention" not in info.get("features", {}):
        raise ValueError(
            f"{root} has no `intervention` feature, so it is not a DAgger dataset. "
            "Record one with `./robot dagger`, or drop --dagger and train on it flat."
        )
    return info


def read_labels(root: Path) -> tuple[np.ndarray, np.ndarray, float]:
    """(operator mask, episode index, fps), both arrays indexed by GLOBAL frame.

    Three scalar columns out of the parquet shards and nothing else: no video is
    opened, so this is a metadata-speed read of an otherwise large dataset.

    The rows are placed by their own `index` column rather than by the order the
    shards happen to concatenate in, because a LeRobotDataset index IS the global
    frame index -- that is what makes the sampling index built here line up with
    the dataset openpi opens.
    """
    import pyarrow.parquet as pq

    info = _read_info(root)
    total = int(info["total_frames"])
    fps = float(info["fps"])

    is_human = np.zeros(total, dtype=bool)
    episode = np.full(total, -1, dtype=np.int64)
    seen = np.zeros(total, dtype=bool)

    files = sorted(glob.glob(str(root / "data" / "**" / "*.parquet"), recursive=True))
    if not files:
        raise FileNotFoundError(f"no parquet shards under {root / 'data'}")
    for f in files:
        table = pq.read_table(f, columns=["index", "episode_index", "intervention"])
        idx = table.column("index").to_numpy()
        if idx.max(initial=-1) >= total or idx.min(initial=0) < 0:
            raise ValueError(f"{f} holds frame indices outside 0..{total - 1}")
        is_human[idx] = np.asarray(table.column("intervention").to_numpy(), dtype=bool)
        episode[idx] = table.column("episode_index").to_numpy()
        seen[idx] = True

    if not seen.all():
        raise ValueError(
            f"{root}: {int((~seen).sum())} of {total} frames are missing from the "
            "shards, so the index would not line up with the dataset openpi opens."
        )
    return is_human, episode, fps


def _episode_weights(is_human: np.ndarray, fps: float, cfg: DaggerConfig) -> np.ndarray:
    """Apply the weight rule within ONE episode.

    Per-episode rather than globally because the pre-intervention ramp must not
    reach backwards across an episode boundary: the last seconds of episode 3 are
    not the run-up to a takeover at the start of episode 4.
    """
    length = is_human.shape[0]
    weights = np.full(length, cfg.auto_weight, dtype=np.float32)

    human_positions = np.flatnonzero(is_human)
    if human_positions.size and cfg.pre_window_s > 0.0:
        # Distance FORWARD to the next operator frame; +inf past the last one.
        positions = np.arange(length)
        nxt = np.searchsorted(human_positions, positions, side="left")
        has_next = nxt < human_positions.size
        distance = np.full(length, np.inf, dtype=np.float32)
        distance[has_next] = (
            human_positions[nxt[has_next]] - positions[has_next]
        ).astype(np.float32)

        seconds = distance / fps
        in_window = ~is_human & (seconds < cfg.pre_window_s)
        ramp = np.clip(seconds[in_window] / cfg.pre_window_s, 0.0, 1.0)
        weights[in_window] = cfg.pre_min_weight + ramp * (
            cfg.auto_weight - cfg.pre_min_weight
        )

    weights[is_human] = cfg.human_weight
    return weights


def frame_weights(root: Path, cfg: DaggerConfig) -> tuple[np.ndarray, np.ndarray, Summary]:
    """Per-frame weights over the whole dataset, plus the operator mask."""
    is_human, episode, fps = read_labels(root)
    weights = np.zeros(is_human.shape[0], dtype=np.float32)
    for ep in np.unique(episode):
        where = np.flatnonzero(episode == ep)
        # The frames of one episode are contiguous in a LeRobot index, but sort
        # anyway: the ramp is computed on position, and a shard read out of order
        # would otherwise ramp toward the wrong frame.
        where.sort()
        weights[where] = _episode_weights(is_human[where], fps, cfg)

    summary = Summary(
        frames_total=int(is_human.shape[0]),
        frames_human=int(is_human.sum()),
        episodes=int(np.unique(episode).size),
        fps=fps,
    )
    return weights, is_human, summary


def expand_index(weights: np.ndarray, cfg: DaggerConfig) -> np.ndarray:
    """Turn per-frame weights into a sampling index openpi can shuffle.

    Each frame appears `round(w_i * k)` times, with k set so the index is
    `epoch_scale` times the frame count. Rounding is a SEEDED BERNOULLI DRAW on
    the fractional part, not largest-remainder: most autonomous frames carry the
    identical weight, and a remainder sort would break those ties by index and
    promote a contiguous prefix -- systematically dropping whole stretches of
    episodes rather than sampling them.
    """
    total = float(weights.sum())
    if total <= 0.0:
        raise ValueError("every frame weight is zero — nothing would be sampled")

    # Normalising against the smallest non-zero weight would diverge: the ramp
    # asymptotes toward pre_min_weight rather than reaching it, so the frame
    # adjacent to a takeover carries auto_weight / (pre_window_s * fps).
    scale = cfg.epoch_scale * weights.shape[0] / total
    scaled = weights.astype(np.float64) * scale
    counts = np.floor(scaled).astype(np.int64)
    fractional = scaled - counts
    counts += np.random.default_rng(cfg.seed).random(scaled.shape) < fractional
    return np.repeat(np.arange(weights.shape[0], dtype=np.int64), counts)


def summarize(index: np.ndarray, is_human: np.ndarray, summary: Summary) -> Summary:
    drawn = np.bincount(index, minlength=is_human.shape[0])
    summary.index_len = int(index.shape[0])
    summary.never_drawn = int((drawn == 0).sum())
    summary.human_share_of_draws = (
        float(drawn[is_human].sum() / summary.index_len) if summary.index_len else 0.0
    )
    return summary


def _dataset_fingerprint(root: Path, cfg: DaggerConfig, is_human: np.ndarray) -> str:
    """Hash what determines the index, so a changed dataset invalidates the cache.

    Over the labels themselves rather than over a file's bytes: the labels are
    spread across the parquet shards here rather than sitting in one annotation
    file, and re-recording an episode changes them without necessarily changing
    any single file's size or name.
    """
    info = _read_info(root)
    digest = hashlib.sha256()
    digest.update(cfg.fingerprint().encode())
    digest.update(f"{info['total_frames']}:{info['total_episodes']}:{info['fps']}".encode())
    digest.update(np.packbits(is_human).tobytes())
    return digest.hexdigest()


def build(root: Path, cfg: DaggerConfig, force: bool = False) -> tuple[np.ndarray, Summary]:
    """The sampling index for this dataset and rule, from cache when it matches."""
    root = Path(root)
    cache = INDEX_CACHE / root.name
    index_path, meta_path = cache / INDEX_FILE, cache / INDEX_META_FILE

    weights, is_human, summary = frame_weights(root, cfg)
    fingerprint = _dataset_fingerprint(root, cfg, is_human)

    if not force and index_path.is_file() and meta_path.is_file():
        try:
            meta = json.loads(meta_path.read_text())
        except json.JSONDecodeError:
            meta = {}
        if meta.get("fingerprint") == fingerprint:
            index = np.load(index_path)
            logger.info("dagger index: reusing %s (%d draws)", index_path, index.shape[0])
            return index, summarize(index, is_human, summary)

    index = expand_index(weights, cfg)
    summary = summarize(index, is_human, summary)
    try:
        cache.mkdir(parents=True, exist_ok=True)
        np.save(index_path, index)
        meta_path.write_text(json.dumps(
            {"fingerprint": fingerprint, "config": dataclasses.asdict(cfg),
             "summary": dataclasses.asdict(summary)}, indent=2) + "\n")
    except OSError as e:   # a read-only cache is not a reason to refuse to train
        logger.warning("could not cache the dagger index (%s); rebuilding each run", e)
    return index, summary


def per_episode(root: Path) -> list[dict]:
    """One row per episode: how long, how much of it you drove, in how many goes.

    The dataset-level share says nothing about WHICH episode is odd, and the odd
    ones are the whole question when deciding what to train on. An episode you
    drove end to end is a demonstration that got filed as a correction; one you
    never touched is the policy succeeding on its own and teaches it what it
    already believes; a dozen one-second takeovers is usually a session where the
    handover key was fighting you rather than twelve separate failures.
    """
    is_human, episode, fps = read_labels(root)
    rows = []
    for ep in np.unique(episode):
        where = np.flatnonzero(episode == ep)
        where.sort()
        mask = is_human[where]
        # A takeover is a run of True; count the rising edges.
        edges = int(np.count_nonzero(mask & ~np.r_[False, mask[:-1]]))
        runs = np.diff(np.flatnonzero(np.r_[True, mask[1:] != mask[:-1], True]))
        human_runs = runs[0 if mask[0] else 1::2] if mask.any() else np.array([0])
        rows.append({
            "episode": int(ep),
            "frames": int(mask.size),
            "seconds": mask.size / fps,
            "human_frames": int(mask.sum()),
            "human_fraction": float(mask.mean()),
            "takeovers": edges,
            "longest_takeover_s": float(human_runs.max() / fps) if mask.any() else 0.0,
        })
    return rows


def format_per_episode(rows: list[dict]) -> str:
    """The per-episode table, with the rows worth a second look called out."""
    out = [f"{'ep':>3s} {'frames':>7s} {'secs':>6s} {'operator':>9s} "
           f"{'takeovers':>10s} {'longest':>8s}  note"]
    for r in rows:
        note = ""
        # Length first: a one-frame episode is a broken save, and saying "no
        # correction at all" about it describes the symptom rather than the fault.
        if r["seconds"] < 1.0:
            note = "BROKEN — too short to be an attempt; drop this episode"
        elif r["seconds"] < 5.0:
            note = "very short for this task; look before keeping it"
        elif r["takeovers"] == 0:
            note = "no correction at all — the policy did this one alone"
        elif r["human_fraction"] > 0.9:
            note = "you drove almost all of it — a demonstration, not a correction"
        elif r["longest_takeover_s"] < 0.5 and r["takeovers"] > 3:
            note = "many tiny takeovers — check the handover key, not the policy"
        out.append(
            f"{r['episode']:3d} {r['frames']:7d} {r['seconds']:6.1f} "
            f"{100 * r['human_fraction']:8.1f}% {r['takeovers']:10d} "
            f"{r['longest_takeover_s']:7.1f}s  {note}")
    return "\n".join(out)


def format_summary(summary: Summary, root: Path) -> str:
    lines = [
        f"dataset                : {root}",
        f"episodes               : {summary.episodes}",
        f"frames                 : {summary.frames_total}",
        f"  operator             : {summary.frames_human} "
        f"({100.0 * summary.human_fraction:.1f}%)",
        f"fps                    : {summary.fps:g}",
        f"sampling index         : {summary.index_len} draws "
        f"({summary.index_len / summary.frames_total:.1f}x the frames)"
        if summary.frames_total else "sampling index         : empty",
        f"  never drawn          : {summary.never_drawn} frames "
        f"({100.0 * summary.never_drawn / summary.frames_total:.1f}%)"
        if summary.frames_total else "",
        f"  operator share       : {100.0 * summary.human_share_of_draws:.1f}% of draws",
    ]
    return "\n".join(line for line in lines if line)


# --------------------------------------------------------------------------- #
# Installing into openpi's loader
# --------------------------------------------------------------------------- #


class IndexRemapDataset:
    """A dataset view that reads its base through a repeated index.

    Module-level rather than a closure so dataloader worker processes can pickle
    it; this config runs num_workers=8.
    """

    def __init__(self, base: Any, index: np.ndarray) -> None:
        self._base = base
        self._index = index

    def __len__(self) -> int:
        return int(self._index.shape[0])

    def __getitem__(self, position: int) -> Any:
        return self._base[int(self._index[position])]


def _check_openpi_seam(data_loader_module: Any) -> None:
    """Verify openpi still builds its dataset the way this patch assumes.

    A silently ineffective monkeypatch is the failure mode that matters: training
    would run to completion on UNWEIGHTED data and look entirely normal. So probe
    the seam and refuse loudly rather than patch and hope.
    """
    target = getattr(data_loader_module, "create_torch_dataset", None)
    if target is None or not callable(target):
        raise RuntimeError(
            "openpi.training.data_loader.create_torch_dataset is gone; "
            "scripts/openpi/dagger_weights.py needs updating for this openpi."
        )
    params = list(inspect.signature(target).parameters)
    if params[:3] != ["data_config", "action_horizon", "model_config"]:
        raise RuntimeError(
            "openpi.training.data_loader.create_torch_dataset changed signature "
            f"(got {params}); scripts/openpi/dagger_weights.py needs updating."
        )
    try:
        source = inspect.getsource(data_loader_module.create_torch_data_loader)
    except OSError:   # source is present in a real checkout
        return
    if "create_torch_dataset(" not in source:
        raise RuntimeError(
            "openpi.training.data_loader.create_torch_data_loader no longer calls "
            "create_torch_dataset as a module global, so patching it would be a "
            "silent no-op; scripts/openpi/dagger_weights.py needs updating."
        )


def install(root: Path, cfg: DaggerConfig, force: bool = False) -> Summary:
    """Make openpi's loader sample this dataset through the DAgger index.

    Patching rather than passing a sampler is forced: `create_torch_data_loader`
    hardcodes `sampler=None` on the JAX path, and forking the vendored openpi for
    this is a worse trade than one probed monkeypatch.
    """
    global _installed

    index, summary = build(Path(root), cfg, force=force)
    if _installed:
        logger.info("dagger sampling already installed")
        return summary

    from openpi.training import data_loader as openpi_data_loader

    _check_openpi_seam(openpi_data_loader)
    original = openpi_data_loader.create_torch_dataset

    @functools.wraps(original)
    def create_torch_dataset_with_dagger(*args: Any, **kwargs: Any) -> Any:
        dataset, data_config = original(*args, **kwargs)
        if len(dataset) != summary.frames_total:
            raise ValueError(
                f"dagger index was built for {summary.frames_total} frames but "
                f"openpi opened a dataset of {len(dataset)}. The dataset and the "
                "index disagree; rebuild with --dagger-rebuild."
            )
        return IndexRemapDataset(dataset, index), data_config

    openpi_data_loader.create_torch_dataset = create_torch_dataset_with_dagger
    _installed = True
    logger.info(
        "dagger sampling installed: %d draws over %d frames, "
        "operator frames are %.1f%% of them",
        summary.index_len, summary.frames_total, 100.0 * summary.human_share_of_draws,
    )
    return summary


def config_from_env(environ: dict[str, str] | None = None) -> DaggerConfig | None:
    """A DaggerConfig from the environment, or None if weighting is off.

    The environment rather than a flag because openpi owns the command line: its
    tyro parser rejects any argument its own dataclasses do not define, so a
    `--dagger-human-weight` on that line would be a parse error. train.sh sets
    these; nothing else reads them.
    """
    env = os.environ if environ is None else environ
    if env.get("DAGGER_ENABLED", "0") != "1":
        return None

    def _f(key: str, default: float) -> float:
        raw = env.get(key, "")
        return float(raw) if raw.strip() else default

    def _i(key: str, default: int) -> int:
        raw = env.get(key, "")
        return int(raw) if raw.strip() else default

    return DaggerConfig(
        human_weight=_f("DAGGER_HUMAN_WEIGHT", 1.0),
        auto_weight=_f("DAGGER_AUTO_WEIGHT", 0.5),
        pre_window_s=_f("DAGGER_PRE_WINDOW_S", 5.0),
        pre_min_weight=_f("DAGGER_PRE_MIN_WEIGHT", 0.0),
        epoch_scale=_i("DAGGER_EPOCH_SCALE", 8),
        seed=_i("DAGGER_SEED", 42),
    )


def dataset_root(repo_id: str) -> Path:
    """Where a repo_id is on disk, following lerobot's own cache layout."""
    base = os.environ.get("LEROBOT_HOME") or os.environ.get("HF_LEROBOT_HOME")
    if base:
        return Path(base) / repo_id
    return Path.home() / ".cache" / "huggingface" / "lerobot" / repo_id


def install_if_configured(environ: dict[str, str] | None = None) -> Summary | None:
    """Install the DAgger index if the environment asks for one. Otherwise nothing."""
    env = os.environ if environ is None else environ
    cfg = config_from_env(env)
    if cfg is None:
        return None
    repo_id = env.get("DAGGER_REPO_ID", "").strip()
    if not repo_id:
        raise ValueError("DAGGER_ENABLED=1 but DAGGER_REPO_ID is empty")
    return install(dataset_root(repo_id), cfg,
                   force=env.get("DAGGER_REBUILD", "0") == "1")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="Inspect a DAgger dataset's sampling weights before spending GPU time.",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", required=True,
                   help="repo_id (zetanschy/v1_cap_to_cup_dagger) or a path")
    p.add_argument("--human-weight", type=float, default=1.0)
    p.add_argument("--auto-weight", type=float, default=0.5)
    p.add_argument("--pre-window-s", type=float, default=5.0)
    p.add_argument("--pre-min-weight", type=float, default=0.0)
    p.add_argument("--epoch-scale", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--rebuild", action="store_true", help="ignore a cached index")
    p.add_argument("--per-episode", action="store_true",
                   help="one row per episode: length, operator share, takeover count")
    a = p.parse_args(argv)

    root = Path(a.dataset)
    if not root.is_dir():
        root = dataset_root(a.dataset)

    cfg = DaggerConfig(human_weight=a.human_weight, auto_weight=a.auto_weight,
                       pre_window_s=a.pre_window_s, pre_min_weight=a.pre_min_weight,
                       epoch_scale=a.epoch_scale, seed=a.seed)
    _, summary = build(root, cfg, force=a.rebuild)
    print(format_summary(summary, root))
    if a.per_episode:
        print()
        print(format_per_episode(per_episode(root)))
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    sys.exit(main())
