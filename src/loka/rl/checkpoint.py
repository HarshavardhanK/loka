"""
Hybrid checkpoint manager for RL training.

Implements the industry-standard strategy for large-scale RL:

    **Last-N + Best-K** — always keep the most recent ``keep_last``
    checkpoints (crash recovery / rollback) plus the top ``keep_best``
    ranked by a primary metric.  Everything else is pruned.

Why not pure best-K?
    In RL, rewards are noisy early on — the "best" checkpoint at step 50
    is meaningless compared to step 5000.  Keeping recent checkpoints
    ensures you can always resume from where you left off if the job
    crashes.

Why not just last-N?
    RL is prone to reward hacking and late-stage collapse.  The best
    checkpoint may occur long before the final one.  Best-K ensures
    you don't lose it.

Ranking metric priority:
    1. ``loka/orbital/success_rate``   — did the mission succeed?
    2. ``loka/orbital/dv_efficiency_mean`` — how fuel-efficient?
    3. ``loka/reward/total_mean``       — fallback to scalar reward

Usage
-----
The manager is called from :meth:`MetricsTracker._flush` after each
batch of metrics is aggregated.  It checks the checkpoint directory
for new saves and prunes old ones.

It can also be run standalone as a background watcher::

    python -m loka.rl.checkpoint --checkpoint-dir /data/checkpoints --watch
"""

from __future__ import annotations

import json
import logging
import shutil
import time
from pathlib import Path
from typing import Any, Optional

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# ── Pydantic schemas ────────────────────────────────────────────────


class CheckpointMeta(BaseModel):
    """Metadata for a single checkpoint."""

    path: str = Field(..., description="Absolute path to checkpoint directory")
    step: int = Field(..., ge=0, description="Training step when saved")
    timestamp: float = Field(..., description="Unix timestamp when saved")

    # Ranking metrics (populated when available)
    success_rate: Optional[float] = Field(None, ge=0.0, le=1.0)
    dv_efficiency: Optional[float] = Field(None, ge=0.0)
    mean_reward: Optional[float] = None

    def score(self) -> tuple[float, float, float]:
        """Return a composite ranking tuple (higher is better).

        Ranked lexicographically:
            1. success_rate  (most important — did it work?)
            2. dv_efficiency (fuel optimality)
            3. mean_reward   (fallback)
        """
        return (
            self.success_rate if self.success_rate is not None else -1.0,
            self.dv_efficiency if self.dv_efficiency is not None else -1.0,
            self.mean_reward if self.mean_reward is not None else -1.0,
        )


class CheckpointConfig(BaseModel):
    """Configuration for the checkpoint manager."""

    keep_last: int = Field(2, ge=1, description="Always keep the N most recent checkpoints")
    keep_best: int = Field(3, ge=1, description="Keep the top-K checkpoints by metric")
    meta_filename: str = Field("loka_ckpt_meta.json", description="Metadata file inside each checkpoint dir")


# ── Checkpoint Manager ──────────────────────────────────────────────


class CheckpointManager:
    """Hybrid last-N + best-K checkpoint pruning.

    Parameters
    ----------
    checkpoint_dir : str or Path
        Root directory where Verl saves checkpoints (each as a subdirectory).
    config : CheckpointConfig, optional
        Pruning configuration.  Defaults to keep_last=2, keep_best=3.

    Examples
    --------
    >>> mgr = CheckpointManager("/data/checkpoints")
    >>> mgr.register(step=50, metrics={"loka/orbital/success_rate": 0.3, ...})
    >>> mgr.prune()  # deletes checkpoints outside the keep window
    """

    def __init__(
        self,
        checkpoint_dir: str | Path,
        config: CheckpointConfig | None = None,
    ):
        self._dir = Path(checkpoint_dir)
        self._cfg = config or CheckpointConfig()
        self._registry: list[CheckpointMeta] = []
        self._load_registry()

    @property
    def checkpoint_dir(self) -> Path:
        return self._dir

    @property
    def registered(self) -> list[CheckpointMeta]:
        """All registered checkpoints, sorted by step (ascending)."""
        return sorted(self._registry, key=lambda c: c.step)

    @property
    def best(self) -> list[CheckpointMeta]:
        """Top-K checkpoints by metric score."""
        return sorted(self._registry, key=lambda c: c.score(), reverse=True)[
            : self._cfg.keep_best
        ]

    # ── Public API ───────────────────────────────────────────────────

    def register(self, step: int, metrics: dict[str, Any] | None = None) -> CheckpointMeta | None:
        """Register a new checkpoint with its metrics.

        Scans the checkpoint directory for a subdirectory matching *step*
        and attaches the provided metrics.

        Parameters
        ----------
        step : int
            Training step (must correspond to a directory in checkpoint_dir).
        metrics : dict, optional
            Aggregated metrics from the current batch (keys from MetricsTracker).

        Returns
        -------
        CheckpointMeta or None
            The registered checkpoint, or ``None`` if no matching directory found.
        """
        # Verl saves checkpoints as: checkpoint_dir/global_step_XXX/
        ckpt_path = self._find_checkpoint_path(step)
        if ckpt_path is None:
            return None

        meta = CheckpointMeta(
            path=str(ckpt_path),
            step=step,
            timestamp=time.time(),
            success_rate=(metrics or {}).get("loka/orbital/success_rate"),
            dv_efficiency=(metrics or {}).get("loka/orbital/dv_efficiency_mean"),
            mean_reward=(metrics or {}).get("loka/reward/total_mean"),
        )

        # Avoid duplicate registration
        existing_steps = {c.step for c in self._registry}
        if step in existing_steps:
            # Update metrics for existing checkpoint
            self._registry = [c for c in self._registry if c.step != step]

        self._registry.append(meta)
        self._save_meta(meta, ckpt_path)
        logger.info(
            "Registered checkpoint step=%d  success=%.2f  dv_eff=%.3f  reward=%.3f",
            step,
            meta.success_rate or 0.0,
            meta.dv_efficiency or 0.0,
            meta.mean_reward or 0.0,
        )
        return meta

    def prune(self) -> list[str]:
        """Delete checkpoints outside the keep window.

        Keeps:
            - The ``keep_last`` most recent checkpoints (by step)
            - The ``keep_best`` highest-scoring checkpoints (by metric)

        Returns
        -------
        list[str]
            Paths of deleted checkpoint directories.
        """
        if not self._registry:
            return []

        # Build the "keep" set
        by_step = sorted(self._registry, key=lambda c: c.step, reverse=True)
        keep_recent = set(c.step for c in by_step[: self._cfg.keep_last])

        by_score = sorted(self._registry, key=lambda c: c.score(), reverse=True)
        keep_best = set(c.step for c in by_score[: self._cfg.keep_best])

        keep_steps = keep_recent | keep_best
        to_delete = [c for c in self._registry if c.step not in keep_steps]

        deleted_paths: list[str] = []
        for ckpt in to_delete:
            p = Path(ckpt.path)
            if p.exists():
                shutil.rmtree(p)
                logger.info("Pruned checkpoint step=%d at %s", ckpt.step, ckpt.path)
                deleted_paths.append(ckpt.path)

        # Update registry
        self._registry = [c for c in self._registry if c.step in keep_steps]
        self._save_registry()

        return deleted_paths

    def register_and_prune(
        self, step: int, metrics: dict[str, Any] | None = None
    ) -> list[str]:
        """Convenience: register a new checkpoint and immediately prune.

        Returns list of pruned checkpoint paths.
        """
        self.register(step, metrics)
        return self.prune()

    def summary(self) -> dict[str, Any]:
        """Return a summary of current checkpoint state."""
        registered = self.registered
        best = self.best

        return {
            "total_registered": len(registered),
            "keep_last": self._cfg.keep_last,
            "keep_best": self._cfg.keep_best,
            "max_checkpoints": self._cfg.keep_last + self._cfg.keep_best,
            "checkpoints": [
                {
                    "step": c.step,
                    "success_rate": c.success_rate,
                    "dv_efficiency": c.dv_efficiency,
                    "mean_reward": c.mean_reward,
                    "is_best": c.step in {b.step for b in best},
                }
                for c in registered
            ],
        }

    # ── Internal helpers ─────────────────────────────────────────────

    def _find_checkpoint_path(self, step: int) -> Path | None:
        """Find the checkpoint directory for a given step."""
        if not self._dir.exists():
            return None

        # Verl naming conventions
        candidates = [
            self._dir / f"global_step_{step}",
            self._dir / f"step_{step}",
            self._dir / f"checkpoint-{step}",
            self._dir / str(step),
        ]
        for candidate in candidates:
            if candidate.is_dir():
                return candidate

        # Fallback: scan for any directory containing the step number
        for p in sorted(self._dir.iterdir()):
            if p.is_dir() and str(step) in p.name:
                return p

        return None

    def _save_meta(self, meta: CheckpointMeta, ckpt_path: Path) -> None:
        """Save checkpoint metadata inside the checkpoint directory."""
        meta_file = ckpt_path / self._cfg.meta_filename
        try:
            meta_file.write_text(meta.model_dump_json(indent=2))
        except OSError:
            logger.warning("Could not write metadata to %s", meta_file)

    def _load_registry(self) -> None:
        """Load all checkpoint metadata from the checkpoint directory."""
        if not self._dir.exists():
            return

        for p in sorted(self._dir.iterdir()):
            if not p.is_dir():
                continue
            meta_file = p / self._cfg.meta_filename
            if meta_file.exists():
                try:
                    data = json.loads(meta_file.read_text())
                    self._registry.append(CheckpointMeta.model_validate(data))
                except Exception:
                    logger.warning("Skipping malformed metadata at %s", meta_file)

    def _save_registry(self) -> None:
        """Persist the registry index to a JSON file in the checkpoint root."""
        index_file = self._dir / "checkpoint_index.json"
        try:
            self._dir.mkdir(parents=True, exist_ok=True)
            entries = [c.model_dump() for c in self.registered]
            index_file.write_text(json.dumps(entries, indent=2, default=str))
        except OSError:
            logger.warning("Could not write checkpoint index to %s", index_file)


# ── Standalone watcher mode ──────────────────────────────────────────


def _watch_loop(checkpoint_dir: str, interval: int = 60) -> None:
    """Poll the checkpoint directory and prune periodically."""
    mgr = CheckpointManager(checkpoint_dir)
    logger.info(
        "Watching %s (keep_last=%d, keep_best=%d, poll=%ds)",
        checkpoint_dir,
        mgr._cfg.keep_last,
        mgr._cfg.keep_best,
        interval,
    )
    while True:
        try:
            mgr._load_registry()
            pruned = mgr.prune()
            if pruned:
                logger.info("Pruned %d checkpoints", len(pruned))
        except Exception:
            logger.exception("Error during checkpoint pruning")
        time.sleep(interval)


if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description="Loka Checkpoint Manager")
    parser.add_argument("--checkpoint-dir", required=True, help="Path to checkpoint directory")
    parser.add_argument("--keep-last", type=int, default=2, help="Keep N most recent (default: 2)")
    parser.add_argument("--keep-best", type=int, default=3, help="Keep K best by metric (default: 3)")
    parser.add_argument("--watch", action="store_true", help="Run as background watcher")
    parser.add_argument("--interval", type=int, default=60, help="Poll interval in seconds (default: 60)")
    parser.add_argument("--summary", action="store_true", help="Print current checkpoint summary and exit")

    args = parser.parse_args()

    if args.summary:
        mgr = CheckpointManager(
            args.checkpoint_dir,
            CheckpointConfig(keep_last=args.keep_last, keep_best=args.keep_best),
        )
        import pprint
        pprint.pprint(mgr.summary())
    elif args.watch:
        _watch_loop(args.checkpoint_dir, args.interval)
    else:
        mgr = CheckpointManager(
            args.checkpoint_dir,
            CheckpointConfig(keep_last=args.keep_last, keep_best=args.keep_best),
        )
        pruned = mgr.prune()
        print(f"Pruned {len(pruned)} checkpoints")
        for p in pruned:
            print(f"  - {p}")
