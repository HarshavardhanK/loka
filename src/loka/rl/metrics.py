"""
Domain-specific metric tracking for orbital-mechanics GRPO training.

This module provides a :class:`MetricsTracker` that accumulates per-sample
statistics from the reward function and flushes aggregated summaries to
Weights & Biases at configurable intervals.

Verl's built-in wandb logger already tracks:
    - train/policy_loss, train/kl_divergence, train/entropy
    - train/mean_reward, train/gradient_norm, train/learning_rate
    - train/clip_fraction, train/response_length

This module adds **Loka-specific** metrics on top:
    - Reward decomposition (format vs physics)
    - Orbital mechanics (success rate, ΔV efficiency, final elements)
    - Action parsing quality (XML, JSON, regex, fallback rates)
    - Curriculum stage tracking
    - Automatic checkpoint pruning (hybrid last-N + best-K)
"""

from __future__ import annotations

import logging
import time
import threading
from collections import defaultdict
from typing import Any, Optional, TYPE_CHECKING

import numpy as np
from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from loka.rl.checkpoint import CheckpointManager

logger = logging.getLogger(__name__)


# ── Pydantic schema for a single sample's metrics ───────────────────


class SampleMetrics(BaseModel):
    """Metrics collected from a single reward computation."""

    # Reward decomposition
    total_reward: float = 0.0
    format_reward: float = 0.0
    physics_reward: float = 0.0

    # Parsing method
    parse_method: str = "fallback"  # xml_json | bare_json | regex | fallback

    # Orbital mechanics (from extra_info)
    success: Optional[bool] = None
    final_a_km: Optional[float] = None
    final_e: Optional[float] = None
    dv_total_kms: Optional[float] = None
    dv_hohmann_kms: Optional[float] = None
    mass_ratio: Optional[float] = None
    steps_used: Optional[int] = None

    # Response stats
    response_length: int = 0
    has_think_tag: bool = False
    has_action_tag: bool = False


# ── Thread-safe metrics accumulator ──────────────────────────────────


class MetricsTracker:
    """Accumulates per-sample metrics and flushes to wandb in batches.

    Optionally manages checkpoint pruning via a :class:`CheckpointManager`.

    Parameters
    ----------
    flush_every : int
        Flush to wandb after this many samples.  Default ``256``
        (one full GRPO batch).
    enabled : bool
        If ``False``, all operations are no-ops (for testing / dry runs).
    checkpoint_manager : CheckpointManager, optional
        If provided, each flush will also register the current step's
        metrics with the checkpoint manager and prune old checkpoints.
    save_freq : int, optional
        How often Verl saves checkpoints (in training steps).  Used to
        detect whether the current flush aligns with a checkpoint save.
        Default ``50`` (matching ``trainer.save_freq``).
    """

    def __init__(
        self,
        flush_every: int = 256,
        enabled: bool = True,
        checkpoint_manager: Optional[CheckpointManager] = None,
        save_freq: int = 50,
    ):
        self._flush_every = flush_every
        self._enabled = enabled
        self._buffer: list[SampleMetrics] = []
        self._lock = threading.Lock()
        self._step = 0
        self._total_samples = 0
        self._start_time = time.monotonic()
        self._wandb = None  # lazy import
        self._ckpt_mgr = checkpoint_manager
        self._save_freq = save_freq

    # ── Public API ───────────────────────────────────────────────────

    def record(self, m: SampleMetrics) -> None:
        """Record a single sample's metrics. Thread-safe."""
        if not self._enabled:
            return
        with self._lock:
            self._buffer.append(m)
            if len(self._buffer) >= self._flush_every:
                self._flush()

    def flush(self) -> None:
        """Force a flush of accumulated metrics."""
        if not self._enabled:
            return
        with self._lock:
            self._flush()

    def get_summary(self) -> dict[str, Any]:
        """Return current accumulated summary without flushing."""
        with self._lock:
            if not self._buffer:
                return {}
            return self._aggregate(self._buffer)

    # ── Internal ─────────────────────────────────────────────────────

    def _flush(self) -> None:
        """Aggregate and log to wandb. Caller must hold the lock."""
        if not self._buffer:
            return

        summary = self._aggregate(self._buffer)
        self._total_samples += len(self._buffer)
        self._step += 1
        self._buffer.clear()

        # Lazy wandb import — only when actually flushing
        if self._wandb is None:
            try:
                import wandb as _wb
                self._wandb = _wb
            except ImportError:
                self._enabled = False
                return

        if self._wandb.run is None:
            # wandb not initialized yet (e.g., during data generation)
            return

        # Add step-level metadata
        summary["loka/step"] = self._step
        summary["loka/total_samples"] = self._total_samples
        summary["loka/throughput_samples_per_min"] = (
            self._total_samples / max(1e-6, (time.monotonic() - self._start_time) / 60)
        )

        self._wandb.log(summary)

        # ── Checkpoint management ────────────────────────────────────
        if self._ckpt_mgr and self._step % self._save_freq == 0:
            try:
                pruned = self._ckpt_mgr.register_and_prune(
                    step=self._step, metrics=summary,
                )
                if pruned:
                    logger.info("Pruned %d checkpoints: %s", len(pruned), pruned)
            except Exception:
                logger.warning("Checkpoint pruning failed", exc_info=True)

    @staticmethod
    def _aggregate(buffer: list[SampleMetrics]) -> dict[str, Any]:
        """Compute summary statistics from a batch of samples."""
        n = len(buffer)
        summary: dict[str, Any] = {}

        # ── Reward decomposition ─────────────────────────────────────
        totals = [m.total_reward for m in buffer]
        formats = [m.format_reward for m in buffer]
        physics = [m.physics_reward for m in buffer]

        summary["loka/reward/total_mean"] = float(np.mean(totals))
        summary["loka/reward/total_std"] = float(np.std(totals))
        summary["loka/reward/total_min"] = float(np.min(totals))
        summary["loka/reward/total_max"] = float(np.max(totals))
        summary["loka/reward/format_mean"] = float(np.mean(formats))
        summary["loka/reward/physics_mean"] = float(np.mean(physics))

        # ── Parse method distribution ────────────────────────────────
        method_counts: dict[str, int] = defaultdict(int)
        for m in buffer:
            method_counts[m.parse_method] += 1
        for method in ["xml_json", "bare_json", "regex", "fallback"]:
            summary[f"loka/parse/{method}_rate"] = method_counts.get(method, 0) / n

        # ── Format compliance ────────────────────────────────────────
        summary["loka/format/think_rate"] = sum(1 for m in buffer if m.has_think_tag) / n
        summary["loka/format/action_rate"] = sum(1 for m in buffer if m.has_action_tag) / n
        summary["loka/format/perfect_rate"] = (
            sum(1 for m in buffer if m.has_think_tag and m.parse_method == "xml_json") / n
        )

        # ── Orbital mechanics ────────────────────────────────────────
        successes = [m for m in buffer if m.success is not None]
        if successes:
            summary["loka/orbital/success_rate"] = (
                sum(1 for m in successes if m.success) / len(successes)
            )

        # ΔV efficiency (only for successful episodes)
        dvs = [m for m in buffer if m.success and m.dv_total_kms and m.dv_hohmann_kms]
        if dvs:
            efficiencies = [m.dv_hohmann_kms / m.dv_total_kms for m in dvs]
            summary["loka/orbital/dv_efficiency_mean"] = float(np.mean(efficiencies))
            summary["loka/orbital/dv_efficiency_std"] = float(np.std(efficiencies))
            summary["loka/orbital/dv_total_mean_kms"] = float(
                np.mean([m.dv_total_kms for m in dvs])
            )

        # Final orbital elements
        elements = [m for m in buffer if m.final_a_km is not None]
        if elements:
            summary["loka/orbital/final_a_mean_km"] = float(
                np.mean([m.final_a_km for m in elements])
            )
            summary["loka/orbital/final_e_mean"] = float(
                np.mean([m.final_e for m in elements if m.final_e is not None])
            )

        # Mass ratio (fuel remaining)
        masses = [m.mass_ratio for m in buffer if m.mass_ratio is not None]
        if masses:
            summary["loka/orbital/mass_ratio_mean"] = float(np.mean(masses))

        # Episode length
        steps = [m.steps_used for m in buffer if m.steps_used is not None]
        if steps:
            summary["loka/orbital/steps_mean"] = float(np.mean(steps))
            summary["loka/orbital/steps_max"] = int(np.max(steps))

        # ── Response stats ───────────────────────────────────────────
        lengths = [m.response_length for m in buffer]
        summary["loka/response/length_mean"] = float(np.mean(lengths))
        summary["loka/response/length_max"] = int(np.max(lengths))

        return summary


# ── Module-level singleton ───────────────────────────────────────────

_tracker: Optional[MetricsTracker] = None


def get_tracker(
    flush_every: int = 256,
    checkpoint_manager: Optional[CheckpointManager] = None,
    save_freq: int = 50,
) -> MetricsTracker:
    """Get or create the global MetricsTracker singleton.

    Parameters
    ----------
    flush_every : int
        Flush to wandb after this many samples.
    checkpoint_manager : CheckpointManager, optional
        If provided on first call, enables automatic checkpoint pruning.
    save_freq : int
        How often Verl saves checkpoints (training steps).
    """
    global _tracker
    if _tracker is None:
        _tracker = MetricsTracker(
            flush_every=flush_every,
            checkpoint_manager=checkpoint_manager,
            save_freq=save_freq,
        )
    return _tracker


def init_wandb_run(
    project: str = "orbital_rl",
    experiment: str = "leo_to_geo_grpo",
    config: Optional[dict] = None,
    tags: Optional[list[str]] = None,
) -> None:
    """Initialize a wandb run with Loka-specific metadata.

    This should be called ONCE at the start of training (before Verl
    initializes its own wandb, which will attach to this run).

    Parameters
    ----------
    project : str
        W&B project name.
    experiment : str
        Run name / experiment name.
    config : dict, optional
        Hyperparameter dict to log (auto-captured from Verl config).
    tags : list[str], optional
        Tags for filtering runs (e.g., ``["grpo", "stage-1", "2-node"]``).
    """
    try:
        import wandb
    except ImportError:
        return

    if wandb.run is not None:
        return  # already initialized

    run_config = {
        "framework": "verl",
        "algorithm": "grpo",
        "task": "leo_to_geo_transfer",
        **(config or {}),
    }

    wandb.init(
        project=project,
        name=experiment,
        config=run_config,
        tags=tags or ["grpo", "orbital-transfer"],
        save_code=True,
        # Let Verl's logger attach to this run
        reinit=False,
    )

    # Define custom x-axis for loka metrics
    wandb.define_metric("loka/*", step_metric="loka/step")
