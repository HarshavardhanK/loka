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
    - Termination analysis (why episodes end)
    - GRPO group-level variance (rollout diversity)
    - Action distribution (thrust/coast patterns)
    - Curriculum stage tracking
    - Episode trajectory tables for W&B
    - Automatic checkpoint pruning (hybrid last-N + best-K)
"""

from __future__ import annotations

import logging
import threading
import time
from collections import defaultdict
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from loka.rl.checkpoint import CheckpointManager

logger = logging.getLogger(__name__)


TerminationReason = Literal["success", "crash", "fuel_exhausted", "timeout", "excessive_dv", "unknown"]


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
    success: bool | None = None
    final_a_km: float | None = None
    final_e: float | None = None
    dv_total_kms: float | None = None
    dv_hohmann_kms: float | None = None
    mass_ratio: float | None = None
    steps_used: int | None = None

    # Response stats
    response_length: int = 0
    has_think_tag: bool = False
    has_action_tag: bool = False

    # Termination analysis
    termination_reason: TerminationReason = "unknown"

    # Action distribution (from parsed action)
    thrust_frac: float | None = Field(None, ge=0.0, le=1.0)
    angle_norm: float | None = Field(None, ge=-1.0, le=1.0)

    # GRPO group tracking
    group_id: str | None = None

    # Curriculum
    curriculum_stage: str | None = None


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
    ema_alpha : float
        Smoothing factor for exponential moving averages.  Default ``0.1``.
    log_episodes : bool
        Whether to log per-episode data to W&B Tables.  Default ``True``.
    episode_table_limit : int
        Max rows per flush in the episode table.  Default ``64``.
    """

    def __init__(
        self,
        flush_every: int = 256,
        enabled: bool = True,
        checkpoint_manager: CheckpointManager | None = None,
        save_freq: int = 50,
        ema_alpha: float = 0.1,
        log_episodes: bool = True,
        episode_table_limit: int = 64,
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

        # EMA state
        self._ema_alpha = ema_alpha
        self._ema: dict[str, float] = {}

        # Episode table config
        self._log_episodes = log_episodes
        self._episode_table_limit = episode_table_limit

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

    def _update_ema(self, key: str, value: float) -> float:
        """Update and return the exponential moving average for *key*."""
        if key in self._ema:
            self._ema[key] = self._ema_alpha * value + (1 - self._ema_alpha) * self._ema[key]
        else:
            self._ema[key] = value
        return self._ema[key]

    def _flush(self) -> None:
        """Aggregate and log to wandb. Caller must hold the lock."""
        if not self._buffer:
            return

        summary = self._aggregate(self._buffer)
        batch = list(self._buffer)
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
            return

        # Add step-level metadata
        summary["loka/step"] = self._step
        summary["loka/total_samples"] = self._total_samples
        summary["loka/throughput_samples_per_min"] = (
            self._total_samples / max(1e-6, (time.monotonic() - self._start_time) / 60)
        )

        # EMA trend lines for key convergence indicators
        for ema_key in [
            "loka/reward/total_mean",
            "loka/orbital/success_rate",
            "loka/format/perfect_rate",
        ]:
            if ema_key in summary:
                ema_val = self._update_ema(ema_key, summary[ema_key])
                summary[f"{ema_key}_ema"] = ema_val

        self._wandb.log(summary)

        # Log episode-level W&B Table
        if self._log_episodes:
            self._log_episode_table(batch)

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

    def _log_episode_table(self, batch: list[SampleMetrics]) -> None:
        """Log a W&B Table with per-episode data for drill-down analysis."""
        if self._wandb is None or self._wandb.run is None:
            return

        columns = [
            "step", "reward", "format_reward", "physics_reward",
            "success", "termination", "parse_method",
            "thrust", "angle_norm",
            "final_a_km", "final_e", "dv_total", "dv_efficiency",
            "mass_ratio", "steps_used", "response_len",
            "curriculum_stage", "group_id",
        ]
        table = self._wandb.Table(columns=columns)

        rows = batch[: self._episode_table_limit]
        for m in rows:
            dv_eff = None
            if m.dv_total_kms and m.dv_hohmann_kms and m.dv_total_kms > 0:
                dv_eff = round(m.dv_hohmann_kms / m.dv_total_kms, 4)
            table.add_data(
                self._step,
                round(m.total_reward, 4),
                round(m.format_reward, 4),
                round(m.physics_reward, 4),
                m.success,
                m.termination_reason,
                m.parse_method,
                round(m.thrust_frac, 4) if m.thrust_frac is not None else None,
                round(m.angle_norm, 4) if m.angle_norm is not None else None,
                round(m.final_a_km, 2) if m.final_a_km is not None else None,
                round(m.final_e, 6) if m.final_e is not None else None,
                round(m.dv_total_kms, 4) if m.dv_total_kms is not None else None,
                dv_eff,
                round(m.mass_ratio, 4) if m.mass_ratio is not None else None,
                m.steps_used,
                m.response_length,
                m.curriculum_stage,
                m.group_id,
            )

        self._wandb.log({"loka/episodes": table})

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
        summary["loka/reward/physics_std"] = float(np.std(physics))
        summary["loka/reward/total_median"] = float(np.median(totals))

        # ── GRPO group variance ──────────────────────────────────────
        groups: dict[str, list[float]] = defaultdict(list)
        for m in buffer:
            if m.group_id is not None:
                groups[m.group_id].append(m.total_reward)

        if groups:
            group_means = [float(np.mean(v)) for v in groups.values()]
            group_stds = [float(np.std(v)) for v in groups.values() if len(v) > 1]
            group_ranges = [max(v) - min(v) for v in groups.values() if len(v) > 1]
            summary["loka/grpo/n_groups"] = len(groups)
            summary["loka/grpo/group_reward_mean"] = float(np.mean(group_means))
            summary["loka/grpo/group_reward_std_of_means"] = float(np.std(group_means))
            if group_stds:
                summary["loka/grpo/intra_group_std_mean"] = float(np.mean(group_stds))
                summary["loka/grpo/intra_group_std_max"] = float(np.max(group_stds))
            if group_ranges:
                summary["loka/grpo/intra_group_range_mean"] = float(np.mean(group_ranges))

            # Per-group best/worst spread (GRPO needs diversity)
            group_maxes = [max(v) for v in groups.values()]
            group_mins = [min(v) for v in groups.values()]
            summary["loka/grpo/best_in_group_mean"] = float(np.mean(group_maxes))
            summary["loka/grpo/worst_in_group_mean"] = float(np.mean(group_mins))

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

        # ── Termination analysis ─────────────────────────────────────
        term_counts: dict[str, int] = defaultdict(int)
        for m in buffer:
            term_counts[m.termination_reason] += 1
        for reason in ["success", "crash", "fuel_exhausted", "timeout", "excessive_dv", "unknown"]:
            summary[f"loka/termination/{reason}_rate"] = term_counts.get(reason, 0) / n

        # ── Action distribution ──────────────────────────────────────
        thrusts = [m.thrust_frac for m in buffer if m.thrust_frac is not None]
        if thrusts:
            summary["loka/action/thrust_mean"] = float(np.mean(thrusts))
            summary["loka/action/thrust_std"] = float(np.std(thrusts))
            summary["loka/action/coast_rate"] = sum(1 for t in thrusts if t < 0.01) / len(thrusts)
            summary["loka/action/full_thrust_rate"] = sum(1 for t in thrusts if t > 0.99) / len(thrusts)

        angles = [m.angle_norm for m in buffer if m.angle_norm is not None]
        if angles:
            summary["loka/action/angle_mean"] = float(np.mean(angles))
            summary["loka/action/angle_std"] = float(np.std(angles))

        # ── Curriculum stage breakdown ───────────────────────────────
        stage_counts: dict[str, list[SampleMetrics]] = defaultdict(list)
        for m in buffer:
            if m.curriculum_stage:
                stage_counts[m.curriculum_stage].append(m)

        for stage, samples in stage_counts.items():
            stage_n = len(samples)
            summary[f"loka/curriculum/{stage}_frac"] = stage_n / n
            with_success = [s for s in samples if s.success is not None]
            if with_success:
                summary[f"loka/curriculum/{stage}_success_rate"] = (
                    sum(1 for s in with_success if s.success) / len(with_success)
                )
            stage_rewards = [s.total_reward for s in samples]
            summary[f"loka/curriculum/{stage}_reward_mean"] = float(np.mean(stage_rewards))

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

_tracker: MetricsTracker | None = None


def get_tracker(
    flush_every: int = 256,
    checkpoint_manager: CheckpointManager | None = None,
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
    config: dict | None = None,
    tags: list[str] | None = None,
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
        reinit=False,
    )

    # Define custom x-axis for loka metrics
    wandb.define_metric("loka/*", step_metric="loka/step")

    # Define summary metrics so W&B picks the right aggregation
    wandb.define_metric("loka/orbital/success_rate", summary="max")
    wandb.define_metric("loka/orbital/dv_efficiency_mean", summary="max")
    wandb.define_metric("loka/reward/total_mean", summary="max")
    wandb.define_metric("loka/reward/total_mean_ema", summary="last")
    wandb.define_metric("loka/orbital/success_rate_ema", summary="last")
