"""
Three-stage curriculum scheduler for orbital mechanics RL.

The curriculum progressively increases mission complexity:

Stage 1 — Circularisation (0–30% of training)
    Near-target orbits with mild eccentricity; teaches burn-phase alignment.

Stage 2 — Hohmann-like transfer (30–70%)
    Full LEO-to-GEO; both semi-major axis and eccentricity rewards active.

Stage 3 — Plane changes & perturbations (70–100%)
    Extended to 3-D with inclination; optional J2 drag (future).
"""


from pydantic import BaseModel, Field, model_validator

# ── Pydantic models for structured stage definitions ─────────────────


class StageConfig(BaseModel):
    """Physical parameters for a single curriculum stage."""

    a_range: tuple[float, float] = Field(..., description="Semi-major axis error range")
    e_range: tuple[float, float] = Field(..., description="Eccentricity range")
    incl_range: tuple[float, float] | None = Field(
        None, description="Inclination range in degrees (Stage 3 only)",
    )


class StageMix(BaseModel):
    """Sampling probabilities across curriculum stages (sum to 1)."""

    circularize: float = Field(..., ge=0.0, le=1.0)
    hohmann: float = Field(..., ge=0.0, le=1.0)
    plane_change: float = Field(..., ge=0.0, le=1.0)

    @model_validator(mode="after")
    def _check_sum(self) -> "StageMix":
        total = self.circularize + self.hohmann + self.plane_change
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"Mix must sum to 1.0, got {total:.6f}")
        return self


class CurriculumScheduler:
    """Return dataset mixing ratios for each training stage.

    Parameters
    ----------
    stage_boundaries : tuple[float, float]
        Two fractions in (0, 1) that delimit stages.  Default ``(0.30, 0.70)``.

    Examples
    --------
    >>> sched = CurriculumScheduler()
    >>> sched.get_mix(global_step=10, total_steps=1000)
    StageMix(circularize=0.7, hohmann=0.25, plane_change=0.05)
    """

    STAGES: dict[str, StageConfig] = {
        "circularize": StageConfig(a_range=(0.95, 1.05), e_range=(0.05, 0.15)),
        "hohmann": StageConfig(a_range=(0.16, 0.16), e_range=(0.0, 0.0)),
        "plane_change": StageConfig(
            a_range=(0.16, 0.16), e_range=(0.0, 0.0), incl_range=(20, 35),
        ),
    }

    # Pre-validated mixes (fail-fast at import time, not during training)
    _MIXES = {
        "circularize": StageMix(circularize=0.70, hohmann=0.25, plane_change=0.05),
        "hohmann": StageMix(circularize=0.15, hohmann=0.60, plane_change=0.25),
        "plane_change": StageMix(circularize=0.05, hohmann=0.25, plane_change=0.70),
    }

    def __init__(self, stage_boundaries: tuple[float, float] = (0.30, 0.70)):
        b1, b2 = stage_boundaries
        if not (0.0 < b1 < b2 < 1.0):
            raise ValueError(
                f"stage_boundaries must satisfy 0 < b1 < b2 < 1, got ({b1}, {b2})"
            )
        self.b1 = b1
        self.b2 = b2

    def get_mix(self, global_step: int, total_steps: int) -> StageMix:
        """Return mixing ratios for the current training progress.

        Parameters
        ----------
        global_step : int
            Current optimisation step.
        total_steps : int
            Total expected steps across all epochs.

        Returns
        -------
        StageMix
            Validated mixing probabilities that sum to 1.0.
        """
        stage = self.get_stage_name(global_step, total_steps)
        return self._MIXES[stage]

    def get_stage_name(self, global_step: int, total_steps: int) -> str:
        """Return the dominant stage label for the current step."""
        progress = global_step / max(total_steps, 1)
        if progress < self.b1:
            return "circularize"
        elif progress < self.b2:
            return "hohmann"
        return "plane_change"
