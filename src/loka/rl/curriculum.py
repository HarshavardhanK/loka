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

from typing import Dict


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
    {'circularize': 0.70, 'hohmann': 0.25, 'plane_change': 0.05}
    """

    STAGES = {
        "circularize": {"a_range": (0.95, 1.05), "e_range": (0.05, 0.15)},
        "hohmann": {"a_range": (0.16, 0.16), "e_range": (0.0, 0.0)},
        "plane_change": {
            "a_range": (0.16, 0.16),
            "e_range": (0.0, 0.0),
            "incl_range": (20, 35),
        },
    }

    def __init__(self, stage_boundaries: tuple = (0.30, 0.70)):
        self.b1, self.b2 = stage_boundaries

    def get_mix(self, global_step: int, total_steps: int) -> Dict[str, float]:
        """Return mixing ratios for the current training progress.

        Parameters
        ----------
        global_step : int
            Current optimisation step.
        total_steps : int
            Total expected steps across all epochs.

        Returns
        -------
        dict[str, float]
            Mapping from stage name to sampling probability (sums to 1).
        """
        progress = global_step / max(total_steps, 1)
        if progress < self.b1:
            return {"circularize": 0.70, "hohmann": 0.25, "plane_change": 0.05}
        elif progress < self.b2:
            return {"circularize": 0.15, "hohmann": 0.60, "plane_change": 0.25}
        else:
            return {"circularize": 0.05, "hohmann": 0.25, "plane_change": 0.70}

    def get_stage_name(self, global_step: int, total_steps: int) -> str:
        """Return the dominant stage label for the current step."""
        progress = global_step / max(total_steps, 1)
        if progress < self.b1:
            return "circularize"
        elif progress < self.b2:
            return "hohmann"
        return "plane_change"
