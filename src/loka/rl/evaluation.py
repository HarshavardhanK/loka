"""
Generalisation and verification battery for orbital-mechanics agents.

Provides functions that test whether a trained agent has learned
transferable physics (novel altitudes, adversarial perturbations,
delta-V efficiency) rather than memorised trajectories.
"""

from typing import Any, Dict, List, Optional, Protocol

import numpy as np
from pydantic import BaseModel, Field

from loka.envs.orbital_transfer import OrbitalTransferEnv


# ── Pydantic result models ───────────────────────────────────────────


class AltitudeResult(BaseModel):
    """Evaluation result for a single altitude test condition."""

    success_rate: float = Field(..., ge=0.0, le=1.0)
    mean_dv_ratio: Optional[float] = Field(
        None, description="Mean dv_agent / dv_hohmann (None if no successes)",
    )


class GeneralizationResult(BaseModel):
    """Full result from :func:`evaluate_generalization`."""

    altitude_tests: Dict[str, AltitudeResult] = Field(
        default_factory=dict,
        description="Keyed by 'LEO_{alt}km'",
    )
    adversarial_perturbation: float = Field(
        ..., ge=0.0, le=1.0,
        description="Success rate under mid-episode velocity kicks",
    )


class EfficiencyResult(BaseModel):
    """Result from :func:`compute_dv_efficiency`."""

    eta_mean: float = Field(0.0, description="Mean dv_hohmann / dv_agent")
    eta_std: float = Field(0.0, description="Std of eta")
    success_rate: float = Field(0.0, ge=0.0, le=1.0)


# ── Agent protocol ───────────────────────────────────────────────────


class AgentProtocol(Protocol):
    """Minimal interface an agent must satisfy for evaluation."""

    def act(self, obs: np.ndarray) -> np.ndarray:
        """Return an action array given an observation."""
        ...


# ── Evaluation functions ─────────────────────────────────────────────


def evaluate_generalization(
    agent: AgentProtocol,
    env_class: type = OrbitalTransferEnv,
    n_episodes: int = 100,
    seed: int = 42,
) -> GeneralizationResult:
    """Run the full generalisation test battery.

    Parameters
    ----------
    agent : AgentProtocol
        Object with an ``act(obs) -> action`` method.
    env_class : type
        Gymnasium environment class (default: ``OrbitalTransferEnv``).
    n_episodes : int
        Episodes per test condition.
    seed : int
        Base random seed for reproducibility.

    Returns
    -------
    GeneralizationResult
        Structured results with altitude tests and adversarial perturbation.
    """
    rng = np.random.RandomState(seed)
    altitude_tests: Dict[str, AltitudeResult] = {}

    # ── Test 1: Novel altitudes ───────────────────────────────────────
    for alt in [300, 500, 600, 800]:
        env = env_class(config={"alt_leo": alt})
        successes: List[bool] = []
        dv_ratios: List[float] = []
        for ep in range(n_episodes):
            obs, _ = env.reset(seed=int(rng.randint(0, 2**31)))
            done = False
            while not done:
                action = agent.act(obs)
                obs, _r, term, trunc, info = env.step(action)
                done = term or trunc
            successes.append(info.get("success", False))
            if info.get("dv_hohmann", 0) > 0:
                dv_ratios.append(info["dv_total"] / info["dv_hohmann"])
        altitude_tests[f"LEO_{alt}km"] = AltitudeResult(
            success_rate=float(np.mean(successes)),
            mean_dv_ratio=float(np.mean(dv_ratios)) if dv_ratios else None,
        )

    # ── Test 2: Adversarial perturbation ──────────────────────────────
    env = env_class()
    perturb_successes: List[bool] = []
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=int(rng.randint(0, 2**31)))
        done, step = False, 0
        while not done:
            action = agent.act(obs)
            obs, _r, term, trunc, info = env.step(action)
            step += 1
            # Inject velocity perturbation at step 100
            if step == 100:
                env.state[2] += rng.normal(0, 0.1)  # vx kick
                env.state[3] += rng.normal(0, 0.1)  # vy kick
            done = term or trunc
        perturb_successes.append(info.get("success", False))

    return GeneralizationResult(
        altitude_tests=altitude_tests,
        adversarial_perturbation=float(np.mean(perturb_successes)),
    )


def compute_dv_efficiency(
    agent: AgentProtocol, n_episodes: int = 50,
) -> EfficiencyResult:
    """Compute delta-V efficiency statistics.

    Returns
    -------
    EfficiencyResult
        ``eta_mean``, ``eta_std``, ``success_rate`` where
        ``eta = dv_hohmann / dv_agent``.
    """
    env = OrbitalTransferEnv()
    etas: List[float] = []
    successes: List[bool] = []
    for _ in range(n_episodes):
        obs, _ = env.reset()
        done = False
        while not done:
            action = agent.act(obs)
            obs, _r, term, trunc, info = env.step(action)
            done = term or trunc
        success = info.get("success", False)
        successes.append(success)
        if success and info["dv_total"] > 0:
            etas.append(info["dv_hohmann"] / info["dv_total"])

    return EfficiencyResult(
        eta_mean=float(np.mean(etas)) if etas else 0.0,
        eta_std=float(np.std(etas)) if etas else 0.0,
        success_rate=float(np.mean(successes)),
    )
