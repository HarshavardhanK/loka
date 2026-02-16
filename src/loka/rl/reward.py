"""
Verl-compatible reward function for orbital mechanics GRPO training.

This module is referenced via ``custom_reward_function.path`` in the Verl
launch config.  Verl calls :func:`compute_score` after each rollout to
score the LLM's text response.
"""

import re

import numpy as np
from pydantic import BaseModel, ValidationError

from loka.rl.bridge import ActionParser


# ── Pydantic model for ground-truth payload ──────────────────────────

class _GroundTruth(BaseModel):
    """Schema for the ``ground_truth`` field in Verl reward_model data."""
    initial_state: list[float] = []
    a_target: float = 42164.0
    e_target: float = 0.0
    dv_hohmann: float = 3.94
    max_steps: int = 10000


# ── Module-level singletons (avoid re-instantiation per call) ────────

_PARSER = ActionParser()

# Compiled regexes for format scoring
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)
_ACTION_RE = re.compile(r"<action>.*?</action>", re.DOTALL)
_TRAILING_RE = re.compile(r"</action>\s*\S")


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: dict = None,
) -> float:
    """Score a single LLM response for the orbital-transfer task.

    The reward is composed of two independent components so that format
    compliance can bootstrap physics learning during early training.

    Parameters
    ----------
    data_source : str
        Dataset identifier (unused but required by Verl API).
    solution_str : str
        Full LLM response containing ``<think>`` and ``<action>`` blocks.
    ground_truth : str
        JSON string with initial state, target parameters, and env config.
    extra_info : dict, optional
        May contain ``"mission_reward"`` from environment simulation.

    Returns
    -------
    float
        Scalar reward in ``[-1, 1]``.
    """
    # Validate ground truth via Pydantic (accepts str or dict)
    if isinstance(ground_truth, str):
        try:
            _gt = _GroundTruth.model_validate_json(ground_truth)
        except ValidationError:
            _gt = _GroundTruth()
    elif isinstance(ground_truth, dict):
        try:
            _gt = _GroundTruth.model_validate(ground_truth)
        except ValidationError:
            _gt = _GroundTruth()
    else:
        _gt = _GroundTruth()

    reward = 0.0

    # ── Format compliance (0.0 – 0.2) ────────────────────────────────
    has_think = bool(_THINK_RE.search(solution_str))
    has_action = bool(_ACTION_RE.search(solution_str))
    has_trailing = bool(_TRAILING_RE.search(solution_str))
    format_score = 0.05 * has_think + 0.10 * has_action - 0.05 * has_trailing

    _action_arr, _parse_reward, method = _PARSER.parse(solution_str)
    if method == "xml_json":
        format_score += 0.05
    reward += max(0.0, format_score)

    # ── Physics simulation (0.0 – 0.8) ───────────────────────────────
    if extra_info and "mission_reward" in extra_info:
        mission_r = float(extra_info["mission_reward"])
        reward += 0.8 * np.clip(mission_r / 160.0, -1.0, 1.0)

    return float(np.clip(reward, -1.0, 1.0))
