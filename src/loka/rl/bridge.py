"""
Text-to-tensor bridge for LLM-based spacecraft control.

Converts environment observations into natural-language prompts and
parses LLM text outputs into continuous thrust actions.

Uses Pydantic v2 for schema validation so the parser never crashes on
malformed model outputs (booleans, nulls, lists, wrong types, etc.).
"""

import re
from typing import List, Dict, Tuple

import numpy as np
from pydantic import BaseModel, field_validator, ValidationError

# ── System prompt (DeepSeek-R1 think/action paradigm) ─────────────────

SYSTEM_PROMPT = (
    "You are a spacecraft guidance controller. At each step you receive "
    "the current orbital state and must output a thrust command.\n"
    "\n"
    "## Output format\n"
    "<think>\n"
    "Reason about the current state and decide your next action.\n"
    "</think>\n"
    "<action>\n"
    '{"thrust": <float 0.0–1.0>, "angle": <float -180.0–180.0>}\n'
    "</action>\n"
    "\n"
    "## Rules\n"
    "- thrust: fraction of max thrust (0 = coast, 1 = full)\n"
    "- angle: degrees relative to velocity vector\n"
    "- Output ONLY valid JSON inside <action> tags\n"
    "- Output NOTHING after </action>\n"
    '- Default safe action: {"thrust": 0.0, "angle": 0.0}'
)


# ── Observation wrapper ───────────────────────────────────────────────


class OrbitalObservationWrapper:
    """Convert 8-dim normalised observation to a compact text prompt.

    The wrapper produces ~50-token observations suitable for LLM
    consumption and can build full chat-format message lists for
    Verl tokenisation.

    Parameters
    ----------
    a_target : float
        Target semi-major axis in km (default: GEO at 42 164 km).
    dv_hohmann : float
        Hohmann delta-V baseline in km/s.
    """

    def __init__(self, a_target: float = 42164.0, dv_hohmann: float = 3.94):
        self.a_target = a_target
        self.dv_hohmann = dv_hohmann

    def state_to_text(
        self,
        obs: np.ndarray,
        step: int,
        max_steps: int,
        dv_used: float = 0.0,
    ) -> str:
        """Convert a normalised observation array to natural language.

        Parameters
        ----------
        obs : ndarray
            8-dim observation from ``OrbitalTransferEnv``.
        step : int
            Current environment step.
        max_steps : int
            Maximum steps in the episode.
        dv_used : float
            Cumulative delta-V used so far (km/s).

        Returns
        -------
        str
            Human-readable state description (~50 tokens).
        """
        a_km = obs[5] * self.a_target
        e = obs[6]
        fuel = obs[4]
        r_km = np.sqrt((obs[0] * 42164) ** 2 + (obs[1] * 42164) ** 2)
        v_kms = np.sqrt((obs[2] * 7.67) ** 2 + (obs[3] * 7.67) ** 2)

        # Compute velocity vector angle for thrust reference
        vx_raw, vy_raw = obs[2], obs[3]
        vel_angle = np.degrees(np.arctan2(vy_raw, vx_raw))

        return (
            f"Step {step}/{max_steps} | Fuel: {fuel:.1%} | "
            f"\u0394V used: {dv_used:.3f}/{self.dv_hohmann:.3f} km/s\n"
            f"a={a_km:.1f} km (target: {self.a_target:.0f}) | "
            f"e={e:.4f} (target: 0.0)\n"
            f"r={r_km:.1f} km | v={v_kms:.3f} km/s | "
            f"vel_heading={vel_angle:.1f}\u00b0\n"
            f"\u0394a/a_target={(obs[5] - 1.0):+.4f} | \u0394e={e:+.4f}"
        )

    def build_messages(
        self,
        obs: np.ndarray,
        step: int,
        max_steps: int,
        dv_used: float = 0.0,
    ) -> List[Dict[str, str]]:
        """Build chat-format message list for Verl tokenisation.

        Parameters
        ----------
        obs : ndarray
            8-dim observation from ``OrbitalTransferEnv``.
        step, max_steps, dv_used :
            Forwarded to :meth:`state_to_text`.

        Returns
        -------
        list[dict]
            ``[{"role": "system", ...}, {"role": "user", ...}]``
        """
        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": self.state_to_text(
                obs, step, max_steps, dv_used,
            )},
        ]


# ── Pydantic model for thrust commands ─────────────────────────────


class ThrustCommand(BaseModel):
    """Validated thrust command parsed from LLM output.

    Pydantic handles all edge cases automatically:
    - Rejects non-object JSON (``false``, ``null``, lists, bare numbers)
    - Coerces numeric strings (``"0.5"`` → ``0.5``)
    - Applies defaults when fields are missing
    - Clamps values via field validators
    """

    thrust: float = 0.0
    angle: float = 0.0

    @field_validator("thrust")
    @classmethod
    def clamp_thrust(cls, v: float) -> float:
        return float(np.clip(v, 0.0, 1.0))

    @field_validator("angle")
    @classmethod
    def clamp_angle(cls, v: float) -> float:
        return float(np.clip(v, -180.0, 180.0))

    def to_action(self) -> np.ndarray:
        """Convert to ``[thrust_frac, angle_normalised]`` for ``env.step()``."""
        return np.array(
            [self.thrust, self.angle / 180.0], dtype=np.float32,
        )


# ── Action parser with cascading fallbacks ────────────────────────────


class ActionParser:
    """Parse LLM text output into ``(thrust_frac, angle_normalised)`` for the env.

    Four strategies are tried in order, each returning a decreasing
    format reward:

    1. XML ``<action>`` tag + validated JSON  (reward +1.0)
    2. Bare JSON anywhere                     (reward +0.5)
    3. Regex field extraction                 (reward +0.25)
    4. Default coast action                   (reward -0.5)

    JSON parsing and validation is handled entirely by
    :class:`ThrustCommand` (Pydantic v2), which rejects non-dict
    payloads, coerces types, fills defaults, and clamps ranges.
    """

    _XML_RE = re.compile(r"<action>\s*(.*?)\s*</action>", re.DOTALL | re.I)
    _JSON_RE = re.compile(
        r'\{[^{}]*"thrust"\s*:\s*[\d.eE+-]+[^{}]*\}', re.DOTALL,
    )
    _FIELD_RE = {
        "thrust": re.compile(r'"thrust"\s*:\s*([-+]?\d*\.?\d+)'),
        "angle": re.compile(r'"angle"\s*:\s*([-+]?\d*\.?\d+)'),
    }
    DEFAULT = np.array([0.0, 0.0], dtype=np.float32)

    def parse(self, text: str) -> Tuple[np.ndarray, float, str]:
        """Parse *text* and return ``(action, format_reward, method)``.

        Parameters
        ----------
        text : str
            Full LLM response (may contain ``<think>`` and ``<action>`` tags).

        Returns
        -------
        action : ndarray
            ``[thrust_frac, angle_normalised]`` ready for ``env.step()``.
        format_reward : float
            Score in ``[-0.5, 1.0]`` reflecting output quality.
        method : str
            One of ``"xml_json"``, ``"bare_json"``, ``"regex"``, ``"fallback"``.
        """
        # Strategy 1: XML + validated JSON (best case)
        m = self._XML_RE.search(text)
        if m:
            cmd = self._validate(m.group(1))
            if cmd is not None:
                return cmd.to_action(), 1.0, "xml_json"

        # Strategy 2: bare JSON anywhere
        m = self._JSON_RE.search(text)
        if m:
            cmd = self._validate(m.group(0))
            if cmd is not None:
                return cmd.to_action(), 0.5, "bare_json"

        # Strategy 3: regex field extraction → pydantic validation
        vals: Dict[str, float] = {}
        for key, pat in self._FIELD_RE.items():
            fm = pat.search(text)
            if fm:
                vals[key] = float(fm.group(1))
        if vals:
            cmd = ThrustCommand(**vals)
            return cmd.to_action(), 0.25, "regex"

        # Strategy 4: default (no thrust)
        return self.DEFAULT.copy(), -0.5, "fallback"

    # ── internal helpers ──────────────────────────────────────────────

    @staticmethod
    def _validate(s: str) -> ThrustCommand | None:
        """Parse and validate a JSON string via Pydantic.

        Handles all LLM quirks: single quotes, trailing commas,
        non-dict payloads (``false``, ``null``, lists), wrong types, etc.
        Returns ``None`` on any validation failure.
        """
        s = s.replace("'", '"')
        s = re.sub(r",\s*}", "}", s)
        try:
            return ThrustCommand.model_validate_json(s)
        except (ValidationError, ValueError):
            return None
