"""
Text-to-tensor bridge for LLM-based spacecraft control.

Converts environment observations into natural-language prompts and
parses LLM text outputs into continuous thrust actions.
"""

import json
import re
from typing import List, Dict, Tuple

import numpy as np

# ── System prompt (DeepSeek-R1 think/action paradigm) ─────────────────

SYSTEM_PROMPT = (
    "You are a spacecraft guidance controller performing a "
    "LEO-to-GEO orbital transfer. At each step you receive the spacecraft state "
    "and must output an optimal thrust command.\n"
    "\n"
    "## Output format (MANDATORY)\n"
    "<think>\n"
    "[Analyze current orbit. Which direction should thrust point? How much fuel "
    "remains? How close are we to target a and e?]\n"
    "</think>\n"
    "<action>\n"
    '{"thrust": 0.XX, "angle": XXX.X}\n'
    "</action>\n"
    "\n"
    "## Constraints\n"
    "- thrust: float in [0.0, 1.0] — fraction of max thrust\n"
    "- angle: float in [-180.0, 180.0] — degrees from velocity vector\n"
    "- Output ONLY valid JSON inside <action> tags\n"
    "- Output NOTHING after </action>\n"
    '- If unsure, output {"thrust": 0.0, "angle": 0.0}\n'
    "\n"
    "## Physics reference\n"
    "- Target: circular orbit at a=42164 km, e=0.0\n"
    "- Hohmann ΔV budget: ~3.94 km/s\n"
    "- Prograde burns (angle≈0°) raise orbit; retrograde burns (angle≈180°) lower it\n"
    "- Burn at apoapsis to raise periapsis; burn at periapsis to raise apoapsis"
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


# ── Action parser with cascading fallbacks ────────────────────────────


class ActionParser:
    """Parse LLM text output into ``(thrust_frac, angle_normalised)`` for the env.

    Four strategies are tried in order, each returning a decreasing
    format reward:

    1. XML ``<action>`` tag + JSON  (reward +1.0)
    2. Bare JSON anywhere           (reward +0.5)
    3. Regex field extraction        (reward +0.25)
    4. Default coast action          (reward -0.5)
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
        # Strategy 1: XML + JSON (best case)
        m = self._XML_RE.search(text)
        if m:
            action = self._try_json(m.group(1))
            if action is not None:
                return action, 1.0, "xml_json"

        # Strategy 2: bare JSON anywhere
        m = self._JSON_RE.search(text)
        if m:
            action = self._try_json(m.group(0))
            if action is not None:
                return action, 0.5, "bare_json"

        # Strategy 3: regex field extraction
        vals: Dict[str, float] = {}
        for key, pat in self._FIELD_RE.items():
            fm = pat.search(text)
            if fm:
                vals[key] = float(fm.group(1))
        if vals:
            t = np.clip(vals.get("thrust", 0.0), 0.0, 1.0)
            a = np.clip(vals.get("angle", 0.0), -180, 180) / 180.0
            return np.array([t, a], dtype=np.float32), 0.25, "regex"

        # Strategy 4: default (no thrust)
        return self.DEFAULT.copy(), -0.5, "fallback"

    # ── internal helpers ──────────────────────────────────────────────

    @staticmethod
    def _try_json(s: str):
        """Attempt lenient JSON parse, returning ``ndarray | None``.

        Returns ``None`` when the parsed value is not a dict (e.g. the
        model emits ``false``, ``null``, a bare number, or a list).
        """
        s = s.replace("'", '"')
        s = re.sub(r",\s*}", "}", s)
        try:
            d = json.loads(s)
            if not isinstance(d, dict):
                return None
            t = np.clip(float(d.get("thrust", 0)), 0.0, 1.0)
            a = np.clip(float(d.get("angle", 0)), -180, 180) / 180.0
            return np.array([t, a], dtype=np.float32)
        except (json.JSONDecodeError, TypeError, ValueError, AttributeError):
            return None
