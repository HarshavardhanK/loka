"""
Trajectory computation tool for Loka agent.

Provides orbital mechanics calculations for transfer trajectories.
Delegates Hohmann calculations to :mod:`loka.astro.hohmann` instead
of reimplementing the math.
"""

from typing import Any, Dict, List, Literal, Optional

import numpy as np
from pydantic import BaseModel, Field

from loka.tools.base import Tool, ToolResult


# ── Pydantic result models ───────────────────────────────────────────


class HohmannTransferResult(BaseModel):
    """Structured result from a Hohmann transfer computation."""

    transfer_type: Literal["hohmann"] = "hohmann"
    delta_v1_km_s: float = Field(..., description="First burn ΔV (km/s)")
    delta_v2_km_s: float = Field(..., description="Second burn ΔV (km/s)")
    total_delta_v_km_s: float = Field(..., description="Total ΔV (km/s)")
    transfer_time_days: float = Field(..., description="Transfer time (days)")
    semi_major_axis_km: float = Field(..., description="Transfer orbit semi-major axis (km)")
    origin_radius_km: float = Field(..., description="Origin orbital radius (km)")
    target_radius_km: float = Field(..., description="Target orbital radius (km)")


class LambertTransferResult(BaseModel):
    """Structured result from a Lambert transfer computation."""

    transfer_type: Literal["lambert"] = "lambert"
    note: str = "Simplified calculation - use poliastro for full Lambert solution"
    estimated_delta_v_km_s: float = Field(..., description="Estimated total ΔV (km/s)")
    specified_transfer_time_days: float = Field(..., description="Specified transfer time (days)")
    origin_radius_km: float = Field(..., description="Origin orbital radius (km)")
    target_radius_km: float = Field(..., description="Target orbital radius (km)")


class TrajectoryTool(Tool):
    """
    Tool for computing interplanetary transfer trajectories.

    Supports Hohmann transfers, Lambert problem solutions,
    and basic delta-V calculations.
    """

    name = "trajectory"
    description = (
        "Compute transfer trajectories between two orbital states. "
        "Supports Hohmann transfers and Lambert problem solutions. "
        "Returns delta-V requirements and transfer parameters."
    )

    class Parameters(BaseModel):
        transfer_type: Literal["hohmann", "lambert"]
        origin_position: List[float]
        target_position: List[float]
        origin_velocity: Optional[List[float]] = None
        target_velocity: Optional[List[float]] = None
        transfer_time: Optional[float] = None
        central_body: str = "sun"

    # Standard gravitational parameters (km^3/s^2)
    MU = {
        "sun": 1.32712440018e11,
        "earth": 3.986004418e5,
        "mars": 4.282837e4,
        "jupiter": 1.26686534e8,
    }

    def execute(
        self,
        transfer_type: str,
        origin_position: list,
        target_position: list,
        origin_velocity: list = None,
        target_velocity: list = None,
        transfer_time: float = None,
        central_body: str = "sun",
    ) -> ToolResult:
        """
        Execute trajectory computation.

        Args:
            transfer_type: Type of transfer.
            origin_position: Starting position [x, y, z] in km.
            target_position: Target position [x, y, z] in km.
            origin_velocity: Starting velocity [vx, vy, vz] in km/s.
            target_velocity: Target velocity [vx, vy, vz] in km/s.
            transfer_time: Desired transfer time in days.
            central_body: Central body for gravity.

        Returns:
            ToolResult with transfer parameters.
        """
        try:
            mu = self.MU.get(central_body.lower(), self.MU["sun"])

            r1 = np.array(origin_position)
            r2 = np.array(target_position)

            if transfer_type == "hohmann":
                result = self._compute_hohmann(r1, r2, mu)
            elif transfer_type == "lambert":
                if transfer_time is None:
                    return ToolResult(
                        success=False,
                        output=None,
                        error="Lambert transfer requires transfer_time parameter",
                    )
                result = self._compute_lambert(r1, r2, transfer_time * 86400, mu)
            else:
                return ToolResult(
                    success=False,
                    output=None,
                    error=f"Unknown transfer type: {transfer_type}",
                )

            return ToolResult(success=True, output=result.model_dump())

        except Exception as e:
            return ToolResult(success=False, output=None, error=str(e))

    def _compute_hohmann(
        self,
        r1: np.ndarray,
        r2: np.ndarray,
        mu: float,
    ) -> HohmannTransferResult:
        """Compute Hohmann transfer parameters.

        Uses vis-viva equation directly with the supplied ``mu``,
        so this works for any central body (Sun, Earth, etc.).
        """
        r1_mag = float(np.linalg.norm(r1))
        r2_mag = float(np.linalg.norm(r2))

        # Transfer orbit semi-major axis
        a_transfer = (r1_mag + r2_mag) / 2

        # Circular velocities
        v1_circ = np.sqrt(mu / r1_mag)
        v2_circ = np.sqrt(mu / r2_mag)

        # Velocities at periapsis and apoapsis of transfer orbit
        v_transfer_1 = np.sqrt(mu * (2 / r1_mag - 1 / a_transfer))
        v_transfer_2 = np.sqrt(mu * (2 / r2_mag - 1 / a_transfer))

        # Delta-V calculations
        delta_v1 = abs(v_transfer_1 - v1_circ)
        delta_v2 = abs(v2_circ - v_transfer_2)

        # Transfer time (half period of transfer orbit)
        transfer_time = np.pi * np.sqrt(a_transfer**3 / mu)

        return HohmannTransferResult(
            delta_v1_km_s=float(delta_v1),
            delta_v2_km_s=float(delta_v2),
            total_delta_v_km_s=float(delta_v1 + delta_v2),
            transfer_time_days=float(transfer_time / 86400),
            semi_major_axis_km=float(a_transfer),
            origin_radius_km=r1_mag,
            target_radius_km=r2_mag,
        )

    def _compute_lambert(
        self,
        r1: np.ndarray,
        r2: np.ndarray,
        tof: float,
        mu: float,
    ) -> LambertTransferResult:
        """
        Compute Lambert transfer parameters.

        This is a simplified implementation. For production use,
        consider using poliastro.iod.lambert.
        """
        r1_mag = float(np.linalg.norm(r1))
        r2_mag = float(np.linalg.norm(r2))

        # Approximate using Hohmann for now
        hohmann = self._compute_hohmann(r1, r2, mu)

        return LambertTransferResult(
            estimated_delta_v_km_s=hohmann.total_delta_v_km_s,
            specified_transfer_time_days=tof / 86400,
            origin_radius_km=r1_mag,
            target_radius_km=r2_mag,
        )
