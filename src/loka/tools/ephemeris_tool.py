"""
Ephemeris query tool for Loka agent.

Provides celestial body position and velocity queries.
"""

from typing import Literal, Optional

from pydantic import BaseModel

from loka.tools.base import Tool, ToolResult
from loka.astro.ephemeris import EphemerisManager


class EphemerisTool(Tool):
    """
    Tool for querying celestial body positions and velocities.

    This tool allows the agent to retrieve precise state vectors
    for solar system bodies at specified times.
    """

    name = "ephemeris"
    description = (
        "Query the position and velocity of a solar system body at a given time. "
        "Returns state vector in the ICRS frame relative to the solar system barycenter."
    )

    class Parameters(BaseModel):
        body: str
        epoch: str
        center: str = "solar_system_barycenter"

    def __init__(self):
        self._ephemeris = EphemerisManager()

    def execute(
        self,
        body: str,
        epoch: str,
        center: str = "solar_system_barycenter",
    ) -> ToolResult:
        """
        Execute ephemeris query.

        Args:
            body: Celestial body name.
            epoch: Time of query (ISO format).
            center: Center body for state vector.

        Returns:
            ToolResult with position and velocity.
        """
        try:
            position, velocity = self._ephemeris.get_state(body, epoch, center)

            output = {
                "body": body,
                "epoch": epoch,
                "center": center,
                "position_km": {
                    "x": float(position[0]),
                    "y": float(position[1]),
                    "z": float(position[2]),
                },
                "velocity_km_s": {
                    "vx": float(velocity[0]),
                    "vy": float(velocity[1]),
                    "vz": float(velocity[2]),
                },
                "position_magnitude_km": float((position**2).sum()**0.5),
                "velocity_magnitude_km_s": float((velocity**2).sum()**0.5),
            }

            return ToolResult(success=True, output=output)

        except Exception as e:
            return ToolResult(success=False, output=None, error=str(e))
