"""
JPL Ephemeris management for Loka.

This module handles loading and querying JPL SPK kernel files
for high-precision celestial body positions.
"""

import os
from datetime import datetime
from pathlib import Path

import numpy as np
from astropy import units as u
from astropy.time import Time

from loka.astro.coordinates import ensure_time


class EphemerisManager:
    """
    Manager for JPL ephemeris data.

    Handles loading SPK kernel files and querying body positions
    with high precision.

    Example:
        >>> eph = EphemerisManager()
        >>> pos, vel = eph.get_state("mars", "2026-07-01")
    """

    # NAIF body codes
    BODY_CODES = {
        "sun": 10,
        "mercury": 199,
        "venus": 299,
        "earth": 399,
        "moon": 301,
        "mars": 499,
        "jupiter": 599,
        "saturn": 699,
        "uranus": 799,
        "neptune": 899,
        "pluto": 999,
        "solar_system_barycenter": 0,
        "earth_moon_barycenter": 3,
    }

    def __init__(self, kernel_path: str | None = None):
        """
        Initialize the ephemeris manager.

        Args:
            kernel_path: Path to SPK kernel file. If None, uses
                         JPL_EPHEMERIS_PATH environment variable.
        """
        self.kernel_path = kernel_path or os.environ.get("JPL_EPHEMERIS_PATH")
        self._kernel = None

        if self.kernel_path:
            self._load_kernel()

    def _load_kernel(self):
        """Load the SPK kernel file."""
        try:
            from jplephem.spk import SPK

            kernel_file = Path(self.kernel_path)
            if kernel_file.is_dir():
                # Find .bsp file in directory
                bsp_files = list(kernel_file.glob("*.bsp"))
                if bsp_files:
                    kernel_file = bsp_files[0]
                else:
                    raise FileNotFoundError(f"No .bsp files found in {kernel_file}")

            self._kernel = SPK.open(str(kernel_file))
            print(f"Loaded ephemeris kernel: {kernel_file}")

        except ImportError:
            print("jplephem not installed. Using astropy ephemeris instead.")
            self._kernel = None

    def get_state(
        self,
        body: str,
        epoch: str | datetime | Time,
        center: str = "solar_system_barycenter",
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Get the state vector (position and velocity) of a body.

        Args:
            body: Name of the celestial body.
            epoch: Time of the query.
            center: Center body for the state vector.

        Returns:
            Tuple of (position, velocity) as numpy arrays in km and km/s.
        """
        t = ensure_time(epoch)
        body_lower = body.lower()
        center_lower = center.lower()

        if body_lower not in self.BODY_CODES:
            raise ValueError(f"Unknown body: {body}")
        if center_lower not in self.BODY_CODES:
            raise ValueError(f"Unknown center: {center}")

        if self._kernel is not None:
            # Use jplephem for high precision
            return self._get_state_jplephem(body_lower, center_lower, t)
        else:
            # Fall back to astropy
            return self._get_state_astropy(body_lower, t)

    def _get_state_jplephem(
        self,
        body: str,
        center: str,
        t: Time,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Get state using jplephem."""
        jd = t.jd

        # Get body position relative to center
        target_code = self.BODY_CODES[body]
        center_code = self.BODY_CODES[center]

        # jplephem returns position in km and velocity in km/day
        try:
            position, velocity = self._kernel[center_code, target_code].compute_and_differentiate(jd)
            velocity = velocity / 86400.0  # Convert km/day to km/s
            return position, velocity
        except KeyError:
            # Try computing via barycenter
            pos1, vel1 = self._kernel[0, target_code].compute_and_differentiate(jd)
            pos2, vel2 = self._kernel[0, center_code].compute_and_differentiate(jd)
            position = pos1 - pos2
            velocity = (vel1 - vel2) / 86400.0
            return position, velocity

    def _get_state_astropy(
        self,
        body: str,
        t: Time,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Get state using astropy (lower precision)."""
        from astropy.coordinates import get_body_barycentric_posvel

        pos, vel = get_body_barycentric_posvel(body, t)

        position = np.array([
            pos.x.to(u.km).value,
            pos.y.to(u.km).value,
            pos.z.to(u.km).value,
        ])

        velocity = np.array([
            vel.x.to(u.km / u.s).value,
            vel.y.to(u.km / u.s).value,
            vel.z.to(u.km / u.s).value,
        ])

        return position, velocity

    @property
    def available_bodies(self) -> list:
        """Return list of available bodies."""
        return list(self.BODY_CODES.keys())
