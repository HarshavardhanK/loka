"""
Analytical Hohmann transfer baseline.

Provides the gold-standard delta-V lower bound against which every
agent trajectory is measured.
"""

import numpy as np
from astropy import units as u
from astropy.constants import GM_earth


def hohmann_baseline(alt_leo_km: float = 400.0, alt_geo_km: float = 35786.0) -> dict:
    """Compute analytical Hohmann transfer delta-V using Astropy constants.

    Parameters
    ----------
    alt_leo_km : float
        LEO altitude above Earth's surface in km (default 400 km).
    alt_geo_km : float
        GEO altitude above Earth's surface in km (default 35 786 km).

    Returns
    -------
    dict
        Transfer parameters including delta-V breakdown, transfer time,
        and fuel mass fractions for several specific impulse values.

    Examples
    --------
    >>> result = hohmann_baseline()
    >>> f"{result['dv_total_kms']:.3f}"
    '3.935'
    """
    mu = GM_earth.to(u.km**3 / u.s**2).value  # 3.986004418e5
    R_E = 6378.137  # km (WGS84 equatorial)

    r1 = R_E + alt_leo_km   # LEO radius
    r2 = R_E + alt_geo_km   # GEO radius
    a_t = (r1 + r2) / 2.0   # transfer orbit semi-major axis

    # Circular velocities
    v1 = np.sqrt(mu / r1)   # LEO circular velocity
    v2 = np.sqrt(mu / r2)   # GEO circular velocity

    # Transfer orbit velocities (vis-viva at each apse)
    v_t_periapsis = np.sqrt(mu * (2.0 / r1 - 1.0 / a_t))
    v_t_apoapsis = np.sqrt(mu * (2.0 / r2 - 1.0 / a_t))

    # Delta-V at each burn
    dv1 = abs(v_t_periapsis - v1)   # departure burn
    dv2 = abs(v2 - v_t_apoapsis)    # circularisation burn
    dv_total = dv1 + dv2

    # Transfer time (half period of transfer ellipse)
    T_transfer = np.pi * np.sqrt(a_t**3 / mu)

    # Fuel mass fraction (Tsiolkovsky) for typical Isp values
    g0 = 9.80665e-3  # km/s^2
    fuel_fractions = {}
    for isp in [300, 1500, 3000]:  # chemical, hall-effect, ion
        mf_ratio = np.exp(-dv_total / (isp * g0))
        fuel_fractions[f"Isp_{isp}s"] = 1.0 - mf_ratio

    return {
        "r_leo_km": r1,
        "r_geo_km": r2,
        "a_transfer_km": a_t,
        "v_leo_kms": v1,
        "v_geo_kms": v2,
        "dv1_kms": dv1,
        "dv2_kms": dv2,
        "dv_total_kms": dv_total,
        "transfer_time_s": T_transfer,
        "transfer_time_hours": T_transfer / 3600,
        "fuel_fractions": fuel_fractions,
    }
