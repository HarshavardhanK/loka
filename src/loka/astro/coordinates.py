"""
Coordinate systems and transformations for celestial mechanics.

This module wraps astropy coordinate functionality for use in Loka.
"""

from typing import Tuple, Optional, Union
from datetime import datetime

import numpy as np
from astropy import units as u
from astropy.coordinates import (
    ICRS,
    GCRS,
    HeliocentricMeanEcliptic,
    CartesianRepresentation,
    SkyCoord,
    get_body_barycentric,
    get_body_barycentric_posvel,
)
from astropy.time import Time

# Supported solar system bodies
SOLAR_SYSTEM_BODIES = [
    "sun",
    "mercury",
    "venus",
    "earth",
    "moon",
    "mars",
    "jupiter",
    "saturn",
    "uranus",
    "neptune",
    "pluto",
]


def get_body_position(
    body: str,
    epoch: Union[str, datetime, Time],
    frame: str = "icrs",
    include_velocity: bool = False,
) -> Union[Tuple[float, float, float], Tuple[Tuple[float, float, float], Tuple[float, float, float]]]:
    """
    Get the position (and optionally velocity) of a solar system body.
    
    Args:
        body: Name of the celestial body.
        epoch: Time of observation (ISO string, datetime, or astropy Time).
        frame: Reference frame ("icrs", "gcrs", "heliocentric").
        include_velocity: Whether to also return velocity.
        
    Returns:
        Position as (x, y, z) in km, or tuple of (position, velocity) if
        include_velocity is True.
        
    Example:
        >>> pos = get_body_position("mars", "2026-07-01")
        >>> print(f"Mars position: {pos}")
    """
    # Convert epoch to astropy Time
    if isinstance(epoch, str):
        t = Time(epoch)
    elif isinstance(epoch, datetime):
        t = Time(epoch)
    else:
        t = epoch
    
    body_lower = body.lower()
    if body_lower not in SOLAR_SYSTEM_BODIES:
        raise ValueError(f"Unknown body: {body}. Must be one of {SOLAR_SYSTEM_BODIES}")
    
    if include_velocity:
        pos, vel = get_body_barycentric_posvel(body_lower, t)
        position = (
            pos.x.to(u.km).value,
            pos.y.to(u.km).value,
            pos.z.to(u.km).value,
        )
        velocity = (
            vel.x.to(u.km / u.s).value,
            vel.y.to(u.km / u.s).value,
            vel.z.to(u.km / u.s).value,
        )
        return position, velocity
    else:
        pos = get_body_barycentric(body_lower, t)
        return (
            pos.x.to(u.km).value,
            pos.y.to(u.km).value,
            pos.z.to(u.km).value,
        )


def transform_coordinates(
    position: Tuple[float, float, float],
    velocity: Optional[Tuple[float, float, float]],
    from_frame: str,
    to_frame: str,
    epoch: Union[str, datetime, Time],
) -> Tuple[Tuple[float, float, float], Optional[Tuple[float, float, float]]]:
    """
    Transform coordinates between reference frames.
    
    Args:
        position: Position as (x, y, z) in km.
        velocity: Velocity as (vx, vy, vz) in km/s, or None.
        from_frame: Source reference frame.
        to_frame: Target reference frame.
        epoch: Time of the state vector.
        
    Returns:
        Transformed (position, velocity) tuple.
    """
    # Convert epoch
    if isinstance(epoch, str):
        t = Time(epoch)
    elif isinstance(epoch, datetime):
        t = Time(epoch)
    else:
        t = epoch
    
    # Create CartesianRepresentation
    cart = CartesianRepresentation(
        x=position[0] * u.km,
        y=position[1] * u.km,
        z=position[2] * u.km,
    )
    
    # Map frame names to astropy frames
    frame_map = {
        "icrs": ICRS,
        "gcrs": GCRS,
        "heliocentric": HeliocentricMeanEcliptic,
    }
    
    if from_frame.lower() not in frame_map:
        raise ValueError(f"Unknown frame: {from_frame}")
    if to_frame.lower() not in frame_map:
        raise ValueError(f"Unknown frame: {to_frame}")
    
    # Create coordinate in source frame
    source_frame = frame_map[from_frame.lower()]
    target_frame = frame_map[to_frame.lower()]
    
    coord = SkyCoord(cart, frame=source_frame(), obstime=t)
    
    # Transform to target frame
    transformed = coord.transform_to(target_frame())
    
    new_pos = (
        transformed.cartesian.x.to(u.km).value,
        transformed.cartesian.y.to(u.km).value,
        transformed.cartesian.z.to(u.km).value,
    )
    
    # TODO: Handle velocity transformation properly
    # This requires differential coordinates
    new_vel = velocity  # Placeholder
    
    return new_pos, new_vel
