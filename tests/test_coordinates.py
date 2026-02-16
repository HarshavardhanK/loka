"""Tests for coordinate system utilities."""

import pytest
import numpy as np
from datetime import datetime


class TestGetBodyPosition:
    """Tests for get_body_position function."""
    
    def test_earth_position_returns_tuple(self):
        """Test that Earth position returns a 3-tuple."""
        from loka.astro.coordinates import get_body_position
        
        pos = get_body_position("earth", "2026-07-01")
        
        assert isinstance(pos, tuple)
        assert len(pos) == 3
        assert all(isinstance(x, float) for x in pos)
    
    def test_mars_position_reasonable(self):
        """Test that Mars position is at a reasonable distance."""
        from loka.astro.coordinates import get_body_position
        
        pos = get_body_position("mars", "2026-07-01")
        
        # Mars is 1.5 AU from Sun on average
        # 1 AU ≈ 1.496e8 km
        distance = np.sqrt(sum(x**2 for x in pos))
        
        # Should be between 0.5 AU and 3 AU from barycenter
        assert 0.5 * 1.496e8 < distance < 3 * 1.496e8
    
    def test_unknown_body_raises_error(self):
        """Test that unknown body raises ValueError."""
        from loka.astro.coordinates import get_body_position
        
        with pytest.raises(ValueError, match="Unknown body"):
            get_body_position("unknown_planet", "2026-07-01")
    
    def test_include_velocity(self):
        """Test that velocity can be included."""
        from loka.astro.coordinates import get_body_position
        
        result = get_body_position("earth", "2026-07-01", include_velocity=True)
        
        assert isinstance(result, tuple)
        assert len(result) == 2
        pos, vel = result
        assert len(pos) == 3
        assert len(vel) == 3


class TestSolarSystemBodies:
    """Tests for solar system body list."""
    
    def test_major_planets_included(self):
        """Test that all major planets are in the list."""
        from loka.astro.coordinates import SOLAR_SYSTEM_BODIES
        
        major_planets = ["mercury", "venus", "earth", "mars", 
                        "jupiter", "saturn", "uranus", "neptune"]
        
        for planet in major_planets:
            assert planet in SOLAR_SYSTEM_BODIES
    
    def test_sun_included(self):
        """Test that the Sun is in the list."""
        from loka.astro.coordinates import SOLAR_SYSTEM_BODIES
        
        assert "sun" in SOLAR_SYSTEM_BODIES
