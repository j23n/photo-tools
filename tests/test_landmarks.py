"""Tests for landmarks helper math."""

import math

from photo_tools.landmarks import _haversine_km


class TestHaversine:
    def test_zero_distance_is_zero(self):
        assert _haversine_km(0, 0, 0, 0) == 0.0

    def test_one_degree_lat_is_about_111_km(self):
        # 1 degree of latitude ≈ 111.19 km
        d = _haversine_km(0, 0, 1, 0)
        assert math.isclose(d, 111.19, rel_tol=0.01)

    def test_known_pair_rome_to_paris(self):
        # Rome (41.9028, 12.4964) → Paris (48.8566, 2.3522) ≈ 1106 km
        d = _haversine_km(41.9028, 12.4964, 48.8566, 2.3522)
        assert math.isclose(d, 1106, rel_tol=0.02)

    def test_antipodes_about_half_circumference(self):
        # Earth circumference / 2 ≈ 20015 km
        d = _haversine_km(0, 0, 0, 180)
        assert math.isclose(d, 20015, rel_tol=0.01)

    def test_symmetric(self):
        a = _haversine_km(40.0, -74.0, 51.5, -0.13)
        b = _haversine_km(51.5, -0.13, 40.0, -74.0)
        assert math.isclose(a, b)
