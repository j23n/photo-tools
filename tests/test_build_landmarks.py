"""Tests for the regional classification used in landmark building."""

import pytest

from photo_tools.build_landmarks import _classify_region


@pytest.mark.parametrize("name,lat,lon,expected", [
    ("Rome",          41.9028,  12.4964, "Europe"),
    ("Paris",         48.8566,   2.3522, "Europe"),
    ("New York",      40.7128, -74.0060, "N.America"),
    ("Mexico City",   19.4326, -99.1332, "N.America"),
    ("Buenos Aires", -34.6037, -58.3816, "S.America"),
    ("Sao Paulo",    -23.5505, -46.6333, "S.America"),
    ("Tokyo",         35.6762, 139.6503, "Asia"),
    ("Mumbai",        19.0760,  72.8777, "Asia"),
    ("Sydney",       -33.8688, 151.2093, "Oceania"),
    ("Auckland",     -36.8485, 174.7633, "Oceania"),
    ("Cape Town",    -33.9249,  18.4241, "Africa"),
    ("Nairobi",       -1.2921,  36.8219, "Africa"),
    ("Addis Ababa",    9.0320,  38.7423, "Africa"),
    ("Mogadishu",      2.0469,  45.3182, "Africa"),
    ("Antananarivo", -18.8792,  47.5079, "Africa"),
    ("Cairo",         30.0444,  31.2357, "Africa"),
    # Arabian peninsula stays in Asia despite being east of the new
    # extended Africa lon bound (the lat>15 band keeps the original lon=35).
    ("Riyadh",        24.7136,  46.6753, "Asia"),
    ("Sanaa",         15.3694,  44.1910, "Asia"),
])
def test_classify_known_cities(name, lat, lon, expected):
    assert _classify_region(lat, lon) == expected, name


def test_open_ocean_falls_back_to_other():
    # Mid Pacific ocean — outside every defined bbox.
    assert _classify_region(0, -150) == "Other"
