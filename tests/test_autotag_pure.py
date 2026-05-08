"""Tests for pure logic in photo_tools.autotag.

Covers GPS coord parsing, EXIF datetime parsing, OCR word validation,
and place-tag formation. Functions that hit the network or models are
not exercised here.
"""

from datetime import datetime
from unittest.mock import patch

import pytest

from photo_tools.autotag import (
    _decide_fix_pipelines,
    _is_plausible_word,
    _parse_exif_datetime,
    get_gps_coords,
    tags_from_gps,
    title,
)


class TestTitle:
    @pytest.mark.parametrize("raw,expected", [
        ("rome", "Rome"),
        ("italy", "Italy"),
        ("  spaced  ", "Spaced"),
    ])
    def test_simple_capitalization(self, raw, expected):
        assert title(raw) == expected

    def test_preserves_roman_numerals(self):
        # The titlecase library detects Roman numerals and uppercases them.
        assert title("municipio roma i") == "Municipio Roma I"

    def test_handles_unicode_diacritics(self):
        # First letter of each word is uppercased; diacritics survive intact.
        assert title("città del vaticano").startswith("Città")


class TestGetGpsCoords:
    def test_basic_positive_coords(self):
        assert get_gps_coords({"GPSLatitude": 41.9, "GPSLongitude": 12.5}) == (41.9, 12.5)

    def test_string_inputs_coerced(self):
        assert get_gps_coords({"GPSLatitude": "41.9", "GPSLongitude": "12.5"}) == (41.9, 12.5)

    def test_southern_hemisphere_negated(self):
        coords = get_gps_coords({
            "GPSLatitude": 33.9,
            "GPSLongitude": 151.2,
            "GPSLatitudeRef": "S",
        })
        assert coords == (-33.9, 151.2)

    def test_western_hemisphere_negated(self):
        coords = get_gps_coords({
            "GPSLatitude": 40.7,
            "GPSLongitude": 74.0,
            "GPSLongitudeRef": "W",
        })
        assert coords == (40.7, -74.0)

    def test_already_negative_not_double_negated(self):
        coords = get_gps_coords({
            "GPSLatitude": -33.9,
            "GPSLongitude": 151.2,
            "GPSLatitudeRef": "S",
        })
        assert coords == (-33.9, 151.2)

    def test_missing_returns_none(self):
        assert get_gps_coords({}) is None
        assert get_gps_coords({"GPSLatitude": 1.0}) is None

    def test_invalid_string_returns_none(self):
        assert get_gps_coords({"GPSLatitude": "north", "GPSLongitude": 12.0}) is None

    def test_out_of_range_rejected(self):
        assert get_gps_coords({"GPSLatitude": 91.0, "GPSLongitude": 0}) is None
        assert get_gps_coords({"GPSLatitude": 0, "GPSLongitude": 181.0}) is None


class TestParseExifDatetime:
    def test_standard_exif_format(self):
        dt = _parse_exif_datetime({"DateTimeOriginal": "2026:05:07 12:34:56"})
        assert dt == datetime(2026, 5, 7, 12, 34)

    def test_iso_format_with_t_separator(self):
        dt = _parse_exif_datetime({"DateTimeOriginal": "2026-05-07T12:34:56"})
        assert dt == datetime(2026, 5, 7, 12, 34)

    def test_falls_back_to_create_date(self):
        dt = _parse_exif_datetime({"CreateDate": "2026:01:02 03:04:05"})
        assert dt == datetime(2026, 1, 2, 3, 4)

    def test_strips_subsecond_and_timezone(self):
        dt = _parse_exif_datetime({"DateTimeOriginal": "2026:05:07 12:34:56.789+02:00"})
        assert dt == datetime(2026, 5, 7, 12, 34)

    @pytest.mark.parametrize("bad", [
        "",
        "2026",
        "not-a-date",
        None,
    ])
    def test_invalid_returns_none(self, bad):
        assert _parse_exif_datetime({"DateTimeOriginal": bad}) is None


class TestIsPlausibleWord:
    @pytest.mark.parametrize("word", [
        "hello", "world", "pizza", "cafe", "OPEN",
    ])
    def test_real_words_pass(self, word):
        assert _is_plausible_word(word) is True

    def test_word_with_no_vowels_rejected(self):
        # 'y' counts as a vowel for OCR purposes (OCR_VOWELS), so a true
        # consonant-only string is needed to exercise this branch.
        assert _is_plausible_word("bczs") is False

    def test_consonant_cluster_rejected(self):
        # Three-consonant cluster not in VALID_ONSETS.
        assert _is_plausible_word("xkcd") is False

    def test_known_onset_passes(self):
        # 'st' is in VALID_ONSETS.
        assert _is_plausible_word("street") is True

    def test_mostly_digits_rejected(self):
        assert _is_plausible_word("a12345") is False


class TestTagsFromGps:
    def test_no_coords_returns_empty(self):
        places, cc, parts = tags_from_gps({})
        assert places == []
        assert cc is None
        assert parts == {}

    def test_full_address_builds_nested_path(self):
        fake_response = {
            "country": "italy",
            "state": "lazio",
            "city": "rome",
            "suburb": "municipio roma i",
            "country_code": "it",
        }
        with patch("photo_tools.autotag.reverse_geocode", return_value=fake_response):
            places, cc, parts = tags_from_gps(
                {"GPSLatitude": 41.9, "GPSLongitude": 12.5})
        assert places == ["Places/Italy/Lazio/Rome/Municipio Roma I"]
        assert cc == "IT"
        assert parts == {
            "Country": "Italy",
            "State": "Lazio",
            "City": "Rome",
            "Sublocation": "Municipio Roma I",
            "CountryCode": "IT",
        }

    def test_collapses_missing_levels(self):
        fake_response = {"country": "france", "country_code": "fr"}
        with patch("photo_tools.autotag.reverse_geocode", return_value=fake_response):
            places, cc, parts = tags_from_gps(
                {"GPSLatitude": 48.8, "GPSLongitude": 2.3})
        assert places == ["Places/France"]
        assert cc == "FR"
        assert parts == {"Country": "France", "CountryCode": "FR"}

    def test_falls_back_to_alternate_keys(self):
        # Nominatim returns 'town' instead of 'city' for smaller settlements.
        fake_response = {
            "country": "germany",
            "town": "freiburg",
            "country_code": "de",
        }
        with patch("photo_tools.autotag.reverse_geocode", return_value=fake_response):
            places, _, parts = tags_from_gps(
                {"GPSLatitude": 48.0, "GPSLongitude": 7.85})
        assert places == ["Places/Germany/Freiburg"]
        assert parts["City"] == "Freiburg"

    def test_empty_geocode_response(self):
        with patch("photo_tools.autotag.reverse_geocode", return_value={}):
            places, cc, parts = tags_from_gps({"GPSLatitude": 0, "GPSLongitude": 0})
        assert places == []
        assert cc is None
        assert parts == {}


class TestDecideFixPipelines:
    def _meta(self, **overrides):
        m = {"keywords": set(), "ocr_ran": False, "coords": None}
        m.update(overrides)
        return m

    def test_runs_all_when_nothing_present(self):
        result = _decide_fix_pipelines(
            self._meta(coords=(1, 2)),
            consider_gps=True, consider_ram=True,
            consider_landmarks=True, consider_ocr=True,
            landmarks_db_exists=True, gps_fallback=None,
        )
        assert result == (True, True, True, True)

    def test_skips_pipelines_with_existing_output(self):
        keywords = {"Places/Italy", "Objects/Animal/Cat", "Landmarks/Colosseum"}
        result = _decide_fix_pipelines(
            self._meta(keywords=keywords, ocr_ran=True, coords=(1, 2)),
            consider_gps=True, consider_ram=True,
            consider_landmarks=True, consider_ocr=True,
            landmarks_db_exists=True, gps_fallback=None,
        )
        assert result == (False, False, False, False)

    def test_geocoding_requires_exif_coords_not_fallback(self):
        # GPS fallback (timeline-inferred) is consumed by landmark pipeline only.
        result = _decide_fix_pipelines(
            self._meta(coords=None),
            consider_gps=True, consider_ram=False,
            consider_landmarks=False, consider_ocr=False,
            landmarks_db_exists=True, gps_fallback=(1, 2),
        )
        assert result[0] is False  # run_gps

    def test_landmarks_uses_fallback_coords(self):
        result = _decide_fix_pipelines(
            self._meta(coords=None),
            consider_gps=False, consider_ram=False,
            consider_landmarks=True, consider_ocr=False,
            landmarks_db_exists=True, gps_fallback=(1, 2),
        )
        assert result[2] is True  # run_landmarks

    def test_landmarks_skipped_without_db(self):
        result = _decide_fix_pipelines(
            self._meta(coords=(1, 2)),
            consider_gps=False, consider_ram=False,
            consider_landmarks=True, consider_ocr=False,
            landmarks_db_exists=False, gps_fallback=None,
        )
        assert result[2] is False

    def test_scenes_keyword_also_blocks_ram(self):
        # Either Objects/* or Scenes/* present means RAM++ already ran.
        result = _decide_fix_pipelines(
            self._meta(keywords={"Scenes/Nature/Forest"}),
            consider_gps=False, consider_ram=True,
            consider_landmarks=False, consider_ocr=False,
            landmarks_db_exists=True, gps_fallback=None,
        )
        assert result[1] is False  # run_ram
