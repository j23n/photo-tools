"""Tests for pure helper functions in photo_tools.helpers.

Only covers logic that doesn't require exiftool, ffmpeg, or any model
to be installed — string/path/list manipulation and merge semantics.
"""

from pathlib import Path

import pytest

from photo_tools.helpers import (
    _build_tag_args,
    _is_non_xmp_write_arg,
    _location_field_args,
    _merge_metas,
    _xmp_only_args,
    alt_xmp_sidecar_path,
    deduplicate,
    is_live_photo_motion,
    is_our_tag,
    leaf_of,
    xmp_sidecar_path,
)


class TestIsOurTag:
    @pytest.mark.parametrize("tag", [
        "Places/Italy",
        "Places/Italy/Lazio",
        "Objects/Animal/Cat",
        "Scenes/Nature/Forest",
        "Landmarks/Colosseum",
    ])
    def test_owned_roots(self, tag):
        assert is_our_tag(tag) is True

    @pytest.mark.parametrize("tag", [
        "People/Alice",       # digiKam-owned
        "Vacation",           # user freeform
        "places/lower",       # case-sensitive root
        "",
    ])
    def test_not_owned(self, tag):
        assert is_our_tag(tag) is False


class TestLeafOf:
    @pytest.mark.parametrize("tag,expected", [
        ("Places/Italy/Lazio/Rome", "Rome"),
        ("Landmarks/Colosseum", "Colosseum"),
        ("Flat", "Flat"),
        ("", ""),
    ])
    def test_returns_last_segment(self, tag, expected):
        assert leaf_of(tag) == expected


class TestDeduplicate:
    def test_preserves_first_seen_order(self):
        assert deduplicate(["a", "b", "a", "c", "b"]) == ["a", "b", "c"]

    def test_drops_falsy_entries(self):
        assert deduplicate(["", "x", None, "x"]) == ["x"]

    def test_empty_input(self):
        assert deduplicate([]) == []


class TestXmpSidecarPath:
    def test_appends_xmp_keeping_original_suffix(self):
        # MWG / digiKam convention — IMG_1234.jpg.xmp not IMG_1234.xmp,
        # so files sharing a stem get distinct sidecars.
        assert xmp_sidecar_path(Path("/p/IMG_1234.jpg")) == Path("/p/IMG_1234.jpg.xmp")
        assert xmp_sidecar_path(Path("/p/IMG_1234.heic")) == Path("/p/IMG_1234.heic.xmp")
        assert xmp_sidecar_path(Path("/p/clip.mov")) == Path("/p/clip.mov.xmp")


class TestAltXmpSidecarPath:
    def test_replaces_suffix(self):
        # Lightroom / Capture One convention.
        assert alt_xmp_sidecar_path(Path("/p/IMG_1234.jpg")) == Path("/p/IMG_1234.xmp")
        assert alt_xmp_sidecar_path(Path("/p/IMG_1234.HEIC")) == Path("/p/IMG_1234.xmp")

    def test_returns_none_for_xmp_input(self):
        # Don't return the input itself for an .xmp file.
        assert alt_xmp_sidecar_path(Path("/p/IMG_1234.xmp")) is None


class TestBuildTagArgs:
    def test_non_people_tag_emits_four_targets(self):
        args = _build_tag_args("Places/Italy/Rome", "+=")
        assert args == [
            "-IPTC:Keywords+=Rome",
            "-XMP-dc:Subject+=Rome",
            "-XMP-digiKam:TagsList+=Places/Italy/Rome",
            "-XMP-lr:HierarchicalSubject+=Places|Italy|Rome",
        ]

    def test_people_tag_includes_person_in_image(self):
        args = _build_tag_args("People/Alice", "+=")
        assert args == [
            "-IPTC:Keywords+=Alice",
            "-XMP-dc:Subject+=Alice",
            "-XMP-digiKam:TagsList+=People/Alice",
            "-XMP-lr:HierarchicalSubject+=People|Alice",
            "-XMP-iptcExt:PersonInImage+=Alice",
        ]

    def test_remove_operator(self):
        args = _build_tag_args("Objects/Cat", "-=")
        assert args == [
            "-IPTC:Keywords-=Cat",
            "-XMP-dc:Subject-=Cat",
            "-XMP-digiKam:TagsList-=Objects/Cat",
            "-XMP-lr:HierarchicalSubject-=Objects|Cat",
        ]


class TestLocationFieldArgs:
    def test_full_set_writes_xmp_and_iptc_pairs(self):
        args = _location_field_args({
            "Country": "Italy",
            "State": "Lazio",
            "City": "Rome",
            "Sublocation": "Municipio Roma I",
            "CountryCode": "IT",
        })
        # Each component fans out to one XMP target and one IPTC IIM target.
        assert "-XMP-photoshop:Country=Italy" in args
        assert "-IPTC:Country-PrimaryLocationName=Italy" in args
        assert "-XMP-photoshop:State=Lazio" in args
        assert "-IPTC:Province-State=Lazio" in args
        assert "-XMP-photoshop:City=Rome" in args
        assert "-IPTC:City=Rome" in args
        assert "-XMP-iptcCore:Location=Municipio Roma I" in args
        assert "-IPTC:Sub-location=Municipio Roma I" in args
        assert "-XMP-iptcCore:CountryCode=IT" in args
        assert "-IPTC:Country-PrimaryLocationCode=IT" in args

    def test_partial_set_writes_only_present_components(self):
        args = _location_field_args({"Country": "France"})
        assert args == [
            "-XMP-photoshop:Country=France",
            "-IPTC:Country-PrimaryLocationName=France",
        ]

    def test_unknown_key_silently_skipped(self):
        # Defensive: a future key in location_fields not in _LOCATION_FIELDS
        # should not crash the writer.
        args = _location_field_args({"Country": "Italy", "Unknown": "x"})
        assert "-XMP-photoshop:Country=Italy" in args
        assert all("Unknown" not in a for a in args)

    def test_iptc_targets_stripped_on_sidecar_phase(self):
        # IPTC writes can't go into a .xmp sidecar; _xmp_only_args
        # must filter them out alongside the keyword IPTC writes.
        args = _location_field_args({"City": "Rome"})
        kept = _xmp_only_args(args)
        assert "-XMP-photoshop:City=Rome" in kept
        assert "-IPTC:City=Rome" not in kept


class TestNonXmpWriteArg:
    @pytest.mark.parametrize("arg", [
        "-IPTC:Keywords+=Cat",
        "-EXIF:DateTimeOriginal=2026:05:07 12:00:00",
        "-QuickTime:CreateDate=2026:01:01",
    ])
    def test_recognizes_non_xmp_groups(self, arg):
        assert _is_non_xmp_write_arg(arg) is True

    @pytest.mark.parametrize("arg", [
        "-XMP-dc:Subject+=Cat",
        "-XMP-digiKam:TagsList+=Places/Italy",
        "-XMP-phototools:CountryCode=IT",
    ])
    def test_xmp_groups_pass_through(self, arg):
        assert _is_non_xmp_write_arg(arg) is False

    def test_non_write_arg_passes(self):
        assert _is_non_xmp_write_arg("-overwrite_original") is False


def test_xmp_only_args_strips_non_xmp_writes():
    args = [
        "-overwrite_original",
        "-IPTC:Keywords+=Cat",
        "-XMP-dc:Subject+=Cat",
        "-XMP-digiKam:TagsList+=Objects/Animal/Cat",
        "-XMP-lr:HierarchicalSubject+=Objects|Animal|Cat",
        "-EXIF:DateTimeOriginal=2026:01:01 00:00:00",
    ]
    assert _xmp_only_args(args) == [
        "-overwrite_original",
        "-XMP-dc:Subject+=Cat",
        "-XMP-digiKam:TagsList+=Objects/Animal/Cat",
        "-XMP-lr:HierarchicalSubject+=Objects|Animal|Cat",
    ]


class TestMergeMetas:
    def test_unions_keyword_lists_with_dedup(self):
        embedded = {"SourceFile": "img.jpg", "Keywords": ["Rome", "Italy"]}
        sidecar = {"SourceFile": "img.jpg.xmp", "Keywords": ["Italy", "Cat"]}
        merged = _merge_metas([embedded, sidecar])
        assert merged["Keywords"] == ["Rome", "Italy", "Cat"]
        assert merged["SourceFile"] == "img.jpg"

    def test_scalar_sidecar_wins(self):
        embedded = {"SourceFile": "img.jpg", "TaggerVersion": "2025.1"}
        sidecar = {"SourceFile": "img.jpg.xmp", "TaggerVersion": "2026.3"}
        merged = _merge_metas([embedded, sidecar])
        assert merged["TaggerVersion"] == "2026.3"

    def test_string_field_treated_as_singleton_list(self):
        embedded = {"SourceFile": "img.jpg", "Subject": "Rome"}
        sidecar = {"SourceFile": "img.jpg.xmp", "Subject": ["Rome", "Italy"]}
        merged = _merge_metas([embedded, sidecar])
        assert merged["Subject"] == ["Rome", "Italy"]

    def test_empty_inputs(self):
        assert _merge_metas([]) == {}

    def test_empty_value_does_not_overwrite(self):
        embedded = {"SourceFile": "img.jpg", "CountryCode": "IT"}
        sidecar = {"SourceFile": "img.jpg.xmp", "CountryCode": ""}
        merged = _merge_metas([embedded, sidecar])
        assert merged["CountryCode"] == "IT"


class TestIsLivePhotoMotion:
    @pytest.mark.parametrize("name", [
        "IMG_1353.HEIC.MOV",
        "IMG_1353.heic.mov",
        "IMG_1353.JPG.mov",
        "IMG_1353.jpeg.mov",
    ])
    def test_recognizes_apple_double_suffix(self, name):
        assert is_live_photo_motion(Path(name)) is True

    @pytest.mark.parametrize("name", [
        "clip.mov",                  # plain video
        "IMG_1353.HEIC",             # the still itself
        "IMG_1353.png.mov",          # png isn't a Live Photo source
        "regular.mp4",
    ])
    def test_rejects_non_motion(self, name):
        assert is_live_photo_motion(Path(name)) is False
