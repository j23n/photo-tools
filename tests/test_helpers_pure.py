"""Tests for pure helper functions in photo_tools.helpers.

Only covers logic that doesn't require exiftool, ffmpeg, or any model
to be installed — string/path/list manipulation and merge semantics.
"""

from pathlib import Path

import pytest

from photo_tools.helpers import (
    _build_tag_args,
    _is_non_xmp_write_arg,
    _merge_metas,
    _xmp_only_args,
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


class TestBuildTagArgs:
    def test_emits_three_targets_with_leaf_for_flat_fields(self):
        args = _build_tag_args("Places/Italy/Rome", "+=")
        assert args == [
            "-IPTC:Keywords+=Rome",
            "-XMP-dc:Subject+=Rome",
            "-XMP-digiKam:TagsList+=Places/Italy/Rome",
        ]

    def test_remove_operator(self):
        args = _build_tag_args("Objects/Cat", "-=")
        assert args == [
            "-IPTC:Keywords-=Cat",
            "-XMP-dc:Subject-=Cat",
            "-XMP-digiKam:TagsList-=Objects/Cat",
        ]


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
        "-EXIF:DateTimeOriginal=2026:01:01 00:00:00",
    ]
    assert _xmp_only_args(args) == [
        "-overwrite_original",
        "-XMP-dc:Subject+=Cat",
        "-XMP-digiKam:TagsList+=Objects/Animal/Cat",
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
