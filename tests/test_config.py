"""Tests for the YAML config loader (defaults + user overlay)."""

import textwrap

import pytest

from photo_tools import config as cfg_mod


@pytest.fixture(autouse=True)
def _reset_config():
    """Reload defaults after each test so module state doesn't leak."""
    cfg_mod._cfg = None
    yield
    cfg_mod._cfg = None
    cfg_mod.load_config()


def test_load_defaults_exposes_known_keys():
    cfg = cfg_mod.load_config()
    assert cfg.clip.model == "ViT-B-32"
    assert cfg.ocr.min_confidence == 0.60
    assert cfg.gps.max_gap_seconds == 1800
    assert cfg.xmp.sidecars is False


def test_get_config_lazy_loads():
    cfg_mod._cfg = None
    cfg = cfg_mod.get_config()
    assert cfg is not None
    assert hasattr(cfg, "clip")


def test_user_overlay_deep_merges(tmp_path):
    overlay = tmp_path / "overrides.yaml"
    overlay.write_text(textwrap.dedent("""
        ocr:
          min_confidence: 0.9
        clip:
          model: ViT-L-14
    """))
    cfg = cfg_mod.load_config(user_config_path=overlay)
    # Overridden values
    assert cfg.ocr.min_confidence == 0.9
    assert cfg.clip.model == "ViT-L-14"
    # Untouched siblings preserved
    assert cfg.ocr.high_confidence == 0.80
    assert cfg.clip.pretrained == "laion2b_s34b_b79k"


def test_overlay_path_missing_falls_back_to_defaults(tmp_path):
    cfg = cfg_mod.load_config(user_config_path=tmp_path / "does-not-exist.yaml")
    assert cfg.clip.model == "ViT-B-32"


def test_deep_merge_replaces_scalars_but_recurses_into_dicts():
    base = {"a": 1, "b": {"x": 1, "y": 2}}
    overlay = {"a": 9, "b": {"y": 99}}
    out = cfg_mod._deep_merge(base, overlay)
    assert out == {"a": 9, "b": {"x": 1, "y": 99}}
