"""Configuration loader for photo-tools.

Loads default_config.yaml from the package, optionally overlays a user
config file. Values are accessed via dot notation: cfg.clip.model, etc.
"""

from pathlib import Path
from types import SimpleNamespace

import yaml

_cfg: SimpleNamespace | None = None


def _deep_merge(base: dict, overlay: dict) -> dict:
    result = base.copy()
    for k, v in overlay.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = _deep_merge(result[k], v)
        else:
            result[k] = v
    return result


def _dict_to_namespace(d: dict) -> SimpleNamespace:
    ns = SimpleNamespace()
    for k, v in d.items():
        if isinstance(v, dict):
            setattr(ns, k, _dict_to_namespace(v))
        else:
            setattr(ns, k, v)
    return ns


def load_config(user_config_path: Path | None = None) -> SimpleNamespace:
    """Load config: defaults + optional user overlay."""
    global _cfg
    defaults_file = Path(__file__).parent / "default_config.yaml"
    with open(defaults_file) as f:
        data = yaml.safe_load(f)

    if user_config_path and user_config_path.exists():
        with open(user_config_path) as f:
            user_data = yaml.safe_load(f) or {}
        data = _deep_merge(data, user_data)

    _cfg = _dict_to_namespace(data)
    return _cfg


def get_config() -> SimpleNamespace:
    """Get config, lazy-loading defaults if not yet initialized."""
    global _cfg
    if _cfg is None:
        load_config()
    return _cfg
