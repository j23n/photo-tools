"""Shared pytest configuration.

The default config is loaded once for the test session so config-dependent
helpers (e.g. ``_sidecars_enabled``) work without per-test setup.
"""

from photo_tools.config import load_config


def pytest_configure(config):
    load_config()
