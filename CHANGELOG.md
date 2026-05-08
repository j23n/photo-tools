# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Added
- Apache-2.0 license, packaging metadata (authors, classifiers, project URLs).
- Test suite under `tests/` covering titlecasing, GPS parsing, OCR word
  validation, config loading, and tag helpers.
- GitHub Actions CI: ruff lint + pytest on Python 3.12.
- `CONTRIBUTING.md` with the workflow and tagger-version bump policy.
- README: requirements section, development section, license section.

### Changed
- `landmarks generate-db` now does a two-pass query (global per-type +
  regional bounded) and applies `REGION_MAX_SHARE` caps to counter the
  Europe-skew of Wikidata sitelink rankings. Cache filename bumped to
  `*.wikidata-v2.json` because v1 dumps lack the regionally-diversified pool.
- Replaced `print(file=sys.stderr)` progress messages in
  `build_landmarks.py` with the project's `log` infrastructure.
- Bare `except Exception:` blocks in `helpers.py`, `duplicates.py`, and
  `build_landmarks.py` now log the swallowed error at debug level.
- Image-viewer subprocess in the duplicates picker now spawns detached so
  a hung viewer can't block the UI.
- Genericized example names in `docs/xmp-schema.md`.

### Removed
- Tracked `.claude/settings.local.json` (now gitignored).

## [0.1.0]

Initial public release.
