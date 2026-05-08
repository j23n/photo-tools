# Contributing to photo-tools

Thanks for your interest. This is a small project; the workflow is simple.

## Quick start

```bash
git clone https://github.com/j23n/photo-tools
cd photo-tools
uv sync                # installs runtime + dev dependencies (ruff, pytest)
uv run pytest          # run the test suite
uv run ruff check .    # lint
uv run ruff format .   # autoformat
```

You'll also need `exiftool`, `ffmpeg`, and `git` on `PATH` — see the
[Requirements](README.md#requirements) section in the README.

## Submitting changes

1. Fork and create a feature branch.
2. Make your change. Keep diffs focused — one logical change per PR.
3. Add or update tests under `tests/` for any behaviour change.
4. Run `uv run pytest` and `uv run ruff check .` locally before pushing.
5. Open a pull request describing the *why* of the change. CI must pass
   (ruff + pytest on Python 3.12).

## Code style

- The project targets Python 3.12+. Use modern syntax
  (`tuple[int, int]`, `str | None`, `match` where it fits).
- Type-annotate public function signatures.
- Module-level docstring on every new module; one-line docstrings on
  public functions where the *why* isn't obvious from the name.
- Logging goes through `photo_tools.logging_setup.get_logger` —
  pick the right subsystem (`tagging`, `geocoding`, `gps`, `ocr`,
  `ram`, `clip`, `landmarks`, `exif`, `dates`, `duplicates`).
- Don't `print()` unless you really mean stdout output the user reads.

## Bumping the tagger version

If your change alters what photo-tools writes to image metadata —
new XMP fields, taxonomy reshape, casing rules — you need to:

1. Bump `TAGGER_VERSION` in `src/photo_tools/constants.py`
   (`YYYY.N` — e.g. `2026.3` → `2026.4`).
2. Update `docs/xmp-schema.md` to describe the new shape.
3. Note the change in `CHANGELOG.md`.

The version bump triggers automatic re-tagging on next `tag` run for
files written under the previous version.

## Reporting bugs

Open an issue at <https://github.com/j23n/photo-tools/issues> with:

- What you ran (the exact command line).
- What you expected.
- What happened (full log output is most useful — `-v` cranks up
  verbosity).
- A small sample image where possible.

## Security

If you find a vulnerability, please email <mail@j23n.com> rather than
opening a public issue.
