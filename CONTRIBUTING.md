# Contributing to AutoPTZ

Thanks for helping! This guide covers local setup and the checks CI enforces.

## Setup

Requires **Python 3.12+**.

```bash
git clone https://github.com/AutoPTZ/autoptz
cd autoptz
python3.12 -m venv .venv            # at the repo root
source .venv/bin/activate           # Windows: .venv\Scripts\activate
python tools/install.py --dev --editable
pre-commit install                  # optional but recommended
```

## Quality gates (what CI runs)

All five must pass on macOS, Windows, and Linux:

```bash
ruff check autoptz/ tests/ tools/          # lint
ruff format --check autoptz/ tests/ tools/ # formatting
mypy autoptz/engine/runtime/ autoptz/config/   # strict types on the typed core
pytest tests/ -v --timeout=60              # unit tests
python -m autoptz --selftest               # smoke test
```

`pre-commit` runs ruff + ruff-format (and basic hygiene) on every commit, so
formatting never drifts. Run `pre-commit run --all-files` to apply it everywhere.

### Typing

`mypy` runs `--strict` on the typed core (`engine/runtime/`, `config/`). The UI
and pipeline are mid typing-migration and excluded in `pyproject.toml`; if you
fully type one of those modules, drop it from the exclude list. Tests aren't held
to strict annotations (see the `tests.*` mypy override).

## Conventions

- **Format/line length is owned by `ruff format`** (100 cols). Don't hand-wrap to
  fight it; run the formatter.
- **Match the surrounding code** — comment density, naming, and idioms.
- **Never hard-fail on a missing model/dep** — degrade to live-preview-only and
  log one actionable message (see existing `_log_*_once` helpers).
- **Cameras are addressed by UUID**, never list index.
- Large modules are being split into focused submodules (e.g.
  `engine/worker/`) — prefer adding new cohesive code in its own module over
  growing the giants.

## Branching

Active development happens on `dev/v2-architecture-rework`. Branch from it, keep
the suite green, and open a PR back into it. See [docs/architecture.md](docs/architecture.md)
for the layout and [docs/building.md](docs/building.md) for release/installer builds.
