## What & why

<!-- What does this change and why? Link any issue: Closes #123 -->

## How

<!-- Key implementation notes / decisions worth calling out for review. -->

## Checklist

- [ ] `ruff check autoptz/ tests/ tools/` passes
- [ ] `ruff format --check autoptz/ tests/ tools/` passes
- [ ] `mypy autoptz/engine/runtime/ autoptz/config/` passes
- [ ] `pytest tests/ --timeout=60` green
- [ ] `python -m autoptz --selftest` passes
- [ ] Docs updated if behavior/config changed

## Testing

<!-- How was this verified? Platforms / accelerators exercised, if relevant. -->
