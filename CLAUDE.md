# lexguard

Privacy concept for RAG and model training on a law firm's data lake. See `README.md`.

## Conventions

- Code, identifiers, docstrings and docs in English. Synthetic document text is Spanish (Chile).
- Idiomatic modern Python (3.12+): dataclasses, enums, protocols, comprehensions, `pathlib`.
- No narrating comments. A docstring only where it states something the code cannot.
- Small modules with real behavior. No placeholder classes, no speculative abstraction.
- Standard library first. Runtime dependencies are `cryptography` and `numpy`; ask before adding one.
- Tests are pytest functions under `tests/<module>/`, using `lexguard.synthetic.generate()`
  for data and `lexguard.testing` for fakes of other modules.

## Module ownership

The shared core is `domain.py`, `ports.py`, `guard.py`, `settings.py`, `crypto.py`,
`storage.py`, `rut.py`, `tokens.py`, `synthetic.py`, `testing.py`, `errors.py`.
Feature work lives in its own package (`deid/`, `access/`, `audit/`, `rag/`, `egress/`,
`training/`, `analytics/`, `rights/`) and depends on other features only through `ports.py`.
A feature branch does not edit the core, `pyproject.toml`, `README.md` or another feature's
package; if the core is missing something, report it instead.

## Commands

```
uv sync
uv run pytest
uv run ruff check
uv run ruff format
```
