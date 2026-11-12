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
`storage.py`, `documents.py`, `rut.py`, `tokens.py`, `synthetic.py`, `testing.py`, `errors.py`.
Feature work lives in its own package (`deid/`, `access/`, `audit/`, `rag/`, `egress/`,
`training/`, `analytics/`, `rights/`) and depends on other features only through `ports.py`.
A feature does not edit the core, `pyproject.toml`, `README.md` or another feature's
package; if the core is missing something, report it instead.
`pipeline.py`, `workspace.py` and `demo.py` are the only modules that wire features together.

## Guard requests

Every feature asks the `Guard` with these shapes. `Resource.matter_id` is always set, except
to read the audit trail, which no matter scopes.

| Operation | Action | Zone | Purpose |
| --- | --- | --- | --- |
| Land a document | `INGEST` | `RAW` | one the matter permits |
| Read raw to de-identify | `READ_RAW` | `RAW` | one the matter permits |
| Retrieve chunks | `RETRIEVE` | `SERVING` | caller's, per (matter, sensitivity) |
| Re-identify an answer | `REIDENTIFY` | `SERVING` | same as the retrieve |
| Send a prompt to a model | `EGRESS` | `SERVING` | same principal and compartment as the retrieve |
| Export or load training data | `EXPORT_TRAINING` | `DEIDENTIFIED` | `MODEL_DEVELOPMENT` |
| Aggregate | `AGGREGATE` | `DEIDENTIFIED` | `ANALYTICS` |
| Locate a data subject | `READ_DEIDENTIFIED` | `DEIDENTIFIED` | `DATA_SUBJECT_REQUEST` |
| Erase a data subject | `ERASE` | `DEIDENTIFIED` | `DATA_SUBJECT_REQUEST` |
| Erase a retained raw copy | `ERASE` | `RAW` | `DATA_SUBJECT_REQUEST` |
| Close a matter | `ERASE` | `RAW` | `RETENTION` |
| Read the audit trail | `READ_AUDIT` | `AUDIT` | `SECURITY_AUDIT` |

`DATA_SUBJECT_REQUEST`, `SECURITY_AUDIT` and `RETENTION` are statutory: they do not need to
appear in `Matter.purposes`.

## Commands

```
uv sync
uv run pytest
uv run ruff check
uv run ruff format
```
