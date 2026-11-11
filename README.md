# lexguard

A working concept of how to build RAG, fine-tuning, classical models and analytics on top of
a law firm's data lake without anyone, human or model, seeing more than they need.

Target context: Chilean firm, documents in Spanish, Ley 21.719 (in force from 1 December 2026)
plus professional secrecy. Everything runs locally on a synthetic corpus; no real data is needed.

## Threat model

| Threat | Control |
| --- | --- |
| A data scientist reads client names in the training set | Raw zone is unreadable to humans; work happens on pseudonymized text |
| A lawyer retrieves chunks from a matter they are walled off from | Matter compartments, ethical walls that override assignment, filtering before ranking |
| Data is reused for a purpose the client never agreed to | Each matter carries its permitted purposes; every request declares one |
| Documents leak to an external model provider | Egress gateway, `local_only` by default, outbound re-scan for residual PII |
| A fine-tuned model memorizes and regurgitates a client fact | Privileged documents excluded, canaries and extraction tests, optional DP training |
| Aggregates single out a person or a small matter | Minimum group size, calibrated noise, privacy budget ledger |
| Stolen disk or bucket | Envelope encryption, one root key per matter, one derived key per zone |
| An engagement ends or a data subject asks for erasure | Crypto-shredding per matter, subject index, tainted training manifests |
| Someone quietly edits the access history | Hash-chained audit log written by a single enforcement point |

## Data flow

```
 law firm data lake
        │  ingest only matters with a legal basis
        ▼
 ┌────────────┐  detect + pseudonymize   ┌────────────────┐
 │  RAW zone  │ ───────────────────────▶ │ DEIDENTIFIED   │
 │ no humans  │      token vault         │ zone           │
 └────────────┘                          └───────┬────────┘
                                                 │
                 ┌───────────────────────────────┼──────────────────────────┐
                 ▼                               ▼                          ▼
          ┌────────────┐                 ┌──────────────┐           ┌─────────────┐
          │ SERVING    │                 │ TRAINING     │           │ analytics   │
          │ RAG index  │                 │ datasets +   │           │ k-threshold │
          │ with ACLs  │                 │ manifests    │           │ + DP noise  │
          └─────┬──────┘                 └──────────────┘           └─────────────┘
                │ re-identify only for an authorized
                ▼ member of the matter
             answer with citations
```

Every arrow crosses the `Guard`: a policy decision (who, which action, for what purpose, on
which matter and zone) followed by an audit record, before anything is read.

## Layout

| Path | Responsibility | Status |
| --- | --- | --- |
| `domain.py`, `ports.py`, `guard.py`, `settings.py` | Shared vocabulary, protocols, enforcement point, trust boundary | done |
| `crypto.py`, `storage.py`, `documents.py` | Per-matter envelope encryption, zoned encrypted store, crypto-shredding | done |
| `rut.py`, `tokens.py` | RUT validation, pseudonym token format | done |
| `synthetic.py`, `testing.py` | Labeled synthetic corpus, in-memory fakes | done |
| `deid/` | PII recognizers for Chilean legal text, pseudonymizer, token vault, residual scan | planned |
| `access/`, `audit/` | Purpose-based ABAC, ethical walls, time-bound grants, hash-chained log | planned |
| `rag/`, `egress/` | Chunking, embedding, security-trimmed retrieval, boundary gateway | planned |
| `training/`, `analytics/`, `rights/` | Dataset manifests, memorization tests, DP aggregates, erasure | planned |

All paths are under `src/lexguard/`, with tests mirrored under `tests/`.

## Design decisions

- **Pseudonyms are keyed per matter.** The same person gets the same token inside a matter,
  so text stays coherent for retrieval and training, and a different token in every other
  matter, so nothing can be joined across clients.
- **Purpose is data, not documentation.** A matter lists what its engagement letter allows.
  Training on a matter that only allows `legal_advice` is denied by policy, not by convention.
- **Keys follow the compartments.** One root per matter, one derived key per zone. Access to
  the de-identified zone gives no ability to decrypt the raw one.
- **The trust boundary is a setting.** `LEXGUARD_BOUNDARY` is `local_only`, `private_cloud`
  or `deidentified_egress`. The default is the strictest.
- **`KeyRing` stands in for a KMS.** In production the master key lives in an HSM-backed
  service and unwrap calls are themselves audited.

## Development

```
uv sync
uv run pytest
uv run ruff check
```
