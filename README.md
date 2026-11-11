# lexguard

A working concept of how to build RAG, fine-tuning, classical models and analytics on top of
a law firm's data lake without anyone, human or model, seeing more than they need.

Target context: Chilean firm, documents in Spanish, Ley 21.719 (in force from 1 December 2026)
plus professional secrecy. Everything runs locally on a synthetic corpus; no real data is needed.

```
uv sync
uv run python -m lexguard
```

The walkthrough ingests a synthetic lake, answers the same question for a lawyer, a paralegal
and a walled-off lawyer, tries to reach a hosted model, builds a training set, releases noisy
aggregates, erases a person, closes a matter and verifies the audit chain.

## Threat model

| Threat | Control | Where |
| --- | --- | --- |
| A data scientist reads client names in the training set | Raw zone is unreadable to humans; work happens on pseudonymized text | `deid/`, `pipeline.py` |
| A lawyer retrieves chunks from a matter they are walled off from | Matter compartments, ethical walls that override assignment, trimming before ranking | `access/`, `rag/` |
| Data is reused for a purpose the client never agreed to | Each matter carries its permitted purposes; every request declares one | `domain.py`, `access/` |
| Documents leak to an external model provider | One egress gateway, `local_only` by default, outbound re-scan for residual PII | `egress/` |
| A fine-tuned model memorizes and regurgitates a client fact | Privileged documents excluded, canary exposure test, DP-SGD mode | `training/` |
| Aggregates single out a person or a small matter | Minimum group size, Laplace noise, privacy budget ledger | `analytics/` |
| Stolen disk or bucket | Envelope encryption, one root key per matter, one derived key per zone | `crypto.py`, `storage.py` |
| An engagement ends or a data subject asks for erasure | Crypto-shredding per matter, irreversible tokens, tainted training manifests | `rights/` |
| Someone quietly edits the access history | HMAC-chained audit log written by a single enforcement point | `guard.py`, `audit/` |

## Data flow

```
 law firm data lake
        │  land only what the policy admits
        ▼
 ┌────────────┐  detect + pseudonymize   ┌────────────────┐
 │  RAW zone  │ ───────────────────────▶ │ DEIDENTIFIED   │
 │ emptied    │      token vault         │ zone           │
 └────────────┘                          └───────┬────────┘
                                                 │
                 ┌───────────────────────────────┼──────────────────────────┐
                 ▼                               ▼                          ▼
          ┌──────────────┐               ┌──────────────┐           ┌─────────────┐
          │ SERVING      │               │ TRAINING     │           │ analytics   │
          │ index with   │               │ shards +     │           │ k-threshold │
          │ compartments │               │ manifests    │           │ + DP noise  │
          └──────┬───────┘               └──────────────┘           └─────────────┘
                 │ re-identify only for a lawyer
                 ▼ assigned to the matter
              answer with citations
```

Every arrow crosses the `Guard`: a policy decision (who, which action, for what purpose, on
which matter, zone and sensitivity) followed by an audit record, before anything is read.

## Layout

| Path under `src/lexguard/` | Responsibility |
| --- | --- |
| `domain.py`, `ports.py`, `guard.py`, `settings.py` | Shared vocabulary, protocols, enforcement point, trust boundary |
| `crypto.py`, `storage.py`, `documents.py` | Per-matter envelope encryption, zoned encrypted store, crypto-shredding |
| `deid/` | PII recognizers for Chilean legal text, per-matter pseudonymizer, sealed token vault |
| `access/`, `audit/` | Declarative purpose-based policy, ethical walls, dual-control grants, chained log |
| `rag/`, `egress/` | Token-safe chunking, local embedder, security-trimmed index, cited answers, model gateway |
| `training/`, `analytics/`, `rights/` | Dataset manifests, DP classifier, memorization test, noisy aggregates, erasure and closure |
| `pipeline.py`, `workspace.py`, `demo.py` | Ingestion, composition root, end-to-end walkthrough |
| `synthetic.py`, `testing.py` | Labeled synthetic corpus, in-memory fakes |

## Design decisions

- **Pseudonyms are keyed per matter.** The same person gets the same token inside a matter,
  so text stays coherent for retrieval and training, and a different token in every other
  matter, so nothing can be joined across clients.
- **Purpose is data, not documentation.** A matter lists what its engagement letter allows.
  Training on a matter that only allows `legal_advice` is denied by policy, not by convention.
- **The raw zone is a landing strip.** The lake stays the system of record; a document is
  removed from the raw zone as soon as it has been pseudonymized.
- **Keys follow the compartments.** One root per matter, one derived key per zone and one for
  the vault. Access to the de-identified zone gives no ability to decrypt the raw one, and
  shredding the root makes every copy of the matter unreadable, backups included.
- **Trim, then rank.** The index drops every compartment the caller is denied before it
  scores anything, and scores use no corpus statistics, so a denied compartment cannot leak
  through ranking.
- **Erasure is irreversible forgetting.** A data subject's tokens are tombstoned in the
  vault, so nobody can re-identify them again, and every training set that used their
  documents is tainted until rebuilt.
- **The trust boundary is a setting.** `LEXGUARD_BOUNDARY` is `local_only`, `private_cloud`
  or `deidentified_egress`. The default is the strictest.

## What the measurements say

Detection is rule-based, with no name lists. Matching is by overlap unless noted.

| Corpus | Precision | Recall |
| --- | --- | --- |
| Synthetic templates, 20 seeds, exact match | 1.00 | 1.00 |
| Handwritten held-out texts, 60 labels, scored once | 0.84 | 0.88 |

On the held-out texts, RUT, e-mail, case numbers and bank accounts were all found. People
reached 0.91 recall at 0.67 precision (capitalized phrases are masked rather than missed) and
organizations 0.56 recall (only names with a legal form are found). Rules alone are not
enough for real documents: add an NER model through `deid.Recognizer`, keep the outbound
residual scan, and review a sample by hand before trusting a new document type.

The classifier and memorization numbers come from a template corpus and only show the
mechanics: the canary test flags a secret planted once (10 of 10 bits) and not an unseen one.

## Limits of the concept

- `KeyRing` lives in memory and stands in for a KMS or HSM; unwrap calls should be audited there.
- The policy trusts the `Principal` it is given. Identity, roles and matter assignment must
  come from the firm's identity provider and matter-intake system.
- The audit chain detects edits, removals and reordering, but a cleanly truncated tail needs
  the head anchored outside the file.
- The serving index is in memory, and training manifests are plain files without a signature.
- DP-SGD reports its parameters and computes no epsilon; an established accountant must.
- There is no connector to a real lake, no concurrency control and no key rotation.

## Development

```
uv run pytest
uv run ruff check
uv run ruff format
```
