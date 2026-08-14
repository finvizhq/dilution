"""Capitalization-table ledger for dilutive instruments.

The ledger is the canonical store: one row per outstanding (or
historical) instrument tranche, mutated chronologically as filings
arrive. Replaces the prior per-filing event-extraction → post-hoc
clustering approach with a stateful cap table that mirrors how a
human dilution analyst tracks instruments.

Modules:
  store        — CRUD over dilution_ledger + apply_mutations
  mutations    — Pydantic models for the mutation vocabulary
  validate     — pre-apply mutation validation
  view         — render an open-ledger snapshot for the walker prompt
  seed         — initial-state extraction from earliest periodic filing
  walker       — chronological filing walker (the orchestrator)
  walker_prompt, walker_llm — the LLM call layer
  anchor       — periodic-filing reconciliation
  cards        — projection from ledger rows to product cards

The walker is the only thing that owns chronological ordering and
mutation application. Everything else operates on the ledger as
plain SQL or pure-function inputs.
"""
