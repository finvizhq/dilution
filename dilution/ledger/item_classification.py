"""Deterministic 8-K dilution-relevance gate via the EDGAR ``items`` field.

Every 8-K carries a canonical item-code list in EDGAR's submissions index
(stored in ``dilution_filings.items``), e.g. ``"1.01,3.02,9.01"``. Those
codes are SEC-canonical and available before any LLM call — the same class
of free, deterministic signal as the ``333-`` file number
(``registration_family.py``) and the Rule 457 fee-table code
(``_exhibit_provisions.py``). This module turns the signal into a verdict:

  skip         — no dilution-relevant item code AND no substantive exhibit
                 attached. The filing cannot carry a cap-table event, so the
                 walker skips the LLM round-trip entirely (mirrors the resale
                 pre-screen in ``walker._walk_one``). Pure cost saving; the
                 skipped filing would have produced ``note_no_event`` anyway.
  must_record  — a hard issuance signal is present (Item 3.02 unregistered
                 sale / 2.03 direct financial obligation / 5.03 charter
                 amendment, or an EX-3.x Certificate of Designation / EX-4.x
                 warrant-or-note exhibit). Computed for a future
                 must-not-be-silent net; not yet wired into the walker.
  process      — ordinary 8-K; hand to the LLM unchanged.

Why a skip needs the EXHIBIT co-gate, not items alone: this universe is
low-quality microcaps that under-tag. 8-Ks tagged only ``9.01`` (exhibits)
or ``2.02,9.01`` (earnings) have created warrants / convertibles / equity
lines because the financing was bundled into an EX-10 agreement or an EX-99
press release. Items alone would drop those (verified 2026-06-01: 16 real
ledger rows across 8 filings). Requiring "no substantive exhibit either"
rescues every one — a real financing always attaches *something*. EX-99 is
in the substantive set deliberately; the price is that earnings 8-Ks (which
all carry EX-99.1) are never skipped, but that keeps the gate zero-miss.

Fail-open: an 8-K with neither an items field nor any fetched doc falls
through to ``process``. 8-K-only by construction — 6-K / 424B / S-3 carry no
items field (verified 100% empty), so this never gates a foreign-issuer 6-K.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass

from db import get_conn

log = logging.getLogger(__name__)

# Item codes that can carry a cap-table event. Mirrors the dilution-relevant
# 8-K items curated in periodic_sections.KEEP_SECTIONS["8-K"] (item_101 etc.),
# plus 2.01 (completion of acquisition → de-SPAC share issuance). Excludes
# 9.01 (exhibit index — present on nearly every 8-K) and the non-dilutive
# 2.02 earnings / 5.02 officers / 5.07 votes / 3.01 listing / 4.01 auditor.
DILUTIVE_ITEMS = frozenset({
    "1.01",  # entry into a material definitive agreement (SPA / ATM / underwriter)
    "1.02",  # termination of a material agreement (close_instrument)
    "2.01",  # completion of acquisition (de-SPAC issuance)
    "2.03",  # creation of a direct financial obligation (convertible / note)
    "2.04",  # triggering event (acceleration / redemption)
    "3.02",  # unregistered sale of equity securities (PIPE / private placement)
    "3.03",  # material modification to rights of security holders
    "5.01",  # change in control
    "5.03",  # amendments to charter (Certificate of Designation → preferred)
    "7.01",  # Reg FD (pricing / offering announcements)
    "8.01",  # other events (closing announcements)
})

# Exhibit families substantive enough to keep an otherwise-empty 8-K in the
# LLM path. EX-99 is included deliberately: financings are routinely
# announced only in an EX-99.1 press release on an 8-K whose items look
# empty. EX-21/23/24/101/104 are never fetched into dilution_raw (fetch_raw
# skips them), so they can't appear here.
SUBSTANTIVE_EXHIBIT_PREFIXES = (
    "EX-1.", "EX-2.", "EX-3.", "EX-4.", "EX-10.", "EX-99.",
)

# Hard "a dilutive instrument is disclosed here" signals → must_record.
MUST_RECORD_ITEMS = frozenset({"3.02", "2.03", "5.03"})
MUST_RECORD_EXHIBIT_PREFIXES = ("EX-3.", "EX-4.")


@dataclass(frozen=True)
class Item8KVerdict:
    """Three-way verdict for an 8-K. ``skip`` and ``must_record`` are
    mutually exclusive (a skipped filing is never must_record); both False
    means ``process``."""
    skip: bool
    must_record: bool
    reason: str


_PROCESS = Item8KVerdict(skip=False, must_record=False, reason="")


def classify_8k(cik: int, accession: str, form: str) -> Item8KVerdict:
    """Verdict from the 8-K's item codes + attached exhibit types.

    Only 8-K / 8-K/A are gated; every other form returns ``process``
    (fail-open). See the module docstring for skip / must_record / process
    semantics and why the skip co-gates on exhibits.
    """
    if not (form or "").upper().startswith("8-K"):
        return _PROCESS

    with get_conn() as conn:
        frow = conn.execute(
            "SELECT items FROM dilution_filings "
            "WHERE cik = ? AND accession_number = ?",
            (cik, accession),
        ).fetchone()
        doc_rows = conn.execute(
            "SELECT doc_type FROM dilution_raw WHERE accession_number = ?",
            (accession,),
        ).fetchall()

    items = (frow["items"] if frow else None) or ""
    doc_types = [(d["doc_type"] or "") for d in doc_rows]

    # Fail-open: we know nothing about this filing → process it.
    if not items and not doc_types:
        return _PROCESS

    item_codes = {c.strip() for c in items.split(",") if c.strip()}
    has_dilutive_item = bool(item_codes & DILUTIVE_ITEMS)
    has_substantive_exhibit = any(
        dt.startswith(p)
        for dt in doc_types
        for p in SUBSTANTIVE_EXHIBIT_PREFIXES
    )

    if not has_dilutive_item and not has_substantive_exhibit:
        return Item8KVerdict(
            skip=True, must_record=False,
            reason=f"no dilutive item + no substantive exhibit "
                   f"(items={items or '∅'})",
        )

    must_items = sorted(item_codes & MUST_RECORD_ITEMS)
    must_exhibits = sorted({
        dt for dt in doc_types
        if any(dt.startswith(p) for p in MUST_RECORD_EXHIBIT_PREFIXES)
    })
    if must_items or must_exhibits:
        reason = "; ".join(filter(None, [
            f"items={','.join(must_items)}" if must_items else "",
            f"exhibits={','.join(must_exhibits)}" if must_exhibits else "",
        ]))
        return Item8KVerdict(skip=False, must_record=True, reason=reason)

    return _PROCESS


# ─── Per-filing create-tool pruning (non-periodic walks) ────────────
# A walk ships ALL of a form's tool schemas (~17K tokens for an 8-K — 71%
# of the call's input), but most create_* tools are irrelevant to any given
# filing. We drop the create schemas a filing can't plausibly need, decided
# from four deterministic signals OR'd together (keep on ANY): body keyword,
# 8-K item code, attached exhibit type, open-ledger type. Over-keeping is
# free (an unused tool just isn't called); only a false drop costs, so the
# OR drives the miss rate to ~0 (verified full-DB: 0/977 prunable creates).
#
# Guardrails proven by the full-DB check:
#   - non-periodic forms only — periodics mention every type, FPIs say
#     "preference shares", and the anchor overhang pass is authoritative.
#   - only the four NAMED creates are prunable. create_equity (generic) and
#     create_equity_line (varied naming — "Common Stock Purchase Agreement",
#     "Pre-Paid Advance") are never pruned; amend_*/record_*/close/
#     note_no_event/apply_split are never touched.
#   - EX-10 (an SPA can issue any type) rescues all four cross-type.
# Backstops if a novel pattern ever slips through: pruning fails SOFT (the
# walk still runs, only a tool is removed) and a missed instrument is
# re-created at the next periodic by the overhang extractor.

# Allowlist: prune ONLY on high-volume EVENT forms whose focused prose names
# its instruments reliably (8-K + pricing/takedown prospectuses). Everything
# else — periodics (FPI "preference shares", anchor is authoritative) AND
# registration statements (S-1/S-3/F-3/MEF cover-page edge cases like an
# S-3MEF embedded ATM) — keeps the FULL tool set. Safe-by-default: an
# unknown form is never pruned.
_PRUNE_FORMS = frozenset({
    "8-K", "424B2", "424B3", "424B4", "424B5", "424B8", "FWP", "SUPPL", "425",
})

# The only prunable create tools: instrument type → tool name.
_CREATE_TOOL_BY_TYPE = {
    "warrant": "create_warrant",
    "preferred": "create_preferred",
    "convertible": "create_convertible",
    "atm": "create_atm",
}

# Body keyword per type — the instrument's own legally-conventional name, so
# recall is ~100%. "preference shares" covers FPI (20-F) terminology.
_CREATE_KEYWORDS = {
    "warrant": re.compile(r"warrant", re.I),
    "preferred": re.compile(r"preferred|preference shares?", re.I),
    "convertible": re.compile(r"convertible|debenture|promissory note", re.I),
    "atm": re.compile(
        r"at[- ]the[- ]market|sales agreement|equity distribution", re.I),
}

# Item-code rescue: 5.03 (Cert of Designation) → preferred; 2.03 (direct
# financial obligation) → convertible.
_CREATE_ITEM_RESCUE = {
    "preferred": frozenset({"5.03"}),
    "convertible": frozenset({"2.03"}),
}

# Type-specific exhibit rescue (EX-3 cert-of-designation → preferred, EX-4
# warrant/indenture, EX-1 underwriting/sales-agreement → atm).
_CREATE_EXHIBIT_RESCUE = {
    "warrant": ("EX-4.",),
    "preferred": ("EX-3.",),
    "convertible": ("EX-4.",),
    "atm": ("EX-1.",),
}
# Cross-type: an SPA (EX-10) can issue any of the four. Keep on its presence
# (closes the lone full-DB warrant residual; over-keeping is free).
_CREATE_EXHIBIT_RESCUE_CROSS = ("EX-10.",)


def _keep_create_type(
    typ: str, *, filing_text: str, item_codes: set[str],
    doc_types: list[str], open_types: set[str],
) -> bool:
    """True if ``create_<typ>`` should stay attached — any of the four
    signals fires. Pure (no I/O) for testability."""
    if _CREATE_KEYWORDS[typ].search(filing_text or ""):
        return True
    if item_codes & _CREATE_ITEM_RESCUE.get(typ, frozenset()):
        return True
    prefixes = _CREATE_EXHIBIT_RESCUE.get(typ, ()) + _CREATE_EXHIBIT_RESCUE_CROSS
    if any(dt.startswith(p) for dt in doc_types for p in prefixes):
        return True
    if typ in open_types:
        return True
    return False


def prune_create_tools(tools, *, form, accession, items, filing_text,
                       active_rows):
    """Drop the create_* schemas a non-periodic filing can't plausibly need.

    Returns the input list unchanged for periodic forms (full tool set kept)
    or when nothing is prunable. Only the four named create tools are ever
    dropped; everything else passes through. One cheap exhibit read by
    accession — items / text / ledger are already in hand at the call site.
    """
    form_base = (form or "").upper().split("/")[0]
    if form_base not in _PRUNE_FORMS:
        return tools

    item_codes = {c.strip() for c in (items or "").split(",") if c.strip()}
    open_types = {(r.get("type") or "") for r in (active_rows or [])}
    with get_conn() as conn:
        doc_rows = conn.execute(
            "SELECT doc_type FROM dilution_raw WHERE accession_number = ?",
            (accession,),
        ).fetchall()
    doc_types = [(d["doc_type"] or "") for d in doc_rows]

    drop = {
        toolname for typ, toolname in _CREATE_TOOL_BY_TYPE.items()
        if not _keep_create_type(
            typ, filing_text=filing_text, item_codes=item_codes,
            doc_types=doc_types, open_types=open_types,
        )
    }
    if not drop:
        return tools
    kept = [t for t in tools if t.name not in drop]
    log.info("  %s — pruned %d create tool(s): %s",
             accession, len(drop), ",".join(sorted(drop)))
    return kept


__all__ = [
    "Item8KVerdict",
    "DILUTIVE_ITEMS",
    "SUBSTANTIVE_EXHIBIT_PREFIXES",
    "MUST_RECORD_ITEMS",
    "MUST_RECORD_EXHIBIT_PREFIXES",
    "classify_8k",
    "prune_create_tools",
]


# ─── Content-side expected-call classes (must_record extension) ──────
# Round-4: the dominant residual eval-defect class is walk-to-walk
# EMISSION variance — temp-0 Gemini intermittently skips create/close
# calls on event filings that plainly disclose them (SCNI's Apr-2026
# 6-K: 4 calls one walk, 1 the next; XTIA's Maxim termination never
# emitted). The metadata must_record net cannot catch these: 6-Ks carry
# no item codes at all (every FPI ticker), and a thin-but-not-silent
# response (one call where the filing discloses two events) passes the
# `not real` gate. These HIGH-PRECISION patterns assert "this event
# filing textually announces transaction X" — when the response carries
# no matching call, walker_llm takes the same one-shot second look.
# A false positive costs one LLM call and resolves as note_no_event, so
# precision matters more than recall: every pattern requires a
# transaction-shaped phrase (counts / dollar amounts + instrument
# nouns), never a bare keyword, and issuance classes additionally
# require an issuance verb nearby. Event forms only — periodics
# re-describe everything and are owned by anchor reconciliation.

_EXPECTED_CLASS_FORMS = frozenset({"8-K", "6-K"})

_ISSUANCE_VERB_RE = re.compile(
    r"\b(?:issued?|issuance|agreed to issue|will issue|sold|sale|"
    r"entered into|granted?|closing)\b", re.I)

# "previously issued warrants…" recaps are history, not an event.
_RECAP_RE = re.compile(
    r"previously (?:issued|reported|disclosed|announced and closed)|"
    r"currently outstanding", re.I)

_EXPECTED_CLASS_PATTERNS: dict[str, re.Pattern] = {
    "warrant": re.compile(
        r"(?:new )?warrants? to purchase (?:up to )?(?:an aggregate of )?"
        r"[\d,]{3,}", re.I),
    "convertible": re.compile(
        r"(?:convertible|promissory) notes? in the (?:aggregate )?"
        r"principal amount of \$[\d,]+"
        r"|aggregate principal amount of \$[\d,]+[^.]{0,120}?"
        r"(?:convertible|promissory) note", re.I),
    "preferred": re.compile(
        r"shares? of (?:newly designated )?series [a-z0-9-]+\s+"
        r"(?:convertible )?preferred stock[^.]{0,140}?"
        r"(?:purchase agreement|exchange agreement|stated value)", re.I),
    "atm": re.compile(
        r"entered into [^.]{0,120}?(?:open market sale|sales|equity"
        r" distribution) agreement[^.]{0,160}?"
        r"(?:up to|aggregate offering price)", re.I),
    "equity_line": re.compile(
        r"entered into [^.]{0,120}?(?:standby equity|equity (?:purchase|"
        r"line of credit)|share purchase) agreement"
        r"[^.]{0,160}?(?:up to \$|commitment)", re.I),
    # Agreement-shaped targets only ("the … Sales Agreement"): warrant
    # and note forms carry "termination of this Warrant/Note" boilerplate
    # in every exhibit, which is a provision, not an event.
    "close": re.compile(
        r"terminat(?:ed|es|ion of) the [^.]{0,80}?"
        r"(?:sales agreement|purchase agreement|sale agreement|"
        r"equity distribution agreement|at[- ]the[- ]market)", re.I),
}

_EXPECTED_EVIDENCE_WINDOW = 240


def expected_call_classes(filing_text: str, form: str) -> dict[str, str]:
    """class → evidence snippet, for event filings whose own text
    announces a dilutive transaction. {} for non-event forms / no hits.
    Pure (no I/O) for testability."""
    form_base = (form or "").upper().split("/")[0]
    if form_base not in _EXPECTED_CLASS_FORMS or not filing_text:
        return {}
    out: dict[str, str] = {}
    for cls, rex in _EXPECTED_CLASS_PATTERNS.items():
        for m in rex.finditer(filing_text):
            lo = max(0, m.start() - _EXPECTED_EVIDENCE_WINDOW)
            hi = min(len(filing_text), m.end() + _EXPECTED_EVIDENCE_WINDOW)
            window = filing_text[lo:hi]
            if _RECAP_RE.search(window):
                continue
            if cls != "close" and not _ISSUANCE_VERB_RE.search(window):
                continue
            out[cls] = " ".join(m.group(0).split())[:200]
            break
    return out
