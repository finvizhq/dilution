"""Section selection for periodic filings (10-K, 10-Q, 20-F).

Drops obvious-noise sections (Risk Factors, Properties, Legal
Proceedings, Mine Safety, internal controls boilerplate, exhibit
indexes) before the markdown reaches the LLM. The goal is to shrink
the prompt without losing dilution signal — risk factors alone are
typically 30-50% of a 10-K.

Uses edgartools' ``Filing.obj().sections`` for boundary detection
rather than regex on markdown. Section headers in our raw markdown
are inconsistently formatted (sometimes inline with prose, sometimes
TOC-only) so regex was a silent-recall risk; edgartools parses the
HTML structure and gives us validated boundaries.

Safe-fail: if the typed object can't be built, sections aren't
parsed, no keep-list sections validate, or the assembled text is
suspiciously small, we return ``None`` and the caller falls back to
the full document. Coverage > size savings (CLAUDE.md operating
policy).
"""

from __future__ import annotations

import logging
from typing import Any

log = logging.getLogger(__name__)


# Keep-list keyed by canonical form. Section keys mirror edgartools'
# ``part_<roman>_item_<n>[<letter>]`` convention. Anything not listed
# here is dropped.
#
# Rationale per item is in the design discussion; in short:
#   - 10-K/10-Q: keep MD&A, financials + notes, unregistered-sales
#     disclosures, subsequent events, related-party / exec-comp.
#   - 20-F: keep Item 3 (includes 3.B Capitalization — sub-items don't
#     split in edgartools), Item 5 (MD&A analog), 6/7 (directors,
#     major shareholders), 8/17/18 (financials), 10 (material
#     contracts), 13 (defaults), 14 (material modifications to
#     security-holder rights).
KEEP_SECTIONS: dict[str, set[str]] = {
    "10-K": {
        "part_ii_item_5",   # Market for Registrant's Common Equity
        "part_ii_item_7",   # MD&A
        "part_ii_item_8",   # Financial Statements + notes
        "part_ii_item_9b",  # Other Information / Subsequent Events
        "part_iii_item_11", # Executive Compensation
        "part_iii_item_12", # Beneficial Ownership
        "part_iii_item_13", # Related-Party Transactions
    },
    "10-Q": {
        "part_i_item_1",    # Financial Statements + notes
        "part_i_item_2",    # MD&A
        "part_ii_item_2",   # Unregistered Sales of Equity Securities
        "part_ii_item_3",   # Defaults Upon Senior Securities
        "part_ii_item_5",   # Other Information
    },
    "20-F": {
        "part_i_item_3",    # Key Information (incl. 3.B Capitalization)
        "part_i_item_5",    # Operating and Financial Review (MD&A)
        "part_i_item_6",    # Directors / Senior Management (equity comp)
        "part_i_item_7",    # Major Shareholders / Related-Party
        "part_i_item_8",    # Financial Information
        "part_i_item_10",   # Additional Information (Material Contracts)
        "part_ii_item_13",  # Defaults / Dividend Arrearages
        "part_ii_item_14",  # Material Modifications to Rights
        "part_iii_item_17", # Financial Statements
        "part_iii_item_18", # Financial Statements
    },
}

# Minimum size of the assembled keep-list text. Below this we assume
# section parsing went sideways and fall back to the full document.
# A real 10-K's keep-list is typically hundreds of KB; anything under
# 5KB is almost certainly a parse failure or a stub filing.
MIN_KEPT_CHARS = 5_000

# Per-section confidence floor. Edgartools tags each section with a
# 0..1 confidence on its boundary detection. Most sections come back
# at ~0.95; anything below this floor on a keep-list section makes us
# bail to the full-doc fallback rather than risk a truncated Item 8.
MIN_SECTION_CONFIDENCE = 0.7


def _normalize_form(form: str) -> str | None:
    """Map a raw form code to a keep-set key. Returns None if the form
    isn't periodic-with-sections (e.g., 40-F, 8-K).

    Handles ``/A`` amendments by stripping the suffix — the section
    structure of 10-K/A matches 10-K.
    """
    if not form:
        return None
    base = form.strip().split("/")[0].strip().upper()
    if base in KEEP_SECTIONS:
        return base
    return None


def select_text(filing, form: str) -> tuple[str | None, dict]:
    """Return (selected_text, stats) for a periodic filing.

    Returns (None, stats) when section selection should be skipped —
    the caller is expected to fall back to the full document. The
    stats dict is suitable for log.info('%s', stats) and includes:
      - reason: only set when text is None
      - kept: list of section keys included
      - dropped: list of section keys excluded
      - kept_chars / dropped_chars
    """
    keep_key = _normalize_form(form)
    if keep_key is None:
        return None, {"reason": f"form {form!r} not in keep-list"}
    keep = KEEP_SECTIONS[keep_key]

    try:
        obj = filing.obj()
    except Exception as e:
        return None, {"reason": f"obj() failed: {e}"}
    if obj is None:
        return None, {"reason": "obj() returned None"}

    sections = getattr(obj, "sections", None)
    if not sections:
        return None, {"reason": "no sections parsed"}
    try:
        section_items = list(sections.items())
    except Exception as e:
        return None, {"reason": f"sections.items() failed: {e}"}

    kept_keys: list[str] = []
    dropped_keys: list[str] = []
    kept_parts: list[str] = []
    kept_chars = 0
    dropped_chars = 0
    low_confidence_keep: list[tuple[str, float]] = []

    for key, sec in section_items:
        try:
            text = sec.text() or ""
        except Exception as e:
            return None, {"reason": f"sec.text() failed on {key}: {e}"}
        n = len(text)
        if key not in keep:
            dropped_keys.append(key)
            dropped_chars += n
            continue
        # In keep-list. If the parser's boundary confidence on a
        # kept section is low, drop the whole filing back to the
        # full-document path rather than risk a truncated Item 8.
        # We bias hard toward coverage (CLAUDE.md).
        confidence = float(getattr(sec, "confidence", 1.0) or 0.0)
        if confidence < MIN_SECTION_CONFIDENCE:
            low_confidence_keep.append((key, confidence))
            continue
        kept_keys.append(key)
        kept_chars += n
        if text.strip():
            # Lead each kept section with a synthesized heading so the
            # LLM has explicit anchoring even when the body markdown
            # is dense / unstructured. ``title`` on Section is just
            # the slug, so build a human-readable label from part +
            # item.
            part = (getattr(sec, "part", "") or "").strip()
            item = (getattr(sec, "item", "") or "").strip()
            if part and item:
                heading = f"Part {part}, Item {item}"
            else:
                heading = (getattr(sec, "title", None) or key).strip()
            kept_parts.append(f"\n\n## {heading}\n\n{text}")

    if low_confidence_keep:
        return None, {
            "reason": f"keep-section low confidence: {low_confidence_keep}",
            "kept": kept_keys,
            "dropped": dropped_keys,
        }
    if not kept_keys:
        return None, {
            "reason": "no keep sections present",
            "dropped": dropped_keys,
        }
    if kept_chars < MIN_KEPT_CHARS:
        return None, {
            "reason": f"kept_chars {kept_chars} below floor {MIN_KEPT_CHARS}",
            "kept": kept_keys,
            "dropped": dropped_keys,
        }

    text = "".join(kept_parts).lstrip("\n")
    return text, {
        "kept": kept_keys,
        "dropped": dropped_keys,
        "kept_chars": kept_chars,
        "dropped_chars": dropped_chars,
    }


def is_periodic_with_sections(form: str) -> bool:
    """True iff this form has a keep-list configured. False for 40-F,
    1-K, 8-K etc., which fall through to the full-document path."""
    return _normalize_form(form) is not None


__all__ = [
    "KEEP_SECTIONS",
    "MIN_KEPT_CHARS",
    "is_periodic_with_sections",
    "select_text",
]
