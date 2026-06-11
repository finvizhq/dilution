"""I.B.6 "baby shelf" cover-legend extraction from prospectus documents.

Form S-3 General Instruction I.B.6, Instruction 7 REQUIRES every issuer
relying on I.B.6 to print its own calculation on the prospectus front
cover: the aggregate market value of common equity held by
non-affiliates (the public float test) and the amount of securities
offered pursuant to I.B.6 in the trailing 12 calendar months. The
standard legend reads:

    "As of August 29, 2025, the aggregate market value of our
    outstanding common stock held by non-affiliates, or public float,
    was approximately $61.7 million, based on 7,113,902 shares of
    outstanding common stock held by non-affiliates at a price of
    $8.67 per share, which was the closing price of our common stock
    on Nasdaq on August 12, 2025. We have not offered any securities
    pursuant to General Instruction I.B.6 of Form S-3 during the
    prior 12-calendar-month period..."

This makes the legend a deterministic REGIME STAMP (thin-prompt-
thick-core: no LLM involved):

  - legend present on a primary prospectus  → that offering was filed
    under I.B.6 (baby-shelf capped);
  - legend absent from a primary prospectus → it was filed under
    I.B.1 (float ≥ $75M at filing, or the Instruction-3 ratchet had
    lifted the cap) — and per C&DI 116.26 (2026-03-19) an ATM
    prospectus supplement filed under I.B.1 keeps its FULL capacity
    even after the issuer later drops below $75M at a Section
    10(a)(3) update. Regime therefore attaches at FILING time; a
    live float×price test cannot reproduce it (KSCP: $50M ATM
    supplement carries no legend → unrestricted, while its live
    float value sits below $75M).

Documents bind the base prospectus after the supplement, so the legend
can appear anywhere in the text — we scan whole documents and take the
first legend cluster (supplement cover precedes any bound base
prospectus).
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from functools import lru_cache

from db import get_conn

log = logging.getLogger(__name__)

# Forms that carry a primary-offering prospectus cover, where Instruction
# 7 forces the legend when (and only when) the offering relies on I.B.6.
# 424B3 / B7 / B8 are typically resale prospectuses — legend absence
# there means nothing, so they never stamp a regime. ASR forms are WKSI
# (I.B.1 by definition) and never carry the legend either; they are
# deliberately NOT regime evidence because WKSI status is handled by
# shelf_status, not this module.
PRIMARY_PROSPECTUS_FORMS = frozenset({
    "424B1", "424B2", "424B4", "424B5",
    "S-3", "S-3/A", "F-3", "F-3/A",
})

# A regime stamp older than this is ignored: S-3 shelves live 3 years,
# ATM supplements get refreshed, and every annual 10-K re-tests the
# float (Section 10(a)(3)). 540 days matches the s1_offering staleness
# convention.
REGIME_STAMP_MAX_AGE_DAYS = 540

# I.B.6 is Form S-3's limited-primary instruction; Form F-3's analog
# (FPIs: AACG, BTOG, CHNR, PPBT, BTCT...) is I.B.5 — both phrased
# "General Instruction(s) I.B.5/6", with optional trailing dot and the
# occasional missing "General". Since S-3 has its own unrelated I.B.5,
# an anchor hit only counts as a legend when baby-shelf context words
# appear nearby (see _CONTEXT_RE in parse_ib6_legend).
_ANCHOR_RE = re.compile(r"(?:General\s+)?Instructions?\s+I\.\s?B\.?\s?[56]",
                        re.IGNORECASE)
_CONTEXT_RE = re.compile(
    r"one-?third|held\s+by\s+non[\s-]{0,2}affiliates|"
    r"\$\s?75(?:,000,000|\s+million)|baby\s+shelf",
    re.IGNORECASE,
)

# Resale-only registrations (selling-stockholder S-3s under I.B.3, and
# resale 424Bs off them) carry no I.B.6 legend because resales aren't
# capped — legend ABSENCE there is not regime evidence. Empirically the
# discriminator is unambiguous: resale docs mention "selling
# stockholders" 14-34× in the cover region, genuine primaries 0×
# (GCTK/XTIA/BJDX/MBRX false 'unrestricted' stamps vs ATPC/KSCP/SDOT
# genuine). Threshold 3 leaves wide margin on both sides.
_RESALE_RE = re.compile(r"selling\s+(?:stock|share)holders?", re.IGNORECASE)
_RESALE_MENTIONS_MIN = 3
_RESALE_SCAN_CHARS = 40_000

# "Any char but a sentence-ending period" — a period FOLLOWED BY A DIGIT
# is a decimal point ("$61.7 million", "$16,660,959.00") and must not
# terminate the sentence scan. Same for periods inside "I.B.6"/"U.S."
# (single-letter abbreviations) and form references ("Form S-3." fused
# directly onto the next sentence — "S-3.As of the date..." appears in
# production text).
_NODOT = (r"(?:[^.]|\.(?=\d)|(?<=\b[A-Z])\.(?=[A-Z])"
          r"|(?<=[SF]-\d)\.|(?<=[SF]-\d\d)\.)")
# Sentence end = a '.' that is none of _NODOT's mid-sentence dots: not a
# decimal, not "Form S-3."-fused, not the dot of a single-letter
# abbreviation ("General Instruction I.B.6" appears MID-sentence when
# the legend interposes it before the value — LMFA/COCH/PPBT).
_SENT_END = r"(?<![SF]-\d)(?<![SF]-\d\d)(?<!\b[A-Z])\.(?!\d)"

# Sentence A — the float calculation. Anchored on "held by non-affiliates"
# inside an "aggregate market value" sentence; tolerant of "common stock" /
# "common equity" / "voting and non-voting" wording, of "is/was
# (approximately)", and of interposed qualifiers ("market value worldwide
# of our outstanding...", PPBT).
_FLOAT_SENT_RE = re.compile(
    rf"aggregate\s+market\s+value\s+(?:\w+\s+){{0,2}}of\s+(?:our|the)\s+"
    rf"{_NODOT}{{0,160}}?"
    rf"held\s+by\s+non[\s-]{{0,2}}affiliates{_NODOT}{{0,500}}?{_SENT_END}",
    re.IGNORECASE | re.DOTALL,
)
# Float value: "(was|is) (approximately) $X (million)". A guard rejects
# threshold mentions ("remains below $75 million", "less than
# $75,000,000") — those state the RULE, not the issuer's float.
_FLOAT_VALUE_RE = re.compile(
    r"(?:was|is)\s+(?:approximately\s+)?(?:US)?\$\s?([\d,]+(?:\.\d+)?)"
    r"\s*(million|billion)?",
    re.IGNORECASE,
)
_THRESHOLD_GUARD_RE = re.compile(
    r"(?:less\s+than|below|under|exceed(?:s|ing)?|more\s+than)\s*"
    r"(?:approximately\s+)?$",
    re.IGNORECASE,
)
# "held by non-affiliates" occurrences — the non-affiliate share count is
# the number nearest BEFORE one of these (covers "N shares ... held by
# non-affiliates", "of which N (shares) were held by non-affiliates",
# "approximately N of which were held by non-affiliates").
_HELD_BY_RE = re.compile(r"held\s+by\s+non[\s-]{0,2}affiliates", re.IGNORECASE)
_NUMBER_RE = re.compile(r"([\d]{1,3}(?:,\d{3})+|\d{4,})")
_DOLLAR_RE = re.compile(r"\$\s?([\d,]+(?:\.\d+)?)")
_PRICE_STYLE_RE = re.compile(
    r"(?:price\s+(?:per\s+(?:share|ADS)\s+)?of|"
    r"per\s+(?:share|ADS)\s+price\s+of|"
    r"at\s+a\s+price\s+of)\s+(?:US)?\$\s?([\d,]+(?:\.\d+)?)",
    re.IGNORECASE,
)
_DATE_RE = re.compile(r"([A-Z][a-z]+\s+\d{1,2},\s+\d{4})")
_AS_OF_RE = re.compile(r"As\s+of\s+([A-Z][a-z]+\s+\d{1,2},\s+\d{4})",
                       re.IGNORECASE)

# Sentence B — trailing-12-month consumption. Zero variant first.
_INSTR_REF = r"(?:General\s+)?Instructions?\s+I\.B\.?\s?[56]"
_SOLD_NONE_RE = re.compile(
    rf"have\s+not\s+(?:offered(?:\s+(?:or|and)\s+sold)?|sold)\s+any\s+"
    rf"securities\s+pursuant\s+to\s+{_INSTR_REF}",
    re.IGNORECASE,
)
_SOLD_AMOUNT_RES = (
    # "...we have offered/sold $X (million) of securities ... pursuant to
    #  General Instruction I.B.6..." (lazy → first $ after the verb, so a
    #  parenthetical second amount doesn't displace it)
    re.compile(
        rf"(?:offered|sold){_NODOT}{{0,200}}?(?:US)?\$\s?([\d,]+(?:\.\d+)?)"
        rf"\s*(million|billion)?{_NODOT}{{0,300}}?"
        rf"pursuant\s+to\s+{_INSTR_REF}",
        re.IGNORECASE | re.DOTALL,
    ),
    # "...pursuant to General Instruction I.B.6 ... we have offered and
    #  sold $X..." (reversed order)
    re.compile(
        rf"pursuant\s+to\s+{_INSTR_REF}{_NODOT}{{0,200}}?"
        rf"(?:offered|sold){_NODOT}{{0,120}}?(?:US)?\$\s?([\d,]+(?:\.\d+)?)"
        rf"\s*(million|billion)?",
        re.IGNORECASE | re.DOTALL,
    ),
)

_MULT = {"million": 1e6, "billion": 1e9}

# Sanity gates — junk captures (page numbers, the $75M threshold, a bare
# "$5" from a mangled table) get nulled rather than poison downstream
# arithmetic. Bounds are deliberately loose: nano-cap floats run ~$1M,
# share counts after 1:50+ reverse splits run ~500K.
_VALUE_BOUNDS = (1e5, 5e9)
_SHARES_BOUNDS = (1e4, 5e10)
_PRICE_BOUNDS = (0.001, 10_000.0)


def _in(v, lo_hi) -> bool:
    return v is not None and lo_hi[0] <= v <= lo_hi[1]


def _num(s: str, unit: str | None = None) -> float | None:
    try:
        v = float(s.replace(",", ""))
    except (ValueError, AttributeError):
        return None
    return v * _MULT.get((unit or "").lower(), 1.0)


def _parse_legend_date(s: str) -> str | None:
    try:
        return datetime.strptime(s, "%B %d, %Y").date().isoformat()
    except ValueError:
        return None


@dataclass(frozen=True)
class Ib6Legend:
    """One parsed cover legend. All value fields may be None when the
    sentence deviates from the template — `present` alone is still a
    valid regime stamp."""
    present: bool
    float_value_usd: float | None = None
    non_affiliate_shares: float | None = None
    price_usd: float | None = None
    price_date: str | None = None
    as_of_date: str | None = None
    sold_12mo_usd: float | None = None   # 0.0 = explicit "not offered any"


def parse_ib6_legend(text: str) -> Ib6Legend | None:
    """Extract the first I.B.6 legend cluster from document text.
    Returns None when the document carries no I.B.6 reference at all."""
    if not text:
        return None
    # First anchor whose surrounding window also carries baby-shelf
    # context words — Form S-3 has its own unrelated I.B.5, and
    # eligibility boilerplate cites I.B.1 ranges, so a bare instruction
    # reference is not a legend.
    anchor = None
    window = ""
    win_start = 0
    for cand in _ANCHOR_RE.finditer(text):
        ws = max(0, cand.start() - 4000)
        win = text[ws: cand.end() + 4000]
        if _CONTEXT_RE.search(win):
            anchor, window, win_start = cand, win, ws
            break
    if anchor is None:
        return None

    # Examine the float sentence nearest the anchor (the legend cluster
    # is float-sentence → consumption-sentence → cap-sentence, so the
    # float sentence normally ends just BEFORE the first anchor).
    # The window keeps an MD&A mention 100K chars away from binding to
    # an unrelated "market value" sentence.
    anchor_rel = anchor.start() - win_start

    # Among ALL candidate float sentences in the window, prefer the one
    # that actually yields numbers — the one-third CAP sentence
    # ("...exceeding more than one-third of the aggregate market value
    # ... held by non-affiliates in any 12-month period...") matches the
    # same surface pattern but carries none (AACG), and cluster order
    # varies: most issuers put the calculation BEFORE the cap sentence,
    # LGVN puts it after. Ties break toward anchor proximity.
    out: dict = {}
    cands = list(_FLOAT_SENT_RE.finditer(window))
    fs = None
    fields: dict = {}
    for m in sorted(cands, key=lambda m: abs(m.start() - anchor_rel)):
        f = _extract_float_fields(m.group(0))
        if len(f) > len(fields):
            fs, fields = m, f
    if fs is None and cands:
        fs = cands[0]
    if fs:
        out.update(fields)
        # "As of <date>," typically PRECEDES the matched span — look in a
        # small prefix window before the sentence start as well.
        prefix = window[max(0, fs.start() - 80): fs.end()]
        if (a := _AS_OF_RE.search(prefix)):
            out["as_of_date"] = _parse_legend_date(a.group(1))
        dates = _DATE_RE.findall(fs.group(0))
        if dates:
            out["price_date"] = _parse_legend_date(dates[-1])

    if _SOLD_NONE_RE.search(window):
        out["sold_12mo_usd"] = 0.0
    else:
        for rx in _SOLD_AMOUNT_RES:
            if (s := rx.search(window)):
                amt = _num(s.group(1), s.group(2))
                if amt is not None and 0 <= amt < 1e10:
                    out["sold_12mo_usd"] = amt
                break

    return Ib6Legend(present=True, **out)


def _extract_float_fields(sent: str) -> dict:
    """Pull {float_value_usd, non_affiliate_shares, price_usd} from the
    float sentence, surviving its observed grammar variants:

      "...$44,509,855, based on 18,557,754 shares of outstanding common
       stock, of which approximately 16,608,155 shares are held by
       non-affiliates, and a price of $2.68 per share..."   (ASTC)
      "...991,443 shares of common stock outstanding, approximately
       880,060 of which were held by non-affiliates, and a per share
       price of $11.00..."                                  (SDOT)

    Strategy: value = first non-threshold "(was|is) $X"; shares = the
    number nearest BEFORE the LAST "held by non-affiliates" mention (the
    headline phrase has no preceding number, the computation clause
    does); price = prefer the candidate most self-consistent with
    shares × price ≈ value, since multi-price sentences (last-reported
    price vs computation price) are common. Missing fields are derived
    from the other two — the legend is self-consistent by construction
    (Instruction 1: value = non-affiliate shares × chosen-date price).
    """
    out: dict = {}

    # value — skip threshold mentions ("remains below $75 million")
    for m in _FLOAT_VALUE_RE.finditer(sent):
        if _THRESHOLD_GUARD_RE.search(sent[max(0, m.start() - 30):
                                           m.start() + 4]):
            continue
        v = _num(m.group(1), m.group(2))
        if _in(v, _VALUE_BOUNDS):
            out["float_value_usd"] = v
        break

    # shares — nearest number in a 60-char lookback before each
    # "held by non-affiliates"; later mentions win (computation clause
    # follows the headline phrase).
    shares = None
    for h in _HELD_BY_RE.finditer(sent):
        back = sent[max(0, h.start() - 60): h.start()]
        nums = [n for n in _NUMBER_RE.findall(back)
                if _in(_num(n), _SHARES_BOUNDS)]
        if nums:
            shares = _num(nums[-1])
    if shares is not None:
        out["non_affiliate_shares"] = shares

    # price — styled candidates first, then any in-bounds dollar amount;
    # when value+shares are known, pick the self-consistent candidate.
    styled = [_num(m.group(1)) for m in _PRICE_STYLE_RE.finditer(sent)]
    loose = [_num(m.group(1)) for m in _DOLLAR_RE.finditer(sent)]
    cands = [p for p in styled if _in(p, _PRICE_BOUNDS)] or \
            [p for p in loose if _in(p, _PRICE_BOUNDS)]
    value = out.get("float_value_usd")
    if cands:
        if value and shares:
            best = min(cands, key=lambda p: abs(shares * p - value))
            # accept only if actually consistent (±2%) — otherwise the
            # sentence deviates from the template and we'd rather derive
            if abs(shares * best - value) / value <= 0.02:
                out["price_usd"] = best
        else:
            out["price_usd"] = cands[0]

    # derivation — fill the missing leg from the other two
    price = out.get("price_usd")
    if value and shares and not price:
        derived = value / shares
        if _in(derived, _PRICE_BOUNDS):
            out["price_usd"] = round(derived, 6)
    elif value and price and not shares:
        derived = value / price
        if _in(derived, _SHARES_BOUNDS):
            out["non_affiliate_shares"] = round(derived)
    elif shares and price and not value:
        derived = shares * price
        if _in(derived, _VALUE_BOUNDS):
            out["float_value_usd"] = round(derived, 2)
    return out


def _candidate_docs(cik: int) -> list[dict]:
    """Primary-prospectus accessions with raw text, newest first.

    424Bs are filed off a registration statement identified by SEC file
    number. A 424B off an S-1/F-1 carries no I.B.6 legend because the
    instruction doesn't apply to S-1s at all (BJDX: S-1 unit offering,
    issuer simultaneously baby-shelf-trapped per its own risk factors) —
    so legend ABSENCE on such a doc is not regime evidence and the doc
    is excluded. The file-number → registration-form map comes from the
    issuer's own filings; 424Bs with an unknown file number are kept
    (legend presence still stamps 'baby'; ib6_cover_status treats their
    absence conservatively)."""
    qmarks = ",".join("?" * len(PRIMARY_PROSPECTUS_FORMS))
    with get_conn() as conn:
        rows = conn.execute(
            f"""SELECT f.accession_number, f.form, f.filing_date,
                       f.file_number
                  FROM dilution_filings f
                 INNER JOIN dilution_raw r
                    ON r.accession_number = f.accession_number
                 WHERE f.cik = ? AND f.form IN ({qmarks})
                 GROUP BY f.accession_number
                 ORDER BY f.filing_date DESC""",
            (cik, *sorted(PRIMARY_PROSPECTUS_FORMS)),
        ).fetchall()
        reg_forms = conn.execute(
            """SELECT DISTINCT file_number, form FROM dilution_filings
                WHERE cik = ? AND file_number IS NOT NULL
                  AND (form LIKE 'S-1%' OR form LIKE 'F-1%'
                       OR form LIKE 'S-3%' OR form LIKE 'F-3%')""",
            (cik,),
        ).fetchall()
    # file number → is it an S-1/F-1 registration? (a number can map to
    # both during transitions; S-3 presence wins — the 424B would then
    # genuinely be shelf-based)
    s1_numbers = {r["file_number"] for r in reg_forms
                  if r["form"].upper().startswith(("S-1", "F-1"))}
    s3_numbers = {r["file_number"] for r in reg_forms
                  if r["form"].upper().startswith(("S-3", "F-3"))}
    out = []
    for r in rows:
        d = dict(r)
        fn = d.get("file_number")
        if (d["form"].startswith("424B") and fn
                and fn in s1_numbers and fn not in s3_numbers):
            continue  # S-1 prospectus — not I.B.6 regime evidence
        d["registration_known"] = bool(
            not d["form"].startswith("424B") or (fn and fn in s3_numbers))
        out.append(d)
    return out


def _accession_text(accession: str) -> str:
    """All raw docs of an accession concatenated (main doc + exhibits —
    exhibits never carry a legend, concatenation is harmless)."""
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT content_md FROM dilution_raw WHERE accession_number = ?",
            (accession,),
        ).fetchall()
    return "\n".join(r["content_md"] or "" for r in rows)


def ib6_cover_status(cik: int, *, today: date | None = None) -> dict:
    """Issuer-level I.B.6 regime from the most recent primary prospectus.

    Returns {regime, legend, source_accession, source_form,
    source_filing_date}. regime is:
      'baby'         — legend present (offering filed under I.B.6)
      'unrestricted' — primary prospectus with NO legend (I.B.1 at
                       filing; per C&DI 116.26 the stamp survives a
                       later float drop for that prospectus)
      None           — no primary prospectus within the freshness
                       window → caller falls back to the computed
                       float test.

    This is the regime of the LATEST primary prospectus, applied
    issuer-wide — per-registration regime tracking (multiple live
    shelves in different regimes) is the (c) follow-up.
    """
    today = today or date.today()
    cutoff = (today - timedelta(days=REGIME_STAMP_MAX_AGE_DAYS)).isoformat()
    for doc in _candidate_docs(cik):
        if doc["filing_date"] < cutoff:
            break  # newest-first: everything after is older still
        text = _accession_text(doc["accession_number"])
        legend = parse_ib6_legend(text)
        if legend is None and not doc.get("registration_known"):
            # 424B whose registration we can't tie to an S-3/F-3:
            # legend absence proves nothing — keep looking.
            continue
        if legend is None and len(_RESALE_RE.findall(
                text[:_RESALE_SCAN_CHARS])) >= _RESALE_MENTIONS_MIN:
            # Resale-only registration — exempt from I.B.6, so the
            # missing legend is expected, not an I.B.1 stamp.
            continue
        return {
            "regime": "baby" if legend else "unrestricted",
            "legend": legend,
            "source_accession": doc["accession_number"],
            "source_form": doc["form"],
            "source_filing_date": doc["filing_date"],
        }
    return {
        "regime": None,
        "legend": None,
        "source_accession": None,
        "source_form": None,
        "source_filing_date": None,
    }


@lru_cache(maxsize=512)
def _cached_status(cik: int, today_iso: str) -> dict:
    return ib6_cover_status(int(cik), today=date.fromisoformat(today_iso))


def ib6_cover_status_cached(cik: int, *, today: date | None = None) -> dict:
    return _cached_status(int(cik), (today or date.today()).isoformat())


# ─── 10-K cover float (dei:EntityPublicFloat) fallback ──────────────
# Every domestic 10-K cover states the aggregate market value of common
# equity held by non-affiliates as of the last business day of Q2 — and
# tags it as dei:EntityPublicFloat in XBRL, so no text parsing is
# needed. FPI 20-F covers don't carry it (verified 18/18 missing across
# the walked FPIs), hence fallback-only. Junk values exist in the wild
# (SDOT tags $620M on a ~$6M float; zeros appear too), so values are
# sanity-gated and this NEVER overrides a prospectus stamp — KSCP and
# XTIA fixtures prove DT keeps the stamp when a newer 10-K disagrees
# (C&DI 116.26 grandfathering).

_PUBLIC_FLOAT_MIN_USD = 1e5      # zeros / penny-junk gate
_PUBLIC_FLOAT_STALE_DAYS = 540   # same convention as the stamp window


def entity_public_float_latest(cik: int) -> dict | None:
    """Latest dei:EntityPublicFloat fact {value, as_of, filed, form},
    or None (FPIs, fetch failure, junk value)."""
    import requests

    import config
    url = (f"https://data.sec.gov/api/xbrl/companyconcept/"
           f"CIK{int(cik):010d}/dei/EntityPublicFloat.json")
    try:
        resp = requests.get(
            url, timeout=10,
            headers={"User-Agent": getattr(
                config, "EDGAR_IDENTITY", "dilution contact@finviz.com")})
        if resp.status_code == 404:
            return None
        resp.raise_for_status()
        facts = resp.json().get("units", {}).get("USD", [])
    except Exception as exc:
        log.warning("EntityPublicFloat fetch failed for cik=%s: %s",
                    cik, exc)
        return None
    facts = [f for f in facts
             if (f.get("val") or 0) >= _PUBLIC_FLOAT_MIN_USD]
    if not facts:
        return None
    latest = max(facts, key=lambda f: (f.get("end") or "",
                                       f.get("filed") or ""))
    return {
        "value": float(latest["val"]),
        "as_of": latest.get("end"),
        "filed": latest.get("filed"),
        "form": latest.get("form"),
    }


@lru_cache(maxsize=512)
def _cached_public_float(cik: int, today_iso: str) -> dict | None:
    return entity_public_float_latest(cik)


def ib6_regime(cik: int, *, today: date | None = None) -> dict:
    """Tiered I.B.6 regime determination.

    1. 'stamp'     — latest genuine-primary prospectus cover legend
                     (presence → baby, absence → I.B.1/unrestricted);
                     resale-only and S-1-based docs excluded.
    2. '10k_float' — no stamp: latest dei:EntityPublicFloat vs $75M.
                     Issuer-computed, dated, but only annual — and it
                     measures the 10(a)(3) re-test input, which the
                     stamp supersedes when both exist.
    3. None        — caller falls through to the live computed test.
    """
    today = today or date.today()
    st = ib6_cover_status_cached(cik, today=today)
    if st["regime"] is not None:
        return {**st, "source": "stamp"}
    pf = _cached_public_float(int(cik), today.isoformat())
    if pf and pf.get("filed"):
        age = (today - date.fromisoformat(pf["filed"])).days
        if age <= _PUBLIC_FLOAT_STALE_DAYS:
            return {
                "regime": ("baby" if pf["value"] < 75_000_000
                           else "unrestricted"),
                "legend": None,
                "source": "10k_float",
                "source_accession": None,
                "source_form": pf.get("form"),
                "source_filing_date": pf.get("filed"),
                "public_float_usd": pf["value"],
                "public_float_as_of": pf.get("as_of"),
            }
    return {"regime": None, "legend": None, "source": None,
            "source_accession": None, "source_form": None,
            "source_filing_date": None}
