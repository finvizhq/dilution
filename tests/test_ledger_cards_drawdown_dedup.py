"""B1 — shelf/take-down drawdown de-dup (ACTU greenshoe double-count).

A follow-on filing that re-books an offering with its over-allotment
exercised re-states the SAME take-down on the SAME pricing date at a larger
share count / dollar amount. `_drawn_to_date` must treat that as a
supersession (keep the max), NOT sum it — and must not let a stored
`drawn_usd` that merely caches the double-counted raw sum win over the
de-duped total. The LLM-pinned-cumulative case (drawn_usd EXCEEDS the
discrete log) must still be honoured.

Ground truth: ACTU SH-2598 booked $15,000,006 / 2,142,858 sh and
$17,250,002 / 2,464,286 sh, both @ $7.00 on 2025-09-10 (base then
base+over-allotment); stored drawn_usd 32,250,008 == the summed raw. The
shelf card rendered total_amount_raised 33,382,848.74 vs the filing-true
18,382,842.74 (= 17,250,002 + the ATM family's 1,132,840.74).
"""

import pytest

from dilution.ledger.cards import _drawdown_sums, _drawn_to_date


def _stage(temp_db, instrument_id, cik, type="shelf"):
    """Company + parent ledger row so drawdown FK (instrument_id) holds."""
    temp_db.add_company(cik, f"T{cik}")
    temp_db.add_instrument(instrument_id, cik=cik, type=type)


def test_greenshoe_restatement_deduped(temp_db):
    _stage(temp_db, "SH-1", 1652935)
    temp_db.add_drawdown("SH-1", cik=1652935, event_date="2025-09-10",
                         amount_usd=15_000_006, shares=2_142_858, price=7.0)
    temp_db.add_drawdown("SH-1", cik=1652935, event_date="2025-09-10",
                         amount_usd=17_250_002, shares=2_464_286, price=7.0)
    # Stored cumulative == the double-counted raw sum -> trust the de-dup.
    out = {"drawn_usd": 32_250_008}
    assert _drawn_to_date(1652935, "SH-1", out) == pytest.approx(17_250_002)


def test_drawdown_sums_raw_vs_deduped(temp_db):
    _stage(temp_db, "X", 1)
    temp_db.add_drawdown("X", cik=1, event_date="2025-09-10",
                         amount_usd=15_000_006, shares=2_142_858, price=7.0)
    temp_db.add_drawdown("X", cik=1, event_date="2025-09-10",
                         amount_usd=17_250_002, shares=2_464_286, price=7.0)
    with temp_db.conn() as c:
        raw, deduped = _drawdown_sums(c, 1, "X")
    assert raw == pytest.approx(32_250_008)
    assert deduped == pytest.approx(17_250_002)


def test_different_date_or_price_is_summed(temp_db):
    # Independent take-downs (different dates AND prices) are NOT collapsed.
    _stage(temp_db, "X", 1, type="atm")
    temp_db.add_drawdown("X", cik=1, event_date="2026-03-31",
                         amount_usd=532_765.24, shares=198_793, price=2.68)
    temp_db.add_drawdown("X", cik=1, event_date="2026-05-13",
                         amount_usd=600_075.5, shares=213_550, price=2.81)
    with temp_db.conn() as c:
        raw, deduped = _drawdown_sums(c, 1, "X")
    assert deduped == pytest.approx(raw)
    assert deduped == pytest.approx(1_132_840.74)


def test_llm_pinned_cumulative_preserved(temp_db):
    # drawn_usd EXCEEDS the discrete log (un-logged take-downs) -> keep it.
    _stage(temp_db, "X", 1, type="atm")
    temp_db.add_drawdown("X", cik=1, event_date="2025-01-01",
                         amount_usd=30_000_000)
    out = {"drawn_usd": 50_000_000}
    assert _drawn_to_date(1, "X", out) == pytest.approx(50_000_000)


def test_no_discrete_draws_uses_stored(temp_db):
    _stage(temp_db, "X", 1)
    out = {"drawn_usd": 12_345_678}
    assert _drawn_to_date(1, "X", out) == pytest.approx(12_345_678)


def test_anchor_path_dedupes_post_asof(temp_db):
    _stage(temp_db, "X", 1)
    temp_db.add_drawdown("X", cik=1, event_date="2025-09-10",
                         amount_usd=15_000_006, shares=2_142_858, price=7.0)
    temp_db.add_drawdown("X", cik=1, event_date="2025-09-10",
                         amount_usd=17_250_002, shares=2_464_286, price=7.0)
    out = {"drawn_usd_anchor": 5_000_000, "drawn_usd_asof": "2025-01-01"}
    # anchor + de-duped post-asof = 5,000,000 + 17,250,002
    assert _drawn_to_date(1, "X", out) == pytest.approx(22_250_002)
