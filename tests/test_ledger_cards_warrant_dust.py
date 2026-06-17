"""B3 — count-dust warrant floor (CELU exhausted-warrant extras).

A warrant exercised down to <=1 share out of a much larger issuance is
exhausted; the residual share is a walker placeholder it couldn't fully
zero (CELU W-5275 Dragasac 652,982 -> 1, exercised 652,981; W-5284
535,275 -> 1). `_warrant_dead` drops it. A warrant genuinely issued at
<=1 share (no exercise activity) is NOT dust and stays rendered.
"""

from dilution.ledger.cards import _warrant_dead


def _row(**out):
    return {
        "status": "active",
        "type": "warrant",
        "instrument_id": "W-1",
        "created_accession": "acc-1",
        "terms": {"expiration": "2030-01-01"},
        "outstanding": out,
        "history": [],
    }


def test_count_one_dust_with_exercise_is_dead():
    r = _row(count=1, initial_count=652982, exercised_to_date=652981)
    assert _warrant_dead(r) is True


def test_fractional_dust_with_exercise_is_dead():
    r = _row(count=0.5, initial_count=100000, exercised_to_date=99999.5)
    assert _warrant_dead(r) is True


def test_count_one_genuine_issuance_stays_alive():
    # 1 share issued, never exercised -> not dust.
    r = _row(count=1, initial_count=1, exercised_to_date=0)
    assert _warrant_dead(r) is False


def test_full_count_stays_alive():
    r = _row(count=500000, initial_count=500000, exercised_to_date=0)
    assert _warrant_dead(r) is False
