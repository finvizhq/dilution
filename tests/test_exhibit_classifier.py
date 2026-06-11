"""Unit tests for ``dilution/exhibit_classifier.py``.

The module exposes one pure function, ``classify_by_description``, a
deterministic substring router returning 'keep' / 'drop' / 'unknown'.
There is no I/O, no DB, no LLM, no network — every test is a direct
assert. KEEP wins over DROP on conflict (a documented
coverage>savings invariant) and matching is literal substring on the
``.upper()``-cased input.
"""

from __future__ import annotations

import pytest

from dilution.exhibit_classifier import (
    DESCRIPTION_DROP_PHRASES,
    DESCRIPTION_KEEP_PHRASES,
    classify_by_description,
)


class TestFalsyInputs:
    """None / empty descriptions short-circuit to 'unknown'."""

    def test_none_returns_unknown(self):
        assert classify_by_description(description=None) == "unknown"

    def test_empty_string_returns_unknown(self):
        assert classify_by_description(description="") == "unknown"

    @pytest.mark.parametrize("falsy", [None, ""])
    def test_falsy_guard_parametrized(self, falsy):
        # The `if not description` guard catches both None and ''.
        assert classify_by_description(description=falsy) == "unknown"

    def test_whitespace_only_is_truthy_but_matches_nothing(self):
        # "   " is truthy so it passes the guard, but matches no phrase.
        assert classify_by_description(description="   ") == "unknown"


class TestUnknownClassification:
    """Generic / ambiguous descriptions fall through as 'unknown'."""

    def test_generic_description_returns_unknown(self):
        assert (
            classify_by_description(description="MISCELLANEOUS EXHIBIT")
            == "unknown"
        )

    @pytest.mark.parametrize(
        "desc",
        [
            "Exhibit 99.1",
            "Letter to Stockholders",
            "Consent of Independent Auditor",
            "Subsidiaries of the Registrant",
            "Power of Attorney",
        ],
    )
    def test_various_non_matching_descriptions(self, desc):
        assert classify_by_description(description=desc) == "unknown"


class TestKeepClassification:
    """Descriptions containing a KEEP phrase return 'keep'."""

    def test_securities_purchase_agreement(self):
        assert (
            classify_by_description(
                description="Securities Purchase Agreement"
            )
            == "keep"
        )

    @pytest.mark.parametrize("phrase", DESCRIPTION_KEEP_PHRASES)
    def test_every_keep_phrase_classifies_keep(self, phrase):
        # Each KEEP phrase, fed verbatim (already upper-case), must
        # classify 'keep'. Some KEEP phrases also embed no DROP phrase,
        # so this is a clean per-phrase coverage sweep.
        assert classify_by_description(description=phrase) == "keep"

    def test_embedded_substring_match(self):
        # Phrase embedded mid-description still matches (substring, not
        # full-string, search).
        assert (
            classify_by_description(
                description="Exhibit 10.1 - Securities Purchase Agreement "
                "dated as of June 1, 2025"
            )
            == "keep"
        )

    def test_leading_trailing_whitespace_still_matches(self):
        assert (
            classify_by_description(
                description="   Securities Purchase Agreement   "
            )
            == "keep"
        )


class TestDropClassification:
    """Descriptions containing only a DROP phrase return 'drop'."""

    def test_employment_agreement(self):
        assert (
            classify_by_description(description="Employment Agreement")
            == "drop"
        )

    @pytest.mark.parametrize("phrase", DESCRIPTION_DROP_PHRASES)
    def test_every_drop_phrase_classifies_drop(self, phrase):
        # Each DROP phrase fed verbatim must classify 'drop' — provided
        # it does not also contain a KEEP phrase. None of the DROP
        # phrases contain a KEEP phrase, so all should be 'drop'.
        assert classify_by_description(description=phrase) == "drop"

    def test_no_drop_phrase_contains_a_keep_phrase(self):
        # Guards the parametrized DROP sweep above: confirms the data
        # truly has no KEEP-inside-DROP overlap that would flip a result.
        for d in DESCRIPTION_DROP_PHRASES:
            du = d.upper()
            assert not any(k in du for k in DESCRIPTION_KEEP_PHRASES), (
                f"DROP phrase {d!r} unexpectedly contains a KEEP phrase"
            )

    def test_single_keep_word_does_not_trigger_keep(self):
        # 'WARRANT' and 'EARNINGS PRESENTATION'... only the DROP phrase
        # ('EARNINGS PRESENTATION') is a full listed phrase; the bare
        # word 'WARRANT' / 'TRANSFER AGENT' are NOT KEEP phrases, so the
        # result is 'drop'. Confirms single words don't trigger KEEP.
        assert (
            classify_by_description(
                description="WARRANT TRANSFER AGENT EARNINGS PRESENTATION"
            )
            == "drop"
        )


class TestKeepWinsOnConflict:
    """KEEP is checked first, so it beats DROP when both are present."""

    def test_keep_beats_drop_concatenated(self):
        # Contains KEEP 'PLACEMENT AGENT AGREEMENT' and DROP
        # 'INDEMNIFICATION AGREEMENT'. KEEP must win.
        assert (
            classify_by_description(
                description="PLACEMENT AGENT AGREEMENT and "
                "INDEMNIFICATION AGREEMENT"
            )
            == "keep"
        )

    def test_keep_beats_drop_order_independent(self):
        # DROP phrase appears first textually; KEEP still wins because
        # the KEEP list is scanned before the DROP list, not by position.
        assert (
            classify_by_description(
                description="INDEMNIFICATION AGREEMENT; "
                "SECURITIES PURCHASE AGREEMENT"
            )
            == "keep"
        )

    def test_warrant_agreement_with_lease(self):
        # 'WARRANT AGREEMENT' (KEEP) + 'LEASE AGREEMENT' (DROP) -> keep.
        assert (
            classify_by_description(
                description="Warrant Agreement and Lease Agreement"
            )
            == "keep"
        )


class TestCaseInsensitivity:
    """The function .upper()s the input; case must not matter."""

    @pytest.mark.parametrize(
        "desc",
        [
            "securities purchase agreement",
            "Securities Purchase Agreement",
            "SECURITIES PURCHASE AGREEMENT",
            "SeCuRiTiEs PuRcHaSe AgReEmEnT",
        ],
    )
    def test_keep_case_variants_all_keep(self, desc):
        assert classify_by_description(description=desc) == "keep"

    @pytest.mark.parametrize(
        "desc",
        [
            "employment agreement",
            "Employment Agreement",
            "EMPLOYMENT AGREEMENT",
        ],
    )
    def test_drop_case_variants_all_drop(self, desc):
        assert classify_by_description(description=desc) == "drop"


class TestLiteralSubstringNotTokenBased:
    """Matching is raw substring, not whitespace/token-normalized."""

    def test_double_space_breaks_match(self):
        # 'Securities  Purchase  Agreement' (double spaces) does NOT
        # contain the exact substring 'SECURITIES PURCHASE AGREEMENT',
        # so it falls through to 'unknown'. Regression guard documenting
        # that matching is literal-substring, not token-based.
        assert (
            classify_by_description(
                description="Securities  Purchase  Agreement"
            )
            == "unknown"
        )

    def test_newline_breaks_match(self):
        assert (
            classify_by_description(
                description="Securities\nPurchase Agreement"
            )
            == "unknown"
        )

    def test_tab_breaks_match(self):
        assert (
            classify_by_description(
                description="Securities\tPurchase Agreement"
            )
            == "unknown"
        )


class TestSpellingAndPunctuationVariants:
    """ATM hyphen/space and prefunded spelling variants both KEEP."""

    @pytest.mark.parametrize(
        "desc", ["AT-THE-MARKET", "AT THE MARKET", "At-The-Market Offering"]
    )
    def test_atm_hyphen_and_space_variants(self, desc):
        assert classify_by_description(description=desc) == "keep"

    @pytest.mark.parametrize(
        "desc",
        [
            "FORM OF PRE-FUNDED WARRANT",
            "FORM OF PREFUNDED WARRANT",
            "Form of Pre-Funded Warrant",
            "Form of Prefunded Warrant",
        ],
    )
    def test_prefunded_spelling_variants(self, desc):
        assert classify_by_description(description=desc) == "keep"


class TestPhraseListInvariants:
    """The function only upper-cases the INPUT — the phrase lists must
    therefore already be all-upper-case, or a lower-cased phrase would
    silently never match."""

    @pytest.mark.parametrize("phrase", DESCRIPTION_KEEP_PHRASES)
    def test_keep_phrases_are_upper(self, phrase):
        assert phrase == phrase.upper()

    @pytest.mark.parametrize("phrase", DESCRIPTION_DROP_PHRASES)
    def test_drop_phrases_are_upper(self, phrase):
        assert phrase == phrase.upper()

    def test_phrase_lists_are_tuples(self):
        # Immutable module-level constants.
        assert isinstance(DESCRIPTION_KEEP_PHRASES, tuple)
        assert isinstance(DESCRIPTION_DROP_PHRASES, tuple)

    def test_no_empty_phrases(self):
        # An empty-string phrase would match every non-empty description.
        assert all(p for p in DESCRIPTION_KEEP_PHRASES)
        assert all(p for p in DESCRIPTION_DROP_PHRASES)

    def test_return_value_is_always_one_of_three(self):
        for desc in [None, "", "x", "SALES AGREEMENT", "OFFER LETTER"]:
            assert classify_by_description(description=desc) in {
                "keep",
                "drop",
                "unknown",
            }


class TestKeywordOnlyArgument:
    """``description`` is a keyword-only parameter."""

    def test_positional_call_raises_type_error(self):
        with pytest.raises(TypeError):
            classify_by_description("Securities Purchase Agreement")


class TestDropInsideKeepCollision:
    """The strongest proof of KEEP-first ordering.

    A KEEP phrase that *literally embeds* a DROP phrase exercises the
    ``has_keep`` short-circuit on a SINGLE real production phrase: if the
    code checked DROP before KEEP (or didn't short-circuit), the phrase
    fed alone would misclassify as 'drop'.
    """

    def test_equity_distribution_agreement_keeps_despite_embedded_drop(self):
        # 'EQUITY DISTRIBUTION AGREEMENT' (KEEP) contains the substring
        # 'DISTRIBUTION AGREEMENT' (DROP). Fed alone it must be 'keep'.
        # Re-derived from source: has_keep matches first, returns 'keep'
        # before the DROP scan is ever reached.
        assert "DISTRIBUTION AGREEMENT" in "EQUITY DISTRIBUTION AGREEMENT"
        assert "DISTRIBUTION AGREEMENT" in DESCRIPTION_DROP_PHRASES
        assert "EQUITY DISTRIBUTION AGREEMENT" in DESCRIPTION_KEEP_PHRASES
        assert (
            classify_by_description(description="Equity Distribution Agreement")
            == "keep"
        )

    def test_every_keep_phrase_that_embeds_a_drop_still_keeps(self):
        # Generalize: ANY KEEP phrase that contains a DROP substring must
        # still resolve to 'keep' when fed verbatim. (Today only
        # 'EQUITY DISTRIBUTION AGREEMENT' qualifies, but this pins the
        # invariant against future list edits that add a colliding pair.)
        colliding = [
            k for k in DESCRIPTION_KEEP_PHRASES
            if any(d in k.upper() for d in DESCRIPTION_DROP_PHRASES)
        ]
        # There is at least one such collision in the production data, so
        # this loop is not vacuous.
        assert colliding, "expected ≥1 KEEP phrase embedding a DROP phrase"
        for k in colliding:
            assert classify_by_description(description=k) == "keep"


class TestInvariantInverseOverlap:
    """Companion to test_no_drop_phrase_contains_a_keep_phrase.

    That test guards the DROP parametrized sweep (no KEEP-inside-DROP).
    Here we assert there is NO identical phrase appearing in BOTH lists,
    which would make the per-phrase sweeps ambiguous about intent.
    """

    def test_no_phrase_appears_in_both_lists(self):
        overlap = set(DESCRIPTION_KEEP_PHRASES) & set(DESCRIPTION_DROP_PHRASES)
        assert overlap == set(), f"phrase(s) in both lists: {overlap}"


class TestKeepWinsPermutations:
    """Order/separator-independent confirmation of KEEP-beats-DROP.

    Survey: KEEP-wins is a documented coverage>savings invariant; sweep a
    few real KEEP×DROP pairings with different separators to make sure the
    win is independent of textual position and joining punctuation.
    """

    @pytest.mark.parametrize(
        "keep_phrase, drop_phrase",
        [
            ("SECURITIES PURCHASE AGREEMENT", "EMPLOYMENT AGREEMENT"),
            ("UNDERWRITING AGREEMENT", "LEASE AGREEMENT"),
            ("CERTIFICATE OF DESIGNATION", "INVESTOR PRESENTATION"),
            ("SALES AGREEMENT", "SEPARATION AGREEMENT"),
        ],
    )
    @pytest.mark.parametrize("sep", [" ", "; ", " and ", "\n", " / "])
    def test_keep_wins_regardless_of_order_and_separator(
        self, keep_phrase, drop_phrase, sep
    ):
        # KEEP-first then DROP-first; both must be 'keep'.
        assert (
            classify_by_description(description=f"{keep_phrase}{sep}{drop_phrase}")
            == "keep"
        )
        assert (
            classify_by_description(description=f"{drop_phrase}{sep}{keep_phrase}")
            == "keep"
        )


class TestNonStringInputs:
    """Failure-path: the function calls ``.upper()`` on any truthy input,
    so non-string truthy values raise rather than returning a verdict.

    Re-derived from source line ``d = description.upper()``: only None/''
    (and other falsy values) are guarded; a truthy non-str reaches the
    attribute/typed call and raises.
    """

    def test_int_input_raises_attribute_error(self):
        # 123 is truthy → bypasses the guard → 123.upper() -> AttributeError.
        with pytest.raises(AttributeError):
            classify_by_description(description=123)

    def test_bytes_input_raises_type_error(self):
        # b'...' has .upper() (returns bytes) → the `p in d` membership
        # test then compares str-in-bytes → TypeError.
        with pytest.raises(TypeError):
            classify_by_description(description=b"SECURITIES PURCHASE AGREEMENT")

    @pytest.mark.parametrize("falsy", [0, 0.0, [], {}, ()])
    def test_other_falsy_values_short_circuit_to_unknown(self, falsy):
        # The guard is `if not description`, so ANY falsy value (not just
        # None/'') returns 'unknown' without ever calling .upper().
        assert classify_by_description(description=falsy) == "unknown"


class TestDeterminism:
    """Pure function: identical input always yields identical output."""

    def test_repeated_calls_are_idempotent(self):
        desc = "Securities Purchase Agreement and Lease Agreement"
        results = {classify_by_description(description=desc) for _ in range(20)}
        assert results == {"keep"}

    def test_classification_does_not_mutate_phrase_lists(self):
        before_keep = tuple(DESCRIPTION_KEEP_PHRASES)
        before_drop = tuple(DESCRIPTION_DROP_PHRASES)
        for d in ["SALES AGREEMENT", "LEASE AGREEMENT", "x", "", None]:
            classify_by_description(description=d)
        assert DESCRIPTION_KEEP_PHRASES == before_keep
        assert DESCRIPTION_DROP_PHRASES == before_drop
