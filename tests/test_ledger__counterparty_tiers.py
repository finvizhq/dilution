"""Unit tests for dilution.ledger._counterparty_tiers.

Pure module: two substring-matchers (`bank_tier`, `investor_class`) plus a
`_norm` helper. No I/O, no DB, no LLM, no config-at-import. We import the
module directly and call the functions. No monkeypatching / temp_db needed.

The behaviorally interesting contracts are:

  * the None/empty/whitespace guard (both functions route through `_norm`);
  * case-insensitivity and legal-suffix substring tolerance;
  * CROSS-BUCKET ordering precedence (first-bucket-wins) — the regression-
    prone lines if someone reorders the bucket tuples;
  * the naive-substring false-positive behavior (short needles like 'nea',
    'roth', 'lind' match unrelated names) — documented, intended, asserted
    so an accidental change is caught.
"""

from __future__ import annotations

import pytest

from dilution.ledger import _counterparty_tiers as ct
from dilution.ledger._counterparty_tiers import bank_tier, investor_class


class TestNorm:
    """_norm(name) -> '' for falsy, else name.strip().lower()."""

    def test_none_returns_empty(self):
        assert ct._norm(None) == ""

    def test_empty_string_returns_empty(self):
        assert ct._norm("") == ""

    def test_whitespace_only_returns_empty(self):
        # strip() on whitespace yields '' which is treated as no-match.
        assert ct._norm("   ") == ""

    def test_strips_and_lowercases(self):
        assert ct._norm("  Foo BAR  ") == "foo bar"

    def test_already_normalized_passthrough(self):
        assert ct._norm("foo bar") == "foo bar"

    def test_uppercase_lowercased(self):
        assert ct._norm("GOLDMAN") == "goldman"

    @pytest.mark.parametrize(
        "raw, expected",
        [
            (None, ""),
            ("", ""),
            ("   ", ""),
            ("\t\n", ""),
            ("X", "x"),
            ("  Mixed Case  ", "mixed case"),
        ],
    )
    def test_norm_sweep(self, raw, expected):
        assert ct._norm(raw) == expected

    def test_falsy_non_str_guarded(self):
        # `if not name` guards falsy non-str inputs; 0 is falsy -> ''.
        assert ct._norm(0) == ""

    def test_internal_whitespace_not_collapsed(self):
        # _norm only strip()s the ends and lower()s; it does NOT collapse
        # interior runs of whitespace. This matters for substring matching:
        # a needle like 'goldman sachs' (single space) would NOT match a
        # canonical 'goldman  sachs' (double space). Pin the contract so an
        # accidental re.sub-style normalization is caught.
        assert ct._norm("Foo   Bar") == "foo   bar"

    @pytest.mark.parametrize(
        "raw, expected",
        [
            ("\tGoldman\n", "goldman"),       # tab/newline stripped from ends
            ("\r\n  Jefferies  \r\n", "jefferies"),
            ("MiXeD", "mixed"),
        ],
    )
    def test_norm_strips_assorted_whitespace_and_lowercases(self, raw, expected):
        assert ct._norm(raw) == expected

    @pytest.mark.parametrize("falsy", [0, 0.0, [], {}, (), False])
    def test_falsy_non_str_sweep_returns_empty(self, falsy):
        # Any falsy value short-circuits the `if not name` guard before
        # .strip()/.lower() is ever reached, so it never raises.
        assert ct._norm(falsy) == ""

    @pytest.mark.parametrize("truthy_non_str", [5, 3.14, ["x"], {"a": 1}, (1, 2)])
    def test_truthy_non_str_raises_attributeerror(self, truthy_non_str):
        # BUG-ish-but-intended: _norm only guards FALSY inputs. A truthy
        # non-str slips past `if not name` and hits .strip(), raising
        # AttributeError. Callers always pass str | None (the canonical
        # name columns), so this never fires in production — but pin the
        # current behavior so an accidental signature/guard change is caught.
        with pytest.raises(AttributeError):
            ct._norm(truthy_non_str)


class TestBankTierGuards:
    """None / empty / whitespace short-circuit to None."""

    @pytest.mark.parametrize("bad", [None, "", "   ", "\t", "\n  \n"])
    def test_falsy_or_blank_returns_none(self, bad):
        assert bank_tier(bad) is None

    def test_unrecognized_microcap_returns_none(self):
        assert bank_tier("Some Random Microcap Securities") is None

    @pytest.mark.parametrize("truthy_non_str", [5, 3.14, ["x"]])
    def test_truthy_non_str_propagates_attributeerror(self, truthy_non_str):
        # bank_tier routes through _norm, which calls .strip() on truthy
        # non-str -> AttributeError propagates. Documents current behavior;
        # callers always pass str | None.
        with pytest.raises(AttributeError):
            bank_tier(truthy_non_str)


class TestBankTierKnownNames:
    """Exact-known names map to their documented tier."""

    def test_goldman_sachs_bulge(self):
        assert bank_tier("Goldman Sachs") == "bulge_bracket"

    def test_jefferies_middle_market(self):
        assert bank_tier("Jefferies") == "middle_market"

    def test_wainwright_boutique(self):
        assert bank_tier("H.C. Wainwright") == "boutique"

    @pytest.mark.parametrize(
        "name, tier",
        [
            ("Goldman Sachs", "bulge_bracket"),
            ("JPMorgan", "bulge_bracket"),
            ("Morgan Stanley", "bulge_bracket"),
            ("Barclays", "bulge_bracket"),
            ("UBS", "bulge_bracket"),
            ("Wells Fargo", "bulge_bracket"),
            ("Jefferies", "middle_market"),
            ("TD Cowen", "middle_market"),
            ("Stifel", "middle_market"),
            ("Cantor Fitzgerald", "middle_market"),
            ("Guggenheim", "middle_market"),
            ("Roth Capital", "pump_trifecta"),
            ("Ladenburg Thalmann", "pump_trifecta"),
            ("Brookline Capital", "pump_trifecta"),
            ("ThinkEquity", "pump_trifecta"),
            ("Boustead Securities", "pump_trifecta"),
            ("Maxim Group", "boutique"),
            ("Aegis Capital", "boutique"),
            ("Chardan Capital", "boutique"),
            ("EF Hutton", "boutique"),
            ("Alliance Global Partners", "boutique"),
        ],
    )
    def test_known_name_sweep(self, name, tier):
        assert bank_tier(name) == tier


class TestBankTierCaseAndWhitespace:
    """Case-insensitivity and leading/trailing whitespace tolerance."""

    @pytest.mark.parametrize("variant", ["GOLDMAN SACHS", "goldman sachs",
                                         "Goldman Sachs", "GoLdMaN sAcHs"])
    def test_case_insensitive(self, variant):
        assert bank_tier(variant) == "bulge_bracket"

    def test_leading_trailing_whitespace(self):
        assert bank_tier("  Jefferies  ") == "middle_market"


class TestBankTierSubstringTolerance:
    """Longer legal suffixes not enumerated are tolerated by substring."""

    def test_roth_capital_partners_llc(self):
        # 'roth capital' is a needle; the LLC suffix rides along.
        assert bank_tier("Roth Capital Partners LLC") == "pump_trifecta"

    def test_citi_embedded_in_citigroup(self):
        # short-form 'citi' needle matches the longer canonical name.
        assert bank_tier("Citigroup Global Markets") == "bulge_bracket"

    def test_short_needle_matches_when_embedded(self):
        # 'citi' is a substring; the trailing ' & Co.' is not enumerated.
        assert bank_tier("Citi & Co.") == "bulge_bracket"


class TestBankTierOrderingPrecedence:
    """Cross-bucket first-match-wins is the documented, regression-prone
    contract (pump_trifecta bucket precedes boutique)."""

    def test_roth_is_pump_trifecta_not_boutique(self):
        # Roth is reputationally a boutique, but _PUMP_TRIFECTA is placed
        # before _BOUTIQUE in _BANK_TIER_BUCKETS so pump_trifecta wins.
        assert bank_tier("Roth") == "pump_trifecta"
        assert bank_tier("Roth Capital") == "pump_trifecta"

    def test_b_riley_resolves_to_middle_market(self):
        # 'b. riley' lives in _MIDDLE_MARKET; 'b. riley principal' is an
        # ELOC funder (a different classifier), so bank_tier of the bare
        # name is middle_market.
        assert bank_tier("B. Riley") == "middle_market"

    def test_bucket_order_is_the_documented_sequence(self):
        # Guard the literal tuple ordering — if a refactor reorders these,
        # the precedence asserts above silently change meaning, so pin it.
        order = [tier for tier, _ in ct._BANK_TIER_BUCKETS]
        assert order == ["pump_trifecta", "bulge_bracket",
                         "middle_market", "boutique"]

    def test_bulge_precedes_middle_when_both_needles_present(self):
        # A single string containing needles from TWO different buckets
        # resolves to the EARLIER bucket. 'goldman' (bulge_bracket) is
        # iterated before 'jefferies' (middle_market), so bulge wins even
        # though both substrings are present. This exercises cross-bucket
        # precedence on a string that isn't the Roth/Boothbay alias overlap.
        assert bank_tier("Goldman Sachs and Jefferies JV") == "bulge_bracket"

    def test_pump_trifecta_precedes_bulge_when_both_present(self):
        # pump_trifecta is the very first bucket, so a contrived string with
        # both 'roth' and 'goldman' resolves to pump_trifecta.
        assert bank_tier("Roth co-managed with Goldman") == "pump_trifecta"


class TestBankTierFalsePositives:
    """Naive substring matching: short needles false-positive on unrelated
    names. This is DOCUMENTED, INTENDED advisory behavior (heuristics for a
    human, never used for routing). Asserted to catch accidental changes."""

    def test_roth_substring_false_positive(self):
        # 'Brothwell Securities' contains 'roth' -> classified pump_trifecta.
        # BUG-ish-but-intended: naive substring; do not "fix" w/o sign-off.
        assert bank_tier("Brothwell Securities") == "pump_trifecta"

    def test_agp_substring_false_positive(self):
        # 'agp' needle in _BOUTIQUE matches 'Flagpartners'.
        assert bank_tier("Flagpartners") == "boutique"

    def test_citi_substring_false_positive(self):
        # 'citi' needle matches 'Felicitin Securities'.
        assert bank_tier("Felicitin Securities") == "bulge_bracket"

    def test_rbc_substring_false_positive(self):
        # 'rbc' (RBC Capital, _MIDDLE_MARKET) is a 3-char needle that lands
        # inside the unrelated 'Garbcan'. BUG-ish-but-intended naive substring.
        assert bank_tier("Garbcan Securities") == "middle_market"

    def test_genuinely_unrelated_name_stays_none(self):
        # Negative control for the false-positive sweep: a name that shares
        # NO bucket substring must classify None. Guards against the matcher
        # quietly becoming over-eager (e.g. someone shortens a needle to ''
        # or 1 char, which would start matching everything).
        assert bank_tier("Pinnacle Brokerage Group") is None


class TestInvestorClassGuards:
    @pytest.mark.parametrize("bad", [None, "", "   ", "\t", "\n  \n"])
    def test_falsy_or_blank_returns_none(self, bad):
        assert investor_class(bad) is None

    def test_unrecognized_obscure_lp_returns_none(self):
        assert investor_class("Zephyr Holdings LP") is None

    @pytest.mark.parametrize("truthy_non_str", [5, 3.14, ["x"]])
    def test_truthy_non_str_propagates_attributeerror(self, truthy_non_str):
        # Same _norm .strip() raise path as bank_tier. Pin current behavior.
        with pytest.raises(AttributeError):
            investor_class(truthy_non_str)


class TestInvestorClassKnownNames:
    def test_baker_bros_long_term_informed(self):
        assert investor_class("Baker Bros") == "long_term_informed"

    def test_baker_brothers_long_term_informed(self):
        assert investor_class("baker brothers") == "long_term_informed"

    def test_lincoln_park_eloc_funder(self):
        assert investor_class("Lincoln Park Capital") == "eloc_funder"

    def test_lind_global_toxic_lender(self):
        assert investor_class("Lind Global Asset Management") == "toxic_lender"

    def test_lind_partners_toxic_lender(self):
        assert investor_class("Lind Partners") == "toxic_lender"

    def test_sabby_pipe_flipper(self):
        assert investor_class("Sabby Volatility Warrant Master Fund") == "pipe_flipper"

    @pytest.mark.parametrize(
        "name, cls",
        [
            ("Perceptive Advisors", "long_term_informed"),
            ("OrbiMed", "long_term_informed"),
            ("RA Capital", "long_term_informed"),
            ("Cormorant Asset Management", "long_term_informed"),
            ("Pfizer", "long_term_informed"),
            ("Aspire Capital", "eloc_funder"),
            ("Yorkville Advisors", "eloc_funder"),
            ("GHS Investments", "eloc_funder"),
            ("Triton Funds", "eloc_funder"),
            ("Streeterville Capital", "toxic_lender"),
            ("Mast Hill Fund", "toxic_lender"),
            ("Bellridge Capital", "toxic_lender"),
            ("Hudson Bay Capital", "pipe_flipper"),
            ("Anson Funds", "pipe_flipper"),
            ("Armistice Capital", "pipe_flipper"),
            ("Empery Asset Management", "pipe_flipper"),
            # Sovereign / supranational long-term-capital cluster (docstring
            # calls these out as a distinct group; the bare acronyms are the
            # canonical short forms the walker emits).
            ("European Investment Bank", "long_term_informed"),
            ("EIB", "long_term_informed"),
            ("IFC", "long_term_informed"),
            ("EBRD", "long_term_informed"),
            # Strategic-acquirer / large-pharma cluster.
            ("Gilead", "long_term_informed"),
            ("AstraZeneca", "long_term_informed"),
            ("Point72", "long_term_informed"),
            # NEA full and short form (the 'nea' short needle is also the
            # source of the Lineage false positive asserted elsewhere).
            ("New Enterprise Associates", "long_term_informed"),
        ],
    )
    def test_known_name_sweep(self, name, cls):
        assert investor_class(name) == cls


class TestInvestorClassCaseAndSuffix:
    @pytest.mark.parametrize("variant", ["PERCEPTIVE ADVISORS",
                                         "perceptive advisors",
                                         "Perceptive Advisors"])
    def test_case_insensitive(self, variant):
        assert investor_class(variant) == "long_term_informed"

    def test_suffix_tolerance_eloc(self):
        # 'lincoln park capital' needle + ', LLC' suffix not enumerated.
        assert investor_class("Lincoln Park Capital Fund, LLC") == "eloc_funder"

    def test_master_fund_suffix_tolerated(self):
        # 'sabby' needle inside the long legal fund name.
        assert investor_class("Sabby Volatility Warrant Master Fund Ltd") == "pipe_flipper"

    def test_partners_suffix_tolerated(self):
        assert investor_class("BVF Partners L.P.") == "long_term_informed"

    def test_leading_trailing_whitespace(self):
        assert investor_class("  OrbiMed  ") == "long_term_informed"


class TestInvestorClassOrderingPrecedence:
    """toxic_lender bucket precedes pipe_flipper, and the shorter
    'boothbay' needle in the toxic bucket matches before 'boothbay
    absolute return' in the pipe_flipper bucket."""

    def test_boothbay_resolves_to_toxic_lender(self):
        # 'boothbay' lives in _TOXIC_LENDERS (iterated before _PIPE_FLIPPERS,
        # which holds 'boothbay absolute return'); first-bucket-wins ->
        # toxic_lender, NOT pipe_flipper.
        assert investor_class("Boothbay Absolute Return Fund") == "toxic_lender"

    def test_bare_boothbay_toxic(self):
        assert investor_class("Boothbay") == "toxic_lender"

    def test_bucket_order_is_the_documented_sequence(self):
        order = [cls for cls, _ in ct._INVESTOR_CLASS_BUCKETS]
        assert order == ["long_term_informed", "eloc_funder",
                         "toxic_lender", "pipe_flipper"]

    def test_eloc_precedes_toxic_when_both_needles_present(self):
        # Cross-bucket precedence on a contrived string carrying needles from
        # TWO different buckets that are NOT an alias overlap (unlike boothbay):
        # 'lincoln park' (eloc_funder) is in an earlier bucket than
        # 'lind global' (toxic_lender), so eloc_funder wins.
        assert investor_class("Lincoln Park and Lind Global JV") == "eloc_funder"

    def test_long_term_precedes_pipe_when_both_needles_present(self):
        # 'baker bros' (long_term_informed, first bucket) beats 'sabby'
        # (pipe_flipper, last bucket) in a contrived combined string.
        assert investor_class("Baker Bros with Sabby") == "long_term_informed"

    def test_last_bucket_only_match_resolves(self):
        # A name whose ONLY matching needle lives in the LAST bucket still
        # classifies — the loop falls through the first three buckets cleanly.
        # 'iroquois' is only in _PIPE_FLIPPERS.
        assert investor_class("Iroquois Master Fund") == "pipe_flipper"


class TestInvestorClassFalsePositives:
    """Naive substring short needles produce false positives. Documented,
    intended advisory behavior; asserted to catch accidental drift."""

    def test_lind_substring_false_positive(self):
        # 'lind' (3-char toxic needle) is a substring of 'Highlinde'.
        # BUG-ish-but-intended: would misclassify an unrelated name.
        assert investor_class("Highlinde Partners") == "toxic_lender"

    def test_lind_in_berlinder(self):
        # 'Berlinder' contains 'lind' -> toxic_lender.
        assert investor_class("Berlinder") == "toxic_lender"

    def test_nea_substring_false_positive(self):
        # 'nea' (in _LONG_TERM_INFORMED for New Enterprise Associates)
        # matches 'Lineage Capital'.
        assert investor_class("Lineage Capital") == "long_term_informed"

    @pytest.mark.parametrize(
        "name, cls",
        [
            # 'bvf' (BVF Partners, _LONG_TERM_INFORMED) inside 'Subvfund'.
            ("Subvfund Capital", "long_term_informed"),
            # 'bms' (Bristol-Myers Squibb, _LONG_TERM_INFORMED) inside 'Lambms'.
            ("Lambms Fund", "long_term_informed"),
        ],
    )
    def test_short_needle_false_positives_sweep(self, name, cls):
        # More 3-char-needle false positives flagged by the survey. Naive
        # substring; documented advisory behavior, NOT a routing input.
        assert investor_class(name) == cls

    def test_no_accidental_needle_stays_none(self):
        # Negative control: 'Pacifico Holdings' contains none of the needles
        # (in particular it does NOT contain 'ifc' as a contiguous substring),
        # so it must classify None — guards against over-eager matching.
        assert investor_class("Pacifico Holdings") is None


class TestCrossClassifierIndependence:
    """bank_tier and investor_class are independent tables; the same string
    can classify differently in each (B. Riley Principal Capital)."""

    def test_b_riley_principal_dual_classification(self):
        s = "B. Riley Principal Capital"
        assert bank_tier(s) == "middle_market"        # 'b. riley' in _MIDDLE_MARKET
        assert investor_class(s) == "eloc_funder"     # 'b. riley principal' in _ELOC_FUNDERS

    def test_goldman_has_no_investor_class(self):
        assert investor_class("Goldman Sachs") is None

    def test_baker_bros_has_no_bank_tier(self):
        assert bank_tier("Baker Bros") is None


class TestModuleSurface:
    """Trivial export / type-alias surface checks."""

    def test_all_exports(self):
        assert set(ct.__all__) == {
            "BankTier", "InvestorClass", "bank_tier", "investor_class"
        }

    def test_public_callables_present(self):
        assert callable(ct.bank_tier)
        assert callable(ct.investor_class)
