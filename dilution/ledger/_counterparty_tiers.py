"""Reference classifications for placement agents and counterparties.

Sourced from the DilutionTracker knowledge articles in `knowledge/`:

  * `pipe.txt` — list of long-term-informed PIPE investors (Baker Bros,
    Perceptive, OrbiMed, RA Capital, BVF, RTW, Cormorant, Adage, etc.)
  * `s1.txt`   — Roth / Ladenburg / Brookline as the "trifecta of
    pump-and-dumps"; banking-tier heuristic for ATM impact
  * `atm.txt`  — middle-market+ banking relationships dampen ATM
    impact at issuers > $150M cap
  * `equity_line.txt` — Lincoln Park & Aspire account for >90% of
    small-cap ELOC agreements
  * `convertible.txt` — Lind Global as the canonical price-insensitive
    convertible note lender

Two independent classifications:

  bank_tier(name)       — for placement_agent_canonical
                          → "bulge_bracket" | "middle_market"
                          | "boutique" | "pump_trifecta" | None

  investor_class(name)  — for counterparty_canonical
                          → "long_term_informed" | "eloc_funder"
                          | "toxic_lender" | "pipe_flipper" | None

These are HEURISTICS. The card layer surfaces them so a human can
assess deal quality at a glance — they are NOT used for any
behavioral routing inside the ledger.

Matching is case-insensitive substring against the canonical name.
The lists below carry the most common short forms / canonicalizations
the walker emits; longer legal forms ("LLC", "Capital Partners",
"Master Fund Ltd") are tolerated by substring fallback.
"""

from __future__ import annotations

from typing import Literal

BankTier = Literal[
    "bulge_bracket",
    "middle_market",
    "boutique",
    "pump_trifecta",
]

InvestorClass = Literal[
    "long_term_informed",
    "eloc_funder",
    "toxic_lender",
    "pipe_flipper",
]


# ─── Placement-agent (bank) tiers ───────────────────────────────────
# Tier order matters: when a name matches multiple buckets the FIRST
# match in iteration order wins, so put the more specific / higher-risk
# bucket first. The pump_trifecta list comes before boutique because
# Roth / Ladenburg are technically boutiques but carry a distinct
# small-cap-pump reputation per s1.txt.

_BULGE_BRACKET = (
    "goldman sachs", "goldman",
    "jpmorgan", "j.p. morgan", "jp morgan",
    "morgan stanley",
    "bank of america", "bofa", "merrill lynch",
    "citigroup", "citi",
    "barclays",
    "deutsche bank",
    "credit suisse",
    "ubs",
    "hsbc",
    "wells fargo",
)

_MIDDLE_MARKET = (
    "jefferies",
    "cowen", "td cowen",
    "stifel",
    "raymond james",
    "piper sandler", "piper jaffray",
    "oppenheimer",
    "william blair",
    "b. riley", "b riley", "br securities",
    "cantor fitzgerald", "cantor",
    "needham",
    "evercore",
    "lazard",
    "rbc capital", "rbc",
    "td securities",
    "scotia capital", "scotiabank",
    "bmo capital", "bmo",
    "guggenheim",
    "leerink",        # SVB Leerink / Leerink Partners
    "svb securities",
    "canaccord",
    "robert w. baird", "baird",
)

# Per s1.txt: "These three banks are the trifecta of pump and dumps so
# it was highly likely manipulation was going to happen." Plus a few
# adjacent small-cap underwriters that specialize in similar deal flow
# and recur frequently in microcap S-1s.
_PUMP_TRIFECTA = (
    "roth capital", "roth",
    "ladenburg thalmann", "ladenburg",
    "brookline capital", "brookline",
    "thinkequity",
    "joseph gunnar",
    "spartan capital",
    "univest securities", "univest",
    "boustead securities", "boustead",
    "ez xchange", "network 1",
    "westpark capital",
)

_BOUTIQUE = (
    "h.c. wainwright", "hc wainwright", "h c wainwright", "wainwright",
    "maxim group", "maxim",
    "aegis capital", "aegis",
    "lake street capital", "lake street",
    "craig-hallum", "craig hallum",
    "northland capital", "northland securities", "northland",
    "lucid capital markets", "lucid",
    "a.g.p.", "a g p", "agp", "alliance global partners",
    "dawson james",
    "chardan capital", "chardan",
    "benchmark company", "the benchmark",
    "rodman & renshaw", "rodman",
    "wallachbeth",
    "ascendiant capital", "ascendiant",
    "litchfield hills",
    "ef hutton", "e.f. hutton", "kingswood",
    "curvature securities", "curvature",
)


_BANK_TIER_BUCKETS: tuple[tuple[BankTier, tuple[str, ...]], ...] = (
    # Pump-trifecta first — Roth is technically a boutique but carries
    # a distinct small-cap-pump reputation worth flagging separately.
    ("pump_trifecta", _PUMP_TRIFECTA),
    ("bulge_bracket", _BULGE_BRACKET),
    ("middle_market", _MIDDLE_MARKET),
    ("boutique",      _BOUTIQUE),
)


# ─── Counterparty (investor) classes ────────────────────────────────
# Per pipe.txt — the canonical list of "long-term informed" small-cap
# PIPE investors. Presence of one of these names in counterparty is
# the article's primary signal of a "Good PIPE".
_LONG_TERM_INFORMED = (
    # Healthcare-focused hedge funds / VCs (article's biggest cluster)
    "baker bros", "baker brothers",
    "perceptive advisors", "perceptive",
    "orbimed",
    "ra capital",
    "bvf partners", "bvf",
    "rtw investments", "rtw",
    "cormorant asset management", "cormorant",
    "adage capital", "adage",
    "redmile group", "redmile",
    "frazier healthcare partners", "frazier",
    "venrock",
    "tcg crossover", "tcg",
    "tang capital",
    "pontifax",
    "invus",
    "acorn bioventures", "acorn",
    "deeptrack capital", "deeptrack",
    "avoro capital", "avoro",
    "nantahala capital", "nantahala",
    "rosalind advisors", "rosalind",
    "abingworth",
    "opaleye management", "opaleye",
    "samsara biocapital", "samsara",
    "fairmount funds", "fairmount",
    "point72",
    # Generalist VCs / PEs the article calls out
    "new enterprise associates", "nea",
    "accelmed",
    # Strategic acquirers / large pharma counted as informed
    "pfizer", "merck", "gsk", "glaxosmithkline",
    "johnson & johnson", "j&j", "jnj",
    "novartis", "roche", "sanofi", "astrazeneca", "eli lilly",
    "bristol-myers squibb", "bristol myers", "bms",
    "amgen", "gilead",
    # Sovereign / supranational long-term capital (rare in microcap
    # space but occurs — e.g. EIB lending to listed European biotechs)
    "european investment bank", "eib",
    "international finance corporation", "ifc",
    "european bank for reconstruction", "ebrd",
)

# Per equity_line.txt — Lincoln Park + Aspire account for >90% of
# small-cap ELOCs. Plus the other recurring ELOC funders we've seen
# in walker output.
_ELOC_FUNDERS = (
    "lincoln park capital", "lincoln park",
    "aspire capital",
    "yorkville advisors", "ya ii", "ya global",
    "m2b funding", "m2b",
    "white lion capital", "white lion",
    "tumim stone capital", "tumim stone", "tumim",
    "ghs investments", "ghs",
    "keystone capital",
    "triton funds", "triton",
    "alumni capital",
    "magnetar capital",  # less common but documented
    "b. riley principal", "br principal",
    "rk stone miami", "rk stone",
    "adi funding", "adi capital",
    "sixth borough capital", "sixth borough",
)

# Per convertible.txt — Lind Global as the archetype price-insensitive
# convertible note lender. Plus the other recurring "toxic" small-cap
# convertible note shops.
_TOXIC_LENDERS = (
    "lind global", "lind partners", "lind",
    "streeterville capital", "streeterville",
    "iliad research", "iliad",
    "tysadco partners", "tysadco",
    "alpha capital anstalt", "alpha capital",
    "bellridge capital", "bellridge",
    "crom structured",
    "mast hill fund", "mast hill",
    "1800 diagonal", "diagonal lending",
    "jefferson street capital", "jefferson street",
    "sixth street", "fast capital",
    "labrys fund",
    "boot capital", "boothbay",
)

# PIPE-flipper hedge funds — buy at discount with warrants, intend to
# sell once registered. Per pipe.txt these are the "Bad PIPE" actors;
# also recur as named warrant holders in inducement transactions.
_PIPE_FLIPPERS = (
    "hudson bay capital", "hudson bay",
    "sabby capital", "sabby management", "sabby",
    "anson funds", "anson investments", "anson",
    "empery asset management", "empery",
    "armistice capital", "armistice",
    "heights capital", "heights",
    "intracoastal capital", "intracoastal",
    "ionic ventures", "ionic",
    "altium capital", "altium",
    "kingsbrook opportunities", "kingsbrook",
    "iroquois master fund", "iroquois",
    "warberg",
    "alto opportunity", "alto",
    "3i lp", "3i",
    "cvi investments",
    "high trail", "hightrail",
    "boothbay absolute return",
    "puissance capital", "puissance",
    "l1 capital",
)


_INVESTOR_CLASS_BUCKETS: tuple[tuple[InvestorClass, tuple[str, ...]], ...] = (
    ("long_term_informed", _LONG_TERM_INFORMED),
    ("eloc_funder",        _ELOC_FUNDERS),
    ("toxic_lender",       _TOXIC_LENDERS),
    ("pipe_flipper",       _PIPE_FLIPPERS),
)


# ─── Lookup helpers ─────────────────────────────────────────────────
def _norm(name: str | None) -> str:
    if not name:
        return ""
    return name.strip().lower()


def bank_tier(name: str | None) -> BankTier | None:
    """Classify a placement_agent_canonical (or _placement_agent) name
    into a banking tier. Substring match — the canonical name may carry
    "LLC" / "Group" / "& Co." suffixes the buckets don't enumerate.
    Returns None for unrecognized names (most lower-tier microcap
    shops aren't worth manually classifying)."""
    n = _norm(name)
    if not n:
        return None
    for tier, names in _BANK_TIER_BUCKETS:
        for needle in names:
            if needle in n:
                return tier
    return None


def investor_class(name: str | None) -> InvestorClass | None:
    """Classify a counterparty_canonical (or counterparty) name into a
    qualitative investor class. None for unrecognized — that's the
    norm; the walker sees thousands of obscure LP names that no static
    table can cover."""
    n = _norm(name)
    if not n:
        return None
    for cls, names in _INVESTOR_CLASS_BUCKETS:
        for needle in names:
            if needle in n:
                return cls
    return None


__all__ = [
    "BankTier",
    "InvestorClass",
    "bank_tier",
    "investor_class",
]
