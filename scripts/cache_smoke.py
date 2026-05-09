"""Smoke-test prompt caching on the ledger walker prompt.

Sends three back-to-back walker calls against grok and reports xAI's
`cached_prompt_text_tokens` for each. With the rulebook in the system
prompt:
  - call 1 (cold): cached ≈ 0
  - call 2 (different user, same system): cached ≈ size(SYSTEM_PROMPT)
  - call 3 (identical to call 1): cached ≈ size(SYSTEM_PROMPT + user)

Calls xai_sdk directly so we can read .usage off the raw response —
the project's _Response wrapper drops it.
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config  # noqa: E402
from dilution.ledger._llm_utils import (  # noqa: E402
    EXTRACT_SEED, EXTRACT_TEMPERATURE,
    unit_preamble as build_unit_preamble,
)
from dilution.ledger.mutations import MutationList  # noqa: E402
from dilution.ledger.walker_prompt import (  # noqa: E402
    SYSTEM_PROMPT, build_user_prompt,
)


XTL_FILING = """\
On March 20, 2026, XTL Biopharmaceuticals Ltd. (the "Company") issued a press release announcing
that the Company plans to change the ratio of its American Depositary Shares ("ADSs") to its
ordinary shares, par value NIS0.1 per share (the "ADS Ratio"), from the current ADS Ratio of one
(1) ADS to one hundred (100) ordinary shares, to a new ADS Ratio of one (1) ADS to four hundred
(400) ordinary shares (the "ADS Ratio Change"). The Company anticipates that the ADS Ratio Change
will be effective on March 25, 2026. For the Company's ADS holders, the change in the ADS Ratio
will have the same effect as a one-for-four reverse ADS split. The ADS Ratio Change will have no
impact on XTL's underlying ordinary shares, and no ordinary shares will be issued or cancelled in
connection with the ADS Ratio Change.
"""

XTL_LEDGER = """\
## Open ledger
type        id      created     counterparty  terms                                              outstanding              status
warrant     W-149   2017-03-07  —             strike=2.3 anti_dilution_type=Customary           count=1,083,312          active
warrant     W-150   2024-08-14  —             strike=0.0001 anti_dilution_type=undisclosed      —                        active
warrant     W-151   2024-08-14  —             strike=1.2 anti_dilution_type=Customary           count=1,500,000          active
"""

NOOP_FILING = """\
On May 5, 2026, the Company issued a press release reaffirming its commitment to its existing
strategic plan. No new securities were issued and no terms of any outstanding instrument were
modified. The Company maintains its existing capital structure.
"""


def _print_usage(label: str, response) -> None:
    usage = getattr(response, "usage", None)
    if usage is None:
        print(f"{label}: no usage on response  type={type(response).__name__}")
        return
    prompt = getattr(usage, "prompt_tokens", 0) or 0
    completion = getattr(usage, "completion_tokens", 0) or 0
    cached = getattr(usage, "cached_prompt_text_tokens", 0) or 0
    total = getattr(usage, "total_tokens", 0) or 0
    pct = (cached / prompt * 100) if prompt else 0.0
    print(f"{label}: prompt={prompt:>6}  cached={cached:>6}  "
          f"({pct:5.1f}%)  completion={completion:>5}  total={total:>6}")


async def _one_call(client, *, unit_pre: str, ledger: str, filing_text: str,
                    accession: str, form: str, filing_date: str):
    from xai_sdk.chat import system as _xs, user as _xu

    user_prompt = build_user_prompt(
        unit_preamble=unit_pre,
        ledger_view=ledger,
        form=form,
        filing_date=filing_date,
        accession=accession,
        items=None,
        period_of_report=filing_date,
        filing_text=filing_text,
    )
    chat = client.chat.create(
        model=config.LLM_MODEL,
        max_tokens=4_000,
        temperature=EXTRACT_TEMPERATURE,
        seed=EXTRACT_SEED,
        response_format=MutationList,
    )
    chat.append(_xs(SYSTEM_PROMPT))
    chat.append(_xu(user_prompt))
    return await chat.sample(), user_prompt


async def main() -> None:
    from xai_sdk.aio.client import Client as XaiAsyncClient

    print(f"model: {config.LLM_MODEL}  provider: {config.LLM_PROVIDER}")
    print(f"system prompt size: {len(SYSTEM_PROMPT):,} chars")
    print()

    client = XaiAsyncClient(api_key=config.XAI_API_KEY)

    import time
    nonce = f"smoke-{int(time.time())}-{__import__('os').urandom(4).hex()}"
    print(f"nonce: {nonce}\n")

    fpi_pre = build_unit_preamble({"is_fpi": True, "ads_ratio": 100})
    domestic_pre = build_unit_preamble({"is_fpi": False})

    # Inject the nonce into the filing text so this exact request can't
    # have been cached by any prior session.
    cold_filing_a = f"<run-id: {nonce}-A>\n\n" + XTL_FILING
    cold_filing_b = f"<run-id: {nonce}-B>\n\n" + NOOP_FILING

    r1, up1 = await _one_call(
        client, unit_pre=fpi_pre, ledger=XTL_LEDGER,
        filing_text=cold_filing_a,
        accession="cold-a", form="6-K", filing_date="2026-03-20",
    )
    _print_usage("call 1 COLD (unique XTL 6-K)   ", r1)
    print(f"  user prompt size: {len(up1):,} chars")

    r2, up2 = await _one_call(
        client, unit_pre=fpi_pre, ledger=XTL_LEDGER,
        filing_text=cold_filing_a,
        accession="cold-a", form="6-K", filing_date="2026-03-20",
    )
    _print_usage("call 2 verbatim repeat         ", r2)
    print(f"  user prompt size: {len(up2):,} chars")

    r3, up3 = await _one_call(
        client, unit_pre=domestic_pre, ledger=XTL_LEDGER,
        filing_text=cold_filing_b,
        accession="cold-b", form="8-K", filing_date="2026-05-05",
    )
    _print_usage("call 3 same system, new user  ", r3)
    print(f"  user prompt size: {len(up3):,} chars")

    r4, up4 = await _one_call(
        client, unit_pre=fpi_pre, ledger=XTL_LEDGER,
        filing_text=cold_filing_a,
        accession="cold-a", form="6-K", filing_date="2026-03-20",
    )
    _print_usage("call 4 back to call 1's body  ", r4)
    print(f"  user prompt size: {len(up4):,} chars")

    print()
    print("Mutations from call 1 (XTL ADS ratio change):")
    print(r1.content[:600])

    await client.close() if hasattr(client, "close") else None


if __name__ == "__main__":
    asyncio.run(main())
