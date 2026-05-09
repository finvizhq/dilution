"""HTML → text conversion that preserves inline numeric cells.

Why: edgartools' ``markdown()`` / ``text()`` treat DIVs and custom iXBRL
wrappers (``<ix:nonFraction>``) as block elements, which inserts
newlines between "$" and the numeric value. For 10-K narratives like
"an exercise price of $2.00 per share" where iXBRL wraps the number,
this destroys the data we need.

Our converter:
- Keeps *semantic* block elements as newline-producers:
  p, h1-h6, li, tr, br, hr, blockquote, pre, section, article,
  header, footer, aside, nav, caption, title
- INLINES everything else (div, span, ix:*) so inline-wrapped numbers
  stay adjacent to their currency marker.
- Inserts tab separators between table cells (td/th) for readability.
- Drops script, style, head, and any elements hidden via style="display:none"

Output is plain text, NOT markdown. Our overhang/events prompts don't
need markdown structure — they just need accurate numbers in a
readable order.
"""

import re

from bs4 import BeautifulSoup, NavigableString

_BLOCK_TAGS = {
    "p", "h1", "h2", "h3", "h4", "h5", "h6",
    "li", "tr", "br", "hr",
    "blockquote", "pre", "section", "article",
    "header", "footer", "aside", "nav",
    "caption", "title",
    # Table container also separates
    "table",
}

_SKIP_TAGS = {"script", "style", "head", "meta", "link", "noscript"}

_CELL_TAGS = {"td", "th"}


def _is_hidden(el) -> bool:
    style = (el.get("style") or "").replace(" ", "").lower()
    if "display:none" in style or "visibility:hidden" in style:
        return True
    return False


def _emit(node, out: list):
    """Walk the soup tree, appending text fragments to `out`."""
    if isinstance(node, NavigableString):
        t = str(node)
        if t:
            out.append(t)
        return

    name = (node.name or "").lower()
    if name in _SKIP_TAGS:
        return
    if _is_hidden(node):
        return

    if name in _BLOCK_TAGS:
        out.append("\n")

    if name in _CELL_TAGS:
        out.append("\t")

    for child in node.children:
        _emit(child, out)

    if name in _BLOCK_TAGS:
        out.append("\n")


def html_to_text(html: str) -> str:
    """Return plaintext preserving inline numeric cells."""
    if not html:
        return ""
    soup = BeautifulSoup(html, "lxml")
    out: list[str] = []
    _emit(soup, out)
    raw = "".join(out)

    # Collapse whitespace: keep at most 2 consecutive newlines,
    # collapse internal spaces/tabs to one, strip per-line.
    lines = [ln.rstrip() for ln in raw.split("\n")]
    # Collapse runs of empty lines
    collapsed = []
    blank_run = 0
    for ln in lines:
        # strip internal runs of whitespace (but keep single tabs between cells)
        ln2 = re.sub(r"[ \t]+", lambda m: "\t" if "\t" in m.group(0) else " ",
                     ln).strip()
        if not ln2:
            blank_run += 1
            if blank_run <= 1:
                collapsed.append("")
        else:
            blank_run = 0
            collapsed.append(ln2)

    return "\n".join(collapsed).strip()
