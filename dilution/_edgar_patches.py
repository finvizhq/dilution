"""Monkey-patch edgartools on import to fix text-extraction bugs.

All patches verified unfixed upstream as of 5.35.0 (identical code at
every site). Applied at import time; each is idempotent via a class
flag, so re-imports won't double-wrap. Audit trail: memory note
``edgartools-text-audit`` (2026-06-03) — synthetic repro + production
DB evidence per patch.

Patch 1 — `$`/iXBRL adjacency splits (edgar.files.html).

Bug: `edgar.files.html.SECHTMLParser._get_text_with_spacing` calls
`_clean_text` (which strips leading/trailing whitespace, edgar/files/
html.py:823) before the adjacency heuristic checks whether to insert
a separator between a NavigableString child and a sibling Tag. By
then the leading-/trailing-space signal is gone. The Tag branch's
fallback heuristic then guesses wrong when the preceding text ends
with `$`, producing artifacts like `$ 0.001per share`, `$ 156,172`,
`6,259,279shares`.

Fix: examine leading/trailing whitespace on the raw `str(child)`
BEFORE `_clean_text` strips it, and insert a single space into the
output list explicitly when adjacency requires it.

Patch 2 — lxml `.tail` loss in the new document parser
(edgar.documents, used by `filing.obj()` → `CurrentReport.sections`,
which periodic_sections.select_text feeds to the walker for 8-Ks).

Bug: `DocumentBuilder._get_element_text` reads `element.text` and each
child's `text_content()`, but never `child.tail` — and lxml stores the
text that FOLLOWS an inline child element in `child.tail`, not in the
parent. Any paragraph whose prose is interleaved with inline spans
(the ubiquitous quoted-defined-term markup: `(the “<span>Company</span>”)
issued 152,000 shares …`) loses everything after the first nested
span. CETY 8-K 0001493152-25-025780 lost two of its eight Item 3.02
conversion events this way; the signature appears in 65 stored 8-K/6-K
bodies across 17 CIKs.

Fix: append `child.tail` after each child's content, mirroring the
upstream whitespace conventions per branch (raw for inline elements,
stripped for h1-h6).

Patch 3 — TOC-path table cells merge (edgar.documents.extractors.
toc_section_extractor). The TOC lazy extractor is the PRIMARY text
path for 10-K/10-Q/20-F sections.

Bug: `SECSectionExtractor._extract_section_content` walks the raw
HTML with iterwalk and emits '\\n\\n' after block elements — but td/th
are not block elements, so adjacent cells in minified HTML
concatenate with NO separator. Production: 207 stored periodic docs,
1,398 merges, incl. fused share counts ("107,981,44128,800,493" on
XTIA, quadruple-fused weighted-average shares on HLLY).

Fix: emit a two-space separator at each td/th end-event.

Patch 4 — preprocessor deletes word-separating spaces
(edgar.documents.processors.preprocessor).

Bug: `_normalize_whitespace` strips ALL whitespace between text and a
tag boundary (`spaces_before_tags`/`spaces_after_tags` sub('')), so
`shares <span>(the "Shares")</span> to` parses as
`shares(the "Shares")to`. Production: 39 stored 8-K bodies with
fusion signatures ("sell up to2,000,000 units" on CETY).

Fix: collapse those runs to a single space instead of deleting.

Patch 5 — builder swallows boundary whitespace + page-skip tails
(edgar.documents.strategies.document_builder).

Bugs: (a) `_process_element` strips text at TextNode creation and
discards the leading/trailing-space signal, so even with Patch 4 the
space dies one layer up; (b) the three page-artifact skips
(page-number container / page-break hr / page-nav container) return
None WITHOUT preserving `element.tail`, unlike the SKIP_ELEMENTS
branch right above them — text after a page-break <hr> vanishes.

Fix: mirror `_process_element`; TextNodes keep one leading space when
the raw text had leading whitespace (ParagraphNode's startswith(' ')
spacing rule then fires) and set has_tail_whitespace metadata when
trailing whitespace was stripped (ParagraphNode's existing metadata
rule re-inserts it); the three page-artifact skips preserve tails.

Patch 6 — fast table renderer drops lines 2+ of multi-line data cells
(edgar.documents.renderers.fast_table; fast rendering is the
production default via ParserConfig.fast_table_rendering=True).

Bug: `_build_table` formats HEADER rows with `_format_multiline_row`
but DATA rows with `_format_row`, which truncates any cell containing
'\\n' to its first line ("take first line only"). A cell like
`<td><div>1,955,122</div><div>(gross)</div></td>` loses "(gross)".

Fix: route data rows through `_format_multiline_row` too (identical
output for single-line rows).

Patch 7 — th/td cell-order swap (edgar.documents.strategies.
table_processing).

Bug: `_process_row` collects `tr.findall('.//td') + tr.findall('.//th')`
— all td's first, then all th's — so a row-header `<th>` lands AFTER
the value cells, misaligning every column when rows mix th and td
("808,071  416,452  Notes payable").

Fix: collect cells in document order via an XPath union.

Patch 8 — old parser loses tbody rows on mixed tables
(edgar.files.html, the exhibits / full-document-fallback path).

Bug: `_process_table` uses direct-child `<tr>`s if any exist and only
falls back to tbody/thead/tfoot otherwise — a table with BOTH direct
`<tr>`s and a `<tbody>` keeps only the direct rows; the entire tbody
body is dropped.

Fix: when the mixed shape is present, unwrap the section wrappers
in-place (document order preserved) before the original logic runs.
Only fires on the pathological markup, where upstream loses whole
table bodies; the thead is_header flag is a strict-improvement
trade-off there.
"""

import re

from bs4 import NavigableString
from edgar.files import html as _eh
from edgar.documents.strategies import document_builder as _edb
from edgar.documents.strategies import table_processing as _etp
from edgar.documents.extractors import toc_section_extractor as _tse
from edgar.documents.processors import preprocessor as _epp
from edgar.documents.renderers import fast_table as _eft
from edgar.documents.nodes import TextNode as _TextNode


_PATCH_FLAG = "_dilution_dollar_patch_applied"
_TAIL_PATCH_FLAG = "_dilution_tail_patch_applied"
_TOC_CELL_PATCH_FLAG = "_dilution_toc_cell_patch_applied"
_WSPACE_PATCH_FLAG = "_dilution_wspace_patch_applied"
_BUILDER_SPACING_PATCH_FLAG = "_dilution_builder_spacing_patch_applied"
_MULTILINE_ROW_PATCH_FLAG = "_dilution_multiline_row_patch_applied"
_CELL_ORDER_PATCH_FLAG = "_dilution_cell_order_patch_applied"
_TBODY_PATCH_FLAG = "_dilution_tbody_patch_applied"
_STICKY_TAIL = re.compile(r"[\$€£¥₹]$")


def _is_sticky_boundary(prev: str) -> bool:
    return bool(_STICKY_TAIL.search(prev.rstrip()))


def _patched_get_text_with_spacing(self, element):
    if element.name == "table":
        return ""

    texts: list[str] = []
    last_was_text = False

    for child in element.children:
        if isinstance(child, NavigableString):
            raw = str(child)
            text = self._clean_text(raw)
            if text:
                leading = bool(raw) and raw[0].isspace()
                trailing = bool(raw) and raw[-1].isspace()
                if leading and texts and not texts[-1].endswith(" "):
                    texts.append(" ")
                texts.append(text)
                if trailing:
                    texts.append(" ")
                    last_was_text = False
                else:
                    last_was_text = True
            elif raw and raw.strip() == "" and last_was_text:
                if texts and not texts[-1].endswith(" "):
                    texts.append(" ")
                last_was_text = False
        elif child.name == "br":
            texts.append("\n")
            last_was_text = False
        elif child.name == "table":
            continue
        else:
            child_text = self._get_text_with_spacing(child)
            stripped = child_text.strip()
            if stripped:
                leading = child_text[:1].isspace()
                trailing = child_text[-1:].isspace()
                if (
                    texts
                    and last_was_text
                    and not texts[-1].endswith(" ")
                    and not leading
                    and not _is_sticky_boundary(texts[-1])
                ):
                    texts.append(" ")
                texts.append(stripped)
                if trailing:
                    texts.append(" ")
                    last_was_text = False
                else:
                    last_was_text = True

    return "".join(texts)


if not getattr(_eh.SECHTMLParser, _PATCH_FLAG, False):
    _eh.SECHTMLParser._get_text_with_spacing = _patched_get_text_with_spacing
    setattr(_eh.SECHTMLParser, _PATCH_FLAG, True)


def _patched_get_element_text(self, element) -> str:
    """Tail-preserving replacement for DocumentBuilder._get_element_text.

    Identical to upstream except that each child's `tail` (lxml's home
    for the text that follows the child element) is appended after the
    child's content. Tails are appended even for SKIP_ELEMENTS children
    — the text after a <script>/<style> block is real prose even though
    the block itself isn't.
    """
    tag = element.tag.lower() if isinstance(element.tag, str) else ""
    is_inline = tag in self.INLINE_ELEMENTS
    text_parts: list[str] = []

    # Get element's direct text
    if element.text:
        # For inline elements, preserve leading/trailing whitespace
        if is_inline:
            text_parts.append(element.text)
        else:
            text_parts.append(element.text.strip())

    # For simple elements, get all text content
    if is_inline or tag in ("h1", "h2", "h3", "h4", "h5", "h6"):
        for child in element:
            ctag = child.tag.lower() if isinstance(child.tag, str) else ""
            if ctag not in self.SKIP_ELEMENTS:
                child_text = child.text_content()
                if child_text:
                    # For inline elements, preserve whitespace in child
                    # content too
                    if is_inline:
                        text_parts.append(child_text)
                    else:
                        text_parts.append(child_text.strip())
            # The fix: text between/after inline children lives in
            # child.tail, which upstream drops entirely.
            if child.tail:
                if is_inline:
                    text_parts.append(child.tail)
                elif child.tail.strip():
                    text_parts.append(child.tail.strip())

    # For inline elements with preserved whitespace, concatenate
    # directly when there's a single run; join with spaces otherwise
    # (upstream convention, kept to minimize output drift).
    if is_inline and len(text_parts) == 1:
        return text_parts[0]
    return " ".join(text_parts)


if not getattr(_edb.DocumentBuilder, _TAIL_PATCH_FLAG, False):
    _edb.DocumentBuilder._get_element_text = _patched_get_element_text
    setattr(_edb.DocumentBuilder, _TAIL_PATCH_FLAG, True)


# --------------------------------------------------------------------
# Patch 3 — TOC iterwalk: separate table cells.
# Mirror of SECSectionExtractor._extract_section_content; the only
# change is the td/th end-event separator (see module docstring).
# --------------------------------------------------------------------

def _patched_extract_section_content(self, html_content, boundary,
                                     include_subsections, clean):
    from lxml import etree
    from lxml import html as lxml_html
    from edgar.documents.utils.anchor_targets import (
        find_anchor_targets,
        is_anchor_match,
    )

    tree = self._tree
    if tree is None:
        if html_content.startswith('<?xml'):
            html_content = re.sub(r'<\?xml[^>]*\?>', '', html_content,
                                  count=1)
        tree = lxml_html.fromstring(html_content)

    start_elements = find_anchor_targets(tree, boundary.anchor_id)
    if not start_elements:
        return ""

    all_text = []
    in_range = False

    block_elements = {'p', 'div', 'table', 'tr', 'li', 'h1', 'h2', 'h3',
                      'h4', 'h5', 'h6', 'blockquote', 'pre', 'section',
                      'article', 'header', 'footer'}
    # The fix: cell boundaries get an explicit separator so adjacent
    # cells in minified HTML can't fuse ("Coventry note92,00010,120").
    cell_elements = {'td', 'th'}

    for event, el in etree.iterwalk(tree, events=('start', 'end')):
        if not hasattr(el, 'get'):
            continue

        el_id = el.get('id', '')
        tag_name = el.tag.lower() if isinstance(el.tag, str) else ''

        if event == 'start':
            if is_anchor_match(el, boundary.anchor_id):
                in_range = True
                continue
            if boundary.end_element_id and is_anchor_match(
                    el, boundary.end_element_id):
                in_range = False
                break
            if (in_range and not include_subsections
                    and self._is_sibling_section(el_id, boundary.name)):
                in_range = False
                break
            if in_range and el.text:
                all_text.append(el.text)

        elif event == 'end':
            if in_range and tag_name in cell_elements:
                all_text.append('  ')
            if in_range and tag_name in block_elements:
                all_text.append('\n\n')
            if in_range and el.tail:
                all_text.append(el.tail)

    combined_text = ''.join(all_text)
    if clean:
        combined_text = self._clean_section_text(combined_text)
    return combined_text


if not getattr(_tse.SECSectionExtractor, _TOC_CELL_PATCH_FLAG, False):
    _tse.SECSectionExtractor._extract_section_content = (
        _patched_extract_section_content
    )
    setattr(_tse.SECSectionExtractor, _TOC_CELL_PATCH_FLAG, True)


# --------------------------------------------------------------------
# Patch 4 — preprocessor: collapse boundary whitespace to one space
# instead of deleting it. Mirror of HTMLPreprocessor.
# _normalize_whitespace; only the two sub('') → sub(' ') change.
# --------------------------------------------------------------------

def _patched_normalize_whitespace(self, html: str) -> str:
    html = self._compiled_patterns['multiple_spaces'].sub(' ', html)
    html = self._compiled_patterns['multiple_newlines'].sub('\n\n', html)
    html = self._compiled_patterns['spaces_between_tags'].sub(' ', html)
    # Upstream deletes these runs outright, fusing `word <span>` into
    # `word<span>`. A single space preserves the word boundary.
    html = self._compiled_patterns['spaces_before_tags'].sub(' ', html)
    html = self._compiled_patterns['spaces_after_tags'].sub(' ', html)
    html = self._compiled_patterns['block_open_tags'].sub(r'\n\1', html)
    html = self._compiled_patterns['block_close_tags'].sub(r'\1\n', html)
    html = self._compiled_patterns['multiple_newlines'].sub('\n\n', html)
    return html.strip()


if not getattr(_epp.HTMLPreprocessor, _WSPACE_PATCH_FLAG, False):
    _epp.HTMLPreprocessor._normalize_whitespace = (
        _patched_normalize_whitespace
    )
    setattr(_epp.HTMLPreprocessor, _WSPACE_PATCH_FLAG, True)


# --------------------------------------------------------------------
# Patch 5 — builder: keep boundary-whitespace signals + preserve tails
# at the three page-artifact skip sites. Mirror of DocumentBuilder.
# _process_element (upstream duplicates its tail handling across two
# branches; the mirror folds them — semantics identical).
# --------------------------------------------------------------------

def _add_text_node(parent, raw, preserve):
    """TextNode from raw text without losing its spacing signals.

    Keeps one leading space when the raw run started with whitespace
    (ParagraphNode's `startswith(' ')` rule re-inserts the separator)
    and records swallowed trailing whitespace via the existing
    has_tail_whitespace metadata (ParagraphNode's metadata rule).
    """
    if preserve:
        parent.add_child(_TextNode(content=raw))
        return
    stripped = raw.strip()
    if stripped:
        content = (' ' + stripped) if raw[0].isspace() else stripped
        tn = _TextNode(content=content)
        if raw[-1].isspace():
            tn.set_metadata('has_tail_whitespace', True)
        parent.add_child(tn)


def _patched_process_element(self, element, parent):
    pw = self.config.preserve_whitespace

    if element.tag in self.SKIP_ELEMENTS:
        if element.tail:
            _add_text_node(parent, element.tail, pw)
        return None

    # Page-artifact skips: upstream returns None here without the tail
    # preservation the SKIP_ELEMENTS branch has — text following a
    # page-break <hr> or a page-number container was silently lost.
    if (self._is_page_number_container(element)
            or self._is_page_break_element(element)
            or self._is_page_navigation_container(element)):
        if element.tail:
            _add_text_node(parent, element.tail, pw)
        return None

    self.context.depth += 1
    try:
        if element.tag.startswith('{'):
            self._enter_xbrl_context(element)

        style = self._extract_style(element)
        node = self._create_node_for_element(element, style)

        if node:
            if self.xbrl_context_stack:
                node.metadata.update(self._get_current_xbrl_metadata())
            parent.add_child(node)

            if self._should_process_children(element, node):
                if element.text:
                    _add_text_node(node, element.text, pw)
                for child in element:
                    self._process_element(child, node)

            # Tail handling — identical for the children-processed and
            # not-processed branches (upstream duplicates this block).
            if element.tail:
                if pw:
                    parent.add_child(_TextNode(content=element.tail))
                elif element.tail.strip():
                    _add_text_node(parent, element.tail, pw)
                elif element.tail.isspace():
                    if hasattr(node, 'set_metadata'):
                        node.set_metadata('has_tail_whitespace', True)
        else:
            for child in element:
                self._process_element(child, parent)
            if element.tail:
                _add_text_node(parent, element.tail, pw)

        if element.tag.startswith('{'):
            self._exit_xbrl_context(element)

        return node
    finally:
        self.context.depth -= 1


if not getattr(_edb.DocumentBuilder, _BUILDER_SPACING_PATCH_FLAG, False):
    _edb.DocumentBuilder._process_element = _patched_process_element
    setattr(_edb.DocumentBuilder, _BUILDER_SPACING_PATCH_FLAG, True)


# --------------------------------------------------------------------
# Patch 6 — fast table renderer: render every line of multi-line data
# cells. Mirror of FastTableRenderer._build_table; data rows now use
# _format_multiline_row like headers already do (identical output for
# single-line rows).
# --------------------------------------------------------------------

def _patched_build_table(self, headers, rows, col_widths, alignments):
    lines = []
    if headers:
        for header_row in headers:
            if any(cell.strip() for cell in header_row):
                lines.extend(self._format_multiline_row(
                    header_row, col_widths, alignments))
        if self.style.header_separator:
            lines.append(self._create_separator_line(col_widths))
    for row in rows:
        if any(cell.strip() for cell in row):
            lines.extend(self._format_multiline_row(
                row, col_widths, alignments))
    return '\n'.join(lines)


if not getattr(_eft.FastTableRenderer, _MULTILINE_ROW_PATCH_FLAG, False):
    _eft.FastTableRenderer._build_table = _patched_build_table
    setattr(_eft.FastTableRenderer, _MULTILINE_ROW_PATCH_FLAG, True)


# --------------------------------------------------------------------
# Patch 7 — table processor: collect row cells in document order.
# Mirror of TableProcessor._process_row; the upstream td-then-th
# concatenation moves row-header <th> cells after the values.
# --------------------------------------------------------------------

def _patched_process_row(self, tr, is_header):
    cells = []
    # XPath unions return document order — mixed th/td rows keep their
    # real cell sequence.
    for cell_elem in tr.xpath('.//td | .//th'):
        cell = self._process_cell(
            cell_elem, is_header or cell_elem.tag == 'th')
        if cell:
            cells.append(cell)
    return cells


if not getattr(_etp.TableProcessor, _CELL_ORDER_PATCH_FLAG, False):
    _etp.TableProcessor._process_row = _patched_process_row
    setattr(_etp.TableProcessor, _CELL_ORDER_PATCH_FLAG, True)


# --------------------------------------------------------------------
# Patch 8 — old parser: don't drop tbody rows when a table mixes
# direct <tr> children with tbody/thead/tfoot sections. Wrapper (not a
# mirror): unwrap the section containers in document order before the
# original row collection runs. Only fires on the mixed shape, where
# upstream would otherwise lose the entire section-wrapped body.
# --------------------------------------------------------------------

def _patched_process_table(self, element):
    if element is not None:
        try:
            direct_trs = element.find_all('tr', recursive=False)
            sections = element.find_all(['tbody', 'thead', 'tfoot'],
                                        recursive=False)
            if direct_trs and sections:
                for section in sections:
                    section.unwrap()
        except Exception:  # noqa: BLE001 - never let the guard break parsing
            pass
    return self._dilution_orig_process_table(element)


if not getattr(_eh.SECHTMLParser, _TBODY_PATCH_FLAG, False):
    # Stash the original on the class so a module re-execution can't
    # capture the wrapper and recurse.
    _eh.SECHTMLParser._dilution_orig_process_table = (
        _eh.SECHTMLParser._process_table
    )
    _eh.SECHTMLParser._process_table = _patched_process_table
    setattr(_eh.SECHTMLParser, _TBODY_PATCH_FLAG, True)
