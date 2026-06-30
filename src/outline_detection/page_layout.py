"""
page_layout.py — Page segmentation and line-density signals for OCR input.

Rules A–H in ``detector.py`` look at orthographic signals inside a flat text
string. The optional Rules I–L instead look at *page structure*: how many
lines each page has. OCR output usually arrives as a sequence of pages, and a
text break very often lines up with a page that is empty or much shorter than
its neighbours (end of one text, start of the next).

This module turns raw text into a list of :class:`Page` objects with accurate
character offsets, so layout-derived boundaries can be merged with the
orthographic candidates produced by the rest of the detector.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List

FORM_FEED = "\f"

# A line is treated as a folio / page number (and skipped when
# ``ignore_page_number_lines`` is on) when it contains nothing but Latin or
# Tibetan digits, whitespace, dashes and dots — e.g. "449", "1-70", "171 447",
# or a lone "-".
_PAGE_NUMBER_LINE = re.compile(r"^[0-9\u0F20-\u0F29\s\-\u2013\u2014.]+$")


@dataclass
class Page:
    """A single page carved out of the source text.

    Attributes:
        start: Character offset of the first character of the page in the
            original text.
        end: Character offset just past the last character of the page.
        content_start: Offset of the first non-whitespace character (used as
            the boundary position). Falls back to ``start`` for an empty page.
        lines: Raw lines of the page (including empty ones), split on ``\\n``.
        nb_lines: Number of non-empty lines after optional page-number
            filtering — the figure the line-density rules compare.
    """

    start: int
    end: int
    content_start: int
    lines: List[str]
    nb_lines: int

    @property
    def is_empty(self) -> bool:
        return self.nb_lines == 0


def _is_page_number_line(stripped: str) -> bool:
    return bool(_PAGE_NUMBER_LINE.match(stripped))


def count_nonempty_lines(page_text: str, *, ignore_page_number_lines: bool = True) -> int:
    """Count non-empty lines in a page.

    Whitespace-only lines never count. When ``ignore_page_number_lines`` is
    set, lines that are only a folio / page number are skipped too, since OCR
    commonly places the number alone on its own line.
    """
    count = 0
    for line in page_text.split("\n"):
        stripped = line.strip()
        if not stripped:
            continue
        if ignore_page_number_lines and _is_page_number_line(stripped):
            continue
        count += 1
    return count


def _first_content_offset(text: str, start: int, end: int) -> int:
    """Offset of the first non-whitespace char in ``text[start:end]``."""
    i = start
    while i < end and text[i].isspace():
        i += 1
    return i if i < end else start


def _spans_from_separators(text: str, sep_regex: "re.Pattern[str]") -> List[tuple]:
    """Return ``(start, end)`` spans for the text between separator matches.

    Consecutive separators yield empty spans on purpose: an empty page is a
    strong text-break signal (Rule I).
    """
    spans = []
    last = 0
    for m in sep_regex.finditer(text):
        spans.append((last, m.start()))
        last = m.end()
    spans.append((last, len(text)))
    return spans


def _blank_line_regex(min_blank_lines: int) -> "re.Pattern[str]":
    # A run of ``min_blank_lines`` or more newlines, each optionally surrounded
    # by spaces / tabs. ``min_blank_lines=2`` means a blank line separates pages.
    n = max(2, int(min_blank_lines))
    return re.compile(r"(?:[ \t]*\n[ \t]*){" + str(n) + r",}")


def resolve_delimiter_regex(delimiter: str, min_blank_lines: int, text: str) -> "re.Pattern[str]":
    """Map a delimiter spec to a compiled separator regex.

    - ``"\\f"`` (default): split on single form feeds; if none are present in
      ``text`` fall back to blank-line splitting.
    - ``"blank"``: force blank-line splitting (``min_blank_lines`` newlines).
    - anything else: treat the string as a regular expression.
    """
    if delimiter == FORM_FEED:
        if FORM_FEED in text:
            return re.compile(re.escape(FORM_FEED))
        return _blank_line_regex(min_blank_lines)
    if delimiter == "blank":
        return _blank_line_regex(min_blank_lines)
    return re.compile(delimiter)


def segment_pages(
    text: str,
    *,
    delimiter: str = FORM_FEED,
    min_blank_lines: int = 2,
    ignore_page_number_lines: bool = True,
) -> List[Page]:
    """Split ``text`` into pages with accurate character offsets.

    Returns a list of :class:`Page`. When the text cannot be split into more
    than one page the result has a single element, which the line-density
    rules treat as "no page structure" (they emit no candidates).
    """
    sep_regex = resolve_delimiter_regex(delimiter, min_blank_lines, text)
    spans = _spans_from_separators(text, sep_regex)

    pages: List[Page] = []
    for start, end in spans:
        page_text = text[start:end]
        nb_lines = count_nonempty_lines(
            page_text, ignore_page_number_lines=ignore_page_number_lines
        )
        pages.append(
            Page(
                start=start,
                end=end,
                content_start=_first_content_offset(text, start, end),
                lines=page_text.split("\n"),
                nb_lines=nb_lines,
            )
        )
    return pages


def char_offset_at_page_start(pages: List[Page], page_index: int) -> int:
    """Boundary offset for the start of ``pages[page_index]``.

    Uses the first non-whitespace character of the page (its ``content_start``)
    so the boundary lands on real text, matching Rule A's convention.
    """
    return pages[page_index].content_start
