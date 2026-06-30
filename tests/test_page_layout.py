"""Tests for the page-layout rules (I–L) and the page_layout module.

Synthetic inputs use ASCII / neutral Tibetan syllables that do NOT trigger the
orthographic rules A–H, so any predicted boundary is attributable to a layout
rule. This keeps the assertions on exact character offsets unambiguous.
"""

from pathlib import Path

from outline_detection import RuleBasedDetector, detect_breakpoints
from outline_detection.page_layout import (
    count_nonempty_lines,
    segment_pages,
)

FIXTURE = Path(__file__).resolve().parents[1] / "data" / "samples" / "page_layout_cases.txt"


def _rules(text, **kwargs):
    """Return the set of rule labels fired for a detailed detection."""
    res = detect_breakpoints(text, detailed=True, **kwargs)
    return {d["rule"] for d in res["details"]}


# ---------------------------------------------------------------------------
# page_layout module
# ---------------------------------------------------------------------------

def test_count_nonempty_lines_skips_blank_and_page_numbers():
    page = "abc\n\n   \ndef\n449\n1-70\n-\nghi"
    # non-empty, non-page-number lines: abc, def, ghi
    assert count_nonempty_lines(page) == 3
    # when page numbers are kept, 449 / 1-70 / - are counted too
    assert count_nonempty_lines(page, ignore_page_number_lines=False) == 6


def test_count_nonempty_lines_tibetan_digit_pagenum():
    # Tibetan digits ༠–༩ on their own line are page numbers
    assert count_nonempty_lines("ཀཁ\n༡༢༣\nགང") == 2


def test_segment_pages_form_feed_offsets():
    text = "AAA\fBBB\fCCC"
    pages = segment_pages(text)
    assert len(pages) == 3
    assert (pages[0].start, pages[0].end) == (0, 3)
    assert pages[1].start == 4 and pages[1].end == 7
    assert pages[2].start == 8 and pages[2].end == 11
    assert text[pages[2].content_start:].startswith("CCC")


def test_segment_pages_empty_middle_page():
    text = "AAA\f\fCCC"
    pages = segment_pages(text)
    assert len(pages) == 3
    assert pages[1].is_empty
    assert pages[1].nb_lines == 0


def test_segment_pages_single_stream():
    # No form feed and no blank lines -> a single page.
    assert len(segment_pages("line one\nline two\nline three")) == 1


def test_segment_pages_blank_line_fallback():
    text = "a\nb\n\nc\nd"
    pages = segment_pages(text)  # no \f -> blank-line split
    assert len(pages) == 2
    assert text[pages[1].content_start:].startswith("c")


# ---------------------------------------------------------------------------
# Rule I — empty page
# ---------------------------------------------------------------------------

def test_rule_i_empty_page_fires_at_next_page():
    text = "aaa\nbbb\nccc\f\fZZZ next text"
    expected = text.index("ZZZ")
    res = detect_breakpoints(text, profile="balanced", rule_i_empty_page=True,
                             detailed=True)
    assert expected in res["breakpoints"]
    assert "I:empty_page" in {d["rule"] for d in res["details"]}


def test_rule_i_off_by_default():
    text = "aaa\nbbb\nccc\f\fZZZ next text"
    assert detect_breakpoints(text, profile="balanced")["breakpoints"] == []


# ---------------------------------------------------------------------------
# Rule J — dense, sparse, sparse
# ---------------------------------------------------------------------------

def _make_page(prefix, n_lines):
    return "\n".join(f"{prefix} line {i}" for i in range(n_lines))


def test_rule_j_sparse_tail():
    page_a = _make_page("AAA", 8)   # dense  (8 > 4)
    page_b = _make_page("BBB", 2)   # sparse (2 < 4)
    page_c = "ZZZ only"             # sparse (1 < 4)
    text = "\f".join([page_a, page_b, page_c])
    expected = text.index("ZZZ")

    res = detect_breakpoints(text, profile="balanced", rule_j_sparse_tail=True,
                             detailed=True)
    assert expected in res["breakpoints"]
    assert "J:sparse_tail" in {d["rule"] for d in res["details"]}


def test_rule_j_not_fired_when_tail_is_dense():
    page_a = _make_page("AAA", 8)
    page_b = _make_page("BBB", 2)
    page_c = _make_page("CCC", 7)   # dense -> not the J pattern
    text = "\f".join([page_a, page_b, page_c])
    assert detect_breakpoints(text, profile="balanced",
                              rule_j_sparse_tail=True)["breakpoints"] == []


# ---------------------------------------------------------------------------
# Rule K — sparse island (ambiguous)
# ---------------------------------------------------------------------------

def test_rule_k_emits_two_candidates():
    page_a = _make_page("AAA", 8)   # dense
    # sparse (2 < 4) but long enough that its two boundaries do not merge
    page_b = "BSTART " + "x" * 40 + "\nsecond sparse line of page B"
    page_c = _make_page("CSTART", 7)  # dense
    text = "\f".join([page_a, page_b, page_c])
    pos_b = text.index("BSTART")
    pos_c = text.index("CSTART")

    res = detect_breakpoints(text, profile="balanced", rule_k_sparse_island=True,
                             detailed=True)
    assert pos_b in res["breakpoints"]
    assert pos_c in res["breakpoints"]
    labels = [d["rule"] for d in res["details"]]
    assert labels.count("K:sparse_island") == 2


def test_rule_k_can_be_disabled_independently():
    page_a = _make_page("AAA", 8)
    page_b = "BSTART only line"
    page_c = _make_page("CSTART", 7)
    text = "\f".join([page_a, page_b, page_c])
    # K off (default) -> no layout boundaries even with I/J enabled
    res = detect_breakpoints(text, profile="balanced",
                             rule_i_empty_page=True, rule_j_sparse_tail=True)
    assert res["breakpoints"] == []


# ---------------------------------------------------------------------------
# Safety: layout rules are inert without page structure / when disabled
# ---------------------------------------------------------------------------

def test_no_layout_candidates_on_single_stream():
    text = _make_page("AAA", 8)  # single page, single newlines only
    res = detect_breakpoints(text, profile="balanced",
                             rule_i_empty_page=True, rule_j_sparse_tail=True,
                             rule_k_sparse_island=True)
    assert res["breakpoints"] == []


def test_layout_off_matches_baseline():
    page_a = _make_page("AAA", 8)
    page_b = _make_page("BBB", 2)
    page_c = "ZZZ only"
    text = "\f".join([page_a, page_b, page_c])
    baseline = detect_breakpoints(text, profile="balanced")["breakpoints"]
    with_flags_off = detect_breakpoints(text, profile="balanced")["breakpoints"]
    assert baseline == with_flags_off == []


def test_explicit_flag_overrides_profile_preset():
    # A profile keeps layout rules off, but an explicit flag must still enable.
    det = RuleBasedDetector(profile="precision", rule_j_sparse_tail=True)
    assert det.use_rule_j is True


# ---------------------------------------------------------------------------
# Fixture regression (blank-line delimited pages)
# ---------------------------------------------------------------------------

def test_fixture_rule_j_boundary():
    text = FIXTURE.read_text(encoding="utf-8")
    expected = text.index("ཅཀཅཀཅཀ")  # start of the lone sparse page (page C)
    res = detect_breakpoints(text, profile="balanced", rule_j_sparse_tail=True,
                             detailed=True)
    assert expected in res["breakpoints"]
    assert "J:sparse_tail" in {d["rule"] for d in res["details"]}
