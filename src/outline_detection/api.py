"""
api.py — High-level programmatic interface.

    >>> from outline_detection import detect_breakpoints
    >>> detect_breakpoints("རྫོགས་སོ།། ༄༅། །next text")
    {'breakpoints': [...]}
"""

from .detector import RuleBasedDetector


def detect_breakpoints(text, profile="balanced", *, min_confidence=None,
                       merge_window=None, detailed=False,
                       rule_i_empty_page=False, rule_j_sparse_tail=False,
                       rule_k_sparse_island=False, rule_l_modern=False,
                       line_threshold=4, page_delimiter="\f",
                       min_blank_lines=2, ignore_page_number_lines=True):
    """Detect text boundaries and return their start indices.

    Args:
        text: Raw (unannotated) Tibetan text.
        profile: One of "recall", "balanced" (default), or "precision".
        min_confidence: Optional override for the profile's confidence threshold.
        merge_window: Optional override for the profile's merge window (chars).
        detailed: If True, also include per-boundary confidence and firing rule.
        rule_i_empty_page: Enable Rule I (empty page marks a break).
        rule_j_sparse_tail: Enable Rule J (dense page then two sparse pages).
        rule_k_sparse_island: Enable Rule K (sparse page between dense pages;
            ambiguous, emits two low-confidence candidates).
        rule_l_modern: Enable Rule L (modern-publication layout; reserved stub).
        line_threshold: ``T`` for the line-density rules (default 4).
        page_delimiter: ``"\\f"`` (default, form feed with blank-line fallback),
            ``"blank"``, or a custom regex string.
        min_blank_lines: Newlines needed to split a page in blank-line mode.
        ignore_page_number_lines: Skip lone folio/page-number lines when
            counting page lines.

    Returns:
        dict with key ``"breakpoints"`` -> list of integer start indices
        (character offsets), sorted ascending. When ``detailed`` is True, an
        additional ``"details"`` key holds a list of
        ``{"index", "confidence", "rule"}`` dicts.
    """
    detector = RuleBasedDetector(
        profile=profile,
        rule_i_empty_page=rule_i_empty_page,
        rule_j_sparse_tail=rule_j_sparse_tail,
        rule_k_sparse_island=rule_k_sparse_island,
        rule_l_modern=rule_l_modern,
        line_threshold=line_threshold,
        page_delimiter=page_delimiter,
        min_blank_lines=min_blank_lines,
        ignore_page_number_lines=ignore_page_number_lines,
    )
    if min_confidence is not None:
        detector.min_confidence = min_confidence
    if merge_window is not None:
        detector.merge_window = merge_window

    predictions = detector.predict(text)
    result = {"breakpoints": [pos for pos, _conf, _rule in predictions]}
    if detailed:
        result["details"] = [
            {"index": pos, "confidence": conf, "rule": rule}
            for pos, conf, rule in predictions
        ]
    return result
