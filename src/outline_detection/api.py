"""
api.py — High-level programmatic interface.

    >>> from outline_detection import detect_breakpoints
    >>> detect_breakpoints("རྫོགས་སོ།། ༄༅། །next text")
    {'breakpoints': [...]}
"""

from .detector import RuleBasedDetector


def detect_breakpoints(text, profile="balanced", *, min_confidence=None,
                       merge_window=None, detailed=False):
    """Detect text boundaries and return their start indices.

    Args:
        text: Raw (unannotated) Tibetan text.
        profile: One of "recall", "balanced" (default), or "precision".
        min_confidence: Optional override for the profile's confidence threshold.
        merge_window: Optional override for the profile's merge window (chars).
        detailed: If True, also include per-boundary confidence and firing rule.

    Returns:
        dict with key ``"breakpoints"`` -> list of integer start indices
        (character offsets), sorted ascending. When ``detailed`` is True, an
        additional ``"details"`` key holds a list of
        ``{"index", "confidence", "rule"}`` dicts.
    """
    detector = RuleBasedDetector(profile=profile)
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
