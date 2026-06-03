"""
utils.py — Shared tokenizer, formula dictionaries, and evaluation tools
for Tibetan text boundary detection.
"""

import json
import re
from collections import Counter
from pathlib import Path

# ===========================================================================
# Tibetan punctuation and Unicode constants
# ===========================================================================

YIG_MGO_CHARS = {"༄", "༅"}
SHAD = "།"
NYIS_SHAD = "༎"
TSHEG = "་"
GTER_TSHEG = "༔"
SBRUL_SHAD = "༈"
RIN_CHEN = "༑"

# Junction regex primitives (boundary_report.md, 82,560 samples)
CLOSING_PUNCT = "།༎༔"
JUNCTION_BEFORE = r'(?:^|\n|[' + CLOSING_PUNCT + r']\s{0,3})'

# ===========================================================================
# Known formulas (from boundary_report.md, 82,560 samples)
# Sorted longest-first so higher-specificity matches win ties.
# ===========================================================================

# Closing formulas — appear near the END of a text (left of </b>)
CLOSING_FORMULAS = [
    "སརྦ་མངྒ་ལཾ",      # 6.0%  (before མངྒ་ལཾ to avoid substring greed)
    "མངྒ་ལཾ",           # 12.5%
    "དགེའོ",            # 10.4%
    "བཀྲ་ཤིས",          # 9.4%
    "རྫོགས་སོ",          # 6.7%
    "དགེ་བ",            # 7.5%
    "གསང་རྒྱ",          # 1.1%
    "ལེགས་སོ",          # 0.9%
    "ཨི་ཐི",            # 0.4%
    "ཤུ་བྷམ",           # 0.1%
    "སམཱཔྟ",            # rare
    "ཐཱི",              # variant of iti
    "ཨིཏི",             # variant of iti
]

# Sanskrit transliterated closings (Rule B/H — petition letters)
SANSKRIT_CLOSINGS = [
    "སརྦ་དཱ་ཀ་ལྱཱ་ཎཾ",
    "སརྦ་དཱ་ཤྲེ་ཡོ་བྷ་བ་ཏུ",
    "སརྦ་དཱ་ཤུ་བྷ",
    "སིདྡྷི",
    "ཤུ་བྷ",
    "ཤུ་བྷམ",
]

# Terma genre seal phrases (Rule B)
TERMA_CLOSINGS = [
    "ས་མ་ཡ",
    "རྒྱ་རྒྱ་རྒྱ",
    "དགེའོ༔",
]

ALL_CLOSING_FORMULAS = CLOSING_FORMULAS + SANSKRIT_CLOSINGS + TERMA_CLOSINGS

# Opening formulas — appear near the START of a text (right of <b>)
OPENING_FORMULAS = [
    "༄༅།",              # 40.0%
    "༄༅༅།",             # 3.0%
    "༄༅།།",             # 1.2%
    "༄༅༔",              # 1.4% terma opener
    "བཞུགས་སོ",          # 36.8% (opening, not closing)
    "བཞུགས༔",
    "བཞུགསཿ",
    "ཞེས་བྱ་བ་བཞུགས་སོ",
    "ན་མོ",             # 19.8%
    "ན་མོ་གུ་རུ",        # subset of ན་མོ
    "ཕྱག་འཚལ་ལོ",       # 13.3%
    "ཨོཾ",              # 7.0%
    "ༀ།",               # 0.3%
    "ཨོཾ་སྭ་སྟི",        # 0.5%
    "ཨོཾ་ཨཱཿཧཱུྃ",       # 0.5%
    "རྒྱ་གར་སྐད་དུ",     # 2.2%
    "བོད་སྐད་དུ",        # 3.0%
    "ཞང་ཞུང་སྐད་དུ",     # rare
    "བླ་མ་ལ་ཕྱག་འཚལ",   # 0.4%
]

YIG_MGO_OPENERS = [f for f in OPENING_FORMULAS if "༄༅" in f]

# Collection titles that appear as running headers (from discovered phrases)
COLLECTION_TITLES = [
    "གསུང་འབུམ།",
    "གསུང་འབུམ་ཆེན་མོ།",
    "གསུང་འབུམ",
    "གླེགས་བམ།",
    "བཀའ་འགྱུར",
    "བཀའ་འགྱུར།",
]


# ===========================================================================
# Tokenizer
# ===========================================================================

def classify_char(ch):
    """Classify a single character into a structural type tag."""
    cp = ord(ch)
    if ch in YIG_MGO_CHARS:
        return "YIG_MGO"
    if ch == SHAD:
        return "SHAD"
    if ch == NYIS_SHAD:
        return "NYIS_SHAD"
    if ch == TSHEG:
        return "TSHEG"
    if ch == GTER_TSHEG:
        return "GTER_TSHEG"
    if ch == SBRUL_SHAD:
        return "SBRUL_SHAD"
    if ch == RIN_CHEN:
        return "RIN_CHEN"
    if ch == "\n":
        return "NEWLINE"
    if ch in " \t\r":
        return "SPACE"
    if ch.isdigit():
        return "DIGIT"
    if 0x0F00 <= cp <= 0x0FFF:
        return "TIBETAN"
    if ch.isalpha():
        return "LATIN"
    return "OTHER"


def tokenize(text):
    """
    Tokenize text into a list of (type, literal, start_pos, end_pos) tuples.
    Consecutive characters of the same type are merged into one token.
    Positions are character offsets into the original string.
    """
    if not text:
        return []

    tokens = []
    current_type = classify_char(text[0])
    current_start = 0
    current_lit = text[0]

    for i, ch in enumerate(text[1:], start=1):
        ch_type = classify_char(ch)
        if ch_type == current_type:
            current_lit += ch
        else:
            tokens.append((current_type, current_lit, current_start, i))
            current_type = ch_type
            current_start = i
            current_lit = ch

    tokens.append((current_type, current_lit, current_start, len(text)))
    return tokens


# ===========================================================================
# Annotation helpers
# ===========================================================================

BOUNDARY_TAGS = ("<b>", "</b>")
BOUNDARY_TAG = "<b>"  # canonical tag written by insert_boundaries / predict output


def has_boundary_marker(text):
    """Return True if text contains any supported boundary marker."""
    return any(tag in text for tag in BOUNDARY_TAGS)


def _boundary_tag_at(text, index):
    """Return the boundary tag starting at index, or None."""
    for tag in BOUNDARY_TAGS:
        if text.startswith(tag, index):
            return tag
    return None


def strip_boundaries(annotated_text):
    """
    Remove all boundary markers from text and return (clean_text, boundary_positions).

    Supports ``<b>`` and ``</b>``. Each position is the character offset in the
    clean text where a marker was removed.
    """
    positions = []
    clean_parts = []
    clean_len = 0
    i = 0
    n = len(annotated_text)

    while i < n:
        tag = _boundary_tag_at(annotated_text, i)
        if tag:
            positions.append(clean_len)
            i += len(tag)
        else:
            clean_parts.append(annotated_text[i])
            clean_len += 1
            i += 1

    return "".join(clean_parts), positions


def insert_boundaries(text, positions, tag=BOUNDARY_TAG):
    """Insert boundary markers at the given character positions."""
    result = text
    for pos in sorted(positions, reverse=True):
        result = result[:pos] + tag + result[pos:]
    return result


def load_annotated(filepath):
    """Load annotated strings from JSON, JSONL, or TXT."""
    path = Path(filepath)
    text = path.read_text(encoding="utf-8")

    if path.suffix == ".json":
        data = json.loads(text)
        if isinstance(data, list):
            return [str(s) for s in data]
        if isinstance(data, dict) and "samples" in data:
            return [str(s) for s in data["samples"]]
        return [str(data)]

    if path.suffix == ".jsonl":
        return [json.loads(line) for line in text.strip().split("\n") if line.strip()]

    if has_boundary_marker(text):
        for sep in ["---\n", "===\n", "***\n"]:
            if sep in text:
                return [
                    chunk.strip() for chunk in text.split(sep)
                    if chunk.strip() and has_boundary_marker(chunk)
                ]
        records = []
        current = []
        for line in text.split("\n"):
            if line.strip() == "" and current:
                merged = "\n".join(current)
                if has_boundary_marker(merged):
                    records.append(merged)
                current = []
            else:
                current.append(line)
        if current:
            merged = "\n".join(current)
            if has_boundary_marker(merged):
                records.append(merged)
        return records if records else [text]

    return [text]


# ===========================================================================
# Evaluation
# ===========================================================================

def evaluate(predicted_positions, true_positions, tolerance=10):
    """
    Evaluate predicted boundary positions against ground truth.

    Args:
        predicted_positions: list of predicted char offsets
        true_positions: list of true char offsets
        tolerance: max char distance to count as a match

    Returns:
        dict with precision, recall, f1, matches, false_positives, false_negatives
    """
    pred = sorted(set(predicted_positions))
    true = sorted(set(true_positions))

    # Greedy matching: for each true position, find closest unmatched prediction
    matched_pred = set()
    matched_true = set()
    matches = []

    for t in true:
        best_dist = tolerance + 1
        best_p = None
        for p in pred:
            if p in matched_pred:
                continue
            dist = abs(p - t)
            if dist <= tolerance and dist < best_dist:
                best_dist = dist
                best_p = p
        if best_p is not None:
            matched_pred.add(best_p)
            matched_true.add(t)
            matches.append((t, best_p, best_dist))

    false_positives = [p for p in pred if p not in matched_pred]
    false_negatives = [t for t in true if t not in matched_true]

    tp = len(matches)
    fp = len(false_positives)
    fn = len(false_negatives)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "true_positives": tp,
        "false_positives_count": fp,
        "false_negatives_count": fn,
        "total_predicted": len(pred),
        "total_true": len(true),
        "matches": matches,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
    }


def format_eval(result):
    """Format evaluation result dict into a readable string."""
    lines = [
        "=" * 60,
        "EVALUATION RESULTS",
        "=" * 60,
        f"  Precision:        {result['precision']:.3f}  ({result['true_positives']}/{result['total_predicted']})",
        f"  Recall:           {result['recall']:.3f}  ({result['true_positives']}/{result['total_true']})",
        f"  F1:               {result['f1']:.3f}",
        f"  True positives:   {result['true_positives']}",
        f"  False positives:  {result['false_positives_count']}",
        f"  False negatives:  {result['false_negatives_count']}",
        "=" * 60,
    ]
    return "\n".join(lines)
