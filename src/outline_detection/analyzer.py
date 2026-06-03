"""
analyzer.py — Boundary Pattern Analyzer for Tibetan Text.

Analyzes annotated strings (with <b> markers) to discover recurring patterns
at text boundaries. Produces a frequency report to guide rule/regex construction.

    from outline_detection.analyzer import BoundaryAnalyzer
    analyzer = BoundaryAnalyzer(annotated_strings)
    report = analyzer.run()
"""

import json
import re
import os
from collections import Counter, defaultdict
from pathlib import Path

from .paths import analysis_dir, ensure_report_dir

try:
    import regex
    HAS_REGEX = True
except ImportError:
    HAS_REGEX = False

# ===========================================================================
# Token type definitions
# ===========================================================================

# Tibetan Unicode block key codepoints
YIG_MGO_CHARS = set("༄༅")
SHAD_CHAR = "།"
NYIS_SHAD = "༎"
TSHEG_CHAR = "་"
GTER_TSHEG = "༔"
DOUBLE_SHAD_SEQ = "། །"

# Known closing formulas (expandable — the analyzer will also discover new ones)
KNOWN_CLOSING_FORMULAS = [
    "དགེའོ",
    "དགེ་བ",
    "རྫོགས་སོ",
    "མངྒ་ལཾ",
    "སརྦ་མངྒ་ལཾ",
    "ཤུ་བྷམ",
    "ཨི་ཐི",
    "བཀྲ་ཤིས",
    "སམཱཔྟ",
    "ལེགས་སོ",
    "གསང་རྒྱ",
]

# Known opening formulas
KNOWN_OPENING_FORMULAS = [
    "༄༅",
    "༄༅།",
    "ན་མོ",
    "ༀ།",
    "ཨོཾ",
    "རྒྱ་གར་སྐད་དུ",
    "བཞུགས་སོ",
    "བོད་སྐད་དུ",
    "ཕྱག་འཚལ་ལོ",
]


# ===========================================================================
# Tokenizer: classify each character/segment into structural types
# ===========================================================================

def classify_char(ch):
    """Classify a single character into a structural type."""
    cp = ord(ch)
    if ch in YIG_MGO_CHARS:
        return "YIG_MGO"
    if ch == SHAD_CHAR:
        return "SHAD"
    if ch == NYIS_SHAD:
        return "NYIS_SHAD"
    if ch == TSHEG_CHAR:
        return "TSHEG"
    if ch == GTER_TSHEG:
        return "GTER_TSHEG"
    if ch == "\n":
        return "NEWLINE"
    if ch in " \t\r":
        return "SPACE"
    if ch.isdigit():
        return "DIGIT"
    # Tibetan block: U+0F00 - U+0FFF
    if 0x0F00 <= cp <= 0x0FFF:
        return "TIBETAN"
    # Latin / other
    if ch.isalpha():
        return "LATIN"
    return "OTHER"


def tokenize_to_types(text):
    """
    Convert a string into a list of (type, literal) tuples,
    merging consecutive characters of the same type into single tokens.
    """
    if not text:
        return []

    tokens = []
    current_type = classify_char(text[0])
    current_lit = text[0]

    for ch in text[1:]:
        ch_type = classify_char(ch)
        if ch_type == current_type:
            current_lit += ch
        else:
            tokens.append((current_type, current_lit))
            current_type = ch_type
            current_lit = ch
    tokens.append((current_type, current_lit))
    return tokens


def type_signature(tokens, max_len=None):
    """Create a type-only signature string from tokens."""
    types = [t[0] for t in tokens]
    if max_len:
        types = types[:max_len]
    return " → ".join(types)


# ===========================================================================
# Core Analyzer
# ===========================================================================

class BoundaryAnalyzer:
    """
    Extracts and analyzes context windows around <b> boundary markers
    in annotated Tibetan text strings.
    """

    BOUNDARY_TAGS = ("<b>", "</b>")

    def __init__(self, annotated_strings, window_size=200):
        """
        Args:
            annotated_strings: list of strings, each containing one or more <b> markers
            window_size: number of characters to extract on each side of <b>
        """
        self.annotated_strings = annotated_strings
        self.window_size = window_size
        self.left_windows = []
        self.right_windows = []
        self.full_windows = []  # (left, right) pairs
        self._extract_windows()

    def _extract_windows(self):
        """Find every boundary tag and extract left/right context windows."""
        for s in self.annotated_strings:
            start = 0
            while True:
                idx, tag = self._find_next_tag(s, start)
                if idx == -1:
                    break

                tag_len = len(tag)
                left = s[max(0, idx - self.window_size):idx]
                right = s[idx + tag_len:idx + tag_len + self.window_size]

                self.left_windows.append(left)
                self.right_windows.append(right)
                self.full_windows.append((left, right))
                start = idx + tag_len

    def _find_next_tag(self, text, start):
        """Return (index, tag) for the earliest boundary marker at or after start."""
        best_idx = -1
        best_tag = None
        for tag in self.BOUNDARY_TAGS:
            idx = text.find(tag, start)
            if idx != -1 and (best_idx == -1 or idx < best_idx):
                best_idx = idx
                best_tag = tag
        return best_idx, best_tag

    # -------------------------------------------------------------------
    # Analysis methods
    # -------------------------------------------------------------------

    def count_punctuation(self):
        """Count Tibetan punctuation occurrences in left and right windows."""
        punct_map = {
            "༄": "YIG_MGO_MDUN_MA (U+0F04)",
            "༅": "YIG_MGO_SGAB_MA (U+0F05)",
            "།": "SHAD (U+0F0D)",
            "༎": "NYIS_SHAD (U+0F0E)",
            "༏": "TSHEG_SHAD (U+0F0F)",
            "༐": "NYIS_TSHEG_SHAD (U+0F10)",
            "༑": "RIN_CHEN_SPUNGS_SHAD (U+0F11)",
            "་": "TSHEG (U+0F0B)",
            "༔": "GTER_TSHEG (U+0F14)",
            "༈": "SBRUL_SHAD (U+0F08)",
            "།།": "DOUBLE_SHAD (sequence)",
            "། །": "SPACED_DOUBLE_SHAD (sequence)",
        }

        left_counts = Counter()
        right_counts = Counter()

        for lw in self.left_windows:
            for char, name in punct_map.items():
                left_counts[name] += lw.count(char)

        for rw in self.right_windows:
            for char, name in punct_map.items():
                right_counts[name] += rw.count(char)

        return {
            "left_window": left_counts.most_common(),
            "right_window": right_counts.most_common(),
        }

    def find_closing_formulas(self, top_n=40):
        """
        Extract Tibetan text segments near the end of left windows
        and find the most common phrases (potential closing formulas).
        """
        # Take the last 80 chars of each left window and extract Tibetan runs
        closing_phrases = Counter()
        closing_tail_chars = 80

        pattern = re.compile(r'[\u0F00-\u0FFF]+')

        for lw in self.left_windows:
            tail = lw[-closing_tail_chars:]
            tibetan_runs = pattern.findall(tail)
            for run in tibetan_runs:
                # Clean tsheg at edges
                clean = run.strip("་")
                if len(clean) >= 3:  # skip very short fragments
                    closing_phrases[clean] += 1

        # Also check for known formulas
        known_hits = Counter()
        for lw in self.left_windows:
            for formula in KNOWN_CLOSING_FORMULAS:
                if formula in lw:
                    known_hits[formula] += 1

        return {
            "discovered_phrases": closing_phrases.most_common(top_n),
            "known_formula_hits": known_hits.most_common(),
            "total_boundaries": len(self.left_windows),
        }

    def find_opening_formulas(self, top_n=40):
        """
        Extract Tibetan text segments near the start of right windows
        and find the most common phrases (potential opening formulas).
        """
        opening_phrases = Counter()
        opening_head_chars = 100

        pattern = re.compile(r'[\u0F00-\u0FFF]+')

        for rw in self.right_windows:
            head = rw[:opening_head_chars]
            tibetan_runs = pattern.findall(head)
            for run in tibetan_runs:
                clean = run.strip("་")
                if len(clean) >= 3:
                    opening_phrases[clean] += 1

        known_hits = Counter()
        for rw in self.right_windows:
            for formula in KNOWN_OPENING_FORMULAS:
                if formula in rw:
                    known_hits[formula] += 1

        return {
            "discovered_phrases": opening_phrases.most_common(top_n),
            "known_formula_hits": known_hits.most_common(),
            "total_boundaries": len(self.right_windows),
        }

    def analyze_digit_patterns(self):
        """Analyze how page numbers / digit clusters appear near boundaries."""
        digit_pattern = re.compile(r'\d+')
        digit_with_context = re.compile(r'(.{0,10})(\d+)(.{0,10})')

        left_digit_freq = Counter()
        right_digit_freq = Counter()
        left_has_digits = 0
        right_has_digits = 0

        digit_contexts_left = Counter()
        digit_contexts_right = Counter()

        for lw in self.left_windows:
            digits = digit_pattern.findall(lw)
            if digits:
                left_has_digits += 1
                for d in digits:
                    left_digit_freq[len(d)] += 1  # count by digit length
            # Capture digit + surrounding context
            for m in digit_with_context.finditer(lw):
                before_types = "".join(
                    "N" if c == "\n" else "S" if c == " " else "T" if ord(c) >= 0x0F00 else "?"
                    for c in m.group(1)
                )
                after_types = "".join(
                    "N" if c == "\n" else "S" if c == " " else "T" if ord(c) >= 0x0F00 else "?"
                    for c in m.group(3)
                )
                digit_contexts_left[f"...{before_types}[DIGITS]{after_types}..."] += 1

        for rw in self.right_windows:
            digits = digit_pattern.findall(rw)
            if digits:
                right_has_digits += 1
                for d in digits:
                    right_digit_freq[len(d)] += 1

        total = len(self.left_windows)
        return {
            "left_windows_with_digits": f"{left_has_digits}/{total} ({100*left_has_digits/max(total,1):.1f}%)",
            "right_windows_with_digits": f"{right_has_digits}/{total} ({100*right_has_digits/max(total,1):.1f}%)",
            "left_digit_lengths": left_digit_freq.most_common(),
            "right_digit_lengths": right_digit_freq.most_common(),
            "left_digit_context_patterns": digit_contexts_left.most_common(20),
        }

    def analyze_newline_patterns(self):
        """Analyze newline distribution and clustering near boundaries."""
        # Count consecutive newline patterns
        newline_pattern = re.compile(r'\n+')

        left_newline_runs = Counter()
        right_newline_runs = Counter()
        left_has_newlines = 0
        right_has_newlines = 0

        for lw in self.left_windows:
            runs = newline_pattern.findall(lw)
            if runs:
                left_has_newlines += 1
                for r in runs:
                    left_newline_runs[len(r)] += 1

        for rw in self.right_windows:
            runs = newline_pattern.findall(rw)
            if runs:
                right_has_newlines += 1
                for r in runs:
                    right_newline_runs[len(r)] += 1

        total = len(self.left_windows)
        return {
            "left_windows_with_newlines": f"{left_has_newlines}/{total} ({100*left_has_newlines/max(total,1):.1f}%)",
            "right_windows_with_newlines": f"{right_has_newlines}/{total} ({100*right_has_newlines/max(total,1):.1f}%)",
            "left_consecutive_newlines": left_newline_runs.most_common(),
            "right_consecutive_newlines": right_newline_runs.most_common(),
        }

    def analyze_type_sequences(self, seq_window=15):
        """
        Tokenize windows into structural types and find the most common
        type-level sequences immediately adjacent to the boundary.
        """
        left_type_seqs = Counter()
        right_type_seqs = Counter()

        for lw in self.left_windows:
            tokens = tokenize_to_types(lw)
            # Take last N tokens
            tail_tokens = tokens[-seq_window:]
            sig = type_signature(tail_tokens)
            left_type_seqs[sig] += 1

        for rw in self.right_windows:
            tokens = tokenize_to_types(rw)
            head_tokens = tokens[:seq_window]
            sig = type_signature(head_tokens)
            right_type_seqs[sig] += 1

        return {
            "left_type_sequences_top30": left_type_seqs.most_common(30),
            "right_type_sequences_top30": right_type_seqs.most_common(30),
        }

    def analyze_ngrams(self, n_range=(2, 5), top_n=25):
        """
        Extract character-type n-grams from the immediate boundary zone.
        Uses a tighter window (last/first 40 chars) for precision.
        """
        ngram_window = 40
        results = {}

        for n in range(n_range[0], n_range[1] + 1):
            left_ngrams = Counter()
            right_ngrams = Counter()

            for lw in self.left_windows:
                tail = lw[-ngram_window:]
                tokens = tokenize_to_types(tail)
                type_list = [t[0] for t in tokens]
                for i in range(len(type_list) - n + 1):
                    gram = " ".join(type_list[i:i+n])
                    left_ngrams[gram] += 1

            for rw in self.right_windows:
                head = rw[:ngram_window]
                tokens = tokenize_to_types(head)
                type_list = [t[0] for t in tokens]
                for i in range(len(type_list) - n + 1):
                    gram = " ".join(type_list[i:i+n])
                    right_ngrams[gram] += 1

            results[f"{n}-gram_left"] = left_ngrams.most_common(top_n)
            results[f"{n}-gram_right"] = right_ngrams.most_common(top_n)

        return results

    def analyze_immediate_boundary(self):
        """
        Look at the very last and very first characters at the boundary
        (within 5-20 chars) for the tightest patterns.
        """
        last_chars = Counter()    # last N chars before <b>
        first_chars = Counter()   # first N chars after <b>
        last_types = Counter()
        first_types = Counter()

        for tight_n in [5, 10, 20]:
            key_l = f"last_{tight_n}_chars"
            key_r = f"first_{tight_n}_chars"
            last_chars[key_l] = Counter()
            first_chars[key_r] = Counter()

            for lw in self.left_windows:
                snippet = repr(lw[-tight_n:]) if len(lw) >= tight_n else repr(lw)
                last_chars[key_l][snippet] += 1

            for rw in self.right_windows:
                snippet = repr(rw[:tight_n]) if len(rw) >= tight_n else repr(rw)
                first_chars[key_r][snippet] += 1

        # Also: what is the single last/first character type?
        for lw in self.left_windows:
            if lw:
                last_types[classify_char(lw[-1])] += 1
        for rw in self.right_windows:
            if rw:
                first_types[classify_char(rw[0])] += 1

        return {
            "last_char_type": last_types.most_common(),
            "first_char_type": first_types.most_common(),
            "last_10_chars_top20": last_chars.get("last_10_chars", Counter()).most_common(20),
            "first_10_chars_top20": first_chars.get("first_10_chars", Counter()).most_common(20),
        }

    def yig_mgo_analysis(self):
        """Specifically measure yig mgo presence and position."""
        right_has_yig_mgo = 0
        yig_mgo_position = Counter()  # position of first ༄ in right window

        for rw in self.right_windows:
            idx = -1
            for i, ch in enumerate(rw):
                if ch in YIG_MGO_CHARS:
                    idx = i
                    break
            if idx >= 0:
                right_has_yig_mgo += 1
                yig_mgo_position[idx] += 1

        # Also check left window (should be rare)
        left_has_yig_mgo = 0
        for lw in self.left_windows:
            if any(ch in YIG_MGO_CHARS for ch in lw):
                left_has_yig_mgo += 1

        total = len(self.right_windows)
        return {
            "right_windows_with_yig_mgo": f"{right_has_yig_mgo}/{total} ({100*right_has_yig_mgo/max(total,1):.1f}%)",
            "left_windows_with_yig_mgo": f"{left_has_yig_mgo}/{total} ({100*left_has_yig_mgo/max(total,1):.1f}%)",
            "yig_mgo_position_in_right_window": yig_mgo_position.most_common(20),
        }

    def find_unique_literal_patterns(self, min_length=5, top_n=50):
        """
        Find literal substrings that appear in many boundary windows
        but are specific enough to be useful (min_length filter).
        Uses a sliding window over the tight boundary zone.
        """
        tight_window = 60
        literal_left = Counter()
        literal_right = Counter()

        for lw in self.left_windows:
            tail = lw[-tight_window:]
            for length in range(min_length, min(30, len(tail) + 1)):
                for start in range(len(tail) - length + 1):
                    substr = tail[start:start + length]
                    # Only count substrings with at least one Tibetan char or punctuation
                    if any(0x0F00 <= ord(c) <= 0x0FFF for c in substr):
                        literal_left[substr] += 1

        for rw in self.right_windows:
            head = rw[:tight_window]
            for length in range(min_length, min(30, len(head) + 1)):
                for start in range(len(head) - length + 1):
                    substr = head[start:start + length]
                    if any(0x0F00 <= ord(c) <= 0x0FFF for c in substr):
                        literal_right[substr] += 1

        # Filter: keep only substrings appearing in at least 2% of boundaries
        min_count = max(2, len(self.left_windows) * 0.02)
        filtered_left = [(s, c) for s, c in literal_left.most_common(top_n * 3)
                         if c >= min_count][:top_n]
        filtered_right = [(s, c) for s, c in literal_right.most_common(top_n * 3)
                          if c >= min_count][:top_n]

        return {
            "left_literal_patterns": filtered_left,
            "right_literal_patterns": filtered_right,
            "min_count_threshold": min_count,
        }

    # -------------------------------------------------------------------
    # Full report
    # -------------------------------------------------------------------

    def run(self):
        """Run all analyses and return a structured report dict."""
        total = len(self.left_windows)
        print(f"Found {total} boundary markers across {len(self.annotated_strings)} input strings.")
        print(f"Window size: {self.window_size} chars on each side.\n")

        if total == 0:
            return {"error": "No <b> or </b> markers found in input."}

        report = {}

        print("[1/8] Analyzing punctuation distribution...")
        report["punctuation"] = self.count_punctuation()

        print("[2/8] Discovering closing formulas (left of <b>)...")
        report["closing_formulas"] = self.find_closing_formulas()

        print("[3/8] Discovering opening formulas (right of <b>)...")
        report["opening_formulas"] = self.find_opening_formulas()

        print("[4/8] Analyzing digit/page-number patterns...")
        report["digit_patterns"] = self.analyze_digit_patterns()

        print("[5/8] Analyzing newline patterns...")
        report["newline_patterns"] = self.analyze_newline_patterns()

        print("[6/8] Mining type-level sequences...")
        report["type_sequences"] = self.analyze_type_sequences()

        print("[7/8] Analyzing yig mgo (head mark) distribution...")
        report["yig_mgo"] = self.yig_mgo_analysis()

        print("[8/8] Extracting character-type n-grams...")
        report["ngrams"] = self.analyze_ngrams()

        return report

    def format_report(self, report):
        """Format the report dict as a Markdown document."""
        if report.get("error"):
            return f"# Tibetan Text Boundary Pattern Analysis\n\n**Error:** {report['error']}\n"

        total = len(self.left_windows)
        lines = [
            "# Tibetan Text Boundary Pattern Analysis",
            "",
            "| Metric | Value |",
            "| --- | --- |",
            f"| Total boundaries analyzed | {total:,} |",
            f"| Input strings | {len(self.annotated_strings):,} |",
            f"| Context window size | {self.window_size} chars (each side) |",
            "",
        ]

        # --- Yig Mgo ---
        lines.append("## Yig mgo (༄༅) — primary boundary signal")
        lines.append("")
        ym = report.get("yig_mgo", {})
        lines.append(f"- **Present in right windows:** {ym.get('right_windows_with_yig_mgo', 'N/A')}")
        lines.append(f"- **Present in left windows:** {ym.get('left_windows_with_yig_mgo', 'N/A')}")
        lines.append("")
        if ym.get("yig_mgo_position_in_right_window"):
            lines.append("### Position of first ༄ in right window")
            lines.append("")
            lines.append("| Char index | Count |")
            lines.append("| --- | ---: |")
            for pos, count in ym["yig_mgo_position_in_right_window"][:10]:
                lines.append(f"| {pos} | {count:,} |")
            lines.append("")

        # --- Punctuation ---
        lines.append("## Punctuation distribution")
        lines.append("")
        punct = report.get("punctuation", {})
        for side, label in (("left_window", "Left (end of text A)"), ("right_window", "Right (start of text B)")):
            lines.append(f"### {label}")
            lines.append("")
            lines.append("| Punctuation | Count |")
            lines.append("| --- | ---: |")
            for name, count in punct.get(side, []):
                lines.append(f"| {name} | {count:,} |")
            lines.append("")

        # --- Closing / Opening formulas ---
        for section_key, title, phrase_label in (
            ("closing_formulas", "Closing formulas", "last 80 chars before boundary"),
            ("opening_formulas", "Opening formulas", "first 100 chars after boundary"),
        ):
            lines.append(f"## {title}")
            lines.append("")
            lines.append(f"*{phrase_label}*")
            lines.append("")
            data = report.get(section_key, {})
            n = max(data.get("total_boundaries", 1), 1)
            lines.append("### Known formula hits")
            lines.append("")
            lines.append("| Formula | Count | % |")
            lines.append("| --- | ---: | ---: |")
            for formula, count in data.get("known_formula_hits", []):
                pct = 100 * count / n
                lines.append(f"| {formula} | {count:,} | {pct:.1f}% |")
            lines.append("")
            lines.append("### Top discovered Tibetan phrases")
            lines.append("")
            lines.append("| Phrase | Count | % |")
            lines.append("| --- | ---: | ---: |")
            for phrase, count in data.get("discovered_phrases", [])[:30]:
                pct = 100 * count / n
                lines.append(f"| {phrase} | {count:,} | {pct:.1f}% |")
            lines.append("")

        # --- Digits ---
        lines.append("## Digit / page-number patterns")
        lines.append("")
        dp = report.get("digit_patterns", {})
        lines.append(f"- **Left windows with digits:** {dp.get('left_windows_with_digits', 'N/A')}")
        lines.append(f"- **Right windows with digits:** {dp.get('right_windows_with_digits', 'N/A')}")
        lines.append("")
        if dp.get("left_digit_lengths"):
            lines.append("### Digit-cluster lengths (left)")
            lines.append("")
            lines.append("| Length | Count |")
            lines.append("| --- | ---: |")
            for length, count in dp["left_digit_lengths"]:
                lines.append(f"| {length}-digit | {count:,} |")
            lines.append("")
        if dp.get("left_digit_context_patterns"):
            lines.append("### Digit context patterns (left)")
            lines.append("")
            lines.append("*T = Tibetan, N = newline, S = space*")
            lines.append("")
            lines.append("| Pattern | Count |")
            lines.append("| --- | ---: |")
            for pat, count in dp["left_digit_context_patterns"][:15]:
                lines.append(f"| `{pat}` | {count:,} |")
            lines.append("")

        # --- Newlines ---
        lines.append("## Newline patterns")
        lines.append("")
        nl = report.get("newline_patterns", {})
        lines.append(f"- **Left windows with newlines:** {nl.get('left_windows_with_newlines', 'N/A')}")
        lines.append(f"- **Right windows with newlines:** {nl.get('right_windows_with_newlines', 'N/A')}")
        lines.append("")
        for side, label in (("left_consecutive_newlines", "Left"), ("right_consecutive_newlines", "Right")):
            if nl.get(side):
                lines.append(f"### Consecutive newline runs ({label.lower()})")
                lines.append("")
                lines.append("| Run length | Count |")
                lines.append("| --- | ---: |")
                for run_len, count in nl[side]:
                    lines.append(f"| {run_len} × `\\n` | {count:,} |")
                lines.append("")

        # --- Type Sequences ---
        lines.append("## Type-level sequences (structural skeletons)")
        lines.append("")
        ts = report.get("type_sequences", {})
        for side_key, heading in (
            ("left_type_sequences_top30", "Top left sequences (~15 tokens before boundary)"),
            ("right_type_sequences_top30", "Top right sequences (~15 tokens after boundary)"),
        ):
            lines.append(f"### {heading}")
            lines.append("")
            lines.append("| Count | Sequence |")
            lines.append("| ---: | --- |")
            for seq, count in ts.get(side_key, [])[:20]:
                lines.append(f"| {count:,} | `{seq}` |")
            lines.append("")

        # --- N-grams ---
        lines.append("## Character-type n-grams (tight 40-char boundary zone)")
        lines.append("")
        ng = report.get("ngrams", {})
        for key in sorted(ng.keys()):
            lines.append(f"### `{key}`")
            lines.append("")
            lines.append("| Count | N-gram |")
            lines.append("| ---: | --- |")
            for gram, count in ng[key][:15]:
                lines.append(f"| {count:,} | `{gram}` |")
            lines.append("")

        return "\n".join(lines)


# ===========================================================================
# Input loading
# ===========================================================================

def load_input(filepath):
    """
    Load annotated strings from file.
    Supports:
      - .json: expects a list of strings
      - .jsonl: one JSON string per line
      - .txt: each line is one annotated string (blank lines separate multi-line entries)
      - any other: read entire file as one string and split on double newlines
    """
    filepath = Path(filepath)
    text = filepath.read_text(encoding="utf-8")

    if filepath.suffix == ".json":
        data = json.loads(text)
        if isinstance(data, list):
            return [str(item) for item in data]
        elif isinstance(data, dict) and "samples" in data:
            return [str(item) for item in data["samples"]]
        else:
            return [str(data)]

    elif filepath.suffix == ".jsonl":
        strings = []
        for line in text.strip().split("\n"):
            line = line.strip()
            if line:
                strings.append(json.loads(line))
        return strings

    else:
        # For .txt and other: if the file contains <b>, treat the whole thing
        # or split on some delimiter
        if "<b>" in text or "</b>" in text:
            # If there are clear record separators (e.g., "---" or "===")
            for sep in ["---\n", "===\n", "***\n"]:
                if sep in text:
                    return [chunk.strip() for chunk in text.split(sep) if chunk.strip()]
            # Otherwise treat each line as a separate annotated string,
            # but merge consecutive non-empty lines into one record
            records = []
            current = []
            for line in text.split("\n"):
                if line.strip() == "" and current:
                    merged = "\n".join(current)
                    if "<b>" in merged or "</b>" in merged:
                        records.append(merged)
                    current = []
                else:
                    current.append(line)
            if current:
                merged = "\n".join(current)
                if "<b>" in merged or "</b>" in merged:
                    records.append(merged)
            # If no splitting worked, treat the whole file as one string
            if not records:
                records = [text]
            return records
        else:
            print("WARNING: No <b> markers found in input file!")
            return [text]


# ===========================================================================
# Main
# ===========================================================================

def _default_analysis_path(input_file):
    stem = Path(input_file).stem
    if stem.startswith("breakpoints_context_snippets"):
        suffix = stem.replace("breakpoints_context_snippets", "").strip("_") or "report"
        name = f"boundary_report_{suffix}.md" if suffix != "report" else "boundary_report.md"
    else:
        name = f"boundary_report_{stem}.md"
    return analysis_dir() / name


def run_analysis(input_file, window=200, output=None):
    """Analyze an annotated file and write markdown + JSON reports."""
    if not os.path.exists(input_file):
        raise FileNotFoundError(input_file)

    print(f"Loading input from: {input_file}")
    strings = load_input(input_file)
    print(f"Loaded {len(strings)} annotated string(s).\n")

    analyzer = BoundaryAnalyzer(strings, window_size=window)
    report = analyzer.run()

    formatted = analyzer.format_report(report)

    # Save to file first (stdout may fail on Windows with Tibetan text)
    output_path = Path(output) if output else _default_analysis_path(input_file)
    ensure_report_dir(output_path)
    output_path.write_text(formatted, encoding="utf-8")
    print(f"\nReport saved to: {output_path}")

    try:
        print("\n" + formatted)
    except UnicodeEncodeError:
        print("\n(Report contains Tibetan text; see file output above.)")

    # Also save raw report as JSON for downstream use
    json_path = output_path.with_suffix(".json")
    ensure_report_dir(json_path)

    def make_serializable(obj):
        if isinstance(obj, Counter):
            return dict(obj)
        if isinstance(obj, dict):
            return {k: make_serializable(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [make_serializable(i) for i in obj]
        if isinstance(obj, tuple):
            return list(obj)
        return obj

    json_report = make_serializable(report)
    json_path.write_text(json.dumps(json_report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"JSON report saved to: {json_path}")
    return output_path
