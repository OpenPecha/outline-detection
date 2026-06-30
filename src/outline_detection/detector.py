"""
detector.py — Rule-based Tibetan text boundary detector.

Pattern rules (A–H) derived from analysis of annotated boundary snippets.
This module holds only the detection logic; evaluation/reporting lives in
``evaluation.py`` and the command-line interface in ``cli.py``.
"""

import re

from .utils import (
    ALL_CLOSING_FORMULAS,
    SANSKRIT_CLOSINGS,
    OPENING_FORMULAS,
    YIG_MGO_OPENERS,
    COLLECTION_TITLES,
    JUNCTION_BEFORE,
    CLOSING_PUNCT,
    YIG_MGO_CHARS,
    SBRUL_SHAD,
)
from .page_layout import (
    FORM_FEED,
    char_offset_at_page_start,
    segment_pages,
)

PROFILE_PRESETS = {
    "recall": {
        "min_confidence": 0.30,
        "merge_window": 25,
        "rule_b": True,
        "rule_c": False,
        "rule_d": False,
        "rule_e": False,
        "rule_g_unguarded": False,
        "rule_c_bare": False,
        "rule_i_empty_page": False,
        "rule_j_sparse_tail": False,
        "rule_k_sparse_island": False,
        "rule_l_modern": False,
    },
    "balanced": {
        "min_confidence": 0.40,
        "merge_window": 20,
        "rule_b": True,
        "rule_c": False,
        "rule_d": False,
        "rule_e": False,
        "rule_g_unguarded": False,
        "rule_c_bare": False,
        "rule_i_empty_page": False,
        "rule_j_sparse_tail": False,
        "rule_k_sparse_island": False,
        "rule_l_modern": False,
    },
    "precision": {
        "min_confidence": 0.50,
        "merge_window": 15,
        "rule_b": False,
        "rule_c": False,
        "rule_d": False,
        "rule_e": False,
        "rule_g_unguarded": False,
        "rule_c_bare": False,
        "rule_i_empty_page": False,
        "rule_j_sparse_tail": False,
        "rule_k_sparse_island": False,
        "rule_l_modern": False,
    },
}

# Confidence assigned to each layout rule (Rules I–L). Page-layout signals are
# weaker than orthographic ones, so the ambiguous Rule K sits below the
# balanced/precision thresholds unless a nearby A/G/H candidate agrees.
LAYOUT_CONF_EMPTY_PAGE = 0.75
LAYOUT_CONF_SPARSE_TAIL = 0.65
LAYOUT_CONF_SPARSE_ISLAND = 0.45


def _opening_alts_without_yig_mgo():
    return "|".join(
        re.escape(f)
        for f in OPENING_FORMULAS
        if f not in YIG_MGO_OPENERS and "༄༅" not in f
    )


class RuleBasedDetector:
    """Predicts text boundary positions using Rules A–H."""

    def __init__(
        self,
        merge_window=20,
        min_confidence=0.40,
        profile=None,
        rule_b=True,
        rule_c=False,
        rule_d=False,
        rule_g_unguarded=False,
        rule_c_bare=False,
        rule_e=False,
        rule_i_empty_page=False,
        rule_j_sparse_tail=False,
        rule_k_sparse_island=False,
        rule_l_modern=False,
        line_threshold=4,
        page_delimiter=FORM_FEED,
        min_blank_lines=2,
        ignore_page_number_lines=True,
    ):
        if profile is not None:
            if profile not in PROFILE_PRESETS:
                raise ValueError(
                    f"Unknown profile {profile!r}; choose from {list(PROFILE_PRESETS)}"
                )
            opts = PROFILE_PRESETS[profile]
            merge_window = opts["merge_window"]
            min_confidence = opts["min_confidence"]
            rule_b = opts["rule_b"]
            rule_c = opts["rule_c"]
            rule_d = opts["rule_d"]
            rule_g_unguarded = opts["rule_g_unguarded"]
            rule_c_bare = opts["rule_c_bare"]
            rule_e = opts["rule_e"]
            # Layout rules (I–L) are intentionally NOT pulled from the preset:
            # they stay off in every profile, but an explicit constructor
            # argument may still enable them alongside a profile.

        self.merge_window = merge_window
        self.min_confidence = min_confidence
        self.profile = profile
        self.use_rule_b = rule_b
        self.use_rule_c = rule_c
        self.use_rule_d = rule_d
        self.rule_g_unguarded = rule_g_unguarded
        self.rule_c_bare = rule_c_bare
        self.rule_e = rule_e

        # Page-layout rules (I–L) — opt-in, see page_layout.py.
        self.use_rule_i = rule_i_empty_page
        self.use_rule_j = rule_j_sparse_tail
        self.use_rule_k = rule_k_sparse_island
        self.use_rule_l = rule_l_modern
        self.line_threshold = line_threshold
        self.page_delimiter = page_delimiter
        self.min_blank_lines = min_blank_lines
        self.ignore_page_number_lines = ignore_page_number_lines

        self._compile_patterns()

    def _compile_patterns(self):
        all_closings = ALL_CLOSING_FORMULAS
        closing_alts = "|".join(re.escape(f) for f in all_closings)
        sanskrit_alts = "|".join(re.escape(f) for f in SANSKRIT_CLOSINGS)
        opening_alts = _opening_alts_without_yig_mgo()
        title_alts = "|".join(re.escape(t) for t in COLLECTION_TITLES)

        # Rule A: yig mgo opener at junction
        self.rule_a = re.compile(
            JUNCTION_BEFORE
            + r'\s*(༄༅(?:༅|༔)?(?:[།༎][།༎]?)?)'
        )

        # Rule B: closing + shad(s) + strong structural break (digits or blank line only)
        self.rule_b = re.compile(
            r'(' + closing_alts + r')'
            + r'\s*[' + CLOSING_PUNCT + r'][\s' + CLOSING_PUNCT + r']*'
            + r'(?:'
            + r'(?:\s*\d+\s*\n)+'
            + r'|'
            + r'\s*\n\s*\n'
            + r')'
        )

        # Rule C: digit line(s) after shad + newline
        self.rule_c = re.compile(
            r'[།༎' + CLOSING_PUNCT + r']\s*\n'
            + r'(\d+\s*\n)+'
            + r'\s*\n?'
        )

        # Rule D: opening formula at junction (non-yig-mgo)
        if opening_alts:
            self.rule_d = re.compile(
                JUNCTION_BEFORE + r'\s*(' + opening_alts + r')'
            )
        else:
            self.rule_d = None

        # Rule E: collection title on own line
        self.rule_e_pat = re.compile(
            r'\n\s*(' + title_alts + r')\s*(?:\n|$)'
        )

        # Rule G: ༈ section opener at junction
        self.rule_g = re.compile(
            JUNCTION_BEFORE + r'\s*(༈)\s*[།༎]?[།༎\s]*'
        )

        # Rule H: Sanskrit closing blessing before split
        self.rule_h = re.compile(
            r'(' + sanskrit_alts + r')'
            + r'[\s' + CLOSING_PUNCT + r']*[།༎][' + CLOSING_PUNCT + r'\s]*'
        )

    def _text_before(self, text, pos, n=80):
        return text[max(0, pos - n):pos]

    def _has_closing_before(self, text, pos, n=80):
        before = self._text_before(text, pos, n)
        return any(f in before for f in ALL_CLOSING_FORMULAS)

    def _has_structural_break_before(self, text, pos, n=80):
        before = self._text_before(text, pos, n)
        if any(f in before for f in ALL_CLOSING_FORMULAS):
            return True
        if re.search(r'\d+\s*\n', before):
            return True
        if "\n\n" in before:
            return True
        if re.search(r'[།༎' + CLOSING_PUNCT + r']\s*[།༎' + CLOSING_PUNCT + r']', before):
            return True
        return False

    def _looks_like_opener_after(self, text, pos, n=80):
        after = text[pos:pos + n].lstrip()
        if not after:
            return False
        if after[0] in YIG_MGO_CHARS or after.startswith(SBRUL_SHAD):
            return True
        return any(after.startswith(f) for f in OPENING_FORMULAS)

    def _boundary_at_opener(self, match, opener_group=1):
        return match.start(opener_group)

    def predict(self, text):
        candidates = []

        # Rule A: yig mgo
        for m in self.rule_a.finditer(text):
            candidates.append((m.start(1), 0.90, "A:yig_mgo"))

        # Rule B: closing + strong structural break (optional; net-negative on single newline)
        if self.use_rule_b:
            for m in self.rule_b.finditer(text):
                conf = 0.78 if re.search(r'\d+\s*\n', m.group(0)) else 0.80
                candidates.append((m.end(), conf, "B:closing+break"))

        # Rule C: gated digit separator (disabled by default — net-negative on this corpus)
        if self.use_rule_c:
            for m in self.rule_c.finditer(text):
                end = m.end()
                has_opener = self._looks_like_opener_after(text, end)
                has_closing = self._has_closing_before(text, m.start())
                if has_opener or has_closing:
                    candidates.append((end, 0.70, "C:page_numbers"))
                elif self.rule_c_bare:
                    candidates.append((end, 0.40, "C:page_numbers_weak"))

        # Rule D: opening formula at junction (disabled by default — ~9% precision)
        if self.use_rule_d and self.rule_d is not None:
            for m in self.rule_d.finditer(text):
                pos = m.start(1)
                if not self._has_structural_break_before(text, pos):
                    continue
                conf = 0.70 if self._has_closing_before(text, pos) else 0.60
                candidates.append((pos, conf, "D:opening_formula"))

        # Rule E: collection title header
        if self.rule_e:
            for m in self.rule_e_pat.finditer(text):
                before = text[max(0, m.start() - 30):m.start()]
                if re.search(r'\d+\s*$', before) or self._has_structural_break_before(
                    text, m.start()
                ):
                    candidates.append((m.end(), 0.50, "E:collection_title"))

        # Rule G: ༈ opener (guarded / unguarded)
        for m in self.rule_g.finditer(text):
            pos = m.start(1)
            if self._has_closing_before(text, pos) or self._has_structural_break_before(
                text, pos
            ):
                candidates.append((pos, 0.82, "G:sbrul_shad"))
            elif self.rule_g_unguarded:
                candidates.append((pos, 0.45, "G:sbrul_shad_weak"))

        # Rule H: Sanskrit closing at split
        for m in self.rule_h.finditer(text):
            end = m.end()
            after = text[end:end + 40].lstrip()
            if not after:
                continue
            if (
                after[0] == "\n"
                or after.startswith(SBRUL_SHAD)
                or after[0] in YIG_MGO_CHARS
                or any(after.startswith(f) for f in YIG_MGO_OPENERS)
            ):
                candidates.append((end, 0.72, "H:sanskrit_closing"))

        # Rules I–L: page-layout signals (opt-in, no-op on continuous text).
        candidates.extend(self._layout_candidates(text))

        return self._merge_candidates(candidates)

    def _layout_candidates(self, text):
        """Generate boundary candidates from page line-density (Rules I–L).

        Returns an empty list when no layout rule is enabled or the text has no
        detectable page structure (a single page), so continuous-stream input
        behaves exactly as it does with only Rules A–H.
        """
        if not (
            self.use_rule_i
            or self.use_rule_j
            or self.use_rule_k
            or self.use_rule_l
        ):
            return []

        pages = segment_pages(
            text,
            delimiter=self.page_delimiter,
            min_blank_lines=self.min_blank_lines,
            ignore_page_number_lines=self.ignore_page_number_lines,
        )
        if len(pages) < 2:
            return []

        cands = []
        threshold = self.line_threshold

        # Rule I: an empty page marks a break -> boundary at the start of the
        # following page (or the empty page itself when it is the last one).
        if self.use_rule_i:
            for idx, page in enumerate(pages):
                if page.is_empty:
                    target = idx + 1 if idx + 1 < len(pages) else idx
                    pos = char_offset_at_page_start(pages, target)
                    cands.append((pos, LAYOUT_CONF_EMPTY_PAGE, "I:empty_page"))

        # Rules J / K: sliding 3-page window keyed on a dense page followed by
        # a sparse one.
        for n in range(len(pages) - 2):
            page_a, page_b, page_c = pages[n], pages[n + 1], pages[n + 2]
            dense_a = page_a.nb_lines > threshold
            sparse_b = page_b.nb_lines < threshold
            if not (dense_a and sparse_b):
                continue

            if page_c.nb_lines < threshold:
                # Rule J: dense, sparse, sparse -> break before page N+2.
                if self.use_rule_j:
                    pos = char_offset_at_page_start(pages, n + 2)
                    cands.append((pos, LAYOUT_CONF_SPARSE_TAIL, "J:sparse_tail"))
            else:
                # Rule K: dense, sparse, dense -> ambiguous. Emit two low
                # confidence candidates and let the merge favour any nearby
                # orthographic (A/G/H) hit.
                if self.use_rule_k:
                    pos_b = char_offset_at_page_start(pages, n + 1)
                    pos_c = char_offset_at_page_start(pages, n + 2)
                    cands.append((pos_b, LAYOUT_CONF_SPARSE_ISLAND, "K:sparse_island"))
                    cands.append((pos_c, LAYOUT_CONF_SPARSE_ISLAND, "K:sparse_island"))

        # Rule L: modern-publication heuristics are reserved for a later phase
        # (running headers/footers, title pages). Intentionally a no-op stub.

        return cands

    def _merge_candidates(self, candidates):
        if not candidates:
            return []

        candidates = [(p, c, r) for p, c, r in candidates if c >= self.min_confidence]
        candidates.sort(key=lambda x: x[0])

        merged = []
        i = 0
        while i < len(candidates):
            group = [candidates[i]]
            j = i + 1
            while j < len(candidates) and candidates[j][0] - candidates[i][0] <= self.merge_window:
                group.append(candidates[j])
                j += 1
            merged.append(max(group, key=lambda x: x[1]))
            i = j

        return merged

    def predict_positions(self, text):
        return [pos for pos, conf, rule in self.predict(text)]


RULE_LABELS = {
    "A:yig_mgo": "Rule A — Yig mgo opener (༄༅)",
    "B:closing+break": "Rule B — Closing formula + line break",
    "C:page_numbers": "Rule C — Page / volume numbers (gated)",
    "C:page_numbers_weak": "Rule C — Page numbers (weak, recall only)",
    "D:opening_formula": "Rule D — Opening formula without yig mgo",
    "E:collection_title": "Rule E — Collection title header",
    "G:sbrul_shad": "Rule G — Section opener (༈)",
    "G:sbrul_shad_weak": "Rule G — ༈ opener (weak, recall only)",
    "H:sanskrit_closing": "Rule H — Sanskrit closing blessing",
    "I:empty_page": "Rule I — Empty page (0 lines)",
    "J:sparse_tail": "Rule J — Dense page then two sparse pages",
    "K:sparse_island": "Rule K — Sparse page between dense pages (ambiguous)",
    "L:modern_publication": "Rule L — Modern publication layout (reserved)",
}
