#!/usr/bin/env python3
"""Build rule_vs_crf_unique.md from evaluation report files."""

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RULE_REPORT = ROOT / "reports/evaluations/rule_based_evaluation_unique.md"
CRF_REPORT = ROOT / "reports/evaluations/crf_full_evaluation_unique.md"
OUT = ROOT / "reports/evaluations/rule_vs_crf_unique.md"


def parse_rule_profiles(text):
    rows = []
    for line in text.splitlines():
        m = re.match(
            r"\| \*\*(\w+)\*\* .*? \| (\d+\.\d+)% \| (\d+\.\d+)% \| (\d+\.\d+)% "
            r"\| ([\d,]+) \| ([\d,]+) \| ([\d,]+) \|",
            line,
        )
        if m:
            profile, p, r, f1, tp, fp, fn = m.groups()
            rows.append({
                "method": "Rule",
                "profile": profile,
                "precision": float(p) / 100,
                "recall": float(r) / 100,
                "f1": float(f1) / 100,
                "tp": int(tp.replace(",", "")),
                "fp": int(fp.replace(",", "")),
                "fn": int(fn.replace(",", "")),
            })
    return rows


def parse_crf(text):
    def grab(name):
        m = re.search(rf"\| {name} \| ([\d.]+) \|", text)
        return float(m.group(1)) if m else None

    tp = int(re.search(r"\| TP \| (\d+) \|", text).group(1))
    fp = int(re.search(r"\| FP \| (\d+) \|", text).group(1))
    fn = int(re.search(r"\| FN \| (\d+) \|", text).group(1))
    return {
        "method": "CRF",
        "profile": "full",
        "precision": grab("Precision"),
        "recall": grab("Recall"),
        "f1": grab("F1"),
        "tp": tp,
        "fp": fp,
        "fn": fn,
    }


def main():
    rule_rows = parse_rule_profiles(RULE_REPORT.read_text(encoding="utf-8"))
    crf_row = parse_crf(CRF_REPORT.read_text(encoding="utf-8"))
    all_rows = rule_rows + [crf_row]

    best_rule = max(rule_rows, key=lambda x: x["f1"])
    lines = [
        "# Rule-based vs CRF — Comparison on unique.json",
        "",
        f"**Benchmark:** `data/breakpoints_context_snippets_unique.json`",
        "**Tolerance:** ±15 characters",
        "",
        "## Summary table",
        "",
        "| Method | Profile | Precision | Recall | F1 | TP | FP | FN |",
        "|--------|---------|-----------|--------|-----|------:|-----:|-----:|",
    ]
    for row in all_rows:
        lines.append(
            f"| {row['method']} | {row['profile']} | {row['precision']:.4f} | "
            f"{row['recall']:.4f} | {row['f1']:.4f} | {row['tp']:,} | "
            f"{row['fp']:,} | {row['fn']:,} |"
        )
    lines.extend([
        "",
        f"**Best rule F1:** {best_rule['profile']} ({best_rule['f1']:.4f})",
        f"**CRF full F1:** {crf_row['f1']:.4f}",
        f"**Delta:** {best_rule['f1'] - crf_row['f1']:+.4f}",
        "",
    ])
    OUT.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
