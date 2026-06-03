"""
evaluation.py — Evaluation, reporting, and file-prediction helpers for the
rule-based Tibetan boundary detector.
"""

import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from .detector import PROFILE_PRESETS, RULE_LABELS, RuleBasedDetector
from .paths import diagnostics_dir, ensure_report_dir, evaluations_dir
from .utils import (
    evaluate,
    insert_boundaries,
    strip_boundaries,
)


def default_report_path(input_path):
    stem = Path(input_path).stem
    if stem.startswith("breakpoints_context_snippets"):
        suffix = stem.replace("breakpoints_context_snippets", "").strip("_") or "report"
        name = f"rule_based_evaluation_{suffix}.md"
    else:
        name = f"rule_based_evaluation_{stem}.md"
    return evaluations_dir() / name


def format_evaluation_report_markdown(
    input_path,
    profile_results,
    tolerance,
    sample_count,
):
    """Build a markdown report readable by technical and non-technical readers."""
    input_name = Path(input_path).name
    generated = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    lines = [
        "# Rule-Based Boundary Detector — Evaluation Report",
        "",
        f"**Generated:** {generated}  ",
        f"**Input data:** `{input_name}`  ",
        f"**Snippets evaluated:** {sample_count:,}  ",
        f"**Match tolerance:** ±{tolerance} characters (a prediction counts as correct if it falls within this distance of the true boundary)",
        "",
        "---",
        "",
        "## Summary (plain language)",
        "",
        "This report measures how well the **rule-based detector** finds places where one Tibetan text ends and another begins. "
        "Human annotators marked the correct positions with `</b>` in each snippet; the detector predicts positions using pattern rules (yig mgo ༄༅, section marks ༈, closing phrases, and similar signals).",
        "",
        "- **Precision** answers: *Of all boundaries the detector predicted, how many were actually correct?* "
        "Higher is better when you want fewer false splits.",
        "- **Recall** answers: *Of all true boundaries in the data, how many did the detector find?* "
        "Higher is better when you want fewer missed boundaries.",
        "- **F1** is a single score that balances precision and recall (higher is better overall).",
        "",
    ]

    if len(profile_results) > 1:
        best = max(profile_results.items(), key=lambda x: x[1]["f1"])
        lines.append(
            f"Among the three built-in profiles, **{best[0]}** achieved the highest F1 "
            f"({best[1]['f1']:.1%}) on this dataset."
        )
        lines.append("")

    lines.extend([
        "---",
        "",
        "## Results by profile",
        "",
        "| Profile | Plain meaning | Precision | Recall | F1 | Correct | Wrong guesses | Missed |",
        "|---------|---------------|-----------|--------|-----|---------|---------------|--------|",
    ])

    profile_blurbs = {
        "recall": "Find more boundaries (more guesses allowed)",
        "balanced": "Default balance",
        "precision": "Fewer wrong guesses (stricter)",
    }
    for name, r in profile_results.items():
        lines.append(
            f"| **{name}** | {profile_blurbs.get(name, '')} | "
            f"{r['precision']:.1%} | {r['recall']:.1%} | {r['f1']:.1%} | "
            f"{r['true_positives']:,} | {r['false_positives']:,} | {r['false_negatives']:,} |"
        )

    lines.extend([
        "",
        "**Counts explained:**",
        "- **Correct** = boundaries found within ±{tolerance} chars of the annotation".format(tolerance=tolerance),
        "- **Wrong guesses** = predicted boundaries that do not match any annotation (false alarms)",
        "- **Missed** = true annotations with no matching prediction nearby",
        "",
        "---",
        "",
        "## Rule breakdown (help vs harm)",
        "",
        "Each prediction is credited to the rule that fired. "
        "**Correct** = true positive (within tolerance of an annotation). "
        "**Wrong** = false alarm. **Net** = correct minus wrong (negative means the rule hurts overall). "
        "**Share of correct** = fraction of all true positives found by this rule.",
        "",
    ])

    for name, r in profile_results.items():
        lines.append(f"### Profile: {name}")
        lines.append("")
        rule_hits = r.get("rule_hits") or {}
        rule_fp = r.get("rule_false_positives") or {}
        all_rules = set(rule_hits) | set(rule_fp)
        if not all_rules:
            lines.append("_No rule firings recorded._")
            lines.append("")
            continue
        lines.append(
            "| Rule | What it looks for | Correct | Wrong | Precision | Net | Share of correct |"
        )
        lines.append(
            "|------|-------------------|--------:|------:|----------:|----:|-----------------:|"
        )
        total_tp = r["true_positives"]
        for rule in sorted(
            all_rules,
            key=lambda x: (rule_hits.get(x, 0) - rule_fp.get(x, 0)),
        ):
            tp = rule_hits.get(rule, 0)
            fp = rule_fp.get(rule, 0)
            label = RULE_LABELS.get(rule, rule)
            prec = tp / (tp + fp) if (tp + fp) else 0
            net = tp - fp
            share = 100 * tp / max(total_tp, 1)
            net_cell = f"{net:+,}" if net else "0"
            lines.append(
                f"| `{rule}` | {label} | {tp:,} | {fp:,} | {prec:.1%} | {net_cell} | {share:.1f}% |"
            )
        lines.append("")
        harmful = [
            rule
            for rule in all_rules
            if rule_hits.get(rule, 0) < rule_fp.get(rule, 0)
        ]
        if harmful:
            labels = ", ".join(f"`{x}`" for x in harmful)
            lines.append(
                f"**Net-negative rules** (more wrong than correct): {labels}."
            )
            lines.append("")

    lines.extend([
        "---",
        "",
        "## Profile settings (technical)",
        "",
        "| Setting | recall | balanced | precision |",
        "|---------|--------|----------|-----------|",
    ])
    for label, key in [
        ("Minimum confidence", "min_confidence"),
        ("Merge window (chars)", "merge_window"),
        ("Rule B enabled", "rule_b"),
        ("Rule C enabled", "rule_c"),
        ("Rule D enabled", "rule_d"),
        ("Rule E enabled", "rule_e"),
        ("Rule G unguarded", "rule_g_unguarded"),
        ("Rule C bare digits", "rule_c_bare"),
    ]:
        cells = [label]
        for pname in ("recall", "balanced", "precision"):
            if pname in profile_results:
                cells.append(str(profile_results[pname]["settings"].get(key, "—")))
            else:
                cells.append("—")
        lines.append("| " + " | ".join(cells) + " |")

    lines.extend([
        "",
        "---",
        "",
        "## Output files",
        "",
        "| File | Description |",
        "|------|-------------|",
        f"| `reports/diagnostics/false_negatives.json` | List of **missed** boundaries from the **last** single-profile run |",
        f"| This report | Updated automatically each time you run `evaluate` (override path with `--report`) |",
        "",
        "---",
        "",
        "## How to reproduce",
        "",
        "```bash",
        f"outline-detect evaluate data/{input_name} --all-profiles --tolerance {tolerance}",
        "```",
        "",
        "Single profile only:",
        "",
        "```bash",
        f"outline-detect evaluate data/{input_name} --profile balanced --tolerance {tolerance}",
        "```",
        "",
    ])
    return "\n".join(lines)


def run_evaluation(annotated_strings, detector, tolerance=15, show_errors=0):
    all_tp = all_fp = all_fn = all_predicted = all_true = 0
    rule_hits = Counter()
    rule_false_positives = Counter()
    fn_contexts = []
    fp_contexts = []

    for idx, annotated in enumerate(annotated_strings):
        clean, true_positions = strip_boundaries(annotated)
        if not true_positions:
            continue

        predictions = detector.predict(clean)
        pred_positions = [p for p, c, r in predictions]
        result = evaluate(pred_positions, true_positions, tolerance=tolerance)

        all_tp += result["true_positives"]
        all_fp += result["false_positives_count"]
        all_fn += result["false_negatives_count"]
        all_predicted += result["total_predicted"]
        all_true += result["total_true"]

        matched_pred = {pred_pos for _, pred_pos, _ in result["matches"]}
        for p, c, r in predictions:
            if p in matched_pred:
                rule_hits[r] += 1
            else:
                rule_false_positives[r] += 1

        for fn_pos in result["false_negatives"]:
            fn_contexts.append({
                "sample_idx": idx,
                "position": fn_pos,
                "left_context": repr(clean[max(0, fn_pos - 60):fn_pos]),
                "right_context": repr(clean[fn_pos:fn_pos + 60]),
            })

        for fp_pos in result["false_positives"]:
            fp_contexts.append({
                "sample_idx": idx,
                "position": fp_pos,
                "left_context": repr(clean[max(0, fp_pos - 40):fp_pos]),
                "right_context": repr(clean[fp_pos:fp_pos + 40]),
            })

    precision = all_tp / (all_tp + all_fp) if (all_tp + all_fp) > 0 else 0
    recall = all_tp / (all_tp + all_fn) if (all_tp + all_fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    profile_label = detector.profile or "custom"
    print("=" * 70)
    print("RULE-BASED DETECTOR — EVALUATION RESULTS")
    print("=" * 70)
    print(f"  Profile:            {profile_label}")
    print(f"  Samples evaluated:  {len(annotated_strings)}")
    print(f"  Tolerance:          ±{tolerance} chars")
    print(f"  Min confidence:     {detector.min_confidence}")
    print(f"  Merge window:       {detector.merge_window} chars")
    print()
    print(f"  Total true boundaries:      {all_true}")
    print(f"  Total predicted boundaries: {all_predicted}")
    print(f"  True positives:             {all_tp}")
    print(f"  False positives:            {all_fp}")
    print(f"  False negatives:            {all_fn}")
    print()
    print(f"  PRECISION:  {precision:.4f}  ({all_tp}/{all_tp + all_fp})")
    print(f"  RECALL:     {recall:.4f}  ({all_tp}/{all_tp + all_fn})")
    print(f"  F1 SCORE:   {f1:.4f}")
    print()

    print("-" * 50)
    print("RULE BREAKDOWN (correct / wrong / precision / net)")
    print("-" * 50)
    all_rules = set(rule_hits) | set(rule_false_positives)
    for rule in sorted(
        all_rules,
        key=lambda x: rule_hits.get(x, 0) - rule_false_positives.get(x, 0),
    ):
        tp = rule_hits.get(rule, 0)
        fp = rule_false_positives.get(rule, 0)
        prec = tp / (tp + fp) if (tp + fp) else 0
        net = tp - fp
        print(f"  {rule:30s}  TP {tp:>5}  FP {fp:>5}  prec {prec:5.1%}  net {net:+6}")
    print()

    if show_errors > 0 and fn_contexts:
        print("-" * 50)
        print(f"FALSE NEGATIVES — showing first {min(show_errors, len(fn_contexts))}")
        print("-" * 50)
        for fn in fn_contexts[:show_errors]:
            print(f"  Sample #{fn['sample_idx']}, pos {fn['position']}:")
            print(f"    LEFT:  {fn['left_context']}")
            print(f"    RIGHT: {fn['right_context']}")
            print()

    fn_path = diagnostics_dir() / "false_negatives.json"
    ensure_report_dir(fn_path)
    fn_path.write_text(json.dumps(fn_contexts, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"  False negative details saved to: {fn_path}")

    return {
        "profile": profile_label,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "true_positives": all_tp,
        "false_positives": all_fp,
        "false_negatives": all_fn,
        "total_predicted": all_predicted,
        "total_true": all_true,
        "samples": len(annotated_strings),
        "tolerance": tolerance,
        "rule_hits": dict(rule_hits),
        "rule_false_positives": dict(rule_false_positives),
        "settings": {
            "min_confidence": detector.min_confidence,
            "merge_window": detector.merge_window,
            "rule_b": detector.use_rule_b,
            "rule_c": detector.use_rule_c,
            "rule_d": detector.use_rule_d,
            "rule_e": detector.rule_e,
            "rule_g_unguarded": detector.rule_g_unguarded,
            "rule_c_bare": detector.rule_c_bare,
        },
        "false_negative_count": len(fn_contexts),
    }


def run_evaluation_all_profiles(
    annotated_strings,
    input_path,
    tolerance=15,
    show_errors=0,
    report_path=None,
):
    """Run recall, balanced, and precision; optionally write a markdown report."""
    profile_results = {}
    for profile_name in PROFILE_PRESETS:
        detector = RuleBasedDetector(profile=profile_name)
        result = run_evaluation(
            annotated_strings,
            detector,
            tolerance=tolerance,
            show_errors=show_errors if profile_name == "balanced" else 0,
        )
        profile_results[profile_name] = result

    if report_path is None:
        report_path = default_report_path(input_path)
    report_path = Path(report_path)
    md = format_evaluation_report_markdown(
        input_path,
        profile_results,
        tolerance,
        len(annotated_strings),
    )
    ensure_report_dir(report_path)
    report_path.write_text(md, encoding="utf-8")
    print(f"  Evaluation report saved to: {report_path.resolve()}")

    return profile_results, report_path


def run_prediction(input_path, output_path, detector):
    text = Path(input_path).read_text(encoding="utf-8")
    predictions = detector.predict(text)

    print(f"Found {len(predictions)} predicted boundaries.\n")
    print("Position  Confidence  Rule")
    print("-" * 50)
    for pos, conf, rule in predictions:
        print(f"{pos:>8}  {conf:.2f}        {rule}")

    positions = [p for p, c, r in predictions]
    output_path = ensure_report_dir(output_path)
    output_path.write_text(insert_boundaries(text, positions), encoding="utf-8")
    print(f"\nAnnotated text saved to: {output_path}")
