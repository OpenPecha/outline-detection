"""
cli.py — Command-line interface for outline_detection.

Subcommands:
    detect    Detect boundaries in text -> {"breakpoints": [...]} JSON
    evaluate  Evaluate the rule detector against annotated data
    analyze   Run boundary pattern analysis on annotated data
    crf       Train / predict / inspect a CRF model (needs [crf] extra)
"""

import argparse
import json
import sys
from pathlib import Path

from .api import detect_breakpoints
from .detector import PROFILE_PRESETS, RuleBasedDetector
from .paths import reports_dir


def _read_input_text(source, inline_text):
    if inline_text is not None:
        return inline_text
    if source in (None, "-"):
        return sys.stdin.read()
    return Path(source).read_text(encoding="utf-8")


def _cmd_detect(args):
    text = _read_input_text(args.input, args.text)
    result = detect_breakpoints(
        text,
        profile=args.profile,
        min_confidence=args.min_confidence,
        merge_window=args.merge_window,
        detailed=args.detailed,
    )
    payload = json.dumps(result, ensure_ascii=False, indent=2 if args.pretty else None)
    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(payload, encoding="utf-8")
        print(f"Wrote {len(result['breakpoints'])} breakpoints to {out}")
    else:
        print(payload)
    return 0


def _cmd_evaluate(args):
    from .evaluation import (
        default_report_path,
        format_evaluation_report_markdown,
        run_evaluation,
        run_evaluation_all_profiles,
    )
    from .paths import ensure_report_dir
    from .utils import load_annotated

    input_path = Path(args.input_file)
    if not input_path.exists():
        print(f"Error: file not found: {input_path}", file=sys.stderr)
        return 1

    strings = load_annotated(str(input_path))
    if args.report == "":
        report_path = default_report_path(input_path)
    elif args.report is not None:
        report_path = Path(args.report)
    else:
        report_path = default_report_path(input_path)

    if args.all_profiles:
        run_evaluation_all_profiles(
            strings,
            str(input_path),
            tolerance=args.tolerance,
            show_errors=args.show_errors,
            report_path=report_path,
        )
    else:
        detector = RuleBasedDetector(profile=args.profile)
        if args.min_confidence is not None:
            detector.min_confidence = args.min_confidence
        if args.merge_window is not None:
            detector.merge_window = args.merge_window
        result = run_evaluation(
            strings,
            detector,
            tolerance=args.tolerance,
            show_errors=args.show_errors,
        )
        md = format_evaluation_report_markdown(
            str(input_path),
            {result["profile"]: result},
            args.tolerance,
            len(strings),
        )
        report_path = ensure_report_dir(report_path)
        report_path.write_text(md, encoding="utf-8")
        print(f"  Evaluation report saved to: {report_path.resolve()}")
    return 0


def _cmd_predict(args):
    from .evaluation import run_prediction

    detector = RuleBasedDetector(profile=args.profile)
    if args.min_confidence is not None:
        detector.min_confidence = args.min_confidence
    if args.merge_window is not None:
        detector.merge_window = args.merge_window
    output = args.output or (reports_dir() / "predicted_boundaries.txt")
    run_prediction(args.input_file, output, detector)
    return 0


def _cmd_analyze(args):
    from .analyzer import run_analysis

    try:
        run_analysis(args.input_file, window=args.window, output=args.output)
    except FileNotFoundError:
        print(f"Error: file not found: {args.input_file}", file=sys.stderr)
        return 1
    return 0


def _cmd_crf(args):
    try:
        from . import crf
    except ImportError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    try:
        if args.crf_command == "train":
            save_model = args.save_model
            if args.save_model == "":
                save_model = crf.default_model_path()
            crf.run_train(
                args.input_file,
                folds=args.folds,
                save_model=save_model,
                c1=args.c1,
                c2=args.c2,
                max_iter=args.max_iter,
                label_radius=args.label_radius,
                tolerance=args.tolerance,
            )
        elif args.crf_command == "predict":
            crf.run_predict(args.input_file, args.model, args.output)
        elif args.crf_command == "inspect":
            crf.run_inspect(args.model_file, top=args.top)
        else:
            print("Specify a crf subcommand: train, predict, or inspect", file=sys.stderr)
            return 1
    except ImportError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    return 0


def build_parser():
    parser = argparse.ArgumentParser(
        prog="outline-detect",
        description="Rule-based Tibetan text boundary (outline) detection.",
    )
    sub = parser.add_subparsers(dest="command")

    # detect
    d = sub.add_parser("detect", help="Detect boundaries -> {\"breakpoints\": [...]} JSON")
    d.add_argument("input", nargs="?", default=None,
                   help="Path to a text file, or '-' for stdin (default: stdin)")
    d.add_argument("--text", default=None, help="Inline text instead of a file")
    d.add_argument("--profile", choices=list(PROFILE_PRESETS), default="balanced")
    d.add_argument("--min-confidence", type=float, default=None)
    d.add_argument("--merge-window", type=int, default=None)
    d.add_argument("--detailed", action="store_true",
                   help="Include per-boundary confidence and firing rule")
    d.add_argument("--pretty", action="store_true", help="Pretty-print JSON")
    d.add_argument("--output", "-o", default=None, help="Write JSON to this file")
    d.set_defaults(func=_cmd_detect)

    # evaluate
    e = sub.add_parser("evaluate", help="Evaluate against annotated data")
    e.add_argument("input_file", help="Annotated JSON/JSONL/TXT file")
    e.add_argument("--profile", choices=list(PROFILE_PRESETS), default="balanced")
    e.add_argument("--all-profiles", action="store_true",
                   help="Evaluate recall, balanced, and precision into one report")
    e.add_argument("--tolerance", type=int, default=15)
    e.add_argument("--min-confidence", type=float, default=None)
    e.add_argument("--merge-window", type=int, default=None)
    e.add_argument("--show-errors", type=int, default=0)
    e.add_argument("--report", nargs="?", const="", default=None,
                   help="Write markdown report (optional path)")
    e.set_defaults(func=_cmd_evaluate)

    # predict (annotated text output)
    p = sub.add_parser("predict", help="Annotate a raw text file with boundary markers")
    p.add_argument("input_file", help="Raw text file")
    p.add_argument("--profile", choices=list(PROFILE_PRESETS), default="balanced")
    p.add_argument("--min-confidence", type=float, default=None)
    p.add_argument("--merge-window", type=int, default=None)
    p.add_argument("--output", "-o", default=None)
    p.set_defaults(func=_cmd_predict)

    # analyze
    a = sub.add_parser("analyze", help="Boundary pattern analysis on annotated data")
    a.add_argument("input_file", help="Annotated JSON/JSONL/TXT file")
    a.add_argument("--window", type=int, default=200)
    a.add_argument("--output", "-o", default=None)
    a.set_defaults(func=_cmd_analyze)

    # crf
    c = sub.add_parser("crf", help="CRF model (needs the [crf] extra)")
    crf_sub = c.add_subparsers(dest="crf_command")

    ct = crf_sub.add_parser("train", help="Train (optionally with cross-validation)")
    ct.add_argument("input_file")
    ct.add_argument("--folds", type=int, default=0)
    ct.add_argument("--save-model", nargs="?", const="", default=None, metavar="PATH",
                    help="Save model (default: reports/models/boundary_crf.pkl)")
    ct.add_argument("--c1", type=float, default=0.1)
    ct.add_argument("--c2", type=float, default=0.1)
    ct.add_argument("--max-iter", type=int, default=150)
    ct.add_argument("--label-radius", type=int, default=1)
    ct.add_argument("--tolerance", type=int, default=15)

    cp = crf_sub.add_parser("predict", help="Predict on raw text")
    cp.add_argument("input_file")
    cp.add_argument("--model", required=True)
    cp.add_argument("--output", "-o", default=None)

    ci = crf_sub.add_parser("inspect", help="Inspect model feature weights")
    ci.add_argument("model_file")
    ci.add_argument("--top", type=int, default=30)

    c.set_defaults(func=_cmd_crf)

    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if not getattr(args, "command", None):
        parser.print_help()
        return 1
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
