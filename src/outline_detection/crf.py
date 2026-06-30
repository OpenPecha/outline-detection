"""
crf.py — CRF sequence-labeling pipeline for Tibetan text boundary detection.

Trains a Conditional Random Field on annotated text to predict boundary positions.

Requires the optional ``crf`` extra:
    pip install "outline-detection[crf]"

    from outline_detection.crf import CRFBoundaryDetector
    detector = CRFBoundaryDetector()
    detector.train(annotated_strings)
    boundaries = detector.predict(text)
"""

import json
import pickle
import re
import warnings
from datetime import datetime, timezone
from pathlib import Path
from collections import Counter

from .paths import models_dir, reports_dir, ensure_report_dir
from .utils import (
    CLOSING_FORMULAS, OPENING_FORMULAS, COLLECTION_TITLES,
    YIG_MGO_CHARS, SHAD, GTER_TSHEG, TSHEG, NYIS_SHAD,
    tokenize, classify_char,
    strip_boundaries, load_annotated, evaluate, format_eval, BOUNDARY_TAG,
    insert_boundaries,
)

warnings.filterwarnings("ignore", category=UserWarning)

_CRF_EXTRA_MSG = (
    'CRF features require extra dependencies. '
    'Install with: pip install "outline-detection[crf]"'
)

# Bump when feature extraction or labeling logic changes (invalidates caches).
FEATURE_CACHE_VERSION = 1


def _import_crf():
    """Lazily import sklearn-crfsuite; raise a friendly error if missing."""
    try:
        import sklearn_crfsuite
        from sklearn_crfsuite import metrics as crf_metrics
        from sklearn.model_selection import KFold
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise ImportError(_CRF_EXTRA_MSG) from exc
    return sklearn_crfsuite, crf_metrics, KFold


# ===========================================================================
# Feature extraction
# ===========================================================================

def token_features(tokens, i):
    """
    Extract features for the token at position i in the token list.
    Each token is (type, literal, start_pos, end_pos).

    Features encode:
    - Token type and literal properties
    - Neighborhood context (±3 tokens)
    - Known formula membership
    - Digit/newline structural patterns
    """
    tok_type, tok_lit, tok_start, tok_end = tokens[i]
    n = len(tokens)

    features = {
        "bias": 1.0,

        # --- Current token ---
        "type": tok_type,
        "len": min(len(tok_lit), 20),  # capped length
        "is_yig_mgo": tok_type == "YIG_MGO",
        "is_shad": tok_type == "SHAD",
        "is_nyis_shad": tok_type == "NYIS_SHAD",
        "is_gter_tsheg": tok_type == "GTER_TSHEG",
        "is_newline": tok_type == "NEWLINE",
        "is_digit": tok_type == "DIGIT",
        "is_space": tok_type == "SPACE",
        "is_tibetan": tok_type == "TIBETAN",

        # Digit specifics
        "digit_len": len(tok_lit) if tok_type == "DIGIT" else 0,
        "is_page_number": (tok_type == "DIGIT" and 1 <= len(tok_lit) <= 4),

        # Newline specifics
        "newline_count": tok_lit.count("\n") if tok_type == "NEWLINE" else 0,
        "is_double_newline": (tok_type == "NEWLINE" and tok_lit.count("\n") >= 2),

        # Known formulas
        "is_closing_formula": any(f in tok_lit for f in CLOSING_FORMULAS) if tok_type == "TIBETAN" else False,
        "is_opening_formula": any(f in tok_lit for f in OPENING_FORMULAS) if tok_type == "TIBETAN" else False,
        "is_collection_title": any(t in tok_lit for t in COLLECTION_TITLES) if tok_type == "TIBETAN" else False,

        # Specific high-value formulas
        "has_mangalam": "མངྒ་ལཾ" in tok_lit if tok_type == "TIBETAN" else False,
        "has_dgeo": "དགེའོ" in tok_lit if tok_type == "TIBETAN" else False,
        "has_rdzogs": "རྫོགས་སོ" in tok_lit if tok_type == "TIBETAN" else False,
        "has_bkra_shis": "བཀྲ་ཤིས" in tok_lit if tok_type == "TIBETAN" else False,
        "has_bzhugs": "བཞུགས" in tok_lit if tok_type == "TIBETAN" else False,
        "has_namo": "ན་མོ" in tok_lit if tok_type == "TIBETAN" else False,
        "has_gsung_bum": "གསུང་འབུམ" in tok_lit if tok_type == "TIBETAN" else False,

        # Position features
        "is_first": i == 0,
        "is_last": i == n - 1,
        "relative_pos": round(i / max(n, 1), 2),
    }

    # --- Tibetan suffix features (last few chars can indicate formula endings) ---
    if tok_type == "TIBETAN" and len(tok_lit) >= 3:
        features["suffix3"] = tok_lit[-3:]
        features["prefix3"] = tok_lit[:3]
    if tok_type == "TIBETAN" and len(tok_lit) >= 5:
        features["suffix5"] = tok_lit[-5:]
        features["prefix5"] = tok_lit[:5]

    # --- Context window: previous tokens ---
    for offset in range(1, 4):
        if i - offset >= 0:
            prev_type, prev_lit, _, _ = tokens[i - offset]
            prefix = f"-{offset}"
            features[f"{prefix}:type"] = prev_type
            features[f"{prefix}:is_yig_mgo"] = prev_type == "YIG_MGO"
            features[f"{prefix}:is_shad"] = prev_type == "SHAD"
            features[f"{prefix}:is_digit"] = prev_type == "DIGIT"
            features[f"{prefix}:is_newline"] = prev_type == "NEWLINE"
            features[f"{prefix}:is_tibetan"] = prev_type == "TIBETAN"
            features[f"{prefix}:is_gter_tsheg"] = prev_type == "GTER_TSHEG"
            if prev_type == "DIGIT":
                features[f"{prefix}:digit_len"] = len(prev_lit)
            if prev_type == "NEWLINE":
                features[f"{prefix}:newline_count"] = prev_lit.count("\n")
            if prev_type == "TIBETAN":
                features[f"{prefix}:is_closing"] = any(f in prev_lit for f in CLOSING_FORMULAS)
                features[f"{prefix}:is_collection"] = any(t in prev_lit for t in COLLECTION_TITLES)
        else:
            features[f"-{offset}:type"] = "BOS"  # beginning of sequence

    # --- Context window: next tokens ---
    for offset in range(1, 4):
        if i + offset < n:
            next_type, next_lit, _, _ = tokens[i + offset]
            prefix = f"+{offset}"
            features[f"{prefix}:type"] = next_type
            features[f"{prefix}:is_yig_mgo"] = next_type == "YIG_MGO"
            features[f"{prefix}:is_shad"] = next_type == "SHAD"
            features[f"{prefix}:is_digit"] = next_type == "DIGIT"
            features[f"{prefix}:is_newline"] = next_type == "NEWLINE"
            features[f"{prefix}:is_tibetan"] = next_type == "TIBETAN"
            if next_type == "TIBETAN":
                features[f"{prefix}:is_opening"] = any(f in next_lit for f in OPENING_FORMULAS)
                features[f"{prefix}:has_bzhugs"] = "བཞུགས" in next_lit
        else:
            features[f"+{offset}:type"] = "EOS"  # end of sequence

    # --- Composite / interaction features ---
    # "shad followed by newline followed by digit" — the page-number pattern
    if i + 2 < n:
        types_ahead = (tokens[i+1][0] if i+1 < n else "", tokens[i+2][0] if i+2 < n else "")
        features["next2_types"] = f"{types_ahead[0]}_{types_ahead[1]}"

    if i >= 2:
        types_behind = (tokens[i-2][0] if i-2 >= 0 else "", tokens[i-1][0] if i-1 >= 0 else "")
        features["prev2_types"] = f"{types_behind[0]}_{types_behind[1]}"

    # Three-token type trigram centered on current
    if 0 < i < n - 1:
        features["trigram"] = f"{tokens[i-1][0]}_{tok_type}_{tokens[i+1][0]}"

    return features


def extract_features_and_labels(tokens, boundary_token_indices):
    """
    Extract feature dicts and labels for all tokens in a sequence.

    Args:
        tokens: list of (type, literal, start, end) tuples
        boundary_token_indices: set of token indices that mark boundaries

    Returns:
        (feature_list, label_list)
    """
    features = []
    labels = []
    for i in range(len(tokens)):
        features.append(token_features(tokens, i))
        if i in boundary_token_indices:
            labels.append("B")  # boundary
        else:
            labels.append("O")  # other
    return features, labels


# ===========================================================================
# Data preparation
# ===========================================================================

def prepare_training_data(annotated_strings, boundary_label_radius=1):
    """
    Convert annotated strings into CRF training sequences.

    For each annotated string:
    1. Strip <b> markers, recording their positions
    2. Tokenize the clean text
    3. Find which tokens are at/near boundary positions
    4. Label those tokens as "B", everything else as "O"

    Args:
        annotated_strings: list of strings with <b> markers
        boundary_label_radius: how many tokens around the boundary to label as "B"
                              (0 = only the exact token, 1 = ±1 token)

    Returns:
        list of (features, labels) pairs, one per input string
    """
    X = []
    y = []
    skipped = 0

    for annotated in annotated_strings:
        clean, boundary_positions = strip_boundaries(annotated)
        if not clean.strip():
            skipped += 1
            continue

        tokens = tokenize(clean)
        if not tokens:
            skipped += 1
            continue

        # Map boundary char positions to token indices
        boundary_token_indices = set()
        for bp in boundary_positions:
            # Find the token whose start is closest to the boundary position
            best_idx = 0
            best_dist = abs(tokens[0][2] - bp)
            for t_idx, (_, _, t_start, t_end) in enumerate(tokens):
                dist = min(abs(t_start - bp), abs(t_end - bp))
                if dist < best_dist:
                    best_dist = dist
                    best_idx = t_idx

            # Label the boundary token and its neighbors
            for offset in range(-boundary_label_radius, boundary_label_radius + 1):
                idx = best_idx + offset
                if 0 <= idx < len(tokens):
                    boundary_token_indices.add(idx)

        features, labels = extract_features_and_labels(tokens, boundary_token_indices)
        X.append(features)
        y.append(labels)

    if skipped:
        print(f"  (Skipped {skipped} empty/invalid strings)")

    return X, y


# ===========================================================================
# Prepared feature cache
# ===========================================================================

def _print_training_stats(X, y):
    print(f"  Training sequences: {len(X)}")
    print(f"  Total tokens: {sum(len(seq) for seq in X)}")
    b_count = sum(labels.count("B") for labels in y)
    o_count = sum(labels.count("O") for labels in y)
    print(f"  B (boundary) tokens: {b_count}")
    print(f"  O (other) tokens:    {o_count}")
    print(f"  B/O ratio:           1:{o_count // max(b_count, 1)}")
    print()


def save_prepared_features(path, X, y, metadata):
    """Persist prepared CRF sequences (X, y) and metadata to disk."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": FEATURE_CACHE_VERSION,
        "X": X,
        "y": y,
        "metadata": metadata,
    }
    print(f"Saving prepared features to {path} ...")
    with open(path, "wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
    size_gb = path.stat().st_size / (1024 ** 3)
    print(f"  Saved ({size_gb:.2f} GB)")


def load_prepared_features(path, boundary_label_radius):
    """Load prepared features; validate version and label radius."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    print(f"Loading prepared features from {path} ...")
    with open(path, "rb") as f:
        payload = pickle.load(f)
    version = payload.get("version")
    if version != FEATURE_CACHE_VERSION:
        raise ValueError(
            f"Feature cache version mismatch: file has {version}, "
            f"expected {FEATURE_CACHE_VERSION}. Delete the cache and rebuild."
        )
    meta = payload.get("metadata") or {}
    cached_radius = meta.get("boundary_label_radius")
    if cached_radius is not None and cached_radius != boundary_label_radius:
        raise ValueError(
            f"Feature cache label_radius={cached_radius} does not match "
            f"requested {boundary_label_radius}. Delete the cache and rebuild."
        )
    X = payload["X"]
    y = payload["y"]
    print(f"  Loaded {len(X)} sequences, {sum(len(s) for s in X)} tokens")
    if meta:
        print(f"  Cache metadata: source={meta.get('source_file')}, "
              f"created={meta.get('created_at')}")
    return X, y


def resolve_prepared_features(annotated_strings, boundary_label_radius, features_cache,
                              source_file=None):
    """
    Return (X, y), loading from cache if the file exists else preparing and saving.
    """
    cache_path = Path(features_cache) if features_cache else None
    if cache_path and cache_path.exists():
        return load_prepared_features(cache_path, boundary_label_radius)

    print(f"Preparing training data from {len(annotated_strings)} strings...")
    X, y = prepare_training_data(annotated_strings, boundary_label_radius)
    _print_training_stats(X, y)

    if cache_path:
        metadata = {
            "source_file": str(source_file) if source_file else None,
            "boundary_label_radius": boundary_label_radius,
            "n_sequences": len(X),
            "n_tokens": sum(len(seq) for seq in X),
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        save_prepared_features(cache_path, X, y, metadata)
    return X, y


# ===========================================================================
# CRF Detector class
# ===========================================================================

class CRFBoundaryDetector:
    """
    CRF-based Tibetan text boundary detector.
    """

    def __init__(self, c1=0.1, c2=0.1, max_iterations=150, boundary_label_radius=1):
        """
        Args:
            c1: L1 regularization coefficient
            c2: L2 regularization coefficient
            max_iterations: max training iterations
            boundary_label_radius: tokens around boundary to label (0=exact, 1=±1)
        """
        self.c1 = c1
        self.c2 = c2
        self.max_iterations = max_iterations
        self.boundary_label_radius = boundary_label_radius
        self.crf = None

    def train(self, annotated_strings=None, features_cache=None, source_file=None,
              X=None, y=None):
        """
        Train the CRF on annotated strings or pre-built (X, y).

        If features_cache path exists, load X/y from disk and skip extraction.
        Otherwise prepare from annotated_strings and optionally save to features_cache.
        """
        if X is None or y is None:
            if annotated_strings is None:
                raise ValueError("Provide annotated_strings or pre-built X, y")
            X, y = resolve_prepared_features(
                annotated_strings,
                self.boundary_label_radius,
                features_cache,
                source_file=source_file,
            )

        print("Training CRF model...")
        sklearn_crfsuite, _crf_metrics, _KFold = _import_crf()
        self.crf = sklearn_crfsuite.CRF(
            algorithm="lbfgs",
            c1=self.c1,
            c2=self.c2,
            max_iterations=self.max_iterations,
            all_possible_transitions=True,
        )
        self.crf.fit(X, y)
        print("  Training complete.")

    def predict(self, text):
        """
        Predict boundary positions in raw text.

        Returns:
            list of (position, confidence) tuples
        """
        if self.crf is None:
            raise RuntimeError("Model not trained. Call train() or load() first.")

        tokens = tokenize(text)
        if not tokens:
            return []

        features = [token_features(tokens, i) for i in range(len(tokens))]
        predicted_labels = self.crf.predict([features])[0]

        # Also get marginal probabilities for confidence
        try:
            marginals = self.crf.predict_marginals([features])[0]
        except Exception:
            marginals = None

        boundaries = []
        for i, label in enumerate(predicted_labels):
            if label == "B":
                pos = tokens[i][2]  # start position of the boundary token
                if marginals is not None:
                    conf = marginals[i].get("B", 0.5)
                else:
                    conf = 0.5
                boundaries.append((pos, conf))

        # Merge nearby predictions (within 15 chars)
        merged = []
        for pos, conf in boundaries:
            if merged and pos - merged[-1][0] < 15:
                # Keep higher confidence
                if conf > merged[-1][1]:
                    merged[-1] = (pos, conf)
            else:
                merged.append((pos, conf))

        return merged

    def predict_positions(self, text):
        """Convenience: return just positions."""
        return [pos for pos, conf in self.predict(text)]

    def save(self, filepath):
        """Save trained model to disk."""
        with open(filepath, "wb") as f:
            pickle.dump({
                "crf": self.crf,
                "c1": self.c1,
                "c2": self.c2,
                "boundary_label_radius": self.boundary_label_radius,
            }, f)
        print(f"Model saved to: {filepath}")

    def load(self, filepath):
        """Load a trained model from disk."""
        with open(filepath, "rb") as f:
            data = pickle.load(f)
        self.crf = data["crf"]
        self.c1 = data.get("c1", 0.1)
        self.c2 = data.get("c2", 0.1)
        self.boundary_label_radius = data.get("boundary_label_radius", 1)
        print(f"Model loaded from: {filepath}")


# ===========================================================================
# Cross-validation
# ===========================================================================

def cross_validate(annotated_strings, n_folds=5, c1=0.1, c2=0.1,
                   max_iterations=150, boundary_label_radius=1, tolerance=15):
    """
    Run k-fold cross-validation and report per-fold + overall metrics.
    """
    print(f"\n{'='*60}")
    print(f"CRF CROSS-VALIDATION ({n_folds} folds)")
    print(f"{'='*60}")
    print(f"  c1={c1}, c2={c2}, max_iter={max_iterations}, label_radius={boundary_label_radius}")
    print(f"  Position tolerance: ±{tolerance} chars")
    print()

    _sklearn_crfsuite, crf_metrics, KFold = _import_crf()
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    indices = list(range(len(annotated_strings)))

    fold_results = []

    for fold, (train_idx, test_idx) in enumerate(kf.split(indices)):
        print(f"--- Fold {fold + 1}/{n_folds} ---")
        train_strings = [annotated_strings[i] for i in train_idx]
        test_strings = [annotated_strings[i] for i in test_idx]

        detector = CRFBoundaryDetector(
            c1=c1, c2=c2,
            max_iterations=max_iterations,
            boundary_label_radius=boundary_label_radius,
        )
        detector.train(train_strings)

        # Evaluate on test fold
        all_tp = 0
        all_fp = 0
        all_fn = 0

        for annotated in test_strings:
            clean, true_positions = strip_boundaries(annotated)
            if not true_positions or not clean.strip():
                continue

            pred_positions = detector.predict_positions(clean)
            result = evaluate(pred_positions, true_positions, tolerance=tolerance)
            all_tp += result["true_positives"]
            all_fp += result["false_positives_count"]
            all_fn += result["false_negatives_count"]

        precision = all_tp / (all_tp + all_fp) if (all_tp + all_fp) > 0 else 0
        recall = all_tp / (all_tp + all_fn) if (all_tp + all_fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        print(f"  P={precision:.4f}  R={recall:.4f}  F1={f1:.4f}  "
              f"(TP={all_tp} FP={all_fp} FN={all_fn})")
        fold_results.append({"precision": precision, "recall": recall, "f1": f1})

    # Summary
    avg_p = sum(r["precision"] for r in fold_results) / n_folds
    avg_r = sum(r["recall"] for r in fold_results) / n_folds
    avg_f1 = sum(r["f1"] for r in fold_results) / n_folds

    print(f"\n{'='*60}")
    print(f"AVERAGE ACROSS {n_folds} FOLDS")
    print(f"{'='*60}")
    print(f"  Precision:  {avg_p:.4f}")
    print(f"  Recall:     {avg_r:.4f}")
    print(f"  F1 Score:   {avg_f1:.4f}")
    print()

    # Also report CRF token-level metrics for the last fold
    print("-" * 50)
    print("TOKEN-LEVEL METRICS (last fold)")
    print("-" * 50)
    X_test, y_test = prepare_training_data(test_strings, boundary_label_radius)
    y_pred = detector.crf.predict(X_test)
    labels = ["B", "O"]
    try:
        report = crf_metrics.flat_classification_report(y_test, y_pred, labels=labels)
        print(report)
    except Exception as e:
        print(f"  (Could not generate token report: {e})")

    return fold_results


# ===========================================================================
# Feature inspection
# ===========================================================================

def inspect_model(model_path, top_n=30):
    """Print the most informative features from a trained CRF model."""
    with open(model_path, "rb") as f:
        data = pickle.load(f)
    crf = data["crf"]

    print(f"\n{'='*60}")
    print("CRF MODEL — TOP FEATURE WEIGHTS")
    print(f"{'='*60}\n")

    # Get state features (feature → label → weight)
    state_features = crf.state_features_

    # Collect features that push toward "B" (boundary)
    b_features = {}
    o_features = {}
    for (feature, label), weight in state_features.items():
        if label == "B":
            b_features[feature] = weight
        elif label == "O":
            o_features[feature] = weight

    # Sort by absolute weight
    top_b_positive = sorted(b_features.items(), key=lambda x: x[1], reverse=True)[:top_n]
    top_b_negative = sorted(b_features.items(), key=lambda x: x[1])[:top_n]

    print(f"Features FAVORING boundary (B label), top {top_n}:")
    print("-" * 60)
    for feat, weight in top_b_positive:
        print(f"  {weight:+8.4f}  {feat}")

    print(f"\nFeatures AGAINST boundary (negative B weight), top {top_n}:")
    print("-" * 60)
    for feat, weight in top_b_negative:
        print(f"  {weight:+8.4f}  {feat}")

    # Transition features
    print(f"\nTransition weights:")
    print("-" * 60)
    for (from_label, to_label), weight in sorted(
        crf.transition_features_.items(), key=lambda x: x[1], reverse=True
    ):
        print(f"  {weight:+8.4f}  {from_label} → {to_label}")


# ===========================================================================
# Main
# ===========================================================================

def default_model_path():
    return models_dir() / "boundary_crf.pkl"


def default_eval_report_path(eval_file):
    stem = Path(eval_file).stem
    return reports_dir() / f"crf_eval_{stem}.md"


def run_crf_evaluation(detector, eval_strings, tolerance=15, eval_file=None,
                       report_path=None):
    """
    Evaluate a trained CRF detector on annotated strings (position-level metrics).
    """
    print(f"\n{'='*60}")
    print("CRF EVALUATION")
    print(f"{'='*60}")
    print(f"  Snippets: {len(eval_strings)}")
    print(f"  Position tolerance: ±{tolerance} chars")
    print()

    all_tp = 0
    all_fp = 0
    all_fn = 0
    evaluated = 0

    for annotated in eval_strings:
        clean, true_positions = strip_boundaries(annotated)
        if not true_positions or not clean.strip():
            continue
        pred_positions = detector.predict_positions(clean)
        result = evaluate(pred_positions, true_positions, tolerance=tolerance)
        all_tp += result["true_positives"]
        all_fp += result["false_positives_count"]
        all_fn += result["false_negatives_count"]
        evaluated += 1

    precision = all_tp / (all_tp + all_fp) if (all_tp + all_fp) > 0 else 0
    recall = all_tp / (all_tp + all_fn) if (all_tp + all_fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    print(f"  Evaluated snippets: {evaluated}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1:        {f1:.4f}")
    print(f"  TP={all_tp}  FP={all_fp}  FN={all_fn}")
    print()

    metrics = {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "true_positives": all_tp,
        "false_positives": all_fp,
        "false_negatives": all_fn,
        "evaluated_snippets": evaluated,
        "tolerance": tolerance,
    }

    if report_path is not None:
        report_path = ensure_report_dir(report_path)
        eval_label = eval_file or "eval set"
        lines = [
            "# CRF evaluation report",
            "",
            f"- **Eval file:** `{eval_label}`",
            f"- **Snippets evaluated:** {evaluated}",
            f"- **Tolerance:** ±{tolerance} chars",
            "",
            "| Metric | Value |",
            "|--------|-------|",
            f"| Precision | {precision:.4f} |",
            f"| Recall | {recall:.4f} |",
            f"| F1 | {f1:.4f} |",
            f"| TP | {all_tp} |",
            f"| FP | {all_fp} |",
            f"| FN | {all_fn} |",
            "",
        ]
        report_path.write_text("\n".join(lines), encoding="utf-8")
        print(f"Evaluation report saved to: {report_path.resolve()}")

    return metrics


def run_train(input_file, folds=0, save_model=None, c1=0.1, c2=0.1,
              max_iter=150, label_radius=1, tolerance=15,
              features_cache=None, eval_file=None, eval_report=None):
    strings = load_annotated(input_file)
    print(f"Loaded {len(strings)} annotated strings.\n")

    if folds > 1:
        cross_validate(
            strings,
            n_folds=folds,
            c1=c1,
            c2=c2,
            max_iterations=max_iter,
            boundary_label_radius=label_radius,
            tolerance=tolerance,
        )

    train_full = save_model is not None or eval_file is not None
    detector = None

    if train_full:
        print("\nTraining on FULL dataset...")
        detector = CRFBoundaryDetector(
            c1=c1,
            c2=c2,
            max_iterations=max_iter,
            boundary_label_radius=label_radius,
        )
        detector.train(
            strings,
            features_cache=features_cache,
            source_file=input_file,
        )
        if save_model is not None:
            save_path = ensure_report_dir(save_model)
            detector.save(str(save_path))

    if eval_file is not None:
        if detector is None:
            raise RuntimeError("--eval-file requires training or a loaded model")
        eval_strings = load_annotated(eval_file)
        print(f"\nLoaded {len(eval_strings)} eval strings from {eval_file}")
        report = eval_report
        if report == "":
            report = default_eval_report_path(eval_file)
        run_crf_evaluation(
            detector,
            eval_strings,
            tolerance=tolerance,
            eval_file=eval_file,
            report_path=report,
        )


def run_evaluate(eval_file, model, tolerance=15, report=None):
    """Load a saved model and evaluate on annotated eval snippets."""
    eval_strings = load_annotated(eval_file)
    print(f"Loaded {len(eval_strings)} eval strings from {eval_file}\n")

    detector = CRFBoundaryDetector()
    detector.load(model)

    report_path = report
    if report == "":
        report_path = default_eval_report_path(eval_file)

    return run_crf_evaluation(
        detector,
        eval_strings,
        tolerance=tolerance,
        eval_file=eval_file,
        report_path=report_path,
    )


def run_predict(input_file, model, output=None):
    if output is None:
        output = reports_dir() / "crf_predicted.txt"
    detector = CRFBoundaryDetector()
    detector.load(model)

    text = Path(input_file).read_text(encoding="utf-8")
    predictions = detector.predict(text)

    print(f"Found {len(predictions)} predicted boundaries.\n")
    for pos, conf in predictions:
        left = text[max(0, pos - 30):pos]
        right = text[pos:pos + 30]
        print(f"  pos={pos:>6}  conf={conf:.3f}  ...{repr(left[-20:])}|{repr(right[:20])}...")

    positions = [p for p, c in predictions]
    annotated = insert_boundaries(text, positions)
    out_path = ensure_report_dir(output)
    out_path.write_text(annotated, encoding="utf-8")
    print(f"\nAnnotated text saved to: {out_path}")


def run_inspect(model_file, top=30):
    inspect_model(model_file, top_n=top)
