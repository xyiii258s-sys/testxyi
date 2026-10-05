"""A small, runnable machine-unlearning demonstration.

The exact unlearning baseline is implemented by removing the requested records
and retraining a fresh model on the retained records.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import numpy as np
from sklearn.datasets import load_digits
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


@dataclass
class Evaluation:
    accuracy: float
    count: int
    mean_confidence: float


def build_model():
    """Create the same deterministic model for both training runs."""

    return make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=1000, random_state=42),
    )


def evaluate(model, x: np.ndarray, y: np.ndarray) -> Evaluation:
    """Return accuracy and confidence on a dataset slice."""

    if len(y) == 0:
        return Evaluation(accuracy=float("nan"), count=0, mean_confidence=float("nan"))

    predictions = model.predict(x)
    probabilities = model.predict_proba(x)
    confidence = probabilities.max(axis=1)
    return Evaluation(
        accuracy=float(accuracy_score(y, predictions)),
        count=len(y),
        mean_confidence=float(confidence.mean()),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Demonstrate exact machine unlearning by retraining.")
    parser.add_argument(
        "--forget-label",
        type=int,
        nargs="+",
        default=[3],
        help="One or more digit labels to remove from the training data (default: 3).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    labels_to_forget = set(args.forget_label)

    if not labels_to_forget.issubset(set(range(10))):
        raise SystemExit("--forget-label must contain digits from 0 to 9")

    digits = load_digits()
    x_train, x_test, y_train, y_test = train_test_split(
        digits.data,
        digits.target,
        test_size=0.25,
        random_state=42,
        stratify=digits.target,
    )

    # Model before unlearning: trained on every record.
    original_model = build_model()
    original_model.fit(x_train, y_train)
    original_test = evaluate(original_model, x_test, y_test)

    forget_train_mask = np.isin(y_train, list(labels_to_forget))
    retain_train_mask = ~forget_train_mask
    forget_test_mask = np.isin(y_test, list(labels_to_forget))
    retain_test_mask = ~forget_test_mask

    # Exact baseline: train a new model without the records to be forgotten.
    unlearned_model = build_model()
    unlearned_model.fit(x_train[retain_train_mask], y_train[retain_train_mask])
    retain_test = evaluate(unlearned_model, x_test[retain_test_mask], y_test[retain_test_mask])
    forget_test = evaluate(unlearned_model, x_test[forget_test_mask], y_test[forget_test_mask])

    labels_text = ", ".join(str(label) for label in sorted(labels_to_forget))
    print("Machine unlearning demo")
    print("=" * 24)
    print(f"Labels to forget: {labels_text}")
    print(f"Original train records: {len(y_train)}")
    print(f"Removed train records: {forget_train_mask.sum()}")
    print(f"Retained train records: {retain_train_mask.sum()}")
    print()
    print(f"Before unlearning - all test accuracy: {original_test.accuracy:.4f}")
    print(f"After unlearning  - retained test accuracy: {retain_test.accuracy:.4f}")
    print(f"After unlearning  - forgotten test accuracy: {forget_test.accuracy:.4f}")
    print(f"Forgotten test samples: {forget_test.count}")
    print(f"Forgotten test mean confidence: {forget_test.mean_confidence:.4f}")


if __name__ == "__main__":
    main()
