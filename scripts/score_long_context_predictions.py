#!/usr/bin/env python3
"""Score TruthLens long-context benchmark predictions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def load_jsonl_by_id(path: Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            rows[row["id"]] = row
    return rows


def safe_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.lower() in {"true", "1", "yes", "issue", "issues_found"}
    return bool(value)


def pct(numerator: int, denominator: int) -> float:
    return round(100.0 * numerator / denominator, 2) if denominator else 0.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("examples_jsonl", type=Path)
    parser.add_argument("predictions_jsonl", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    examples = load_jsonl_by_id(args.examples_jsonl)
    predictions = load_jsonl_by_id(args.predictions_jsonl)

    true_positive = 0
    false_positive = 0
    true_negative = 0
    false_negative = 0
    type_correct = 0
    target_turn_correct = 0
    issue_count = 0
    predicted_issue_count = 0
    by_category: dict[str, dict[str, int]] = {}
    by_band: dict[str, dict[str, int]] = {}
    missing_predictions: list[str] = []

    for example_id, example in examples.items():
        prediction = predictions.get(example_id)
        if prediction is None:
            missing_predictions.append(example_id)
            continue

        expected_issue = bool(example["expected_has_issue"])
        predicted_issue = safe_bool(prediction.get("predicted_has_issue", False))
        category = example["category"]
        band = example["metadata"]["target_band"]

        by_category.setdefault(category, {"total": 0, "correct_issue": 0})
        by_band.setdefault(band, {"total": 0, "correct_issue": 0})
        by_category[category]["total"] += 1
        by_band[band]["total"] += 1

        if expected_issue:
            issue_count += 1
        if predicted_issue:
            predicted_issue_count += 1

        if expected_issue and predicted_issue:
            true_positive += 1
            by_category[category]["correct_issue"] += 1
            by_band[band]["correct_issue"] += 1
            if prediction.get("predicted_type") == example.get("expected_type"):
                type_correct += 1
            expected_turn = example.get("labels", {}).get("target_turn")
            if prediction.get("predicted_target_turn") == expected_turn:
                target_turn_correct += 1
        elif expected_issue and not predicted_issue:
            false_negative += 1
        elif not expected_issue and predicted_issue:
            false_positive += 1
        else:
            true_negative += 1
            by_category[category]["correct_issue"] += 1
            by_band[band]["correct_issue"] += 1

    scored = len(examples) - len(missing_predictions)
    precision = pct(true_positive, true_positive + false_positive)
    recall = pct(true_positive, true_positive + false_negative)
    accuracy = pct(true_positive + true_negative, scored)
    false_positive_rate = pct(false_positive, false_positive + true_negative)

    report = {
        "examples": len(examples),
        "scored": scored,
        "missing_predictions": len(missing_predictions),
        "expected_issue_count": issue_count,
        "predicted_issue_count": predicted_issue_count,
        "accuracy_percent": accuracy,
        "precision_percent": precision,
        "recall_percent": recall,
        "false_positive_rate_percent": false_positive_rate,
        "type_accuracy_on_true_positives_percent": pct(type_correct, true_positive),
        "target_turn_accuracy_on_true_positives_percent": pct(target_turn_correct, true_positive),
        "confusion": {
            "true_positive": true_positive,
            "false_positive": false_positive,
            "true_negative": true_negative,
            "false_negative": false_negative,
        },
        "by_category": {
            key: {
                **value,
                "issue_accuracy_percent": pct(value["correct_issue"], value["total"]),
            }
            for key, value in sorted(by_category.items())
        },
        "by_target_band": {
            key: {
                **value,
                "issue_accuracy_percent": pct(value["correct_issue"], value["total"]),
            }
            for key, value in sorted(by_band.items())
        },
    }

    print(json.dumps(report, indent=2, sort_keys=True))
    if missing_predictions:
        print("Missing prediction ids:")
        for example_id in missing_predictions[:20]:
            print(f"- {example_id}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
