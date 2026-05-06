#!/usr/bin/env python3
"""Write a detailed confusion analysis for TruthLens Auditor Eval V2/V2.1."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from score_truthlens_auditor_eval_v2 import (  # noqa: E402
    PRIMARY_FAILURES,
    SECONDARY_TAGS,
    load_examples,
    load_predictions,
    normalized_label_failure,
    normalized_label_tags,
    prediction_issues,
)


def pct(numerator: int, denominator: int) -> float:
    return round(100.0 * numerator / denominator, 2) if denominator else 0.0


def f1_from_counts(true_positive: int, false_positive: int, false_negative: int) -> float:
    precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
    recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
    return round(100.0 * 2 * precision * recall / (precision + recall), 2) if precision + recall else 0.0


def choose_prediction_for_label(
    label: dict[str, Any],
    issues: list[dict[str, Any]],
    used_indices: set[int],
) -> tuple[str, int | None]:
    expected_failure = normalized_label_failure(label)
    for index, issue in enumerate(issues):
        if index in used_indices:
            continue
        if issue["primary_failure"] == expected_failure and issue["turn"] == label["target_turn"]:
            return "exact", index
    for index, issue in enumerate(issues):
        if index in used_indices:
            continue
        if issue["turn"] == label["target_turn"]:
            return "turn_only", index
    for index, issue in enumerate(issues):
        if index in used_indices:
            continue
        if issue["primary_failure"] == expected_failure:
            return "type_only", index
    return "miss", None


def build_analysis(
    examples: dict[str, dict[str, Any]],
    predictions: dict[str, dict[str, Any]],
    min_confidence_score: int,
) -> dict[str, Any]:
    prediction_items = {
        example_id: prediction_issues(row, min_confidence_score=min_confidence_score)
        for example_id, row in predictions.items()
    }

    per_failure: dict[str, dict[str, Any]] = {
        failure: {
            "expected_positive": 0,
            "trap_negative": 0,
            "true_positive": 0,
            "false_negative": 0,
            "false_positive": 0,
            "true_negative": 0,
            "trap_false_positive": 0,
            "turn_only_wrong_primary_failure": 0,
            "type_only_wrong_turn": 0,
            "wrong_primary_failure_predictions": Counter(),
            "false_positive_predictions": [],
            "trap_false_positive_predictions": [],
            "wrong_turn_predictions": [],
            "miss_examples": [],
        }
        for failure in sorted(PRIMARY_FAILURES)
    }
    primary_confusion = {
        failure: {predicted: 0 for predicted in sorted(PRIMARY_FAILURES)}
        for failure in sorted(PRIMARY_FAILURES)
    }
    secondary_report = {
        tag: {"expected": 0, "matched": 0, "predicted": 0, "extra": 0}
        for tag in sorted(SECONDARY_TAGS)
    }
    confidence_buckets = {
        value: {"predicted": 0, "true_positive": 0, "false_positive": 0, "trap_false_positive": 0}
        for value in range(1, 11)
    }

    turn_only_examples: list[dict[str, Any]] = []
    unmatched_predictions: list[dict[str, Any]] = []
    all_used_indices: dict[str, set[int]] = defaultdict(set)

    for example_id, example in examples.items():
        issues = prediction_items.get(example_id, [])
        used_indices = all_used_indices[example_id]

        trap_labels_by_failure: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for label in example["labels"]:
            expected_failure = normalized_label_failure(label)
            if label["expected_positive"]:
                per_failure[expected_failure]["expected_positive"] += 1
                for tag in normalized_label_tags(label):
                    secondary_report[tag]["expected"] += 1
                match_type, index = choose_prediction_for_label(label, issues, used_indices)
                if match_type == "exact" and index is not None:
                    issue = issues[index]
                    used_indices.add(index)
                    per_failure[expected_failure]["true_positive"] += 1
                    expected_tags = set(normalized_label_tags(label))
                    predicted_tags = set(issue["secondary_tags"])
                    for tag in expected_tags & predicted_tags:
                        secondary_report[tag]["matched"] += 1
                    for tag in predicted_tags - expected_tags:
                        secondary_report[tag]["extra"] += 1
                    confidence_buckets[issue["confidence_score"]]["true_positive"] += 1
                elif match_type == "turn_only" and index is not None:
                    issue = issues[index]
                    used_indices.add(index)
                    per_failure[expected_failure]["false_negative"] += 1
                    per_failure[expected_failure]["turn_only_wrong_primary_failure"] += 1
                    predicted_failure = issue["primary_failure"]
                    per_failure[expected_failure]["wrong_primary_failure_predictions"][predicted_failure] += 1
                    primary_confusion[expected_failure][predicted_failure] += 1
                    turn_only_examples.append(
                        {
                            "example_id": example_id,
                            "issue_id": label["issue_id"],
                            "expected_primary_failure": expected_failure,
                            "predicted_primary_failure": predicted_failure,
                            "target_turn": label["target_turn"],
                            "confidence_score": issue["confidence_score"],
                            "quote": issue["quote"][:220],
                        }
                    )
                elif match_type == "type_only" and index is not None:
                    issue = issues[index]
                    used_indices.add(index)
                    per_failure[expected_failure]["false_negative"] += 1
                    per_failure[expected_failure]["type_only_wrong_turn"] += 1
                    per_failure[expected_failure]["wrong_turn_predictions"].append(
                        {
                            "example_id": example_id,
                            "issue_id": label["issue_id"],
                            "expected_primary_failure": expected_failure,
                            "target_turn": label["target_turn"],
                            "predicted_turn": issue["turn"],
                            "confidence_score": issue["confidence_score"],
                            "quote": issue["quote"][:220],
                        }
                    )
                else:
                    per_failure[expected_failure]["false_negative"] += 1
                    per_failure[expected_failure]["miss_examples"].append(
                        {
                            "example_id": example_id,
                            "issue_id": label["issue_id"],
                            "expected_primary_failure": expected_failure,
                            "target_turn": label["target_turn"],
                            "evidence_turns": label["evidence_turns"],
                        }
                    )
            else:
                per_failure[expected_failure]["trap_negative"] += 1
                trap_labels_by_failure[expected_failure].append(label)

        for failure, trap_labels in trap_labels_by_failure.items():
            matched = False
            for trap in trap_labels:
                for index, issue in enumerate(issues):
                    if index in used_indices:
                        continue
                    if issue["primary_failure"] == failure and issue["turn"] == trap["target_turn"]:
                        matched = True
                        break
                if matched:
                    break
            if matched:
                per_failure[failure]["trap_false_positive"] += 1
            else:
                per_failure[failure]["true_negative"] += 1

        trap_lookup = {
            (normalized_label_failure(label), int(label["target_turn"])): label
            for label in example["labels"]
            if not label["expected_positive"]
        }
        for index, issue in enumerate(issues):
            for tag in issue["secondary_tags"]:
                secondary_report[tag]["predicted"] += 1
            confidence_buckets[issue["confidence_score"]]["predicted"] += 1
            if index in used_indices:
                continue
            failure = issue["primary_failure"]
            per_failure[failure]["false_positive"] += 1
            confidence_buckets[issue["confidence_score"]]["false_positive"] += 1
            record = {
                "example_id": example_id,
                "predicted_primary_failure": failure,
                "secondary_tags": issue["secondary_tags"],
                "confidence_score": issue["confidence_score"],
                "predicted_turn": issue["turn"],
                "quote": issue["quote"][:220],
                "trap_match": False,
            }
            trap_label = trap_lookup.get((failure, issue["turn"]))
            if trap_label is not None:
                record["trap_match"] = True
                record["trap_issue_id"] = trap_label["issue_id"]
                confidence_buckets[issue["confidence_score"]]["trap_false_positive"] += 1
                per_failure[failure]["trap_false_positive_predictions"].append(record)
            else:
                per_failure[failure]["false_positive_predictions"].append(record)
            unmatched_predictions.append(record)

    summary = {
        "examples_in_split": len(examples),
        "predicted_examples": len(predictions),
        "min_confidence_score": min_confidence_score,
        "exact_true_positives": sum(item["true_positive"] for item in per_failure.values()),
        "false_negatives": sum(item["false_negative"] for item in per_failure.values()),
        "false_positives": sum(item["false_positive"] for item in per_failure.values()),
        "turn_only_wrong_primary_failure": sum(item["turn_only_wrong_primary_failure"] for item in per_failure.values()),
        "type_only_wrong_turn": sum(item["type_only_wrong_turn"] for item in per_failure.values()),
        "trap_true_negatives": sum(item["true_negative"] for item in per_failure.values()),
        "trap_false_positives": sum(item["trap_false_positive"] for item in per_failure.values()),
    }
    summary["strict_precision_percent"] = pct(summary["exact_true_positives"], summary["exact_true_positives"] + summary["false_positives"])
    summary["strict_recall_percent"] = pct(summary["exact_true_positives"], summary["exact_true_positives"] + summary["false_negatives"])
    summary["strict_f1_percent"] = f1_from_counts(
        summary["exact_true_positives"],
        summary["false_positives"],
        summary["false_negatives"],
    )

    failure_rows = {}
    for failure, values in per_failure.items():
        failure_rows[failure] = {
            "expected_positive": values["expected_positive"],
            "trap_negative": values["trap_negative"],
            "true_positive": values["true_positive"],
            "false_negative": values["false_negative"],
            "false_positive": values["false_positive"],
            "true_negative": values["true_negative"],
            "trap_false_positive": values["trap_false_positive"],
            "precision_percent": pct(values["true_positive"], values["true_positive"] + values["false_positive"]),
            "recall_percent": pct(values["true_positive"], values["expected_positive"]),
            "f1_percent": f1_from_counts(values["true_positive"], values["false_positive"], values["false_negative"]),
            "turn_only_wrong_primary_failure": values["turn_only_wrong_primary_failure"],
            "type_only_wrong_turn": values["type_only_wrong_turn"],
            "wrong_primary_failure_predictions": dict(values["wrong_primary_failure_predictions"].most_common()),
            "false_positive_examples": values["false_positive_predictions"][:25],
            "trap_false_positive_examples": values["trap_false_positive_predictions"][:25],
            "wrong_turn_examples": values["wrong_turn_predictions"][:25],
            "miss_examples": values["miss_examples"][:25],
        }

    return {
        "summary": summary,
        "by_primary_failure": failure_rows,
        "primary_failure_confusion_turn_only": primary_confusion,
        "secondary_tags": secondary_report,
        "confidence_buckets": confidence_buckets,
        "turn_only_wrong_primary_failure_examples": turn_only_examples[:200],
        "unmatched_predictions": unmatched_predictions[:400],
    }


def write_markdown(path: Path, analysis: dict[str, Any], split: str) -> None:
    summary = analysis["summary"]
    lines = [
        f"# TruthLens Auditor Eval Detailed Analysis ({split})",
        "",
        "## Summary",
        "",
        f"- Examples in split: `{summary['examples_in_split']}`",
        f"- Predicted examples: `{summary['predicted_examples']}`",
        f"- Min confidence score: `{summary['min_confidence_score']}`",
        f"- Exact true positives: `{summary['exact_true_positives']}`",
        f"- False negatives: `{summary['false_negatives']}`",
        f"- False positives: `{summary['false_positives']}`",
        f"- Turn-only wrong-primary matches: `{summary['turn_only_wrong_primary_failure']}`",
        f"- Type-only wrong-turn matches: `{summary['type_only_wrong_turn']}`",
        f"- Trap true negatives: `{summary['trap_true_negatives']}`",
        f"- Trap false positives: `{summary['trap_false_positives']}`",
        f"- Strict precision: `{summary['strict_precision_percent']}%`",
        f"- Strict recall: `{summary['strict_recall_percent']}%`",
        f"- Strict F1: `{summary['strict_f1_percent']}%`",
        "",
        "## Primary Failure Table",
        "",
        "| Primary failure | TP | FP | TN (traps) | FN | Turn-only wrong primary | Type-only wrong turn | Precision | Recall | F1 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for failure, row in analysis["by_primary_failure"].items():
        lines.append(
            "| "
            + " | ".join(
                [
                    failure,
                    str(row["true_positive"]),
                    str(row["false_positive"]),
                    str(row["true_negative"]),
                    str(row["false_negative"]),
                    str(row["turn_only_wrong_primary_failure"]),
                    str(row["type_only_wrong_turn"]),
                    f"{row['precision_percent']}%",
                    f"{row['recall_percent']}%",
                    f"{row['f1_percent']}%",
                ]
            )
            + " |"
        )

    lines.extend(
        [
            "",
            "## Turn-Only Wrong-Primary Confusion",
            "",
            "Rows are the expected primary failure. Values are how Gemma labeled the same target turn instead.",
            "",
            "```json",
            json.dumps(analysis["primary_failure_confusion_turn_only"], indent=2),
            "```",
            "",
            "## Notes",
            "",
            "- `TN (traps)` means category-specific trap examples that the model correctly left unflagged.",
            "- Confidence buckets and example-level details are in the JSON output.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_dir", type=Path)
    parser.add_argument("predictions", type=Path)
    parser.add_argument("--split", choices=["unit", "packed_long", "packed"], required=True)
    parser.add_argument("--completed-only", action="store_true")
    parser.add_argument("--min-confidence-score", type=int, default=1)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    examples = load_examples(args.dataset_dir, args.split)
    predictions = load_predictions(args.predictions)
    if args.completed_only:
        examples = {example_id: row for example_id, row in examples.items() if example_id in predictions}
    analysis = build_analysis(examples, predictions, args.min_confidence_score)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(analysis, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_markdown(args.output_md, analysis, args.split)
    print(f"Wrote {args.output_json}")
    print(f"Wrote {args.output_md}")


if __name__ == "__main__":
    main()
