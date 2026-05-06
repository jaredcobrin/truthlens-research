#!/usr/bin/env python3
"""Score TruthLens Auditor Eval V2/V2.1 predictions at issue level."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from truthlens.reporting import (  # noqa: E402
    LEGACY_TO_PRIMARY_FAILURE,
    PRIMARY_FAILURES,
    SECONDARY_TAGS,
    normalize_confidence_score,
    normalize_primary_failure,
    normalize_secondary_tag,
    normalize_turn,
)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def load_examples(dataset_dir: Path, split: str | None) -> dict[str, dict[str, Any]]:
    paths: list[Path] = []
    if split in {None, "unit"}:
        paths.append(dataset_dir / "unit_examples.jsonl")
    if split in {None, "packed_long", "packed"}:
        paths.append(dataset_dir / "packed_long_examples.jsonl")

    examples: dict[str, dict[str, Any]] = {}
    for path in paths:
        for row in load_jsonl(path):
            examples[row["id"]] = row
    return examples


def load_predictions(path: Path) -> dict[str, dict[str, Any]]:
    return {row["id"]: row for row in load_jsonl(path)}


def normalized_label_failure(label: dict[str, Any]) -> str:
    primary = normalize_primary_failure(label.get("primary_failure") or label.get("category"))
    if primary:
        return primary
    return LEGACY_TO_PRIMARY_FAILURE.get(str(label.get("legacy_category") or label.get("category") or ""), "")


def normalized_label_tags(label: dict[str, Any]) -> list[str]:
    raw_tags = label.get("secondary_tags") or []
    if isinstance(raw_tags, str):
        raw_tags = [raw_tags]
    tags: list[str] = []
    if isinstance(raw_tags, list):
        for item in raw_tags:
            tag = normalize_secondary_tag(item)
            if tag and tag not in tags:
                tags.append(tag)
    return tags


def prediction_issues(row: dict[str, Any], min_confidence_score: int = 1) -> list[dict[str, Any]]:
    raw_issues = (
        row.get("issues")
        or row.get("final_issues")
        or row.get("findings")
        or row.get("final_findings")
        or []
    )
    if not isinstance(raw_issues, list):
        return []

    normalized: list[dict[str, Any]] = []
    for issue in raw_issues:
        if not isinstance(issue, dict):
            continue
        turn = normalize_turn(issue.get("turn") or issue.get("target_turn"))
        primary_failure = normalize_primary_failure(
            issue.get("primary_failure") or issue.get("type") or issue.get("category")
        )
        confidence_score = normalize_confidence_score(
            issue.get("confidence_score", issue.get("confidence", 8))
        )
        if primary_failure not in PRIMARY_FAILURES or turn is None:
            continue
        if confidence_score < min_confidence_score:
            continue
        raw_tags = issue.get("secondary_tags") or issue.get("tags") or []
        if isinstance(raw_tags, str):
            raw_tags = [raw_tags]
        tags: list[str] = []
        if isinstance(raw_tags, list):
            for item in raw_tags:
                tag = normalize_secondary_tag(item)
                if tag and tag not in tags:
                    tags.append(tag)
        evidence_turns: list[int] = []
        raw_evidence = issue.get("evidence") or issue.get("evidence_turns") or []
        if isinstance(raw_evidence, list):
            for item in raw_evidence:
                evidence_turn = normalize_turn(item.get("turn")) if isinstance(item, dict) else normalize_turn(item)
                if evidence_turn is not None:
                    evidence_turns.append(evidence_turn)
        normalized.append(
            {
                "primary_failure": primary_failure,
                "secondary_tags": tags,
                "confidence_score": confidence_score,
                "turn": turn,
                "evidence_turns": sorted(set(evidence_turns)),
                "quote": str(issue.get("quote", issue.get("problematic_quote", ""))).strip(),
                "raw": issue,
            }
        )
    return dedupe_issues(normalized)


def dedupe_issues(issues: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[tuple[str, int, str]] = set()
    deduped: list[dict[str, Any]] = []
    for issue in issues:
        key = (issue["primary_failure"], issue["turn"], issue["quote"][:80])
        if key in seen:
            continue
        seen.add(key)
        deduped.append(issue)
    return deduped


def pct(numerator: int, denominator: int) -> float:
    return round(100.0 * numerator / denominator, 2) if denominator else 0.0


def f1(precision: float, recall: float) -> float:
    return round(2 * precision * recall / (precision + recall), 2) if precision + recall else 0.0


def score(
    examples: dict[str, dict[str, Any]],
    predictions: dict[str, dict[str, Any]],
    min_confidence_score: int = 1,
) -> dict[str, Any]:
    expected_positive: list[tuple[str, dict[str, Any]]] = []
    expected_traps: list[tuple[str, dict[str, Any]]] = []
    for example_id, example in examples.items():
        for label in example["labels"]:
            if label["expected_positive"]:
                expected_positive.append((example_id, label))
            else:
                expected_traps.append((example_id, label))

    prediction_items = {
        example_id: prediction_issues(row, min_confidence_score=min_confidence_score)
        for example_id, row in predictions.items()
    }

    by_failure = {
        failure: {
            "expected_positive": 0,
            "true_positive": 0,
            "false_negative": 0,
            "false_positive": 0,
            "trap_count": 0,
            "trap_false_positive": 0,
        }
        for failure in sorted(PRIMARY_FAILURES)
    }
    by_relation: dict[str, dict[str, int]] = {}
    secondary = {
        tag: {"expected": 0, "matched": 0, "predicted": 0, "extra": 0}
        for tag in sorted(SECONDARY_TAGS)
    }
    confidence_buckets = {
        score_value: {"predicted": 0, "true_positive": 0, "false_positive": 0, "trap_false_positive": 0}
        for score_value in range(1, 11)
    }

    matched_labels: set[tuple[str, str]] = set()
    matched_predictions: set[tuple[str, int]] = set()
    target_turn_correct = 0
    evidence_turn_exact = 0
    turn_only_matches = 0
    turn_only_prediction_indices: set[tuple[str, int]] = set()

    for example_id, label in expected_positive:
        primary_failure = normalized_label_failure(label)
        by_failure[primary_failure]["expected_positive"] += 1
        relation = label["chunk_relation"]
        by_relation.setdefault(relation, {"expected_positive": 0, "true_positive": 0})
        by_relation[relation]["expected_positive"] += 1
        for tag in normalized_label_tags(label):
            secondary[tag]["expected"] += 1

        issues = prediction_items.get(example_id, [])
        exact_index = None
        turn_only_index = None
        for index, issue in enumerate(issues):
            if issue["turn"] == label["target_turn"] and turn_only_index is None:
                turn_only_index = index
            if issue["primary_failure"] == primary_failure and issue["turn"] == label["target_turn"]:
                exact_index = index
                break

        if turn_only_index is not None:
            turn_only_matches += 1
            turn_only_prediction_indices.add((example_id, turn_only_index))

        if exact_index is None:
            by_failure[primary_failure]["false_negative"] += 1
            continue

        issue = issues[exact_index]
        matched_labels.add((example_id, label["issue_id"]))
        matched_predictions.add((example_id, exact_index))
        by_failure[primary_failure]["true_positive"] += 1
        by_relation[relation]["true_positive"] += 1
        target_turn_correct += 1
        if set(label["evidence_turns"]).issubset(set(issue["evidence_turns"])):
            evidence_turn_exact += 1
        expected_tags = set(normalized_label_tags(label))
        predicted_tags = set(issue["secondary_tags"])
        for tag in expected_tags & predicted_tags:
            secondary[tag]["matched"] += 1
        for tag in predicted_tags - expected_tags:
            secondary[tag]["extra"] += 1

    trap_lookup: dict[str, list[dict[str, Any]]] = {}
    for example_id, label in expected_traps:
        trap_lookup.setdefault(example_id, []).append(label)
        by_failure[normalized_label_failure(label)]["trap_count"] += 1

    false_positive = 0
    trap_false_positive = 0
    unmatched_predictions: list[dict[str, Any]] = []
    for example_id, issues in prediction_items.items():
        if example_id not in examples:
            continue
        for index, issue in enumerate(issues):
            bucket = confidence_buckets[issue["confidence_score"]]
            bucket["predicted"] += 1
            for tag in issue["secondary_tags"]:
                secondary[tag]["predicted"] += 1
            if (example_id, index) in matched_predictions:
                bucket["true_positive"] += 1
                continue
            false_positive += 1
            bucket["false_positive"] += 1
            by_failure[issue["primary_failure"]]["false_positive"] += 1
            matched_trap = next(
                (
                    label
                    for label in trap_lookup.get(example_id, [])
                    if normalized_label_failure(label) == issue["primary_failure"]
                    and label["target_turn"] == issue["turn"]
                ),
                None,
            )
            if matched_trap:
                trap_false_positive += 1
                bucket["trap_false_positive"] += 1
                by_failure[issue["primary_failure"]]["trap_false_positive"] += 1
            unmatched_predictions.append(
                {
                    "example_id": example_id,
                    "primary_failure": issue["primary_failure"],
                    "secondary_tags": issue["secondary_tags"],
                    "confidence_score": issue["confidence_score"],
                    "turn": issue["turn"],
                    "quote": issue["quote"][:220],
                    "trap_match": bool(matched_trap),
                }
            )

    true_positive = len(matched_labels)
    false_negative = len(expected_positive) - true_positive
    precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
    recall = true_positive / len(expected_positive) if expected_positive else 0.0

    total_predicted_issues = sum(
        len(issues) for example_id, issues in prediction_items.items() if example_id in examples
    )
    turn_only_false_positive = max(0, total_predicted_issues - len(turn_only_prediction_indices))
    turn_only_precision = (
        turn_only_matches / (turn_only_matches + turn_only_false_positive)
        if turn_only_matches + turn_only_false_positive
        else 0.0
    )
    turn_only_recall = turn_only_matches / len(expected_positive) if expected_positive else 0.0

    failure_report: dict[str, dict[str, Any]] = {}
    for failure, values in by_failure.items():
        cat_precision = (
            values["true_positive"] / (values["true_positive"] + values["false_positive"])
            if values["true_positive"] + values["false_positive"]
            else 0.0
        )
        cat_recall = (
            values["true_positive"] / values["expected_positive"]
            if values["expected_positive"]
            else 0.0
        )
        failure_report[failure] = {
            **values,
            "precision_percent": pct(values["true_positive"], values["true_positive"] + values["false_positive"]),
            "recall_percent": pct(values["true_positive"], values["expected_positive"]),
            "f1_percent": f1(cat_precision * 100, cat_recall * 100),
            "trap_false_positive_rate_percent": pct(values["trap_false_positive"], values["trap_count"]),
        }

    relation_report = {
        relation: {**values, "recall_percent": pct(values["true_positive"], values["expected_positive"])}
        for relation, values in sorted(by_relation.items())
    }
    secondary_report = {
        tag: {
            **values,
            "recall_percent": pct(values["matched"], values["expected"]),
            "extra_rate_percent": pct(values["extra"], values["predicted"]),
        }
        for tag, values in secondary.items()
    }
    missing_predictions = sorted(set(examples) - set(predictions))
    return {
        "examples": len(examples),
        "predicted_examples": len(set(predictions) & set(examples)),
        "missing_predictions": len(missing_predictions),
        "min_confidence_score": min_confidence_score,
        "expected_positive_issues": len(expected_positive),
        "expected_traps": len(expected_traps),
        "predicted_issues": sum(len(issues) for example_id, issues in prediction_items.items() if example_id in examples),
        "precision_percent": round(precision * 100, 2),
        "recall_percent": round(recall * 100, 2),
        "f1_percent": f1(precision * 100, recall * 100),
        "target_turn_accuracy_on_true_positives_percent": pct(target_turn_correct, true_positive),
        "evidence_turn_exact_on_true_positives_percent": pct(evidence_turn_exact, true_positive),
        "turn_only_issue_detection": {
            "true_positive": turn_only_matches,
            "false_positive": turn_only_false_positive,
            "false_negative": len(expected_positive) - turn_only_matches,
            "precision_percent": round(turn_only_precision * 100, 2),
            "recall_percent": round(turn_only_recall * 100, 2),
            "f1_percent": f1(turn_only_precision * 100, turn_only_recall * 100),
        },
        "trap_false_positive_rate_percent": pct(trap_false_positive, len(expected_traps)),
        "confusion": {
            "true_positive": true_positive,
            "false_positive": false_positive,
            "false_negative": false_negative,
            "trap_false_positive": trap_false_positive,
        },
        "by_primary_failure": failure_report,
        "by_chunk_relation": relation_report,
        "secondary_tags": secondary_report,
        "confidence_buckets": confidence_buckets,
        "missing_prediction_ids": missing_predictions[:50],
        "unmatched_predictions_sample": unmatched_predictions[:50],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_dir", type=Path)
    parser.add_argument("predictions_jsonl", type=Path)
    parser.add_argument("--split", choices=["unit", "packed_long", "packed"])
    parser.add_argument("--min-confidence-score", type=int, default=1)
    parser.add_argument(
        "--only-predicted",
        action="store_true",
        help="Score only examples that appear in the predictions file. Useful for smoke runs.",
    )
    parser.add_argument(
        "--allow-missing",
        action="store_true",
        help="Do not exit nonzero when predictions are missing.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    examples = load_examples(args.dataset_dir, args.split)
    predictions = load_predictions(args.predictions_jsonl)
    if args.only_predicted:
        examples = {example_id: example for example_id, example in examples.items() if example_id in predictions}
    report = score(examples, predictions, min_confidence_score=args.min_confidence_score)
    print(json.dumps(report, indent=2, sort_keys=True))
    if report["missing_predictions"] and not args.allow_missing:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
