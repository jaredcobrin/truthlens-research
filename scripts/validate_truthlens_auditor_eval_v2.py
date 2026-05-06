#!/usr/bin/env python3
"""Validate the TruthLens Auditor Eval V2 dataset."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from build_truthlens_auditor_eval_v2 import (  # noqa: E402
    CHUNK_MAX_TOKENS,
    CHUNK_OVERLAP_TURNS,
    LEGACY_CATEGORIES,
    PRIMARY_FAILURES,
    SECONDARY_TAGS,
    chunk_conversation,
    chunk_ids_for_turn,
    conversation_tokens,
    relation_for_turns,
)


REQUIRED_UNIT_POSITIVES = 10
REQUIRED_UNIT_TRAPS = 5
REQUIRED_PACKED_POSITIVES = 10
REQUIRED_PACKED_TRAPS = 5
PACKED_MIN_TOKENS = 220_000
PACKED_MAX_TOKENS = 320_000


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def turn_map(conversation: list[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    return {int(turn["turn"]): turn for turn in conversation}


def count_occurrences(conversation: list[dict[str, Any]], needle: str) -> int:
    return sum(turn["content"].count(needle) for turn in conversation)


def visible_text(conversation: list[dict[str, Any]]) -> str:
    return "\n".join(turn["content"] for turn in conversation)


def validate_label(example: dict[str, Any], label: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    conversation = example["conversation"]
    turns = turn_map(conversation)

    primary_failure = label.get("primary_failure") or label.get("category")
    if primary_failure not in PRIMARY_FAILURES:
        errors.append(f"invalid primary_failure {primary_failure!r}")

    legacy_category = label.get("legacy_category")
    if legacy_category not in LEGACY_CATEGORIES:
        errors.append(f"invalid legacy_category {legacy_category!r}")

    secondary_tags = label.get("secondary_tags")
    if not isinstance(secondary_tags, list):
        errors.append("secondary_tags must be a list")
    else:
        for tag in secondary_tags:
            if tag not in SECONDARY_TAGS:
                errors.append(f"invalid secondary tag {tag!r}")

    issue_id = str(label.get("issue_id", ""))
    if not issue_id:
        errors.append("missing issue_id")

    target_turn = label.get("target_turn")
    evidence_turns = label.get("evidence_turns")
    if not isinstance(target_turn, int) or target_turn not in turns:
        errors.append(f"invalid target_turn {target_turn!r}")
        return errors
    if not isinstance(evidence_turns, list) or not evidence_turns:
        errors.append("missing evidence_turns")
        return errors
    if any(not isinstance(turn, int) or turn not in turns for turn in evidence_turns):
        errors.append(f"invalid evidence_turns {evidence_turns!r}")
        return errors

    if turns[target_turn]["role"] != "assistant":
        errors.append("target_turn is not an assistant turn")

    target_quote = label.get("target_quote")
    if not isinstance(target_quote, str) or not target_quote.strip():
        errors.append("missing target_quote")
    elif target_quote not in turns[target_turn]["content"]:
        errors.append("target_quote is not in target_turn content")
    elif count_occurrences(conversation, target_quote) != 1:
        errors.append("target_quote does not appear exactly once")

    evidence_quotes = label.get("evidence_quotes")
    if not isinstance(evidence_quotes, list) or not evidence_quotes:
        errors.append("missing evidence_quotes")
    else:
        evidence_blob = "\n".join(turns[turn]["content"] for turn in evidence_turns)
        for quote in evidence_quotes:
            if not isinstance(quote, str) or not quote.strip():
                errors.append("empty evidence_quote")
            elif quote not in evidence_blob:
                errors.append(f"evidence_quote not found in evidence turns: {quote[:80]!r}")
            elif count_occurrences(conversation, quote) != 1:
                errors.append(f"evidence_quote does not appear exactly once: {quote[:80]!r}")

    expected_relation = label.get("chunk_relation")
    actual_relation = relation_for_turns(conversation, evidence_turns, target_turn)
    if expected_relation != actual_relation:
        errors.append(f"chunk_relation mismatch: expected {expected_relation}, actual {actual_relation}")

    if label.get("expected_positive") is False and not label.get("trap_kind"):
        errors.append("false-positive trap is missing trap_kind")
    if label.get("expected_positive") not in {True, False}:
        errors.append("expected_positive must be boolean")

    if not isinstance(label.get("source_dataset"), str) or not label["source_dataset"]:
        errors.append("missing source_dataset")

    text = visible_text(conversation)
    forbidden_fragments = [
        issue_id,
        "expected_positive",
        "target_quote",
        "evidence_turns",
        "tla_v2_",
        *PRIMARY_FAILURES,
        *SECONDARY_TAGS,
        *LEGACY_CATEGORIES,
    ]
    for fragment in forbidden_fragments:
        if fragment and fragment in text:
            errors.append(f"visible label leakage: {fragment}")
            break

    chunks_with_target = chunk_ids_for_turn(conversation, target_turn)
    if not chunks_with_target:
        errors.append("target_turn is not in any chunk")

    return errors


def validate_example(example: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    example_id = example.get("id", "<missing id>")
    split = example.get("split")
    if split not in {"unit", "packed_long"}:
        errors.append(f"{example_id}: invalid split {split!r}")

    conversation = example.get("conversation")
    if not isinstance(conversation, list) or not conversation:
        return [f"{example_id}: missing conversation"]

    expected_turns = list(range(1, len(conversation) + 1))
    actual_turns = [turn.get("turn") for turn in conversation]
    if actual_turns != expected_turns:
        errors.append(f"{example_id}: turn numbers are not contiguous")

    for turn in conversation:
        if turn.get("role") not in {"user", "assistant"}:
            errors.append(f"{example_id}: invalid role at turn {turn.get('turn')}")
        if not isinstance(turn.get("content"), str) or not turn["content"].strip():
            errors.append(f"{example_id}: empty content at turn {turn.get('turn')}")

    estimated_tokens = conversation_tokens(conversation)
    metadata_tokens = example.get("metadata", {}).get("estimated_tokens")
    if metadata_tokens != estimated_tokens:
        errors.append(f"{example_id}: metadata token count {metadata_tokens} != {estimated_tokens}")

    if split == "packed_long":
        if not (PACKED_MIN_TOKENS <= estimated_tokens <= PACKED_MAX_TOKENS):
            errors.append(f"{example_id}: packed token count out of range: {estimated_tokens}")
        chunk_count = len(chunk_conversation(conversation, CHUNK_MAX_TOKENS, CHUNK_OVERLAP_TURNS))
        if chunk_count < 10:
            errors.append(f"{example_id}: expected at least 10 chunks, got {chunk_count}")
        if example.get("metadata", {}).get("chunk_count") != chunk_count:
            errors.append(f"{example_id}: metadata chunk_count mismatch")

    labels = example.get("labels")
    if not isinstance(labels, list) or not labels:
        errors.append(f"{example_id}: missing labels")
        return errors
    if split == "unit" and len(labels) != 1:
        errors.append(f"{example_id}: unit examples must have exactly one label")
    if split == "packed_long" and len(labels) < 10:
        errors.append(f"{example_id}: packed examples should contain many labels")

    seen_issue_ids: set[str] = set()
    for label in labels:
        if label.get("issue_id") in seen_issue_ids:
            errors.append(f"{example_id}: duplicate issue_id {label.get('issue_id')}")
        seen_issue_ids.add(label.get("issue_id"))
        for error in validate_label(example, label):
            errors.append(f"{example_id}/{label.get('issue_id')}: {error}")

    return errors


def summarize_counts(examples: list[dict[str, Any]]) -> dict[str, Any]:
    counts_by_primary = {
        category: {
            "unit_positive": 0,
            "unit_trap": 0,
            "packed_positive": 0,
            "packed_trap": 0,
        }
        for category in PRIMARY_FAILURES
    }
    counts_by_legacy = {
        category: {
            "unit_positive": 0,
            "unit_trap": 0,
            "packed_positive": 0,
            "packed_trap": 0,
        }
        for category in LEGACY_CATEGORIES
    }
    relations: dict[str, int] = {}
    for example in examples:
        split = example["split"]
        for label in example["labels"]:
            key = f"{'unit' if split == 'unit' else 'packed'}_{'positive' if label['expected_positive'] else 'trap'}"
            counts_by_primary[label["primary_failure"]][key] += 1
            counts_by_legacy[label["legacy_category"]][key] += 1
            relations[label["chunk_relation"]] = relations.get(label["chunk_relation"], 0) + 1
    return {
        "counts_by_primary_failure": counts_by_primary,
        "counts_by_legacy_category": counts_by_legacy,
        "relations": relations,
    }


def validate_counts(examples: list[dict[str, Any]]) -> list[str]:
    errors: list[str] = []
    summary = summarize_counts(examples)
    for category, values in summary["counts_by_primary_failure"].items():
        if values["unit_positive"] < REQUIRED_UNIT_POSITIVES:
            errors.append(f"{category}: primary unit positives {values['unit_positive']} < {REQUIRED_UNIT_POSITIVES}")
        if values["unit_trap"] < REQUIRED_UNIT_TRAPS:
            errors.append(f"{category}: primary unit traps {values['unit_trap']} < {REQUIRED_UNIT_TRAPS}")
        if values["packed_positive"] < REQUIRED_PACKED_POSITIVES:
            errors.append(f"{category}: primary packed positives {values['packed_positive']} < {REQUIRED_PACKED_POSITIVES}")
        if values["packed_trap"] < REQUIRED_PACKED_TRAPS:
            errors.append(f"{category}: primary packed traps {values['packed_trap']} < {REQUIRED_PACKED_TRAPS}")

    for category, values in summary["counts_by_legacy_category"].items():
        if values["unit_positive"] < REQUIRED_UNIT_POSITIVES:
            errors.append(f"{category}: legacy unit positives {values['unit_positive']} < {REQUIRED_UNIT_POSITIVES}")
        if values["unit_trap"] < REQUIRED_UNIT_TRAPS:
            errors.append(f"{category}: legacy unit traps {values['unit_trap']} < {REQUIRED_UNIT_TRAPS}")
        if values["packed_positive"] < REQUIRED_PACKED_POSITIVES:
            errors.append(f"{category}: legacy packed positives {values['packed_positive']} < {REQUIRED_PACKED_POSITIVES}")
        if values["packed_trap"] < REQUIRED_PACKED_TRAPS:
            errors.append(f"{category}: legacy packed traps {values['packed_trap']} < {REQUIRED_PACKED_TRAPS}")

    relations = summary["relations"]
    for relation in ["same_chunk", "adjacent_cross_chunk", "far_cross_chunk"]:
        if relations.get(relation, 0) == 0:
            errors.append(f"missing chunk relation coverage: {relation}")
    return errors


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_dir", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    unit_path = args.dataset_dir / "unit_examples.jsonl"
    packed_path = args.dataset_dir / "packed_long_examples.jsonl"
    manifest_path = args.dataset_dir / "label_manifest.json"

    examples = load_jsonl(unit_path) + load_jsonl(packed_path)
    failures: list[str] = []
    for example in examples:
        failures.extend(validate_example(example))
    failures.extend(validate_counts(examples))

    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest_total = manifest.get("example_counts", {}).get("total")
        if manifest_total != len(examples):
            failures.append(f"manifest total {manifest_total} != loaded total {len(examples)}")
    else:
        failures.append("missing label_manifest.json")

    summary = summarize_counts(examples)
    print(f"examples: {len(examples)}")
    print("counts_by_primary_failure:")
    print(json.dumps(summary["counts_by_primary_failure"], indent=2, sort_keys=True))
    print("counts_by_legacy_category:")
    print(json.dumps(summary["counts_by_legacy_category"], indent=2, sort_keys=True))
    print("chunk_relations:")
    print(json.dumps(summary["relations"], indent=2, sort_keys=True))

    if failures:
        print("validation: FAIL")
        for failure in failures[:80]:
            print(f"- {failure}")
        raise SystemExit(1)

    print("validation: PASS")


if __name__ == "__main__":
    main()
