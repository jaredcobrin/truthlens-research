#!/usr/bin/env python3
"""Validate TruthLens long-context benchmark JSONL files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


CHUNK_MAX_TOKENS = 180_000
CHUNK_OVERLAP_TURNS = 10


def estimate_tokens(text: str) -> int:
    return max(1, len(text) // 4)


def conversation_tokens(conversation: list[dict[str, Any]]) -> int:
    return sum(estimate_tokens(turn["content"]) for turn in conversation)


def chunk_conversation(
    conversation: list[dict[str, Any]],
    max_tokens: int = CHUNK_MAX_TOKENS,
    overlap_turns: int = CHUNK_OVERLAP_TURNS,
) -> list[list[dict[str, Any]]]:
    chunks: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    current_tokens = 0

    for turn in conversation:
        tokens = estimate_tokens(turn["content"])
        if current and current_tokens + tokens > max_tokens:
            chunks.append(current)
            current = current[-overlap_turns:]
            current_tokens = sum(estimate_tokens(t["content"]) for t in current)

        current.append(turn)
        current_tokens += tokens

    if current:
        chunks.append(current)

    return chunks


def turn_by_id(conversation: list[dict[str, Any]], turn_id: int) -> dict[str, Any]:
    for turn in conversation:
        if turn["turn"] == turn_id:
            return turn
    raise KeyError(f"Missing turn {turn_id}")


def chunk_ids_for_turn(conversation: list[dict[str, Any]], turn_id: int) -> list[int]:
    ids: list[int] = []
    for index, chunk in enumerate(chunk_conversation(conversation), start=1):
        if any(turn["turn"] == turn_id for turn in chunk):
            ids.append(index)
    return ids


def count_occurrences(conversation: list[dict[str, Any]], needle: str | None) -> int:
    if not needle:
        return 0
    return sum(turn["content"].count(needle) for turn in conversation)


def validate_example(example: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    conversation = example.get("conversation", [])
    labels = example.get("labels", {})
    metadata = example.get("metadata", {})
    category = example.get("category")
    turns = [turn.get("turn") for turn in conversation]

    if turns != list(range(1, len(conversation) + 1)):
        errors.append("turn numbers are not contiguous")

    for index, turn in enumerate(conversation, start=1):
        if turn.get("turn") != index:
            errors.append(f"turn {index} has wrong turn number")
        if turn.get("role") not in {"user", "assistant"}:
            errors.append(f"turn {index} has invalid role")
        if not isinstance(turn.get("content"), str) or not turn["content"].strip():
            errors.append(f"turn {index} has empty content")

    estimated = conversation_tokens(conversation)
    target = metadata.get("target_estimated_tokens")
    if not isinstance(target, int):
        errors.append("missing integer target_estimated_tokens")
    elif not (target <= estimated <= int(target * 1.04) + 2_000):
        errors.append(f"estimated tokens {estimated} outside expected range for target {target}")

    expected_has_issue = example.get("expected_has_issue")
    if category == "clean_no_issue" and expected_has_issue:
        errors.append("clean example should not expect an issue")
    if category in {"local_contradiction", "cross_chunk_contradiction", "far_distance_contradiction"}:
        if expected_has_issue is not True:
            errors.append(f"{category} should expect an issue")
        if example.get("expected_type") != "CONTRADICTION":
            errors.append(f"{category} should expect CONTRADICTION")
    if category == "allowed_revision" and expected_has_issue:
        errors.append("allowed revision should not expect an issue")

    evidence_turns = labels.get("evidence_turns") or []
    target_turn = labels.get("target_turn")

    if category == "clean_no_issue":
        if evidence_turns or target_turn:
            errors.append("clean example has evidence/target labels")
        return errors

    if count_occurrences(conversation, labels.get("injected_evidence_text")) != 1:
        errors.append("injected evidence text does not appear exactly once")
    if count_occurrences(conversation, labels.get("injected_target_text")) != 1:
        errors.append("injected target text does not appear exactly once")

    if not evidence_turns or target_turn is None:
        errors.append("missing evidence or target turn")
        return errors

    try:
        evidence = turn_by_id(conversation, evidence_turns[0])
        target_item = turn_by_id(conversation, target_turn)
    except KeyError as exc:
        errors.append(str(exc))
        return errors

    if evidence.get("role") != "assistant":
        errors.append("evidence turn is not assistant")
    if target_item.get("role") != "assistant":
        errors.append("target turn is not assistant")

    evidence_chunks = set(chunk_ids_for_turn(conversation, evidence_turns[0]))
    target_chunks = set(chunk_ids_for_turn(conversation, target_turn))
    same_chunk = bool(evidence_chunks & target_chunks)

    if category == "local_contradiction" and not same_chunk:
        errors.append("local contradiction is not visible in one evaluation chunk")
    if category in {"cross_chunk_contradiction", "far_distance_contradiction"} and same_chunk:
        errors.append("cross/far contradiction is visible in one evaluation chunk")
    if category == "far_distance_contradiction":
        distance = metadata.get("distance_tokens") or 0
        target = metadata.get("target_estimated_tokens") or 0
        minimum_distance = min(250_000, int(target * 0.55))
        if distance < minimum_distance:
            errors.append(
                f"far-distance contradiction is too close: {distance} < {minimum_distance}"
            )

    if category == "allowed_revision":
        new_evidence_turn = labels.get("new_evidence_turn")
        if not new_evidence_turn:
            errors.append("allowed revision missing new evidence turn")
        else:
            try:
                new_evidence = turn_by_id(conversation, new_evidence_turn)
            except KeyError as exc:
                errors.append(str(exc))
            else:
                if new_evidence.get("role") != "user":
                    errors.append("allowed revision new evidence turn is not user")
                if not (evidence_turns[0] < new_evidence_turn < target_turn):
                    errors.append("allowed revision new evidence is not between evidence and target")

    return errors


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("examples_jsonl", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    count = 0
    failures: list[tuple[str, list[str]]] = []
    category_counts: dict[str, int] = {}
    band_counts: dict[str, int] = {}

    with args.examples_jsonl.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            example = json.loads(line)
            count += 1
            category = example.get("category", "unknown")
            band = example.get("metadata", {}).get("target_band", "unknown")
            category_counts[category] = category_counts.get(category, 0) + 1
            band_counts[band] = band_counts.get(band, 0) + 1
            errors = validate_example(example)
            if errors:
                failures.append((example.get("id", f"row_{count}"), errors))

    print(f"examples: {count}")
    print(f"categories: {json.dumps(category_counts, sort_keys=True)}")
    print(f"target_bands: {json.dumps(band_counts, sort_keys=True)}")

    if failures:
        print("validation: FAIL")
        for example_id, errors in failures[:20]:
            print(f"- {example_id}: {errors}")
        raise SystemExit(1)

    print("validation: PASS")


if __name__ == "__main__":
    main()
