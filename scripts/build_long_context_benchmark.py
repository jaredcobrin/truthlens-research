#!/usr/bin/env python3
"""
Build the TruthLens V1 long-context contradiction benchmark.

The builder uses the Hugging Face Dataset Viewer API directly, so it does not
require the `datasets` Python package. It creates conversation-shaped JSONL
examples with deterministic long context lengths and auditable injected labels.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any


HF_DATASET_SERVER = "https://datasets-server.huggingface.co"
DEFAULT_OUTPUT_DIR = Path("data/truthlens_long_context_v1")
SEED = 42
ASSISTANT_CHUNK_ESTIMATED_TOKENS = 10_000
ASSISTANT_CHUNK_CHARS = ASSISTANT_CHUNK_ESTIMATED_TOKENS * 4
CHUNK_MAX_TOKENS = 180_000
CHUNK_OVERLAP_TURNS = 10
SOURCE_POOL_CHARS = 10_000_000

CATEGORIES = [
    "clean_no_issue",
    "local_contradiction",
    "cross_chunk_contradiction",
    "far_distance_contradiction",
    "allowed_revision",
]

TOKEN_BANDS = [
    ("300k", 300_000),
    ("500k", 500_000),
    ("750k", 750_000),
    ("1m", 1_000_000),
]

EXAMPLES_PER_CATEGORY_PER_BAND = 5


@dataclass(frozen=True)
class SeedPair:
    source_dataset: str
    source_config: str
    source_split: str
    source_row_idx: int
    label_name: str
    premise: str
    hypothesis: str


def estimate_tokens(text: str) -> int:
    return max(1, len(text) // 4)


def conversation_tokens(conversation: list[dict[str, Any]]) -> int:
    return sum(estimate_tokens(turn["content"]) for turn in conversation)


def hf_get(endpoint: str, params: dict[str, Any], retries: int = 4) -> dict[str, Any]:
    query = urllib.parse.urlencode(params)
    url = f"{HF_DATASET_SERVER}/{endpoint}?{query}"
    last_error: Exception | None = None

    for attempt in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=90) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            last_error = exc
            if exc.code == 429:
                time.sleep(8 * (attempt + 1))
            else:
                time.sleep(1.5 * (attempt + 1))
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            last_error = exc
            time.sleep(1.5 * (attempt + 1))

    raise RuntimeError(f"Dataset Viewer request failed: {url}") from last_error


def fetch_rows(
    dataset: str,
    config: str,
    split: str,
    offset: int,
    length: int,
) -> list[dict[str, Any]]:
    data = hf_get(
        "rows",
        {
            "dataset": dataset,
            "config": config,
            "split": split,
            "offset": offset,
            "length": length,
        },
    )
    return data.get("rows", [])


def normalize_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def clean_seed_text(text: str, max_chars: int = 360) -> str:
    text = normalize_text(text)
    text = re.sub(r"\s+", " ", text)
    if len(text) <= max_chars:
        return text

    clipped = text[:max_chars].rsplit(" ", 1)[0].strip()
    return clipped.rstrip(".,;:") + "."


def fetch_pg19_source_pool(target_chars: int) -> tuple[str, list[dict[str, Any]]]:
    chunks: list[str] = []
    sources: list[dict[str, Any]] = []
    collected = 0

    for split, max_rows in [("test", 100), ("validation", 50), ("train", 250)]:
        for offset in range(max_rows):
            rows = fetch_rows("emozilla/pg19", "default", split, offset, 1)
            time.sleep(0.3)
            if not rows:
                break

            row = rows[0]
            item = row["row"]
            text = normalize_text(item.get("text", ""))
            if len(text) < 50_000:
                continue

            chunks.append(text)
            collected += len(text)
            sources.append(
                {
                    "dataset": "emozilla/pg19",
                    "config": "default",
                    "split": split,
                    "row_idx": row.get("row_idx"),
                    "title": item.get("short_book_title"),
                    "url": item.get("url"),
                    "chars": len(text),
                }
            )
            print(f"source pool: {collected:,}/{target_chars:,} chars", end="\r", flush=True)

            if collected >= target_chars:
                print()
                return "\n\n".join(chunks), sources

    raise RuntimeError(f"Only collected {collected:,} PG19 chars; need {target_chars:,}.")


def fetch_mnli_contradictions(limit: int) -> list[SeedPair]:
    seeds: list[SeedPair] = []
    splits = ["validation_matched", "validation_mismatched", "train"]

    for split in splits:
        offset = 0
        while len(seeds) < limit:
            rows = fetch_rows("nyu-mll/multi_nli", "default", split, offset, 100)
            if not rows:
                break

            for row in rows:
                item = row["row"]
                if item.get("label") != 2:
                    continue

                premise = clean_seed_text(item.get("premise", ""))
                hypothesis = clean_seed_text(item.get("hypothesis", ""))
                if len(premise) < 20 or len(hypothesis) < 20:
                    continue

                seeds.append(
                    SeedPair(
                        source_dataset="nyu-mll/multi_nli",
                        source_config="default",
                        source_split=split,
                        source_row_idx=int(row.get("row_idx", -1)),
                        label_name="contradiction",
                        premise=premise,
                        hypothesis=hypothesis,
                    )
                )

                if len(seeds) >= limit:
                    return seeds

            offset += 100

    if len(seeds) < limit:
        raise RuntimeError(f"Only found {len(seeds)} MultiNLI contradiction seeds.")
    return seeds


def fetch_contract_contradictions(limit: int) -> list[SeedPair]:
    seeds: list[SeedPair] = []
    configs = ["contractnli_a", "contractnli_b"]

    for config in configs:
        for split in ["validation", "test", "train"]:
            offset = 0
            while len(seeds) < limit:
                rows = fetch_rows("kiddothe2b/contract-nli", config, split, offset, 100)
                if not rows:
                    break

                for row in rows:
                    item = row["row"]
                    if item.get("label") != 0:
                        continue

                    premise = clean_seed_text(item.get("premise", ""), max_chars=520)
                    hypothesis = clean_seed_text(item.get("hypothesis", ""))
                    if len(premise) < 30 or len(hypothesis) < 20:
                        continue

                    seeds.append(
                        SeedPair(
                            source_dataset="kiddothe2b/contract-nli",
                            source_config=config,
                            source_split=split,
                            source_row_idx=int(row.get("row_idx", -1)),
                            label_name="contradiction",
                            premise=premise,
                            hypothesis=hypothesis,
                        )
                    )

                    if len(seeds) >= limit:
                        return seeds

                offset += 100

    if len(seeds) < limit:
        raise RuntimeError(f"Only found {len(seeds)} ContractNLI contradiction seeds.")
    return seeds


def seed_to_json(seed: SeedPair) -> dict[str, Any]:
    return {
        "source_dataset": seed.source_dataset,
        "source_config": seed.source_config,
        "source_split": seed.source_split,
        "source_row_idx": seed.source_row_idx,
        "label_name": seed.label_name,
        "premise": seed.premise,
        "hypothesis": seed.hypothesis,
    }


def seed_from_json(data: dict[str, Any]) -> SeedPair:
    return SeedPair(
        source_dataset=data["source_dataset"],
        source_config=data["source_config"],
        source_split=data["source_split"],
        source_row_idx=int(data["source_row_idx"]),
        label_name=data["label_name"],
        premise=data["premise"],
        hypothesis=data["hypothesis"],
    )


def load_or_fetch_source_pool(output_dir: Path, target_chars: int) -> tuple[str, list[dict[str, Any]]]:
    source_cache = output_dir / "_source_pool_cache.txt"
    metadata_cache = output_dir / "_source_pool_sources.json"

    if source_cache.exists() and metadata_cache.exists():
        text = source_cache.read_text(encoding="utf-8")
        if len(text) >= target_chars:
            sources = json.loads(metadata_cache.read_text(encoding="utf-8"))
            print(f"Loaded source pool cache: {len(text):,} chars")
            return text, sources

    text, sources = fetch_pg19_source_pool(target_chars)
    source_cache.write_text(text, encoding="utf-8")
    metadata_cache.write_text(json.dumps(sources, indent=2, ensure_ascii=False), encoding="utf-8")
    return text, sources


def load_or_fetch_seed_cache(
    output_dir: Path,
    name: str,
    limit: int,
    fetcher: Any,
) -> list[SeedPair]:
    cache = output_dir / f"_{name}_seed_cache.json"
    if cache.exists():
        data = json.loads(cache.read_text(encoding="utf-8"))
        if len(data) >= limit:
            print(f"Loaded {name} seed cache: {len(data)}")
            return [seed_from_json(item) for item in data[:limit]]

    seeds = fetcher(limit)
    cache.write_text(
        json.dumps([seed_to_json(seed) for seed in seeds], indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return seeds


def slice_pool(pool: str, start: int, length: int) -> str:
    if length <= 0:
        return ""

    start = start % len(pool)
    if start + length <= len(pool):
        return pool[start : start + length]

    parts: list[str] = []
    remaining = length
    cursor = start
    while remaining > 0:
        take = min(remaining, len(pool) - cursor)
        parts.append(pool[cursor : cursor + take])
        remaining -= take
        cursor = 0
    return "".join(parts)


def build_base_conversation(pool_text: str, start: int, target_tokens: int) -> list[dict[str, Any]]:
    target_chars = target_tokens * 4
    source_text = slice_pool(pool_text, start, target_chars + ASSISTANT_CHUNK_CHARS)
    cursor = 0
    turn = 1
    segment = 1
    conversation: list[dict[str, Any]] = []

    while conversation_tokens(conversation) < target_tokens:
        conversation.append(
            {
                "turn": turn,
                "role": "user",
                "content": f"Please continue the source document. Segment {segment}.",
            }
        )
        turn += 1

        chunk = source_text[cursor : cursor + ASSISTANT_CHUNK_CHARS].strip()
        if len(chunk) < ASSISTANT_CHUNK_CHARS // 2:
            cursor = 0
            chunk = source_text[cursor : cursor + ASSISTANT_CHUNK_CHARS].strip()

        conversation.append({"turn": turn, "role": "assistant", "content": chunk})
        turn += 1
        segment += 1
        cursor += ASSISTANT_CHUNK_CHARS

    return conversation


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


def assistant_turns(conversation: list[dict[str, Any]]) -> list[int]:
    return [turn["turn"] for turn in conversation if turn["role"] == "assistant"]


def turn_at_token_position(conversation: list[dict[str, Any]], token_position: int) -> int:
    running = 0
    fallback = assistant_turns(conversation)[0]
    for turn in conversation:
        running += estimate_tokens(turn["content"])
        if turn["role"] == "assistant":
            fallback = turn["turn"]
        if running >= token_position and turn["role"] == "assistant":
            return turn["turn"]
    return fallback


def turn_by_id(conversation: list[dict[str, Any]], turn_id: int) -> dict[str, Any]:
    for turn in conversation:
        if turn["turn"] == turn_id:
            return turn
    raise KeyError(f"Missing turn {turn_id}")


def inject(turn: dict[str, Any], text: str) -> None:
    turn["content"] = f"{text}\n\n{turn['content']}"


def choose_local_turns(conversation: list[dict[str, Any]]) -> tuple[int, int]:
    chunks = chunk_conversation(conversation)
    middle = chunks[len(chunks) // 2]
    assistants = [turn["turn"] for turn in middle if turn["role"] == "assistant"]
    if len(assistants) < 2:
        assistants = assistant_turns(conversation)
    index = max(0, len(assistants) // 2 - 1)
    return assistants[index], assistants[index + 1]


def choose_cross_chunk_turns(conversation: list[dict[str, Any]]) -> tuple[int, int]:
    chunks = chunk_conversation(conversation)
    if len(chunks) < 2:
        return choose_local_turns(conversation)

    pair_index = max(0, len(chunks) // 2 - 1)
    previous = chunks[pair_index]
    current = chunks[pair_index + 1]
    overlap_turn_ids = {turn["turn"] for turn in current[:CHUNK_OVERLAP_TURNS]}
    previous_unique_assistants = [
        turn["turn"]
        for turn in previous
        if turn["role"] == "assistant" and turn["turn"] not in overlap_turn_ids
    ]
    current_new_assistants = [
        turn["turn"] for turn in current[CHUNK_OVERLAP_TURNS:] if turn["role"] == "assistant"
    ]

    if not previous_unique_assistants or not current_new_assistants:
        return choose_local_turns(conversation)

    return previous_unique_assistants[-1], current_new_assistants[0]


def choose_far_turns(conversation: list[dict[str, Any]]) -> tuple[int, int]:
    total = conversation_tokens(conversation)
    evidence = turn_at_token_position(conversation, min(50_000, total // 5))
    target = turn_at_token_position(conversation, int(total * 0.86))
    if evidence == target:
        assistants = assistant_turns(conversation)
        evidence = assistants[1]
        target = assistants[-2]
    return evidence, target


def choose_seed(
    rng: random.Random,
    mnli_seeds: list[SeedPair],
    contract_seeds: list[SeedPair],
    index: int,
) -> SeedPair:
    if contract_seeds and index % 5 == 0:
        return contract_seeds[(index // 5) % len(contract_seeds)]
    return mnli_seeds[index % len(mnli_seeds)]


def seed_metadata(seed: SeedPair) -> dict[str, Any]:
    return {
        "source_dataset": seed.source_dataset,
        "source_config": seed.source_config,
        "source_split": seed.source_split,
        "source_row_idx": seed.source_row_idx,
        "label_name": seed.label_name,
    }


def token_distance_between_turns(conversation: list[dict[str, Any]], first: int, second: int) -> int:
    if first > second:
        first, second = second, first

    running = 0
    inside = False
    for turn in conversation:
        if turn["turn"] == first:
            inside = True
        elif turn["turn"] == second:
            break
        elif inside:
            running += estimate_tokens(turn["content"])
    return running


def chunk_ids_for_turn(conversation: list[dict[str, Any]], turn_id: int) -> list[int]:
    ids: list[int] = []
    for index, chunk in enumerate(chunk_conversation(conversation), start=1):
        if any(turn["turn"] == turn_id for turn in chunk):
            ids.append(index)
    return ids


def build_example(
    *,
    example_id: str,
    category: str,
    target_band: str,
    target_tokens: int,
    base_text_pool: str,
    base_start: int,
    seed: SeedPair | None,
) -> dict[str, Any]:
    conversation = build_base_conversation(base_text_pool, base_start, target_tokens)
    labels: dict[str, Any] = {
        "evidence_turns": [],
        "target_turn": None,
        "expected_quote": None,
        "source_seed": None,
        "new_evidence_turn": None,
    }

    expected_has_issue = category not in {"clean_no_issue", "allowed_revision"}
    expected_type = "CONTRADICTION" if expected_has_issue else None

    if category != "clean_no_issue":
        assert seed is not None

        if category == "local_contradiction":
            evidence_turn, target_turn = choose_local_turns(conversation)
        elif category == "cross_chunk_contradiction":
            evidence_turn, target_turn = choose_cross_chunk_turns(conversation)
        else:
            evidence_turn, target_turn = choose_far_turns(conversation)

        record_id = example_id.upper().replace("-", "_")
        evidence_text = f"Document record {record_id} initial statement: {seed.premise}"
        target_text = f"Document record {record_id} later statement: {seed.hypothesis}"

        if category == "allowed_revision":
            user_turn = target_turn - 1
            if turn_by_id(conversation, user_turn)["role"] != "user":
                user_turn = max(1, target_turn - 1)
            user_text = (
                f"New evidence for document record {record_id}: the corrected source says: "
                f"{seed.hypothesis}"
            )
            target_text = (
                f"Document record {record_id} corrected statement after the new evidence: "
                f"{seed.hypothesis}"
            )
            inject(turn_by_id(conversation, evidence_turn), evidence_text)
            inject(turn_by_id(conversation, user_turn), user_text)
            inject(turn_by_id(conversation, target_turn), target_text)
            labels["new_evidence_turn"] = user_turn
        else:
            inject(turn_by_id(conversation, evidence_turn), evidence_text)
            inject(turn_by_id(conversation, target_turn), target_text)

        labels.update(
            {
                "evidence_turns": [evidence_turn],
                "target_turn": target_turn,
                "expected_quote": seed.hypothesis,
                "source_seed": seed_metadata(seed),
                "injected_evidence_text": evidence_text,
                "injected_target_text": target_text,
            }
        )

    estimated_tokens = conversation_tokens(conversation)
    evidence_turns = labels["evidence_turns"]
    target_turn = labels["target_turn"]
    distance = (
        token_distance_between_turns(conversation, evidence_turns[0], target_turn)
        if evidence_turns and target_turn
        else None
    )

    return {
        "id": example_id,
        "category": category,
        "expected_has_issue": expected_has_issue,
        "expected_type": expected_type,
        "conversation": conversation,
        "labels": labels,
        "metadata": {
            "target_band": target_band,
            "target_estimated_tokens": target_tokens,
            "estimated_tokens": estimated_tokens,
            "distance_tokens": distance,
            "base_dataset": "emozilla/pg19",
            "base_source_offset_chars": base_start,
            "assistant_chunk_estimated_tokens": ASSISTANT_CHUNK_ESTIMATED_TOKENS,
            "chunk_max_tokens": CHUNK_MAX_TOKENS,
            "chunk_overlap_turns": CHUNK_OVERLAP_TURNS,
            "chunks_with_evidence": (
                chunk_ids_for_turn(conversation, evidence_turns[0]) if evidence_turns else []
            ),
            "chunks_with_target": (
                chunk_ids_for_turn(conversation, target_turn) if target_turn else []
            ),
            "generation_seed": SEED,
        },
    }


def count_occurrences(conversation: list[dict[str, Any]], needle: str) -> int:
    if not needle:
        return 0
    return sum(turn["content"].count(needle) for turn in conversation)


def validate_example(example: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    conversation = example["conversation"]
    turns = [turn["turn"] for turn in conversation]

    if turns != list(range(1, len(conversation) + 1)):
        errors.append("turn numbers are not contiguous")

    estimated = conversation_tokens(conversation)
    target = example["metadata"]["target_estimated_tokens"]
    if not (target <= estimated <= int(target * 1.04) + 2_000):
        errors.append(f"estimated tokens {estimated} outside expected range for target {target}")

    labels = example["labels"]
    category = example["category"]
    evidence_turns = labels.get("evidence_turns") or []
    target_turn = labels.get("target_turn")

    if category == "clean_no_issue":
        if evidence_turns or target_turn:
            errors.append("clean example should not have evidence/target labels")
        return errors

    for key in ["injected_evidence_text", "injected_target_text"]:
        text = labels.get(key)
        if count_occurrences(conversation, text) != 1:
            errors.append(f"{key} does not appear exactly once")

    if not evidence_turns or target_turn is None:
        errors.append("non-clean example missing evidence or target turn")
        return errors

    try:
        evidence = turn_by_id(conversation, evidence_turns[0])
        target_item = turn_by_id(conversation, target_turn)
    except KeyError as exc:
        errors.append(str(exc))
        return errors

    if evidence["role"] != "assistant":
        errors.append("evidence turn is not assistant")
    if target_item["role"] != "assistant":
        errors.append("target turn is not assistant")

    evidence_chunks = set(example["metadata"]["chunks_with_evidence"])
    target_chunks = set(example["metadata"]["chunks_with_target"])
    same_chunk = bool(evidence_chunks & target_chunks)

    if category == "local_contradiction" and not same_chunk:
        errors.append("local contradiction is not visible inside one evaluation chunk")
    if category in {"cross_chunk_contradiction", "far_distance_contradiction"} and same_chunk:
        errors.append("cross/far contradiction is visible inside one evaluation chunk")
    if category == "far_distance_contradiction":
        distance = example["metadata"].get("distance_tokens") or 0
        target = example["metadata"]["target_estimated_tokens"]
        minimum_distance = min(250_000, int(target * 0.55))
        if distance < minimum_distance:
            errors.append(
                f"far-distance contradiction too close: {distance} < {minimum_distance}"
            )
    if category == "allowed_revision":
        new_evidence_turn = labels.get("new_evidence_turn")
        if not new_evidence_turn:
            errors.append("allowed revision missing new evidence turn")
        elif turn_by_id(conversation, new_evidence_turn)["role"] != "user":
            errors.append("allowed revision new evidence turn is not user")
        elif not (evidence_turns[0] < new_evidence_turn < target_turn):
            errors.append("allowed revision new evidence is not between evidence and target")

    return errors


def write_jsonl(path: Path, examples: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for example in examples:
            handle.write(json.dumps(example, ensure_ascii=False, separators=(",", ":")))
            handle.write("\n")


def write_readme(path: Path) -> None:
    path.write_text(
        """# TruthLens Long-Context Benchmark V1

This folder contains a generated Pilot 100 benchmark for testing TruthLens
contradiction detection above a 256k-token context window.

## Files

- `examples.jsonl`: one complete conversation-shaped benchmark example per line.
- `manifest.json`: generation settings, source datasets, and validation summary.

## Categories

- `clean_no_issue`
- `local_contradiction`
- `cross_chunk_contradiction`
- `far_distance_contradiction`
- `allowed_revision`

## Validate

```bash
python3 scripts/validate_long_context_benchmark.py data/truthlens_long_context_v1/examples.jsonl
```

## Score Predictions

Create a predictions JSONL file with at least:

```json
{"id":"tlc_v1_local_contradiction_300k_000","predicted_has_issue":true,"predicted_type":"CONTRADICTION","predicted_target_turn":42}
```

Then run:

```bash
python3 scripts/score_long_context_predictions.py data/truthlens_long_context_v1/examples.jsonl predictions.jsonl
```
""",
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--source-pool-chars", type=int, default=SOURCE_POOL_CHARS)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Fetching base source text from emozilla/pg19...")
    base_pool, base_sources = load_or_fetch_source_pool(output_dir, args.source_pool_chars)

    print("Fetching contradiction seeds from MultiNLI...")
    mnli_seeds = load_or_fetch_seed_cache(output_dir, "mnli", 140, fetch_mnli_contradictions)
    print(f"MultiNLI seeds: {len(mnli_seeds)}")

    print("Fetching contradiction seeds from ContractNLI...")
    contract_seeds = load_or_fetch_seed_cache(output_dir, "contractnli", 40, fetch_contract_contradictions)
    print(f"ContractNLI seeds: {len(contract_seeds)}")

    examples: list[dict[str, Any]] = []
    seed_index = 0
    for category in CATEGORIES:
        for target_band, target_tokens in TOKEN_BANDS:
            for number in range(EXAMPLES_PER_CATEGORY_PER_BAND):
                example_id = f"tlc_v1_{category}_{target_band}_{number:03d}"
                base_start = rng.randrange(0, max(1, len(base_pool) - 1))
                seed = None
                if category != "clean_no_issue":
                    seed = choose_seed(rng, mnli_seeds, contract_seeds, seed_index)
                    seed_index += 1

                example = build_example(
                    example_id=example_id,
                    category=category,
                    target_band=target_band,
                    target_tokens=target_tokens,
                    base_text_pool=base_pool,
                    base_start=base_start,
                    seed=seed,
                )

                errors = validate_example(example)
                if errors:
                    raise RuntimeError(f"{example_id} failed validation: {errors}")

                examples.append(example)
                print(f"built {len(examples):3d}/100 {example_id}")

    examples_path = output_dir / "examples.jsonl"
    manifest_path = output_dir / "manifest.json"
    readme_path = output_dir / "README.md"

    write_jsonl(examples_path, examples)
    write_readme(readme_path)

    category_counts = {category: 0 for category in CATEGORIES}
    band_counts = {band: 0 for band, _ in TOKEN_BANDS}
    for example in examples:
        category_counts[example["category"]] += 1
        band_counts[example["metadata"]["target_band"]] += 1

    manifest = {
        "name": "truthlens_long_context_v1",
        "version": "pilot_100",
        "generated_at_unix": int(time.time()),
        "generation_seed": args.seed,
        "example_count": len(examples),
        "category_counts": category_counts,
        "target_band_counts": band_counts,
        "token_estimate": "len(text) // 4 per turn",
        "chunking_assumptions": {
            "chunk_max_tokens": CHUNK_MAX_TOKENS,
            "chunk_overlap_turns": CHUNK_OVERLAP_TURNS,
            "assistant_chunk_estimated_tokens": ASSISTANT_CHUNK_ESTIMATED_TOKENS,
        },
        "source_datasets": [
            "emozilla/pg19",
            "nyu-mll/multi_nli",
            "kiddothe2b/contract-nli",
        ],
        "base_sources_used": base_sources,
        "files": {
            "examples": str(examples_path),
            "readme": str(readme_path),
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {examples_path}")
    print(f"Wrote {manifest_path}")
    print(f"Wrote {readme_path}")


if __name__ == "__main__":
    main()
