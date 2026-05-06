#!/usr/bin/env python3
"""Run TruthLens Auditor Eval V2/V2.1 through the hybrid audit-memory pipeline."""

from __future__ import annotations

import argparse
import getpass
import json
import os
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from truthlens.claim_extraction import HeuristicClaimExtractor, LLMClaimExtractor  # noqa: E402
from truthlens.llm import OpenAICompatibleChatClient  # noqa: E402
from truthlens.pipeline import AuditMemoryPipeline, LLMChunkAuditor  # noqa: E402
from truthlens.retrieval import ClaimRetriever, RetrievalConfig  # noqa: E402


DEFAULT_EXAMPLES = Path("data/truthlens_auditor_eval_v2_1/unit_examples.jsonl")
DEFAULT_OUTPUT_DIR = Path("results/truthlens_memory_audit_eval_v2_1")
DEFAULT_CLAIM_PROMPT = Path("prompts/truthlens_claim_extractor_v1.md")
DEFAULT_AUDITOR_PROMPT = Path("prompts/truthlens_chunk_auditor_with_memory_v1.md")
DEFAULT_BASE_URL = "https://api.novita.ai/openai/v1"
DEFAULT_MODEL = "google/gemma-4-31b-it"
CHUNK_MAX_TOKENS = 12_000
CHUNK_OVERLAP_TURNS = 4


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def load_existing_predictions(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return {row["id"] for row in load_jsonl(path)}


def resolve_api_key(args: argparse.Namespace) -> str:
    if args.api_key_stdin:
        return getpass.getpass("API key: ").strip()
    if args.api_key_plain_stdin:
        return input().strip()
    return (
        os.environ.get("NOVITA_API_KEY")
        or os.environ.get("GOOGLE_AI_STUDIO_KEY")
        or os.environ.get("GOOGLE_API_KEY")
        or os.environ.get("OPENAI_API_KEY")
        or ""
    )


def build_pipeline(args: argparse.Namespace, api_key: str) -> AuditMemoryPipeline:
    client = OpenAICompatibleChatClient(
        api_key=api_key,
        base_url=args.base_url,
        model=args.model,
        timeout=args.timeout,
        retries=args.retries,
    )
    claim_extractor = (
        HeuristicClaimExtractor()
        if args.heuristic_claims
        else LLMClaimExtractor(
            client=client,
            prompt_file=args.claim_prompt_file,
            temperature=args.claim_temperature,
            max_output_tokens=args.claim_max_output_tokens,
            response_format_json=args.response_format_json,
        )
    )
    auditor = LLMChunkAuditor(
        client=client,
        prompt_file=args.auditor_prompt_file,
        temperature=args.temperature,
        max_output_tokens=args.max_output_tokens,
        response_format_json=args.response_format_json,
        max_prompt_tokens=args.max_prompt_tokens,
        min_confidence_score=args.min_confidence_score,
    )

    retriever = ClaimRetriever(
        config=RetrievalConfig(
            top_k=args.retrieval_top_k,
            min_score=args.retrieval_min_score,
        )
    )
    return AuditMemoryPipeline(
        claim_extractor=claim_extractor,
        auditor=auditor,
        retriever=retriever,
        min_confidence_score=args.min_confidence_score,
        strict_issue_gates=args.strict_issue_gates,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--examples", type=Path, default=DEFAULT_EXAMPLES)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--claim-prompt-file", type=Path, default=DEFAULT_CLAIM_PROMPT)
    parser.add_argument("--auditor-prompt-file", type=Path, default=DEFAULT_AUDITOR_PROMPT)
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--id", action="append", dest="ids")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--heuristic-claims", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--claim-temperature", type=float, default=0.0)
    parser.add_argument("--max-output-tokens", type=int, default=4096)
    parser.add_argument("--claim-max-output-tokens", type=int, default=4096)
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--retries", type=int, default=6)
    parser.add_argument("--chunk-max-tokens", type=int, default=CHUNK_MAX_TOKENS)
    parser.add_argument("--chunk-overlap-turns", type=int, default=CHUNK_OVERLAP_TURNS)
    parser.add_argument("--max-chunks", type=int, help="Process only the first N chunks for smoke tests.")
    parser.add_argument("--max-prompt-tokens", type=int, default=16_000)
    parser.add_argument("--min-confidence-score", type=int, default=8)
    parser.add_argument("--strict-issue-gates", dest="strict_issue_gates", action="store_true", default=True)
    parser.add_argument("--no-strict-issue-gates", dest="strict_issue_gates", action="store_false")
    parser.add_argument("--retrieval-top-k", type=int, default=30)
    parser.add_argument("--retrieval-min-score", type=float, default=0.35)
    parser.add_argument("--response-format-json", action="store_true")
    parser.add_argument("--api-key-stdin", action="store_true")
    parser.add_argument("--api-key-plain-stdin", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir
    raw_dir = output_dir / "raw"
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = output_dir / "predictions.jsonl"

    api_key = resolve_api_key(args)
    if not api_key:
        raise SystemExit("Missing API key. Set NOVITA_API_KEY.")

    examples = load_jsonl(args.examples)
    if args.ids:
        wanted = set(args.ids)
        examples = [example for example in examples if example["id"] in wanted]
    if args.limit is not None:
        examples = examples[: args.limit]

    completed = set() if args.no_resume else load_existing_predictions(predictions_path)
    todo = [example for example in examples if example["id"] not in completed]
    print(f"examples selected: {len(examples)}")
    print(f"already completed: {len(examples) - len(todo)}")
    print(f"remaining: {len(todo)}")

    pipeline = build_pipeline(args, api_key)
    mode = "a" if predictions_path.exists() and not args.no_resume else "w"
    with predictions_path.open(mode, encoding="utf-8") as predictions_handle:
        for index, example in enumerate(todo, start=1):
            print(
                f"[{index}/{len(todo)}] {example['id']} "
                f"split={example.get('split')} labels={len(example.get('labels', []))}"
            )
            result = pipeline.run(
                example_id=example["id"],
                conversation=example["conversation"],
                max_tokens=args.chunk_max_tokens,
                overlap_turns=args.chunk_overlap_turns,
                expected_label_count=len(example.get("labels", [])),
                max_chunks=args.max_chunks,
                progress_callback=lambda message: print(f"  {message}", flush=True),
            )
            predictions_handle.write(
                json.dumps(result.prediction, ensure_ascii=False, separators=(",", ":")) + "\n"
            )
            predictions_handle.flush()
            (raw_dir / f"{example['id']}.json").write_text(
                json.dumps(result.raw_record, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )

    (output_dir / "run_manifest.json").write_text(
        json.dumps(
            {
                "model": args.model,
                "base_url": args.base_url,
                "examples": str(args.examples),
                "claim_prompt_file": str(args.claim_prompt_file),
                "auditor_prompt_file": str(args.auditor_prompt_file),
                "predictions": str(predictions_path),
                "raw_dir": str(raw_dir),
                "heuristic_claims": args.heuristic_claims,
                "chunk_max_tokens": args.chunk_max_tokens,
                "chunk_overlap_turns": args.chunk_overlap_turns,
                "max_chunks": args.max_chunks,
                "max_prompt_tokens": args.max_prompt_tokens,
                "min_confidence_score": args.min_confidence_score,
                "strict_issue_gates": args.strict_issue_gates,
                "retrieval_top_k": args.retrieval_top_k,
                "retrieval_min_score": args.retrieval_min_score,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(f"Wrote {predictions_path}")
    print(f"Wrote raw records under {raw_dir}")


if __name__ == "__main__":
    main()
