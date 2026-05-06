#!/usr/bin/env python3
"""
Run the full TruthLens auditor prompt on V2 examples.

This runner is OpenAI-compatible and chunk-aware. It reads the auditor prompt
from prompts/truthlens_auditor_v1.md, carries forward claims_to_remember, and
emits prediction rows shaped for score_truthlens_auditor_eval_v2.py.
"""

from __future__ import annotations

import argparse
import getpass
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from truthlens.reporting import issue_from_mapping  # noqa: E402


DEFAULT_MODEL = "google/gemma-4-31b-it"
DEFAULT_BASE_URL = "https://api.novita.ai/openai/v1"
DEFAULT_PROMPT_FILE = Path("prompts/truthlens_auditor_v1.md")
DEFAULT_EXAMPLES = Path("data/truthlens_auditor_eval_v2_1/unit_examples.jsonl")
DEFAULT_OUTPUT_DIR = Path("results/truthlens_auditor_eval_v2_1")
CHUNK_MAX_TOKENS = 12_000
CHUNK_OVERLAP_TURNS = 4


def estimate_tokens(text: str) -> int:
    return max(1, len(text) // 4)


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
            current = current[-overlap_turns:] if overlap_turns else []
            current_tokens = sum(estimate_tokens(item["content"]) for item in current)
        current.append(turn)
        current_tokens += tokens
    if current:
        chunks.append(current)
    return chunks


def format_conversation(chunk: list[dict[str, Any]]) -> str:
    parts: list[str] = []
    for turn in chunk:
        role = "User" if turn["role"] == "user" else "AI Assistant"
        parts.append(f"[Turn {turn['turn']}] {role}:\n{turn['content']}")
    return "\n\n".join(parts)


def compact_claim_registry(claims: list[dict[str, Any]], max_claims: int = 120) -> str:
    if not claims:
        return "None."
    lines = []
    for item in claims[-max_claims:]:
        turn = item.get("turn", "?")
        claim = str(item.get("claim", "")).strip()
        if claim:
            lines.append(f"- Turn {turn}: {claim[:600]}")
    return "\n".join(lines) if lines else "None."


def build_chunk_prompt(
    *,
    chunk: list[dict[str, Any]],
    chunk_index: int,
    chunk_count: int,
    claim_registry: list[dict[str, Any]],
    min_confidence_score: int,
) -> str:
    return f"""This is chunk {chunk_index} of {chunk_count} from a longer AI assistant conversation.
Turns in this chunk are numbered {chunk[0]['turn']} to {chunk[-1]['turn']}.
Minimum confidence score for visible findings: {min_confidence_score}

Claims/context remembered from previous chunks:
{compact_claim_registry(claim_registry)}

Audit this current chunk. Detect issues inside the current chunk and issues
between current turns and remembered prior claims/context. Return only the JSON
schema requested by the system prompt.

Conversation chunk:
{format_conversation(chunk)}
"""


def api_post_json(
    *,
    url: str,
    api_key: str,
    payload: dict[str, Any],
    timeout: int,
    retries: int,
) -> dict[str, Any]:
    body = json.dumps(payload).encode("utf-8")
    last_error: Exception | None = None
    last_detail = ""
    for attempt in range(retries):
        request = urllib.request.Request(
            url,
            data=body,
            method="POST",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "User-Agent": "truthlens-auditor-eval/0.2",
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            last_error = exc
            last_detail = exc.read().decode("utf-8", errors="replace")
            retry_after = exc.headers.get("retry-after")
            if retry_after and retry_after.isdigit():
                sleep_seconds = int(retry_after)
            elif exc.code in {408, 409, 429, 500, 502, 503, 504}:
                sleep_seconds = min(120, 5 * (attempt + 1))
            else:
                raise RuntimeError(f"API error {exc.code}: {last_detail}") from exc
            time.sleep(sleep_seconds)
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            last_error = exc
            time.sleep(min(120, 5 * (attempt + 1)))
    raise RuntimeError(f"API request failed after retries. Last detail: {last_detail}") from last_error


def call_model(
    *,
    api_key: str,
    base_url: str,
    model: str,
    system_prompt: str,
    user_prompt: str,
    temperature: float,
    max_output_tokens: int,
    timeout: int,
    retries: int,
    response_format_json: bool,
) -> str:
    endpoint = base_url.rstrip("/") + "/chat/completions"
    payload: dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": temperature,
        "max_tokens": max_output_tokens,
    }
    if response_format_json:
        payload["response_format"] = {"type": "json_object"}
    response = api_post_json(url=endpoint, api_key=api_key, payload=payload, timeout=timeout, retries=retries)
    return response["choices"][0]["message"]["content"]


def parse_model_json(raw: str) -> dict[str, Any]:
    text = raw.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match:
            return {"status": "parse_error", "issues": [], "claims_to_remember": [], "parse_error": raw[:1200]}
        try:
            parsed = json.loads(match.group(0))
        except json.JSONDecodeError:
            return {"status": "parse_error", "issues": [], "claims_to_remember": [], "parse_error": raw[:1200]}
    if not isinstance(parsed, dict):
        return {"status": "parse_error", "issues": [], "claims_to_remember": [], "parse_error": raw[:1200]}
    if not isinstance(parsed.get("issues", []), list):
        parsed["issues"] = []
    if not isinstance(parsed.get("claims_to_remember", []), list):
        parsed["claims_to_remember"] = []
    return parsed


def normalize_turn(value: Any) -> int | None:
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        match = re.search(r"\d+", value)
        if match:
            return int(match.group(0))
    return None


def normalize_type(value: Any) -> str:
    return re.sub(r"[^A-Z0-9_]+", "_", str(value or "").strip().upper())


def normalize_issue(issue: dict[str, Any]) -> dict[str, Any] | None:
    parsed = issue_from_mapping(issue)
    if parsed is None:
        return None
    return parsed.to_dict()


def dedupe_issues(issues: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[tuple[str, int, str]] = set()
    deduped: list[dict[str, Any]] = []
    for issue in issues:
        normalized = normalize_issue(issue)
        if normalized is None:
            continue
        key = (normalized["primary_failure"], normalized["turn"], normalized["quote"][:100])
        if key in seen:
            continue
        seen.add(key)
        deduped.append(normalized)
    return deduped


def clean_claims(claims: list[Any]) -> list[dict[str, Any]]:
    cleaned: list[dict[str, Any]] = []
    for claim in claims:
        if not isinstance(claim, dict):
            continue
        turn = normalize_turn(claim.get("turn"))
        text = str(claim.get("claim", "")).strip()
        if turn is not None and text:
            cleaned.append({"turn": turn, "claim": text[:800]})
    return cleaned


def evaluate_example(
    *,
    example: dict[str, Any],
    system_prompt: str,
    api_key: str,
    base_url: str,
    model: str,
    temperature: float,
    max_output_tokens: int,
    timeout: int,
    retries: int,
    sleep_seconds: float,
    dry_run: bool,
    chunk_max_tokens: int,
    chunk_overlap_turns: int,
    response_format_json: bool,
    min_confidence_score: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    chunks = chunk_conversation(example["conversation"], chunk_max_tokens, chunk_overlap_turns)
    claim_registry: list[dict[str, Any]] = []
    all_issues: list[dict[str, Any]] = []
    chunk_results: list[dict[str, Any]] = []

    for index, chunk in enumerate(chunks, start=1):
        prompt = build_chunk_prompt(
            chunk=chunk,
            chunk_index=index,
            chunk_count=len(chunks),
            claim_registry=claim_registry,
            min_confidence_score=min_confidence_score,
        )
        if dry_run:
            raw = ""
            parsed = {
                "status": "clean",
                "issues": [],
                "claims_to_remember": [],
                "dry_run_prompt_estimated_tokens": estimate_tokens(system_prompt) + estimate_tokens(prompt),
            }
        else:
            raw = call_model(
                api_key=api_key,
                base_url=base_url,
                model=model,
                system_prompt=system_prompt,
                user_prompt=prompt,
                temperature=temperature,
                max_output_tokens=max_output_tokens,
                timeout=timeout,
                retries=retries,
                response_format_json=response_format_json,
            )
            parsed = parse_model_json(raw)
            if sleep_seconds:
                time.sleep(sleep_seconds)

        issues = dedupe_issues(parsed.get("issues", []))
        claims = clean_claims(parsed.get("claims_to_remember", []))
        all_issues.extend(issues)
        claim_registry.extend(claims)
        chunk_results.append(
            {
                "chunk_index": index,
                "chunk_count": len(chunks),
                "turn_start": chunk[0]["turn"],
                "turn_end": chunk[-1]["turn"],
                "estimated_prompt_tokens": estimate_tokens(system_prompt) + estimate_tokens(prompt),
                "raw_output": raw,
                "parsed": parsed,
                "issues": issues,
                "claims_added": claims,
            }
        )

    raw_issues = dedupe_issues(all_issues)
    final_issues = [
        issue for issue in raw_issues if int(issue.get("confidence_score", 8)) >= min_confidence_score
    ]
    prediction = {
        "id": example["id"],
        "status": "issues_found" if final_issues else "clean",
        "issue_count": len(final_issues),
        "raw_issue_count": len(raw_issues),
        "min_confidence_score": min_confidence_score,
        "issues": final_issues,
        "chunk_count": len(chunks),
        "final_claim_count": len(claim_registry),
    }
    raw_record = {
        "id": example["id"],
        "model": model,
        "expected_label_count": len(example.get("labels", [])),
        "prediction": prediction,
        "chunks": chunk_results,
    }
    return prediction, raw_record


def load_existing_predictions(path: Path) -> set[str]:
    if not path.exists():
        return set()
    ids: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                ids.add(json.loads(line)["id"])
    return ids


def iter_examples(path: Path) -> list[dict[str, Any]]:
    return [row for row in load_jsonl(path)]


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--examples", type=Path, default=DEFAULT_EXAMPLES)
    parser.add_argument("--prompt-file", type=Path, default=DEFAULT_PROMPT_FILE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--id", action="append", dest="ids")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--max-output-tokens", type=int, default=4096)
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--retries", type=int, default=6)
    parser.add_argument("--sleep-seconds", type=float, default=0.5)
    parser.add_argument("--chunk-max-tokens", type=int, default=CHUNK_MAX_TOKENS)
    parser.add_argument("--chunk-overlap-turns", type=int, default=CHUNK_OVERLAP_TURNS)
    parser.add_argument("--min-confidence-score", type=int, default=8)
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

    if args.api_key_stdin:
        api_key = getpass.getpass("API key: ").strip()
    elif args.api_key_plain_stdin:
        api_key = input().strip()
    else:
        api_key = (
            os.environ.get("NOVITA_API_KEY")
            or os.environ.get("GOOGLE_AI_STUDIO_KEY")
            or os.environ.get("GOOGLE_API_KEY")
            or os.environ.get("OPENAI_API_KEY")
            or ""
        )
    if not api_key and not args.dry_run:
        raise SystemExit("Missing API key. Set NOVITA_API_KEY or use --dry-run.")

    system_prompt = args.prompt_file.read_text(encoding="utf-8")
    examples = iter_examples(args.examples)
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

    mode = "a" if predictions_path.exists() and not args.no_resume else "w"
    with predictions_path.open(mode, encoding="utf-8") as predictions_handle:
        for number, example in enumerate(todo, start=1):
            chunks = chunk_conversation(example["conversation"], args.chunk_max_tokens, args.chunk_overlap_turns)
            print(
                f"[{number}/{len(todo)}] {example['id']} "
                f"split={example.get('split')} labels={len(example.get('labels', []))} chunks={len(chunks)}"
            )
            prediction, raw_record = evaluate_example(
                example=example,
                system_prompt=system_prompt,
                api_key=api_key,
                base_url=args.base_url,
                model=args.model,
                temperature=args.temperature,
                max_output_tokens=args.max_output_tokens,
                timeout=args.timeout,
                retries=args.retries,
                sleep_seconds=args.sleep_seconds,
                dry_run=args.dry_run,
                chunk_max_tokens=args.chunk_max_tokens,
                chunk_overlap_turns=args.chunk_overlap_turns,
                response_format_json=args.response_format_json,
                min_confidence_score=args.min_confidence_score,
            )
            predictions_handle.write(json.dumps(prediction, ensure_ascii=False, separators=(",", ":")) + "\n")
            predictions_handle.flush()
            (raw_dir / f"{example['id']}.json").write_text(
                json.dumps(raw_record, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )

    (output_dir / "run_manifest.json").write_text(
        json.dumps(
            {
                "model": args.model,
                "base_url": args.base_url,
                "examples": str(args.examples),
                "prompt_file": str(args.prompt_file),
                "predictions": str(predictions_path),
                "raw_dir": str(raw_dir),
                "dry_run": args.dry_run,
                "chunk_max_tokens": args.chunk_max_tokens,
                "chunk_overlap_turns": args.chunk_overlap_turns,
                "min_confidence_score": args.min_confidence_score,
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
