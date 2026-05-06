"""Sequential hybrid audit-memory pipeline."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Protocol

from .chunking import chunk_turns, estimate_tokens, format_chunk
from .llm import ChatClient, parse_json_object
from .memory import ClaimMemory
from .models import AuditClaim, AuditIssue, ConversationChunk
from .reporting import build_prediction, parse_issues
from .retrieval import ClaimRetriever, format_memory_bundle


class ClaimExtractor(Protocol):
    def extract(self, chunk: ConversationChunk) -> list[AuditClaim]:
        """Extract audit-relevant claims from a chunk."""


class ChunkAuditor(Protocol):
    def audit(
        self,
        *,
        chunk: ConversationChunk,
        current_claims: list[AuditClaim],
        memory_bundle: str,
    ) -> tuple[list[AuditIssue], dict]:
        """Audit a chunk with structured memory and return issues plus diagnostics."""


class LLMChunkAuditor:
    def __init__(
        self,
        *,
        client: ChatClient,
        prompt_file: Path,
        temperature: float = 0.1,
        max_output_tokens: int = 4096,
        response_format_json: bool = True,
        max_prompt_tokens: int | None = None,
        min_confidence_score: int = 8,
    ) -> None:
        self.client = client
        self.system_prompt = prompt_file.read_text(encoding="utf-8")
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens
        self.response_format_json = response_format_json
        self.max_prompt_tokens = max_prompt_tokens
        self.min_confidence_score = min_confidence_score

    def audit(
        self,
        *,
        chunk: ConversationChunk,
        current_claims: list[AuditClaim],
        memory_bundle: str,
    ) -> tuple[list[AuditIssue], dict]:
        user_prompt = build_auditor_user_prompt(
            chunk=chunk,
            current_claims=current_claims,
            memory_bundle=memory_bundle,
            min_confidence_score=self.min_confidence_score,
        )
        raw = self.client.complete(
            system_prompt=self.system_prompt,
            user_prompt=user_prompt,
            temperature=self.temperature,
            max_output_tokens=self.max_output_tokens,
            response_format_json=self.response_format_json,
        )
        parsed = parse_json_object(raw)
        issues = parse_issues(parsed)
        return issues, {
            "raw_output": raw,
            "parsed": parsed,
            "estimated_prompt_tokens": estimate_tokens(self.system_prompt) + estimate_tokens(user_prompt),
        }


@dataclass
class PipelineResult:
    prediction: dict
    raw_record: dict


def build_auditor_user_prompt(
    *,
    chunk: ConversationChunk,
    current_claims: list[AuditClaim],
    memory_bundle: str,
    min_confidence_score: int = 8,
) -> str:
    current_claim_payload = [claim.to_dict() for claim in current_claims]
    return f"""Audit this chunk using the structured memory below.

Chunk id: {chunk.chunk_id}
Turns in chunk: {chunk.turn_start}-{chunk.turn_end}
Minimum confidence score for visible findings: {min_confidence_score}

Structured prior memory:
{memory_bundle}

Claims extracted from the current chunk:
{json.dumps(current_claim_payload, indent=2, ensure_ascii=False)}

Conversation chunk:
{format_chunk(chunk)}
"""


def bounded_memory_bundle(
    *,
    auditor: ChunkAuditor,
    chunk: ConversationChunk,
    current_claims: list[AuditClaim],
    retrieval_results: list,
    summaries: list,
) -> tuple[str, list, list, int]:
    """Trim retrieved memory until the auditor prompt fits its configured budget."""
    max_prompt_tokens = getattr(auditor, "max_prompt_tokens", None)
    system_prompt = getattr(auditor, "system_prompt", "")
    kept_results = list(retrieval_results)
    kept_summaries = list(summaries)

    while True:
        bundle = format_memory_bundle(results=kept_results, summaries=kept_summaries)
        prompt = build_auditor_user_prompt(
            chunk=chunk,
            current_claims=current_claims,
            memory_bundle=bundle,
            min_confidence_score=getattr(auditor, "min_confidence_score", 8),
        )
        estimated = estimate_tokens(system_prompt) + estimate_tokens(prompt)
        if not max_prompt_tokens or estimated <= max_prompt_tokens:
            return bundle, kept_results, kept_summaries, estimated
        if kept_results:
            kept_results.pop()
            continue
        if kept_summaries:
            kept_summaries.pop()
            continue
        return bundle, kept_results, kept_summaries, estimated


class AuditMemoryPipeline:
    def __init__(
        self,
        *,
        claim_extractor: ClaimExtractor | None = None,
        auditor: ChunkAuditor | None = None,
        retriever: ClaimRetriever | None = None,
        min_confidence_score: int = 8,
        strict_issue_gates: bool = True,
    ) -> None:
        if claim_extractor is None:
            raise ValueError("claim_extractor is required")
        if auditor is None:
            raise ValueError("auditor is required")
        self.claim_extractor = claim_extractor
        self.auditor = auditor
        self.retriever = retriever or ClaimRetriever()
        self.min_confidence_score = max(1, min(10, int(min_confidence_score)))
        self.strict_issue_gates = strict_issue_gates

    def run(
        self,
        *,
        example_id: str,
        conversation: list[dict],
        max_tokens: int,
        overlap_turns: int,
        expected_label_count: int | None = None,
        max_chunks: int | None = None,
        progress_callback: Callable[[str], None] | None = None,
    ) -> PipelineResult:
        chunks = chunk_turns(conversation, max_tokens=max_tokens, overlap_turns=overlap_turns)
        original_chunk_count = len(chunks)
        if max_chunks is not None:
            chunks = chunks[:max_chunks]
        memory = ClaimMemory()
        all_issues: list[AuditIssue] = []
        chunk_records: list[dict] = []

        for chunk in chunks:
            if progress_callback:
                progress_callback(
                    f"chunk {chunk.chunk_id}/{original_chunk_count} "
                    f"turns={chunk.turn_start}-{chunk.turn_end} tokens={chunk.estimated_tokens}"
                )
            current_claims = self.claim_extractor.extract(chunk)
            retrieved = self.retriever.retrieve(
                memory=memory,
                chunk=chunk,
                current_claims=current_claims,
            )
            summaries = self.retriever.relevant_summaries(
                memory=memory,
                current_claims=current_claims,
            )
            memory_bundle, retrieved, summaries, bounded_prompt_tokens = bounded_memory_bundle(
                auditor=self.auditor,
                chunk=chunk,
                current_claims=current_claims,
                retrieval_results=retrieved,
                summaries=summaries,
            )
            issues, audit_record = self.auditor.audit(
                chunk=chunk,
                current_claims=current_claims,
                memory_bundle=memory_bundle,
            )
            added_claims = memory.add_claims(current_claims)
            all_issues.extend(issues)
            if progress_callback:
                progress_callback(
                    f"chunk {chunk.chunk_id}/{original_chunk_count} done "
                    f"claims={len(current_claims)} retrieved={len(retrieved)} issues={len(issues)} "
                    f"prompt_tokens={audit_record.get('estimated_prompt_tokens')}"
                )

            chunk_records.append(
                {
                    "chunk_index": chunk.chunk_id,
                    "chunk_count": original_chunk_count,
                    "turn_start": chunk.turn_start,
                    "turn_end": chunk.turn_end,
                    "estimated_chunk_tokens": chunk.estimated_tokens,
                    "claims_extracted": [claim.to_dict() for claim in current_claims],
                    "claims_added": [claim.to_dict() for claim in added_claims],
                    "retrieved_prior_claims": [result.to_dict() for result in retrieved],
                    "topic_summaries": [summary.to_dict() for summary in summaries],
                    "bounded_prompt_estimated_tokens": bounded_prompt_tokens,
                    "issues": [issue.to_dict() for issue in issues],
                    **audit_record,
                }
            )

        prediction = build_prediction(
            example_id=example_id,
            issues=all_issues,
            chunk_count=original_chunk_count,
            final_claim_count=len(memory.claims),
            topic_summary_count=len(memory.topic_summaries),
            min_confidence_score=self.min_confidence_score,
            strict_issue_gates=self.strict_issue_gates,
            turn_roles={
                int(turn["turn"]): str(turn["role"])
                for turn in conversation
                if "turn" in turn and "role" in turn
            },
            claim_quotes={claim.original_quote for claim in memory.claims},
        )
        raw_record = {
            "id": example_id,
            "expected_label_count": expected_label_count,
            "processed_chunk_count": len(chunks),
            "original_chunk_count": original_chunk_count,
            "prediction": prediction,
            "memory": {
                "claims": [claim.to_dict() for claim in memory.claims],
                "topic_summaries": {
                    topic: summary.to_dict() for topic, summary in memory.topic_summaries.items()
                },
            },
            "chunks": chunk_records,
        }
        return PipelineResult(prediction=prediction, raw_record=raw_record)
