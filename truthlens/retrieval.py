"""Local claim retrieval for TruthLens audit memory."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Protocol

from .chunking import format_chunk
from .memory import ClaimMemory
from .models import AuditClaim, ConversationChunk, RetrievalResult, TopicSummary


class EmbeddingProvider(Protocol):
    def embed(self, texts: list[str]) -> list[list[float]]:
        """Return one vector per input text."""


@dataclass(frozen=True)
class RetrievalConfig:
    top_k: int = 30
    min_score: float = 0.35
    include_user_instruction_bonus: float = 1.0


TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9_-]{2,}")
STOPWORDS = {
    "assistant",
    "response",
    "answer",
    "because",
    "therefore",
    "about",
    "after",
    "before",
    "with",
    "from",
    "that",
    "this",
    "have",
    "will",
    "would",
    "could",
    "should",
    "case",
    "user",
}


def tokens(text: str) -> set[str]:
    return {token for token in TOKEN_RE.findall(text.lower()) if token not in STOPWORDS}


def lexical_similarity(left: str, right: str) -> float:
    left_tokens = tokens(left)
    right_tokens = tokens(right)
    if not left_tokens or not right_tokens:
        return 0.0
    intersection = len(left_tokens & right_tokens)
    union = len(left_tokens | right_tokens)
    return intersection / union if union else 0.0


class ClaimRetriever:
    def __init__(
        self,
        *,
        config: RetrievalConfig | None = None,
        embedding_provider: EmbeddingProvider | None = None,
    ) -> None:
        self.config = config or RetrievalConfig()
        self.embedding_provider = embedding_provider

    def retrieve(
        self,
        *,
        memory: ClaimMemory,
        chunk: ConversationChunk,
        current_claims: list[AuditClaim],
    ) -> list[RetrievalResult]:
        query_topics = {claim.topic for claim in current_claims}
        query_subtopics = {subtopic for claim in current_claims for subtopic in claim.subtopics}
        query_texts = [claim.normalized_claim for claim in current_claims]
        chunk_text = format_chunk(chunk)

        results: list[RetrievalResult] = []
        for claim in memory.active_claims_before_turn(chunk.turn_start):
            score, reasons = self._score_claim(
                claim=claim,
                query_topics=query_topics,
                query_subtopics=query_subtopics,
                query_texts=query_texts,
                chunk_text=chunk_text,
            )
            if score >= self.config.min_score:
                results.append(RetrievalResult(claim=claim, score=score, reasons=reasons))

        results.sort(key=lambda item: (item.score, item.claim.importance_score, item.claim.turn_number), reverse=True)
        return results[: self.config.top_k]

    def relevant_summaries(
        self,
        *,
        memory: ClaimMemory,
        current_claims: list[AuditClaim],
        max_summaries: int = 8,
    ) -> list[TopicSummary]:
        topics = {claim.topic for claim in current_claims}
        topics.update(subtopic for claim in current_claims for subtopic in claim.subtopics)
        return memory.summaries_for_topics(topics, max_summaries=max_summaries)

    def _score_claim(
        self,
        *,
        claim: AuditClaim,
        query_topics: set[str],
        query_subtopics: set[str],
        query_texts: list[str],
        chunk_text: str,
    ) -> tuple[float, list[str]]:
        reasons: list[str] = []
        score = 0.0

        if claim.topic in query_topics:
            score += 3.0
            reasons.append("same_topic")

        overlap = set(claim.subtopics) & query_subtopics
        if overlap:
            score += 1.5 + min(1.0, 0.25 * len(overlap))
            reasons.append("subtopic_overlap")

        lexical = max(
            [lexical_similarity(claim.normalized_claim, text) for text in query_texts]
            + [lexical_similarity(claim.normalized_claim, chunk_text)]
        )
        if lexical:
            score += min(2.5, lexical * 8.0)
            reasons.append(f"lexical_{lexical:.2f}")

        if claim.related_user_instruction:
            score += self.config.include_user_instruction_bonus
            reasons.append("active_user_instruction")

        if claim.source_reference:
            score += 0.5
            reasons.append("source_claim")

        score += math.sqrt(max(0.0, claim.importance_score)) * 0.4
        return score, reasons


def format_memory_bundle(
    *,
    results: list[RetrievalResult],
    summaries: list[TopicSummary],
    max_quote_chars: int = 700,
) -> str:
    if not results and not summaries:
        return "No prior structured audit memory was retrieved."

    lines: list[str] = []
    if summaries:
        lines.append("Topic summaries for routing only, not evidence:")
        for summary in summaries:
            lines.append(f"- Topic {summary.topic}: {summary.summary[:900]}")

    if results:
        lines.append("Retrieved prior claims with exact evidence quotes:")
        for result in results:
            claim = result.claim
            quote = claim.original_quote[:max_quote_chars]
            lines.append(
                "- "
                f"{claim.claim_id} | turn={claim.turn_number} | "
                f"speaker={claim.speaker} | topic={claim.topic} | score={result.score:.2f} | "
                f"claim={claim.normalized_claim[:500]} | quote={quote}"
            )
    return "\n".join(lines)
