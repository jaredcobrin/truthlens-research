"""Structured claim memory for long-context auditing."""

from __future__ import annotations

from collections import defaultdict

from .models import AuditClaim, TopicSummary


class ClaimMemory:
    def __init__(self) -> None:
        self.claims: list[AuditClaim] = []
        self.claims_by_id: dict[str, AuditClaim] = {}
        self.topic_index: dict[str, set[str]] = defaultdict(set)
        self.subtopic_index: dict[str, set[str]] = defaultdict(set)
        self.topic_summaries: dict[str, TopicSummary] = {}
        self._seen_claim_keys: set[tuple[int, str]] = set()

    def add_claims(self, claims: list[AuditClaim]) -> list[AuditClaim]:
        added: list[AuditClaim] = []
        for claim in claims:
            key = (claim.turn_number, claim.original_quote)
            if key in self._seen_claim_keys:
                continue
            self._seen_claim_keys.add(key)
            self.claims.append(claim)
            self.claims_by_id[claim.claim_id] = claim
            self.topic_index[claim.topic].add(claim.claim_id)
            for subtopic in claim.subtopics:
                self.subtopic_index[subtopic].add(claim.claim_id)
            added.append(claim)

        for claim in added:
            self._refresh_topic_summary(claim.topic)
        return added

    def active_claims(self) -> list[AuditClaim]:
        return list(self.claims)

    def active_claims_before_turn(self, turn_number: int) -> list[AuditClaim]:
        return [claim for claim in self.claims if claim.turn_number < turn_number]

    def claims_for_topic(self, topic: str) -> list[AuditClaim]:
        claims = [self.claims_by_id[claim_id] for claim_id in self.topic_index.get(topic, set())]
        return sorted(claims, key=lambda claim: claim.turn_number)

    def summaries_for_topics(self, topics: set[str], max_summaries: int = 8) -> list[TopicSummary]:
        summaries = [
            summary
            for topic, summary in self.topic_summaries.items()
            if topic in topics or any(subtopic in topics for subtopic in summary.subtopics)
        ]
        summaries.sort(key=lambda item: item.updated_chunk_id, reverse=True)
        return summaries[:max_summaries]

    def _refresh_topic_summary(self, topic: str) -> None:
        topic_claims = self.claims_for_topic(topic)
        if not topic_claims:
            return

        important = sorted(topic_claims, key=lambda claim: (claim.importance_score, claim.turn_number), reverse=True)
        selected = list(reversed(important[:6]))
        summary = " | ".join(
            f"T{claim.turn_number} {claim.speaker}: {claim.normalized_claim[:180]}"
            for claim in selected
        )
        subtopics = sorted({subtopic for claim in topic_claims for subtopic in claim.subtopics})
        self.topic_summaries[topic] = TopicSummary(
            topic=topic,
            summary=summary,
            claim_ids=[claim.claim_id for claim in selected],
            subtopics=subtopics,
            updated_chunk_id=max(claim.chunk_id for claim in topic_claims),
        )
