"""Shared data models for TruthLens audit memory."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal


Role = Literal["user", "assistant"]


@dataclass(frozen=True)
class Turn:
    turn: int
    role: Role
    content: str

    @classmethod
    def from_mapping(cls, data: dict[str, Any]) -> "Turn":
        role = data.get("role")
        if role not in {"user", "assistant"}:
            raise ValueError(f"invalid role: {role!r}")
        return cls(turn=int(data["turn"]), role=role, content=str(data["content"]))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ConversationChunk:
    chunk_id: int
    turns: list[Turn]
    estimated_tokens: int

    @property
    def turn_start(self) -> int:
        return self.turns[0].turn

    @property
    def turn_end(self) -> int:
        return self.turns[-1].turn

    def to_dict(self) -> dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "turn_start": self.turn_start,
            "turn_end": self.turn_end,
            "estimated_tokens": self.estimated_tokens,
            "turns": [turn.to_dict() for turn in self.turns],
        }


@dataclass
class AuditClaim:
    claim_id: str
    normalized_claim: str
    original_quote: str
    speaker: Role
    turn_number: int
    chunk_id: int
    topic: str
    subtopics: list[str] = field(default_factory=list)
    importance_score: float = 0.5
    confidence_level: str = "unknown"
    source_reference: str | None = None
    related_user_instruction: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class TopicSummary:
    topic: str
    summary: str
    claim_ids: list[str] = field(default_factory=list)
    subtopics: list[str] = field(default_factory=list)
    updated_chunk_id: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RetrievalResult:
    claim: AuditClaim
    score: float
    reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "claim": self.claim.to_dict(),
            "score": round(self.score, 4),
            "reasons": self.reasons,
        }


@dataclass
class AuditIssue:
    primary_failure: str
    severity: str
    turn: int
    title: str
    quote: str
    evidence: list[dict[str, Any]] = field(default_factory=list)
    secondary_tags: list[str] = field(default_factory=list)
    confidence_score: int = 8
    reason: str = ""
    why_it_matters: str = ""
    suggested_rewrite: str = ""
    repair_prompt: str = ""

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        # Keep a compatibility alias so older downstream readers that expect
        # "type" can still inspect predictions while new scoring uses
        # "primary_failure".
        data["type"] = self.primary_failure
        return data


@dataclass
class AuditState:
    claims: list[AuditClaim] = field(default_factory=list)
    topic_summaries: dict[str, TopicSummary] = field(default_factory=dict)
    issues: list[AuditIssue] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "claims": [claim.to_dict() for claim in self.claims],
            "topic_summaries": {
                topic: summary.to_dict() for topic, summary in self.topic_summaries.items()
            },
            "issues": [issue.to_dict() for issue in self.issues],
        }
