"""Audit-relevant claim extraction."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from .chunking import format_chunk
from .llm import ChatClient, parse_json_object
from .models import AuditClaim, ConversationChunk


DEFAULT_MAX_CLAIMS_PER_CHUNK = 40
IMPORTANT_CUES = (
    "case ",
    "baseline note",
    "later note",
    "source excerpt",
    "source-based answer",
    "standing instruction",
    "user context",
    "user goal",
    "user premise",
    "assistant response",
    "assistant advice",
    "assistant recommendation",
    "initial assessment",
    "revised assessment",
    "status update",
    "allergic",
    "do not",
    "must",
    "never",
    "definitely",
    "cannot determine",
    "not enough evidence",
    "costs",
    "therefore",
)


_UNICODE_FOLD = str.maketrans({
    "‘": "'", "’": "'", "‚": "'", "‛": "'",
    "“": '"', "”": '"', "„": '"', "‟": '"',
    "–": "-", "—": "-", "―": "-",
    " ": " ",   # non-breaking space
    "​": "", "‌": "", "‍": "", "﻿": "",
})


def _fold(s: str) -> str:
    return s.translate(_UNICODE_FOLD)


def _quote_appears_in(quote: str, content: str) -> bool:
    """LLMs frequently normalize unicode quotes/dashes when generating, so the
    extracted quote stops matching the source verbatim. Accept fold-equivalent
    matches (', ', ', ", ", –, —, NBSP, zero-widths) so those claims still
    pass verification instead of getting silently dropped."""
    return quote in content or _fold(quote) in _fold(content)


def clean_string(value: Any, limit: int = 1200) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()[:limit]


def infer_topic(text: str) -> str:
    lowered = text.lower()
    case_match = re.search(r"\bcase\s+(\d{4})", lowered)
    if case_match:
        return f"case-{case_match.group(1)}"
    project_match = re.search(r"\bproject\s+([a-z0-9_-]+)", lowered)
    if project_match:
        return f"project-{project_match.group(1)}"
    vendor_match = re.search(r"\bvendor\s+([a-z])", lowered)
    if vendor_match:
        return "vendor-comparison"
    option_match = re.search(r"\boption\s+([ab])", lowered)
    if option_match:
        return "option-comparison"
    if "warfarin" in lowered or "ibuprofen" in lowered:
        return "medical-caveat"
    if "allergic" in lowered:
        return "user-allergy"
    if "source excerpt" in lowered or "source-based" in lowered:
        return "source-fidelity"
    if "status update" in lowered or "standing instruction" in lowered:
        return "status-update-instructions"
    if "budget" in lowered or "cheapest" in lowered:
        return "budget-goal"
    tokens = re.findall(r"[a-z][a-z0-9_-]{3,}", lowered)
    stopwords = {
        "assistant",
        "response",
        "answer",
        "user",
        "that",
        "this",
        "with",
        "from",
        "have",
        "will",
        "because",
        "therefore",
    }
    useful = [token for token in tokens if token not in stopwords]
    return "-".join(useful[:3]) if useful else "general"


def infer_subtopics(text: str) -> list[str]:
    lowered = text.lower()
    subtopics: list[str] = []
    cues = {
        "instruction": ("instruction", "constraint", "do not", "exactly", "format"),
        "source": ("source", "excerpt", "citation", "quote"),
        "confidence": ("definitely", "certain", "will", "exactly", "not enough evidence"),
        "preference": ("prefer", "goal", "budget", "cheapest", "allergic"),
        "reasoning": ("therefore", "because", "proves", "reasoning"),
        "medical": ("warfarin", "ibuprofen", "clinician", "pharmacist"),
    }
    for subtopic, words in cues.items():
        if any(word in lowered for word in words):
            subtopics.append(subtopic)
    return subtopics[:6]


def infer_confidence(text: str) -> str:
    lowered = text.lower()
    if any(word in lowered for word in ("definitely", "certainly", "will definitely", "proves")):
        return "high"
    if any(word in lowered for word in ("may", "might", "unclear", "not enough", "cannot determine")):
        return "low"
    return "medium"


def infer_importance(text: str) -> float:
    lowered = text.lower()
    score = 0.45
    if any(word in lowered for word in ("allergic", "warfarin", "ibuprofen", "budget", "hard cap")):
        score += 0.3
    if any(word in lowered for word in ("do not", "must", "exactly", "source", "definitely")):
        score += 0.2
    if any(word in lowered for word in ("case ", "baseline note", "later note")):
        score += 0.2
    return min(1.0, score)


def source_reference_for(text: str) -> str | None:
    match = re.search(r"(source excerpt|source-based answer|citation|file|document)", text, re.I)
    return match.group(1).lower() if match else None


def related_instruction_for(text: str) -> str | None:
    lowered = text.lower()
    if any(word in lowered for word in ("do not", "must", "exactly", "instruction", "constraint")):
        return clean_string(text, 300)
    return None


def claim_from_fields(
    *,
    chunk: ConversationChunk,
    index: int,
    turn_number: int,
    speaker: str,
    original_quote: str,
    normalized_claim: str | None,
    topic: str | None,
    subtopics: list[Any] | None,
    importance_score: Any = None,
    confidence_level: str | None = None,
    source_reference: str | None = None,
    related_user_instruction: str | None = None,
) -> AuditClaim | None:
    quote = clean_string(original_quote)
    if not quote:
        return None
    turn = next((item for item in chunk.turns if item.turn == turn_number), None)
    if turn is None or not _quote_appears_in(quote, turn.content):
        return None
    speaker_value = "user" if speaker == "user" else "assistant"
    try:
        importance = float(importance_score)
    except (TypeError, ValueError):
        importance = infer_importance(quote)
    return AuditClaim(
        claim_id=f"cl_{chunk.chunk_id}_{turn_number}_{index:03d}",
        normalized_claim=clean_string(normalized_claim or quote, 700),
        original_quote=quote,
        speaker=speaker_value,
        turn_number=turn_number,
        chunk_id=chunk.chunk_id,
        topic=clean_string(topic or infer_topic(quote), 120) or "general",
        subtopics=[clean_string(item, 80) for item in (subtopics or infer_subtopics(quote)) if clean_string(item, 80)],
        importance_score=max(0.0, min(1.0, importance)),
        confidence_level=clean_string(confidence_level or infer_confidence(quote), 40) or "unknown",
        source_reference=source_reference or source_reference_for(quote),
        related_user_instruction=related_user_instruction or related_instruction_for(quote),
    )


class LLMClaimExtractor:
    def __init__(
        self,
        *,
        client: ChatClient,
        prompt_file: Path,
        temperature: float = 0.0,
        max_output_tokens: int = 4096,
        response_format_json: bool = True,
    ) -> None:
        self.client = client
        self.system_prompt = prompt_file.read_text(encoding="utf-8")
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens
        self.response_format_json = response_format_json

    def extract(self, chunk: ConversationChunk) -> list[AuditClaim]:
        user_prompt = f"""Extract audit-relevant claims from this conversation chunk.

Chunk id: {chunk.chunk_id}
Turns: {chunk.turn_start}-{chunk.turn_end}

Conversation chunk:
{format_chunk(chunk)}
"""
        raw = self.client.complete(
            system_prompt=self.system_prompt,
            user_prompt=user_prompt,
            temperature=self.temperature,
            max_output_tokens=self.max_output_tokens,
            response_format_json=self.response_format_json,
        )
        parsed = parse_json_object(raw)
        raw_claims = parsed.get("claims", [])
        if not isinstance(raw_claims, list):
            return []

        claims: list[AuditClaim] = []
        for item in raw_claims:
            if not isinstance(item, dict):
                continue
            try:
                turn_number = int(item.get("turn_number"))
            except (TypeError, ValueError):
                continue
            claim = claim_from_fields(
                chunk=chunk,
                index=len(claims) + 1,
                turn_number=turn_number,
                speaker=str(item.get("speaker", "")),
                original_quote=str(item.get("original_quote", "")),
                normalized_claim=str(item.get("normalized_claim", "")),
                topic=str(item.get("topic", "")),
                subtopics=item.get("subtopics") if isinstance(item.get("subtopics"), list) else [],
                importance_score=item.get("importance_score"),
                confidence_level=str(item.get("confidence_level", "")),
                source_reference=item.get("source_reference"),
                related_user_instruction=item.get("related_user_instruction"),
            )
            if claim is not None:
                claims.append(claim)
            if len(claims) >= DEFAULT_MAX_CLAIMS_PER_CHUNK:
                break
        return claims
