"""Turn-based chunking helpers."""

from __future__ import annotations

from typing import Any, Iterable

from .models import ConversationChunk, Turn


def estimate_tokens(text: str) -> int:
    """Cheap token estimate used consistently by the benchmark scripts."""
    return max(1, len(text) // 4)


def turn_tokens(turn: Turn) -> int:
    return estimate_tokens(turn.content)


def normalize_turns(conversation: Iterable[Turn | dict[str, Any]]) -> list[Turn]:
    turns: list[Turn] = []
    for item in conversation:
        if isinstance(item, Turn):
            turns.append(item)
        else:
            turns.append(Turn.from_mapping(item))
    return turns


def chunk_turns(
    conversation: Iterable[Turn | dict[str, Any]],
    max_tokens: int,
    overlap_turns: int,
) -> list[ConversationChunk]:
    turns = normalize_turns(conversation)
    chunks: list[ConversationChunk] = []
    current: list[Turn] = []
    current_tokens = 0

    for turn in turns:
        tokens = turn_tokens(turn)
        if current and current_tokens + tokens > max_tokens:
            chunks.append(
                ConversationChunk(
                    chunk_id=len(chunks) + 1,
                    turns=current,
                    estimated_tokens=sum(turn_tokens(item) for item in current),
                )
            )
            current = current[-overlap_turns:] if overlap_turns else []
            current_tokens = sum(turn_tokens(item) for item in current)

        current.append(turn)
        current_tokens += tokens

    if current:
        chunks.append(
            ConversationChunk(
                chunk_id=len(chunks) + 1,
                turns=current,
                estimated_tokens=sum(turn_tokens(item) for item in current),
            )
        )

    return chunks


def format_turns(turns: Iterable[Turn]) -> str:
    parts: list[str] = []
    for turn in turns:
        role = "User" if turn.role == "user" else "AI Assistant"
        parts.append(f"[Turn {turn.turn}] {role}:\n{turn.content}")
    return "\n\n".join(parts)


def format_chunk(chunk: ConversationChunk) -> str:
    return format_turns(chunk.turns)
