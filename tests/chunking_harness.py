"""
Offline chunking harness for TruthLens v1.

This does not call the evaluator model. It stress-tests the long-context
conversation splitting behavior so you can verify chunk coverage before
spending API calls on >256k-token conversations.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Literal


Role = Literal["user", "assistant"]


@dataclass(frozen=True)
class Turn:
    turn: int
    role: Role
    content: str


def estimate_tokens(text: str) -> int:
    return max(1, len(text) // 4)


def turn_tokens(turn: Turn) -> int:
    return estimate_tokens(turn.content)


def chunk_conversation(
    conversation: list[Turn],
    max_tokens: int = 180_000,
    overlap_turns: int = 10,
) -> list[list[Turn]]:
    """Split turns into overlapping chunks, preserving every original turn."""
    chunks: list[list[Turn]] = []
    current_chunk: list[Turn] = []
    current_tokens = 0

    for turn in conversation:
        tokens = turn_tokens(turn)

        if current_chunk and current_tokens + tokens > max_tokens:
            chunks.append(current_chunk)
            current_chunk = current_chunk[-overlap_turns:]
            current_tokens = sum(turn_tokens(t) for t in current_chunk)

        current_chunk.append(turn)
        current_tokens += tokens

    if current_chunk:
        chunks.append(current_chunk)

    return chunks


def make_filler_tokens(token_count: int, marker: str) -> str:
    # 4 chars per estimated token, matching estimate_tokens above.
    body = (marker + " ") * max(1, token_count // max(1, estimate_tokens(marker + " ")))
    while estimate_tokens(body) < token_count:
        body += marker + " "
    return body


def synthetic_conversation(
    total_estimated_tokens: int,
    tokens_per_turn: int = 5_000,
    contradiction_across_boundary: bool = False,
) -> list[Turn]:
    """Create a deterministic long conversation with optional boundary issue."""
    conversation: list[Turn] = []
    running_tokens = 0
    turn_number = 1

    while running_tokens < total_estimated_tokens:
        role: Role = "user" if turn_number % 2 else "assistant"
        content = make_filler_tokens(tokens_per_turn, f"turn-{turn_number}")
        conversation.append(Turn(turn=turn_number, role=role, content=content))
        running_tokens += turn_tokens(conversation[-1])
        turn_number += 1

    if contradiction_across_boundary and len(conversation) >= 50:
        # These two turns should land in different chunks with default settings.
        conversation[35] = Turn(
            turn=36,
            role="assistant",
            content="Project Alpha uses PostgreSQL as its only production database. "
            + make_filler_tokens(tokens_per_turn, "alpha-postgres"),
        )
        conversation[-4] = Turn(
            turn=conversation[-4].turn,
            role="assistant",
            content="Project Alpha has never used PostgreSQL; production is MongoDB only. "
            + make_filler_tokens(tokens_per_turn, "alpha-mongodb"),
        )

    return conversation


def unique_turns(chunks: Iterable[list[Turn]]) -> list[int]:
    seen: set[int] = set()
    ordered: list[int] = []
    for chunk in chunks:
        for turn in chunk:
            if turn.turn not in seen:
                seen.add(turn.turn)
                ordered.append(turn.turn)
    return ordered


def assert_chunking_integrity(
    conversation: list[Turn],
    chunks: list[list[Turn]],
    max_tokens: int,
    overlap_turns: int,
) -> None:
    assert chunks, "expected at least one chunk"
    assert [] not in chunks, "chunker produced an empty chunk"

    expected_turns = [turn.turn for turn in conversation]
    actual_turns = unique_turns(chunks)
    assert actual_turns == expected_turns, "chunker lost, duplicated, or reordered original turns"

    for i, chunk in enumerate(chunks, start=1):
        size = sum(turn_tokens(turn) for turn in chunk)
        largest_turn = max(turn_tokens(turn) for turn in chunk)
        assert size <= max_tokens or largest_turn > max_tokens, (
            f"chunk {i} is {size} estimated tokens, above max_tokens={max_tokens}"
        )

    for previous, current in zip(chunks, chunks[1:]):
        expected_overlap = [turn.turn for turn in previous[-overlap_turns:]]
        actual_overlap = [turn.turn for turn in current[:overlap_turns]]
        assert actual_overlap == expected_overlap, "chunk overlap does not preserve prior tail turns"


def run_case(name: str, conversation: list[Turn], max_tokens: int = 180_000) -> None:
    overlap_turns = 10
    chunks = chunk_conversation(conversation, max_tokens=max_tokens, overlap_turns=overlap_turns)
    assert_chunking_integrity(conversation, chunks, max_tokens, overlap_turns)

    total_tokens = sum(turn_tokens(turn) for turn in conversation)
    chunk_sizes = [sum(turn_tokens(turn) for turn in chunk) for chunk in chunks]
    print(f"{name}: PASS")
    print(f"  turns={len(conversation)} estimated_tokens={total_tokens:,} chunks={len(chunks)}")
    print(f"  chunk_sizes={chunk_sizes}")
    print(f"  first_last_turns={[(chunk[0].turn, chunk[-1].turn) for chunk in chunks]}")


def main() -> None:
    run_case(
        "single chunk below threshold",
        synthetic_conversation(total_estimated_tokens=120_000),
    )
    run_case(
        "multi chunk above 256k context",
        synthetic_conversation(total_estimated_tokens=310_000),
    )
    run_case(
        "cross chunk contradiction fixture",
        synthetic_conversation(total_estimated_tokens=310_000, contradiction_across_boundary=True),
    )
    run_case(
        "single giant turn",
        [Turn(turn=1, role="assistant", content=make_filler_tokens(220_000, "giant"))],
    )


if __name__ == "__main__":
    main()
