#!/usr/bin/env python3
"""Offline tests for the TruthLens hybrid audit memory package."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from truthlens.chunking import chunk_turns  # noqa: E402
from truthlens.memory import ClaimMemory  # noqa: E402
from truthlens.models import AuditClaim, AuditIssue, Turn  # noqa: E402
from truthlens.reporting import build_prediction, dedupe_issues, parse_issues  # noqa: E402
from truthlens.retrieval import ClaimRetriever  # noqa: E402


def claim(
    claim_id: str,
    text: str,
    *,
    speaker: str = "assistant",
    turn_number: int = 1,
    chunk_id: int = 1,
    topic: str = "project-alpha",
    subtopics: list[str] | None = None,
    importance: float = 0.8,
) -> AuditClaim:
    return AuditClaim(
        claim_id=claim_id,
        normalized_claim=text.lower(),
        original_quote=text,
        speaker=speaker,  # type: ignore[arg-type]
        turn_number=turn_number,
        chunk_id=chunk_id,
        topic=topic,
        subtopics=subtopics or ["database"],
        importance_score=importance,
        confidence_level="medium",
    )


def issue(
    primary_failure: str,
    quote: str,
    *,
    turn: int = 2,
    evidence: list[dict] | None = None,
    confidence_score: int = 10,
    reason: str = "",
) -> AuditIssue:
    return AuditIssue(
        primary_failure=primary_failure,
        severity="HIGH",
        turn=turn,
        title="Finding",
        quote=quote,
        evidence=evidence or [{"turn": 1, "quote": "Evidence quote."}],
        confidence_score=confidence_score,
        reason=reason,
    )


class TruthLensMemoryTests(unittest.TestCase):
    def test_chunking_preserves_turns_and_overlap(self) -> None:
        turns = [
            Turn(turn=i, role="assistant" if i % 2 == 0 else "user", content=("x" * 120))
            for i in range(1, 15)
        ]
        chunks = chunk_turns(turns, max_tokens=100, overlap_turns=2)
        unique_turns = []
        seen = set()
        for chunk in chunks:
            for turn in chunk.turns:
                if turn.turn not in seen:
                    unique_turns.append(turn.turn)
                    seen.add(turn.turn)
        self.assertEqual(unique_turns, list(range(1, 15)))
        for previous, current in zip(chunks, chunks[1:]):
            self.assertEqual(
                [turn.turn for turn in previous.turns[-2:]],
                [turn.turn for turn in current.turns[:2]],
            )

    def test_claim_store_keeps_exact_quotes(self) -> None:
        memory = ClaimMemory()
        original = "Project Alpha uses PostgreSQL as its only production database."
        memory.add_claims([claim("c1", original)])
        self.assertEqual(memory.claims[0].original_quote, original)
        self.assertEqual(memory.claims_by_id["c1"].original_quote, original)

    def test_retrieval_finds_same_topic_and_similar_claim(self) -> None:
        memory = ClaimMemory()
        memory.add_claims(
            [
                claim(
                    "c1",
                    "Project Alpha uses PostgreSQL as its only production database.",
                    turn_number=2,
                    topic="project-alpha",
                    subtopics=["database"],
                )
            ]
        )
        current = [
            claim(
                "c2",
                "Project Alpha has never used PostgreSQL; production is MongoDB only.",
                turn_number=10,
                chunk_id=2,
                topic="project-alpha",
                subtopics=["database"],
            )
        ]
        chunk = chunk_turns(
            [
                Turn(turn=10, role="assistant", content=current[0].original_quote),
            ],
            max_tokens=1000,
            overlap_turns=0,
        )[0]
        results = ClaimRetriever().retrieve(memory=memory, chunk=chunk, current_claims=current)
        self.assertTrue(results)
        self.assertEqual(results[0].claim.claim_id, "c1")
        self.assertIn("same_topic", results[0].reasons)


    def test_issue_dedupe_keeps_different_target_turns(self) -> None:
        first = AuditIssue(
            primary_failure="CONFLICTING_CLAIMS",
            severity="MEDIUM",
            turn=10,
            title="Conflict",
            quote="The later statement conflicts.",
            confidence_score=8,
        )
        second = AuditIssue(
            primary_failure="CONFLICTING_CLAIMS",
            severity="MEDIUM",
            turn=20,
            title="Conflict",
            quote="The later statement conflicts.",
            confidence_score=8,
        )
        self.assertEqual(len(dedupe_issues([first, second])), 2)

    def test_new_schema_parses_confidence_and_secondary_tags(self) -> None:
        issues = parse_issues(
            {
                "issues": [
                    {
                        "primary_failure": "USER_CONTEXT_OR_MEMORY_ERROR",
                        "secondary_tags": ["GOAL_MISALIGNMENT", "HIGH_STAKES"],
                        "confidence_score": 9,
                        "severity": "HIGH",
                        "turn": 4,
                        "quote": "Try the peanut sauce.",
                    }
                ]
            }
        )
        self.assertEqual(len(issues), 1)
        self.assertEqual(issues[0].primary_failure, "USER_CONTEXT_OR_MEMORY_ERROR")
        self.assertEqual(issues[0].secondary_tags, ["GOAL_MISALIGNMENT", "HIGH_STAKES"])
        self.assertEqual(issues[0].confidence_score, 9)

    def test_old_schema_falls_back_to_primary_failure(self) -> None:
        issues = parse_issues(
            {
                "issues": [
                    {
                        "type": "MEMORY_OR_CONTEXT_MISUSE",
                        "severity": "HIGH",
                        "turn": 4,
                        "quote": "Try the peanut sauce.",
                    }
                ]
            }
        )
        self.assertEqual(issues[0].primary_failure, "USER_CONTEXT_OR_MEMORY_ERROR")
        self.assertIn("GOAL_MISALIGNMENT", issues[0].secondary_tags)

    def test_prediction_filters_low_confidence_findings(self) -> None:
        high = AuditIssue(
            primary_failure="SOURCE_OR_EVIDENCE_ERROR",
            severity="HIGH",
            turn=2,
            title="Source drift",
            quote="Unsupported source claim.",
            confidence_score=8,
        )
        low = AuditIssue(
            primary_failure="SOURCE_OR_EVIDENCE_ERROR",
            severity="MEDIUM",
            turn=3,
            title="Maybe source drift",
            quote="Borderline source claim.",
            confidence_score=6,
        )
        prediction = build_prediction(
            example_id="example",
            issues=[high, low],
            chunk_count=1,
            final_claim_count=0,
            topic_summary_count=0,
            min_confidence_score=8,
        )
        self.assertEqual(prediction["raw_issue_count"], 2)
        self.assertEqual(prediction["issue_count"], 1)
        self.assertEqual(prediction["issues"][0]["confidence_score"], 8)

    def test_strict_gate_drops_background_filler_even_at_high_confidence(self) -> None:
        finding = issue(
            "INSTRUCTION_OR_CONSTRAINT_ERROR",
            "woman felt sure that her pet had escaped and would some day find his way back to her",
            evidence=[{"turn": 1, "quote": "Please continue the background document."}],
        )
        prediction = build_prediction(
            example_id="example",
            issues=[finding],
            chunk_count=1,
            final_claim_count=0,
            topic_summary_count=0,
            min_confidence_score=8,
            strict_issue_gates=True,
            turn_roles={1: "user", 2: "assistant"},
        )
        self.assertEqual(prediction["issue_count"], 0)
        self.assertIn("likely_background_filler", prediction["dropped_issues"][0]["reasons"])

    def test_strict_gate_accepts_source_drift_with_source_excerpt(self) -> None:
        finding = issue(
            "SOURCE_OR_EVIDENCE_ERROR",
            "Case 7302-A source-based answer: Allie Goertz wrote a song about Milhouse, named after an influential political figure.",
            evidence=[
                {
                    "turn": 1,
                    "quote": "Case 7302-A source excerpt: Milhouse was named after President Richard Nixon's middle name.",
                }
            ],
        )
        prediction = build_prediction(
            example_id="example",
            issues=[finding],
            chunk_count=1,
            final_claim_count=0,
            topic_summary_count=0,
            strict_issue_gates=True,
            turn_roles={1: "user", 2: "assistant"},
        )
        self.assertEqual(prediction["issue_count"], 1)

    def test_strict_gate_drops_source_answer_without_source_evidence(self) -> None:
        finding = issue(
            "SOURCE_OR_EVIDENCE_ERROR",
            "Case 7352-B source-based answer: standard gauge track",
            evidence=[{"turn": 1, "quote": "Case 7352-B question: What gauge is the track?"}],
        )
        prediction = build_prediction(
            example_id="example",
            issues=[finding],
            chunk_count=1,
            final_claim_count=0,
            topic_summary_count=0,
            strict_issue_gates=True,
            turn_roles={1: "user", 2: "assistant"},
        )
        self.assertEqual(prediction["issue_count"], 0)
        self.assertIn("source_error_without_source_evidence", prediction["dropped_issues"][0]["reasons"])

    def test_strict_gate_requires_instruction_evidence(self) -> None:
        finding = issue(
            "INSTRUCTION_OR_CONSTRAINT_ERROR",
            "Case 7400-A baseline note: Credibility is a vital factor, and Jim Lehrer does, indeed, have it.",
            evidence=[{"turn": 1, "quote": "Case 7400-A baseline note: Jim Lehrer moderated the debate."}],
        )
        prediction = build_prediction(
            example_id="example",
            issues=[finding],
            chunk_count=1,
            final_claim_count=0,
            topic_summary_count=0,
            strict_issue_gates=True,
            turn_roles={1: "assistant", 2: "assistant"},
        )
        self.assertEqual(prediction["issue_count"], 0)
        self.assertIn("instruction_error_without_user_constraint", prediction["dropped_issues"][0]["reasons"])

    def test_strict_gate_requires_memory_context_evidence(self) -> None:
        finding = issue(
            "USER_CONTEXT_OR_MEMORY_ERROR",
            "Case 7401-A assistant recommendation: Try the tasting menu.",
            evidence=[{"turn": 1, "quote": "Case 7401-A background note: The restaurant is popular."}],
        )
        prediction = build_prediction(
            example_id="example",
            issues=[finding],
            chunk_count=1,
            final_claim_count=0,
            topic_summary_count=0,
            strict_issue_gates=True,
            turn_roles={1: "assistant", 2: "assistant"},
        )
        self.assertEqual(prediction["issue_count"], 0)
        self.assertIn("memory_error_without_user_context", prediction["dropped_issues"][0]["reasons"])

    def test_strict_gate_requires_decision_risk_for_missing_caveat(self) -> None:
        finding = issue(
            "DECISION_CRITICAL_OMISSION",
            "Case 7402-A assistant answer: The plan is acceptable.",
            evidence=[{"turn": 1, "quote": "Case 7402-A context: The plan has several details."}],
        )
        prediction = build_prediction(
            example_id="example",
            issues=[finding],
            chunk_count=1,
            final_claim_count=0,
            topic_summary_count=0,
            strict_issue_gates=True,
            turn_roles={1: "user", 2: "assistant"},
        )
        self.assertEqual(prediction["issue_count"], 0)
        self.assertIn("decision_omission_without_decision_risk", prediction["dropped_issues"][0]["reasons"])


if __name__ == "__main__":
    unittest.main()
