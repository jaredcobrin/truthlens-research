# TruthLens V1 Long-Context Test Plan

## Goal

Verify that conversations above the model's native context limit are split,
evaluated, and recombined without losing turns or missing failures that span
chunk boundaries.

## What To Test First

1. Chunk mechanics, offline
   - Generate conversations estimated at 120k, 310k, and 600k tokens.
   - Assert no empty chunks are produced.
   - Assert every original turn appears at least once.
   - Assert original turn order is preserved.
   - Assert each adjacent chunk shares the expected overlap turns.
   - Assert a single oversized pasted turn does not create an empty chunk.

2. Boundary-sensitive fixtures
   - Put a claim near the end of chunk 1.
   - Put a contradictory claim in chunk 2 or chunk 3.
   - Confirm the aggregation path carries the earlier claim forward.
   - Confirm final findings cite both the issue turn and the earlier turn.

3. API smoke tests
   - Use a reduced chunk size, such as 2k estimated tokens, so ordinary short
     conversations force the chunked path.
   - Mock or log each outbound model request before using real API calls.
   - Confirm the direct path and chunked path return the same schema.

4. Full long-context model tests
   - Run one synthetic 260k to 310k estimated-token conversation with a known
     cross-chunk contradiction.
   - Run one clean 260k to 310k estimated-token conversation.
   - Track false positives separately from missed boundary contradictions.

## Current Harness

Run:

```bash
python3 tests/chunking_harness.py
```

The harness currently tests:

- one conversation below the chunking threshold
- one conversation above 256k estimated tokens
- one cross-chunk contradiction fixture
- one single-turn oversized paste

## Implementation Notes

Use a lower production chunk threshold than the advertised model context. For a
256k-token context window, 180k estimated tokens per chunk is a reasonable v1
default because the system prompt, prior-claim registry, XML instructions, and
model output budget all consume context too.

The pseudocode in the brief should be adjusted before it becomes production
code: only append the current chunk when it is non-empty. Otherwise a single
first turn larger than `max_tokens` can produce an empty chunk.

For true cross-chunk contradiction detection, overlapping turns alone is not
enough once the contradiction is farther away than the overlap. The aggregator
needs a compact claim registry extracted from each chunk, then each later chunk
prompt should include prior claims.
