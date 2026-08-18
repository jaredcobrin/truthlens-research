# TruthLens

Reads a multi-turn chat conversation and flags turns where the assistant was
untrustworthy — a claim that contradicts one made much earlier, an instruction
that got dropped, a confident answer with nothing behind it.

A flag only counts if it quotes the exact text that proves it. If the quote the
model returns is not verbatim in the transcript, the flag is discarded.

## How it works

```
conversation
  -> split into 12k-token chunks, 4 turns of overlap
  -> pull the claims out of each chunk, carry them forward as memory
  -> for the current chunk, retrieve the earlier claims that relate to it
  -> ask the model to audit this chunk against those retrieved claims
  -> drop any finding whose quote is not verbatim in the transcript
  -> report: turn, failure type, confidence
```

Carrying claims forward is what lets it catch a turn-90 statement contradicting
a turn-12 one without holding the whole conversation in context at once.

The seven failure types: `CONFLICTING_CLAIMS`,
`UNSUPPORTED_OR_OVERCONFIDENT_CLAIM`, `SOURCE_OR_EVIDENCE_ERROR`,
`USER_CONTEXT_OR_MEMORY_ERROR`, `INSTRUCTION_OR_CONSTRAINT_ERROR`,
`REASONING_OR_BAD_PREMISE_ERROR`, `DECISION_CRITICAL_OMISSION`.

## The test set

165 conversations: **110 with a failure planted in a known turn, and 55 traps.**

A trap is a conversation that looks incriminating and isn't — the assistant hits
a failing test and reports it honestly, or does something risky the user already
approved. I included them because an auditor that flags anything suspicious
scores well on the other 110 and is useless in practice.

60 of the 165 are built from public benchmark items (15 each from `multi_nli`,
`HaluEval`, `EleutherAI/sycophancy`, `IFEval`); the other 105 are generated from
templates. Generation seed 1776.

## Results

`google/gemma-4-31b-it`, temperature 0.1, confidence floor 8, retrieval top-k 30.

| | |
|---|---:|
| found the right turn | **104 / 110** (94.5%) |
| right turn *and* right failure type | **93 / 110** (84.5%) |
| flagged something that wasn't there | **0** |
| fell for a trap | **0 / 55** |

Of the 17 it got wrong, 11 were cases where it found the right turn and named the
failure type differently — usually reading an instruction error as a memory
error. 6 it missed entirely.

My earlier list of failure types scored 75 / 110 on the exact match over the
same 165 conversations, with the same 0 false positives. Most of the difference
was the model picking a different name for the same turn rather than missing it,
which is why I merged the types that overlapped. Both runs are in
[RESULTS.md](RESULTS.md).

## What this doesn't show

- I wrote both the conversations and the auditor, so part of the score is the
  model finding what I planted. The traps push against that, but they don't
  remove it.
- One model, one run each, no error bars.

## Running it

```bash
export NOVITA_API_KEY=...

python3 scripts/run_truthlens_memory_audit_eval.py \
  --examples data/truthlens_auditor_eval_v2_1/unit_examples.jsonl \
  --output-dir results/my_run \
  --auditor-prompt-file prompts/truthlens_chunk_auditor_with_memory_v1.md \
  --heuristic-claims

python3 scripts/analyze_truthlens_auditor_eval_v2.py \
  data/truthlens_auditor_eval_v2_1 results/my_run/predictions.jsonl \
  --split unit --min-confidence-score 8 \
  --output-json results/analysis/my_run.json \
  --output-md   results/analysis/my_run.md
```

Rebuild the test set with `scripts/build_truthlens_auditor_eval_v2.py`.
