# TruthLens Auditor Eval Detailed Analysis (unit)

## Summary

- Examples in split: `165`
- Predicted examples: `165`
- Min confidence score: `8`
- Exact true positives: `93`
- False negatives: `17`
- False positives: `0`
- Turn-only wrong-primary matches: `11`
- Type-only wrong-turn matches: `0`
- Trap true negatives: `55`
- Trap false positives: `0`
- Strict precision: `100.0%`
- Strict recall: `84.55%`
- Strict F1: `91.63%`

## Primary Failure Table

| Primary failure | TP | FP | TN (traps) | FN | Turn-only wrong primary | Type-only wrong turn | Precision | Recall | F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| CONFLICTING_CLAIMS | 9 | 0 | 5 | 1 | 0 | 0 | 100.0% | 90.0% | 94.74% |
| DECISION_CRITICAL_OMISSION | 9 | 0 | 5 | 1 | 0 | 0 | 100.0% | 90.0% | 94.74% |
| INSTRUCTION_OR_CONSTRAINT_ERROR | 9 | 0 | 10 | 11 | 10 | 0 | 100.0% | 45.0% | 62.07% |
| REASONING_OR_BAD_PREMISE_ERROR | 39 | 0 | 20 | 1 | 0 | 0 | 100.0% | 97.5% | 98.73% |
| SOURCE_OR_EVIDENCE_ERROR | 7 | 0 | 5 | 3 | 1 | 0 | 100.0% | 70.0% | 82.35% |
| UNSUPPORTED_OR_OVERCONFIDENT_CLAIM | 10 | 0 | 5 | 0 | 0 | 0 | 100.0% | 100.0% | 100.0% |
| USER_CONTEXT_OR_MEMORY_ERROR | 10 | 0 | 5 | 0 | 0 | 0 | 100.0% | 100.0% | 100.0% |

## Turn-Only Wrong-Primary Confusion

Rows are the expected primary failure. Values are how Gemma labeled the same target turn instead.

```json
{
  "CONFLICTING_CLAIMS": {
    "CONFLICTING_CLAIMS": 0,
    "DECISION_CRITICAL_OMISSION": 0,
    "INSTRUCTION_OR_CONSTRAINT_ERROR": 0,
    "REASONING_OR_BAD_PREMISE_ERROR": 0,
    "SOURCE_OR_EVIDENCE_ERROR": 0,
    "UNSUPPORTED_OR_OVERCONFIDENT_CLAIM": 0,
    "USER_CONTEXT_OR_MEMORY_ERROR": 0
  },
  "DECISION_CRITICAL_OMISSION": {
    "CONFLICTING_CLAIMS": 0,
    "DECISION_CRITICAL_OMISSION": 0,
    "INSTRUCTION_OR_CONSTRAINT_ERROR": 0,
    "REASONING_OR_BAD_PREMISE_ERROR": 0,
    "SOURCE_OR_EVIDENCE_ERROR": 0,
    "UNSUPPORTED_OR_OVERCONFIDENT_CLAIM": 0,
    "USER_CONTEXT_OR_MEMORY_ERROR": 0
  },
  "INSTRUCTION_OR_CONSTRAINT_ERROR": {
    "CONFLICTING_CLAIMS": 0,
    "DECISION_CRITICAL_OMISSION": 0,
    "INSTRUCTION_OR_CONSTRAINT_ERROR": 0,
    "REASONING_OR_BAD_PREMISE_ERROR": 0,
    "SOURCE_OR_EVIDENCE_ERROR": 0,
    "UNSUPPORTED_OR_OVERCONFIDENT_CLAIM": 0,
    "USER_CONTEXT_OR_MEMORY_ERROR": 10
  },
  "REASONING_OR_BAD_PREMISE_ERROR": {
    "CONFLICTING_CLAIMS": 0,
    "DECISION_CRITICAL_OMISSION": 0,
    "INSTRUCTION_OR_CONSTRAINT_ERROR": 0,
    "REASONING_OR_BAD_PREMISE_ERROR": 0,
    "SOURCE_OR_EVIDENCE_ERROR": 0,
    "UNSUPPORTED_OR_OVERCONFIDENT_CLAIM": 0,
    "USER_CONTEXT_OR_MEMORY_ERROR": 0
  },
  "SOURCE_OR_EVIDENCE_ERROR": {
    "CONFLICTING_CLAIMS": 0,
    "DECISION_CRITICAL_OMISSION": 0,
    "INSTRUCTION_OR_CONSTRAINT_ERROR": 0,
    "REASONING_OR_BAD_PREMISE_ERROR": 1,
    "SOURCE_OR_EVIDENCE_ERROR": 0,
    "UNSUPPORTED_OR_OVERCONFIDENT_CLAIM": 0,
    "USER_CONTEXT_OR_MEMORY_ERROR": 0
  },
  "UNSUPPORTED_OR_OVERCONFIDENT_CLAIM": {
    "CONFLICTING_CLAIMS": 0,
    "DECISION_CRITICAL_OMISSION": 0,
    "INSTRUCTION_OR_CONSTRAINT_ERROR": 0,
    "REASONING_OR_BAD_PREMISE_ERROR": 0,
    "SOURCE_OR_EVIDENCE_ERROR": 0,
    "UNSUPPORTED_OR_OVERCONFIDENT_CLAIM": 0,
    "USER_CONTEXT_OR_MEMORY_ERROR": 0
  },
  "USER_CONTEXT_OR_MEMORY_ERROR": {
    "CONFLICTING_CLAIMS": 0,
    "DECISION_CRITICAL_OMISSION": 0,
    "INSTRUCTION_OR_CONSTRAINT_ERROR": 0,
    "REASONING_OR_BAD_PREMISE_ERROR": 0,
    "SOURCE_OR_EVIDENCE_ERROR": 0,
    "UNSUPPORTED_OR_OVERCONFIDENT_CLAIM": 0,
    "USER_CONTEXT_OR_MEMORY_ERROR": 0
  }
}
```

## Notes

- `TN (traps)` means category-specific trap examples that the model correctly left unflagged.
- Confidence buckets and example-level details are in the JSON output.
