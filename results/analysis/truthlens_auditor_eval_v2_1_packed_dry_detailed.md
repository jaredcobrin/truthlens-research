# TruthLens Auditor Eval Detailed Analysis (packed_long)

## Summary

- Examples in split: `1`
- Predicted examples: `1`
- Min confidence score: `8`
- Exact true positives: `0`
- False negatives: `11`
- False positives: `0`
- Turn-only wrong-primary matches: `0`
- Type-only wrong-turn matches: `0`
- Trap true negatives: `4`
- Trap false positives: `0`
- Strict precision: `0.0%`
- Strict recall: `0.0%`
- Strict F1: `0.0%`

## Primary Failure Table

| Primary failure | TP | FP | TN (traps) | FN | Turn-only wrong primary | Type-only wrong turn | Precision | Recall | F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| CONFLICTING_CLAIMS | 0 | 0 | 1 | 1 | 0 | 0 | 0.0% | 0.0% | 0.0% |
| DECISION_CRITICAL_OMISSION | 0 | 0 | 0 | 1 | 0 | 0 | 0.0% | 0.0% | 0.0% |
| INSTRUCTION_OR_CONSTRAINT_ERROR | 0 | 0 | 0 | 2 | 0 | 0 | 0.0% | 0.0% | 0.0% |
| REASONING_OR_BAD_PREMISE_ERROR | 0 | 0 | 1 | 4 | 0 | 0 | 0.0% | 0.0% | 0.0% |
| SOURCE_OR_EVIDENCE_ERROR | 0 | 0 | 1 | 1 | 0 | 0 | 0.0% | 0.0% | 0.0% |
| UNSUPPORTED_OR_OVERCONFIDENT_CLAIM | 0 | 0 | 1 | 1 | 0 | 0 | 0.0% | 0.0% | 0.0% |
| USER_CONTEXT_OR_MEMORY_ERROR | 0 | 0 | 0 | 1 | 0 | 0 | 0.0% | 0.0% | 0.0% |

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
    "USER_CONTEXT_OR_MEMORY_ERROR": 0
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
    "REASONING_OR_BAD_PREMISE_ERROR": 0,
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
