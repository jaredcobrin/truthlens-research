# TruthLens Auditor Eval Detailed Analysis (packed_long)

## Summary

- Examples in split: `2`
- Predicted examples: `2`
- Min confidence score: `8`
- Exact true positives: `17`
- False negatives: `5`
- False positives: `0`
- Turn-only wrong-primary matches: `2`
- Type-only wrong-turn matches: `0`
- Trap true negatives: `8`
- Trap false positives: `0`
- Strict precision: `100.0%`
- Strict recall: `77.27%`
- Strict F1: `87.18%`

## Primary Failure Table

| Primary failure | TP | FP | TN (traps) | FN | Turn-only wrong primary | Type-only wrong turn | Precision | Recall | F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| CONFLICTING_CLAIMS | 2 | 0 | 1 | 0 | 0 | 0 | 100.0% | 100.0% | 100.0% |
| DECISION_CRITICAL_OMISSION | 2 | 0 | 1 | 0 | 0 | 0 | 100.0% | 100.0% | 100.0% |
| INSTRUCTION_OR_CONSTRAINT_ERROR | 1 | 0 | 0 | 3 | 2 | 0 | 100.0% | 25.0% | 40.0% |
| REASONING_OR_BAD_PREMISE_ERROR | 8 | 0 | 2 | 0 | 0 | 0 | 100.0% | 100.0% | 100.0% |
| SOURCE_OR_EVIDENCE_ERROR | 1 | 0 | 2 | 1 | 0 | 0 | 100.0% | 50.0% | 66.67% |
| UNSUPPORTED_OR_OVERCONFIDENT_CLAIM | 1 | 0 | 2 | 1 | 0 | 0 | 100.0% | 50.0% | 66.67% |
| USER_CONTEXT_OR_MEMORY_ERROR | 2 | 0 | 0 | 0 | 0 | 0 | 100.0% | 100.0% | 100.0% |

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
    "USER_CONTEXT_OR_MEMORY_ERROR": 2
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
