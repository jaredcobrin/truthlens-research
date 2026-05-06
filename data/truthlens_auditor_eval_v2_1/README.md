# TruthLens Auditor Eval V2.1

This dataset tests the TruthLens auditor prompt with mechanism-based
`primary_failure` labels, optional `secondary_tags`, and issue-level evidence.
It is built for fast prompt iteration without requiring 100 separate 256k-1m
token runs.

## Files

- `unit_examples.jsonl`: short category-discrimination examples.
- `packed_long_examples.jsonl`: dense 220k-320k token long-context examples.
- `label_manifest.json`: full ground truth labels and aggregate counts.
- `manifest.json`: compact generation manifest.

## Label Semantics

Each example has a `labels` array. Positive labels have
`expected_positive: true`; false-positive traps have `expected_positive: false`.
The important scoring fields are:

- `primary_failure`
- `secondary_tags`
- `legacy_category`
- `target_turn`
- `evidence_turns`
- `target_quote`
- `evidence_quotes`
- `chunk_relation`
- `source_dataset`
- `scoring_notes`

## Intended Use

Start with the unit set for cheap prompt debugging:

```bash
python3 scripts/validate_truthlens_auditor_eval_v2.py data/truthlens_auditor_eval_v2_1
```

Then run one packed example as a long-context smoke test before running all
packed examples.
