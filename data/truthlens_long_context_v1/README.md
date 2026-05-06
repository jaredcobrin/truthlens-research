# TruthLens Long-Context Benchmark V1

This folder contains a generated Pilot 100 benchmark for testing TruthLens
contradiction detection above a 256k-token context window.

## Files

- `examples.jsonl`: one complete conversation-shaped benchmark example per line.
- `manifest.json`: generation settings, source datasets, and validation summary.

## Categories

- `clean_no_issue`
- `local_contradiction`
- `cross_chunk_contradiction`
- `far_distance_contradiction`
- `allowed_revision`

## Validate

```bash
python3 scripts/validate_long_context_benchmark.py data/truthlens_long_context_v1/examples.jsonl
```

## Score Predictions

Create a predictions JSONL file with at least:

```json
{"id":"tlc_v1_local_contradiction_300k_000","predicted_has_issue":true,"predicted_type":"CONTRADICTION","predicted_target_turn":42}
```

Then run:

```bash
python3 scripts/score_long_context_predictions.py data/truthlens_long_context_v1/examples.jsonl predictions.jsonl
```

## Run Gemma Evaluation

Set your Google AI Studio key in the shell:

```bash
export GOOGLE_AI_STUDIO_KEY="your-key-here"
```

Run a smoke test first:

```bash
python3 scripts/run_gemma_long_context_eval.py --limit 1
```

Run the full Pilot 100:

```bash
python3 scripts/run_gemma_long_context_eval.py
```

The runner writes:

- `results/truthlens_long_context_v1/predictions.jsonl`
- `results/truthlens_long_context_v1/raw/<example-id>.json`

Score the completed run:

```bash
python3 scripts/score_long_context_predictions.py \
  data/truthlens_long_context_v1/examples.jsonl \
  results/truthlens_long_context_v1/predictions.jsonl
```

The runner is resumable. If it stops because of rate limits or network errors,
rerun the same command and it will skip completed example IDs.

If your Google AI Studio quota is limited to about 16k input tokens per minute,
the default 180k-token chunk mode will fail. You can run a quota-safe fallback,
but it is much slower and does not test the intended 180k chunk size:

```bash
python3 scripts/run_gemma_long_context_eval.py \
  --chunk-max-tokens 12000 \
  --chunk-overlap-turns 0 \
  --sleep-seconds 65
```
