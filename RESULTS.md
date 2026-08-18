# Every run

| run | model used? | conversations | right turn | + right type | false alarms | traps failed |
|---|---|---:|---:|---:|---:|---:|
| `truthlens_auditor_eval_v2_1_unit_novita` | yes | 165 | 104/110 | 93/110 | 0 | 0/55 |
| `truthlens_auditor_eval_v2_unit` | yes | 165 | 106/110 | 75/110 | 0 | 0/55 |
| `truthlens_auditor_eval_v2_1_packed_novita_2` | yes | 2 | 19/22 | 16/22 | 10 | 2/8 |
| `truthlens_auditor_eval_v2_1_packed_strict_novita_2` | yes | 2 | 19/22 | 17/22 | 0 | 0/8 |
| `truthlens_auditor_eval_v2_1_packed_dry` | no | 1 | 0/11 | 0/11 | 0 | 0/4 |
| `truthlens_auditor_eval_v2_1_unit_dry` | no | 1 | 0/1 | 0/1 | 0 | 0/0 |

**The `model used?` column matters.** Runs marked `no` have `dry_run: true` in
their `run_manifest.json` — the model was switched off and the pipeline ran
against stubs. They exist to test the scoring code and say nothing about how
well anything was caught.

The two 165-conversation runs use the same conversations (byte-identical, same
generation seed) and the same model. They differ only in the list of failure
types the auditor was asked to choose from.

Regenerate any row:

```bash
python3 scripts/analyze_truthlens_auditor_eval_v2.py \
  data/truthlens_auditor_eval_v2_1 results/<run>/predictions.jsonl \
  --split unit --min-confidence-score 8 \
  --output-json results/analysis/<run>.json --output-md results/analysis/<run>.md
```

Per-chunk raw model output lives under each run's `raw/` and is not committed.
